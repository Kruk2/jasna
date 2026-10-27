"""Inference-only UNet 4x, with weights read from its ONNX export."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def resize(x: torch.Tensor, scale: float, mode: str = "bilinear") -> torch.Tensor:
    # The export resizes in float32, then rounds back to the network's dtype.
    options = {} if mode == "nearest" else {"align_corners": False}
    return F.interpolate(x.float(), scale_factor=scale, mode=mode, **options).to(x.dtype)


def match_color(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    image = image.float()
    reference = reference.float()
    mean = image.mean((2, 3), keepdim=True)
    ref_mean = reference.mean((2, 3), keepdim=True)
    centered = image - mean
    std = centered.square().mean((2, 3), keepdim=True).sqrt().clamp_min(1e-8)
    ref_std = (reference - ref_mean).square().mean((2, 3), keepdim=True).sqrt()
    matched = centered * (ref_std / std) + ref_mean
    return (image + (matched - image) * 0.8).clamp(0, 1)


class Unet4x(nn.Module):
    def __init__(self, onnx_bytes: bytes) -> None:
        super().__init__()
        import onnx
        from onnx.numpy_helper import to_array

        graph = onnx.load_model_from_string(onnx_bytes).graph
        convolutions = [node for node in graph.node if node.op_type == "Conv"]
        transposes = [node for node in graph.node if node.op_type == "ConvTranspose"]
        shapes = [[dim.dim_value for dim in value.type.tensor_type.shape.dim] for value in graph.input]
        if len(convolutions) != 60 or len(transposes) != 4 or shapes != [
            [1, 256, 256, 3], [1, 256, 256, 3], [1, 1024, 1024, 3],
        ]:
            raise ValueError("Unsupported UNet 4x ONNX architecture")
        tensors = {value.name: torch.from_numpy(to_array(value).copy()) for value in graph.initializer}
        conv_iter = iter(convolutions)
        transpose_iter = iter(transposes)

        def convolution(*, reflect: bool = False, transpose: bool = False) -> nn.Module:
            node = next(transpose_iter if transpose else conv_iter)
            attributes = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
            weight, bias = (tensors[name] for name in node.input[1:])
            kernel = tuple(weight.shape[2:])
            stride = tuple(attributes["strides"])
            padding = (1, 1) if reflect else tuple(attributes["pads"][:2])
            if transpose:
                layer = nn.ConvTranspose2d(weight.shape[0], weight.shape[1], kernel, stride, padding, device="meta")
            else:
                layer = nn.Conv2d(
                    weight.shape[1], weight.shape[0], kernel, stride, padding,
                    padding_mode="reflect" if reflect else "zeros", device="meta",
                )
            layer.weight = nn.Parameter(weight, requires_grad=False)
            layer.bias = nn.Parameter(bias, requires_grad=False)
            return layer

        self.flow = nn.ModuleList(convolution() for _ in range(17))
        self.encoder = nn.ModuleList()
        self.down = nn.ModuleList()
        for _ in range(4):
            self.encoder.append(nn.ModuleList(convolution(reflect=True) for _ in range(2)))
            self.down.append(convolution())
        self.bottleneck = nn.ModuleList(convolution(reflect=True) for _ in range(3))
        self.up = nn.ModuleList(convolution(transpose=True) for _ in range(4))
        self.decoder = nn.ModuleList()
        self.heads = nn.ModuleList()
        for _ in range(4):
            self.decoder.append(nn.ModuleList(convolution(reflect=True) for _ in range(2)))
            self.heads.append(convolution(reflect=True))
        self.super_resolution = nn.ModuleList(convolution(reflect=True) for _ in range(3))
        self.color_correction = convolution()
        self.refine = nn.ModuleList(
            nn.ModuleList(convolution(reflect=True) for _ in range(3)) for _ in range(4)
        )

        prelu = next(node for node in graph.node if node.op_type == "PRelu")
        self.register_buffer("slope", tensors[prelu.input[1]])
        grid_sample = next(node for node in graph.node if node.op_type == "GridSample")
        producers = {node.output[0]: node for node in graph.node}
        grid_add = producers[producers[grid_sample.input[1]].input[0]]
        self.register_buffer("grid", tensors[grid_add.input[0]])
        self.eval()

    def activate(self, x: torch.Tensor) -> torch.Tensor:
        return F.prelu(x, self.slope)

    def forward(self, lr_prev: torch.Tensor, lr_curr: torch.Tensor, hr_prev: torch.Tensor) -> torch.Tensor:
        previous = lr_prev.permute(0, 3, 1, 2)
        current = lr_curr.permute(0, 3, 1, 2)
        high_previous = hr_prev.permute(0, 3, 1, 2)

        flow = torch.cat((current, previous), dim=1)
        for index in range(0, 6, 2):
            flow = self.activate(self.flow[index](flow))
            flow = F.avg_pool2d(self.activate(self.flow[index + 1](flow)), 2)
        flow = self.activate(self.flow[6](flow))
        flow = flow + self.flow[8](self.activate(self.flow[7](flow)))
        flow = flow + self.flow[10](self.activate(self.flow[9](flow)))
        for index in (11, 13):
            flow = self.activate(self.flow[index](resize(flow, 2, "nearest")))
            flow = self.activate(self.flow[index + 1](flow))
        flow = self.activate(self.flow[15](resize(flow, 2, "nearest")))
        flow = self.flow[16](flow).tanh() * 32
        flow = resize(flow, 4) * 4
        grid = (self.grid + flow / 511.5).permute(0, 2, 3, 1)
        warped = F.grid_sample(high_previous, grid, padding_mode="border", align_corners=True)
        packed = F.pixel_unshuffle(warped, 4)
        empty = current.new_zeros((1, 6, 256, 256))
        features = torch.cat((current, empty, packed), dim=1)
        skips = []
        for block, down in zip(self.encoder, self.down):
            for conv in block:
                features = self.activate(conv(features))
            skips.append(features)
            features = self.activate(down(features))
        for conv in self.bottleneck:
            features = self.activate(conv(features))
        heads = []
        for up, block, head, skip in zip(self.up, self.decoder, self.heads, reversed(skips)):
            features = self.activate(up(features))
            zeros = current.new_zeros((1, 6, features.shape[2], features.shape[3]))
            features = torch.cat((features, skip, zeros), dim=1)
            for conv in block:
                features = self.activate(conv(features))
            heads.append(head(features).tanh())
        for conv in self.super_resolution[:2]:
            features = self.activate(conv(features))
        features = F.pixel_shuffle(self.super_resolution[2](features), 4)
        high = self.color_correction(features).tanh()

        pyramid = [current]
        for _ in range(4):
            pyramid.append(resize(pyramid[-1], 0.5))
        refined = resize(pyramid[-1], 2)
        for block, head, reference in zip(self.refine, heads, reversed(pyramid[:-1])):
            refined = torch.cat((head, refined, reference), dim=1)
            refined = self.activate(block[0](refined))
            refined = self.activate(block[1](refined))
            refined = block[2](refined)
            refined = resize(refined, 2 if reference.shape[2] < 256 else 4)
        image = (high + refined).clamp(0, 1)
        reference = F.interpolate(current.float(), size=(1024, 1024), mode="bicubic", align_corners=False)
        return match_color(image, reference).to(current.dtype).permute(0, 2, 3, 1)
