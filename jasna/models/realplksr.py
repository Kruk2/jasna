"""Inference-only RealPLKSR (Partial Large Kernel CNNs for Efficient Super-Resolution, Lee et
al. 2024), the neosr ``realplksr`` variant with GroupNorm and a pixel-shuffle output."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from jasna.models.safetensors_conv import conv_from_weights

REALPLKSR_NORM_GROUPS = 4
# On bright, high-contrast detail the large-kernel conv outputs exceed fp16's 65504 (~1.2e5 seen at
# 720p); the overflow reaches the group norm and turns the whole frame into NaN. Those channels are
# stored divided by this and the convs reading them multiplied by it: same fp32 result, fp16 headroom.
LARGE_KERNEL_SHRINK = 16


def group_norm(features: torch.Tensor, norm: nn.GroupNorm) -> torch.Tensor:
    """``norm(features)`` as one scale and shift per channel from fp32 group statistics: ROCm's
    group norm kernel takes about 4x as long. ONNX exports keep the GroupNorm, which TensorRT fuses
    (1.6x faster than this form there)."""
    if torch.onnx.is_in_onnx_export():
        return norm(features)
    batch, channels = features.shape[:2]
    groups = norm.num_groups
    var, mean = torch.var_mean(features.reshape(batch, groups, -1).float(), dim=2, keepdim=True, correction=0)
    scale = norm.weight.float().view(1, groups, -1) * torch.rsqrt(var + norm.eps)
    shift = norm.bias.float().view(1, groups, -1) - mean * scale
    return torch.addcmul(
        shift.reshape(batch, channels, 1, 1).to(features.dtype), features, scale.reshape(batch, channels, 1, 1).to(features.dtype)
    )


class _Block(nn.Module):
    def __init__(self, weights: Mapping[str, torch.Tensor], name: str) -> None:
        super().__init__()
        self.mix_in = conv_from_weights(weights, f"{name}.channel_mixer.0")
        self.mix_out = conv_from_weights(weights, f"{name}.channel_mixer.2")
        self.large_kernel = conv_from_weights(weights, f"{name}.lk.conv")
        self.attention = conv_from_weights(weights, f"{name}.attn.f.0")
        self.refine = conv_from_weights(weights, f"{name}.refine")
        dim = self.refine.out_channels
        self.norm = nn.GroupNorm(REALPLKSR_NORM_GROUPS, dim, device="meta")
        self.norm.weight = nn.Parameter(weights[f"{name}.norm.weight"], requires_grad=False)
        self.norm.bias = nn.Parameter(weights[f"{name}.norm.bias"], requires_grad=False)
        self.partial = self.large_kernel.in_channels
        self.large_kernel.weight = nn.Parameter(self.large_kernel.weight / LARGE_KERNEL_SHRINK, requires_grad=False)
        self.large_kernel.bias = nn.Parameter(self.large_kernel.bias / LARGE_KERNEL_SHRINK, requires_grad=False)
        regrow = torch.ones(dim, 1, 1, dtype=self.refine.weight.dtype)
        regrow[: self.partial] = LARGE_KERNEL_SHRINK
        for conv in (self.attention, self.refine):
            conv.weight = nn.Parameter(conv.weight * regrow, requires_grad=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        mixed = self.mix_out(F.mish(self.mix_in(features)))
        mixed = torch.cat([self.large_kernel(mixed[:, : self.partial]), mixed[:, self.partial :]], 1)
        mixed = mixed * torch.sigmoid(self.attention(mixed))
        return group_norm(self.refine(mixed), self.norm) + features


class RealPlksr(nn.Module):
    """RGB in ``[0, 1]`` (``[N, 3, H, W]``) to RGB at ``scale`` times the size."""

    def __init__(self, weights: Mapping[str, torch.Tensor], *, scale: int) -> None:
        super().__init__()
        self.scale = scale
        blocks = sorted({int(name.split(".")[1]) for name in weights if ".lk." in name})
        self.head = conv_from_weights(weights, "feats.0")
        self.blocks = nn.ModuleList(_Block(weights, f"feats.{index}") for index in blocks)
        # The training network has a dropout layer between the last block and the tail.
        self.tail = conv_from_weights(weights, f"feats.{blocks[-1] + 2}")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.head(image)
        for block in self.blocks:
            features = block(features)
        residual = image.repeat_interleave(self.scale**2, dim=1)
        return F.pixel_shuffle(self.tail(features) + residual, self.scale)


def load_realplksr(path: Path, *, scale: int) -> RealPlksr:
    from safetensors.torch import load_file

    return RealPlksr(load_file(str(path)), scale=scale).eval()
