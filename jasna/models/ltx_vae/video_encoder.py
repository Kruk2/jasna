"""Causal 3D-conv video VAE encoder (``ltx_core/model/video_vae/{video_vae,convolution,resnet,sampling}.py``)."""

import logging
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from jasna.models.ltx_vae.ops import PerChannelStatistics, PixelNorm, patchify
from jasna.models.ltx_vae.tiling import SpatioTemporalScaleFactors
from jasna.tensor_utils import compile_static

logger = logging.getLogger(__name__)

ENCODER_COMPILE_MODE = "default"
FP16_DOWN_BLOCKS = 7
"""With the real weights the activations up to ``down_blocks[7]`` stay below ~1e3; the
low-resolution tail after them reaches ~2e5, past fp16's range."""
KITCHEN_DOWN_BLOCKS = 1
"""On AMD comfy-kitchen's fp16 conv3d beats MIOpen only on the 128-channel full-resolution layers
(``conv_in`` and ``down_blocks[0]``), and needs no workspace there, where MIOpen's first-run solver
search asks for ~13 GiB at 512 px. MIOpen's bf16 conv3d is faster on every later shape."""

_kitchen_conv3d = False


class CausalConv3d(nn.Module):
    """3x3x3 conv; time is padded causally by repeating the first frame."""

    time_kernel_size = 3

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 3, padding=(0, 1, 1), padding_mode="zeros")

    def _causal_frames(self, x: torch.Tensor) -> torch.Tensor:
        # Views of x keep its memory format: a contiguous ``repeat`` beside a channels-last x makes
        # the concatenation a strided transpose, 60x slower on ROCm (an eager encoder 2.26 -> 1.40 s).
        first_frame = x[:, :, :1, :, :]
        return torch.concatenate((first_frame,) * (self.time_kernel_size - 1) + (x,), dim=2)

    def _runs_on_kitchen(self) -> bool:
        return _kitchen_conv3d and self.conv.weight.dtype == torch.float16

    def _kitchen_forward(self, x: torch.Tensor, residual: torch.Tensor | None) -> torch.Tensor:
        padded = F.pad(self._causal_frames(x), (1, 1, 1, 1)).contiguous(memory_format=torch.channels_last_3d)
        return torch.ops.comfy_kitchen.fp16_conv3d(padded, self.conv.weight, self.conv.bias, residual, [1, 1, 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._runs_on_kitchen():
            return self._kitchen_forward(x, None)
        return self.conv(self._causal_frames(x))

    def add_to(self, residual: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """``residual + self(x)``; kitchen adds the residual in the conv's epilogue."""
        if self._runs_on_kitchen():
            return self._kitchen_forward(x, residual)
        return residual + self(x)


class ResnetBlock3D(nn.Module):
    """PixelNorm -> SiLU -> conv, twice, plus identity skip."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = PixelNorm()
        self.non_linearity = nn.SiLU()
        self.conv1 = CausalConv3d(channels, channels)
        self.norm2 = PixelNorm()
        self.conv2 = CausalConv3d(channels, channels)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm1(input_tensor)
        hidden_states = self.non_linearity(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.non_linearity(hidden_states)
        return self.conv2.add_to(input_tensor, hidden_states)


class UNetMidBlock3D(nn.Module):
    def __init__(self, channels: int, num_layers: int) -> None:
        super().__init__()
        self.res_blocks = nn.ModuleList([ResnetBlock3D(channels) for _ in range(num_layers)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.res_blocks:
            hidden_states = resnet(hidden_states)
        return hidden_states


def space_to_depth(x: torch.Tensor, stride: tuple[int, int, int]) -> torch.Tensor:
    """``b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w``."""
    b, c, d, h, w = x.shape
    p1, p2, p3 = stride
    x = x.reshape(b, c, d // p1, p1, h // p2, p2, w // p3, p3).permute(0, 1, 3, 5, 7, 2, 4, 6)
    return x.reshape(b, c * p1 * p2 * p3, d // p1, h // p2, w // p3)


class SpaceToDepthDownsample(nn.Module):
    """Strided downsample: conv + space-to-depth, plus a group-mean space-to-depth skip."""

    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int, int]) -> None:
        super().__init__()
        self.stride = stride
        self.group_size = in_channels * math.prod(stride) // out_channels
        self.conv = CausalConv3d(in_channels, out_channels // math.prod(stride))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride[0] == 2:
            x = torch.cat([x[:, :, :1, :, :], x], dim=2)

        x_in = space_to_depth(x, self.stride)
        b, c, d, h, w = x_in.shape
        x_in = x_in.view(b, c // self.group_size, self.group_size, d, h, w).mean(dim=2)

        x = self.conv(x)
        x = space_to_depth(x, self.stride)
        return x + x_in


_DOWNSAMPLE_STRIDES = {
    "compress_space_res": (1, 2, 2),
    "compress_time_res": (2, 1, 1),
    "compress_all_res": (2, 2, 2),
}


def _make_encoder_block(block_name: str, block_config: dict[str, Any], in_channels: int) -> tuple[nn.Module, int]:
    if block_name == "res_x":
        return UNetMidBlock3D(in_channels, block_config["num_layers"]), in_channels
    if block_name in _DOWNSAMPLE_STRIDES:
        out_channels = in_channels * block_config.get("multiplier", 2)
        return SpaceToDepthDownsample(in_channels, out_channels, _DOWNSAMPLE_STRIDES[block_name]), out_channels
    raise ValueError(f"unsupported encoder block: {block_name}")


class VideoEncoder(nn.Module):
    """Encodes ``(B, 3, F, H, W)`` video in [-1, 1] into normalized bf16 latent means.

    ``F' = 1 + (F-1)/8, H' = H/32, W' = W/32``; F must be ``1 + 8k`` (extra frames are cropped).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        encoder_blocks: list[tuple[str, dict[str, Any]]],
        patch_size: int,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.latent_channels = out_channels
        self.video_scale_factors = SpatioTemporalScaleFactors.from_blocks(encoder_blocks, patch_size)
        self.per_channel_statistics = PerChannelStatistics(latent_channels=out_channels)

        feature_channels = out_channels
        self.conv_in = CausalConv3d(in_channels * patch_size**2, feature_channels)

        self.down_blocks = nn.ModuleList([])
        for block_name, block_config in encoder_blocks:
            block, feature_channels = _make_encoder_block(block_name, block_config, feature_channels)
            self.down_blocks.append(block)

        self.conv_norm_out = PixelNorm()
        self.conv_act = nn.SiLU()
        # One extra channel: the checkpoint's (unused) constant log-variance.
        self.conv_out = CausalConv3d(feature_channels, out_channels + 1)

    def to_fp16_head(self, down_blocks: int) -> "VideoEncoder":
        """fp16 for ``conv_in`` and the first ``down_blocks`` down blocks; the rest keeps its dtype."""
        for layer in (self.conv_in, *self.down_blocks[:down_blocks]):
            layer.half()
        return self

    def to_fp16_with_float32_tail(self) -> "VideoEncoder":
        """fp16 for the high-resolution layers (nearly all the FLOPs), fp32 for the tail."""
        return self.float().to_fp16_head(FP16_DOWN_BLOCKS)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        temporal_factor = self.video_scale_factors.time
        frames_count = sample.shape[2]
        if ((frames_count - 1) % temporal_factor) != 0:
            frames_to_crop = (frames_count - 1) % temporal_factor
            logger.warning(
                "Invalid number of frames %s for encode; cropping last %s frames to satisfy 1 + %s*k.",
                frames_count,
                frames_to_crop,
                temporal_factor,
            )
            sample = sample[:, :, :-frames_to_crop, ...]

        encode = _encode_cropped if sample.is_cuda else encode_cropped
        return encode(self, sample.to(_weight_dtype(self.conv_in))).to(torch.bfloat16)


def _weight_dtype(layer: nn.Module) -> torch.dtype:
    return next(layer.parameters()).dtype


def _layer_input(sample: torch.Tensor, layer: nn.Module) -> torch.Tensor:
    """``sample`` in ``layer``'s weight dtype, and contiguous (in the same copy) for a layer whose
    conv weights are, as in an eager AMD encoder's tail."""
    weight = next(layer.parameters())
    if weight.is_contiguous(memory_format=torch.channels_last_3d):
        return sample.to(weight.dtype)
    if sample.is_cuda and sample.dtype == torch.float16 and weight.dtype == torch.bfloat16:
        from jasna.models.ltx_vae.layout_hip import channels_first_bf16

        return channels_first_bf16(sample)
    return sample.to(dtype=weight.dtype, memory_format=torch.contiguous_format)


def encode_cropped(encoder: VideoEncoder, sample: torch.Tensor) -> torch.Tensor:
    """Latent means of ``sample`` (in ``conv_in``'s dtype); every later layer computes in its
    own weights' dtype and memory format."""
    sample = patchify(sample, patch_size_hw=encoder.patch_size)
    sample = encoder.conv_in(sample)

    for down_block in encoder.down_blocks:
        sample = down_block(_layer_input(sample, down_block))

    sample = _layer_input(sample, encoder.conv_out)
    sample = encoder.conv_norm_out(sample)
    sample = encoder.conv_act(sample)
    sample = encoder.conv_out(sample)

    means = sample[:, :-1, ...]
    return encoder.per_channel_statistics.normalize(means)


_encode_cropped = encode_cropped


def enable_kitchen_conv3d() -> None:
    """Run the fp16 layers' convs through comfy-kitchen's ``fp16_conv3d`` (an opaque op, so a
    compiled encoder keeps one graph)."""
    global _kitchen_conv3d
    import comfy_kitchen  # noqa: F401

    _kitchen_conv3d = True


def compile_video_encoder(device: torch.device, dtype: torch.dtype) -> None:
    """Compile the encoder's layers (computing in ``dtype``) with static shapes (one graph per
    window shape); CUDA inputs use it from then on."""
    global _encode_cropped
    if _encode_cropped is encode_cropped:
        _encode_cropped = compile_static(encode_cropped, mode=ENCODER_COMPILE_MODE, device=device, dtype=dtype)
