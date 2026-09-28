"""Causal 3D-conv video VAE encoder (``ltx_core/model/video_vae/{video_vae,convolution,resnet,sampling}.py``)."""

import logging
import math
from typing import Any

import torch
from torch import nn

from jasna.models.ltx_vae.ops import PerChannelStatistics, PixelNorm, patchify
from jasna.models.ltx_vae.tiling import SpatioTemporalScaleFactors

logger = logging.getLogger(__name__)


class CausalConv3d(nn.Module):
    """3x3x3 conv; time is padded causally by repeating the first frame."""

    time_kernel_size = 3

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 3, padding=(0, 1, 1), padding_mode="zeros")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_frame_pad = x[:, :, :1, :, :].repeat((1, 1, self.time_kernel_size - 1, 1, 1))
        x = torch.concatenate((first_frame_pad, x), dim=2)
        return self.conv(x)


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
        hidden_states = self.conv2(hidden_states)
        return input_tensor + hidden_states


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
    """Encodes ``(B, 3, F, H, W)`` video in [-1, 1] into normalized latent means.

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

        sample = patchify(sample, patch_size_hw=self.patch_size)
        sample = self.conv_in(sample)

        for down_block in self.down_blocks:
            sample = down_block(sample)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        means = sample[:, :-1, ...]
        return self.per_channel_statistics.normalize(means)
