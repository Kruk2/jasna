"""Inference-only SPAN (Swift Parameter-free Attention Network, Wan et al., CVPR 2024 NTIRE).

Each re-parameterizable ``Conv3XC`` of the training network is stored already folded into
the single 3x3 convolution it computes at inference, so the network is plain convolutions,
SiLU and the parameter-free attention.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

SPAN_BLOCKS = 6
SPAN_RGB_MEAN = (0.4488, 0.4371, 0.4040)


def _conv(weights: Mapping[str, torch.Tensor], name: str) -> nn.Conv2d:
    weight = weights[f"{name}.weight"]
    out_channels, in_channels, kernel, _ = weight.shape
    conv = nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2, device="meta")
    conv.weight = nn.Parameter(weight, requires_grad=False)
    conv.bias = nn.Parameter(weights[f"{name}.bias"], requires_grad=False)
    return conv


class Span(nn.Module):
    """RGB in ``[0, 1]`` (``[N, 3, H, W]``) to RGB in about ``[0, 1]`` at ``scale`` times the size."""

    def __init__(self, weights: Mapping[str, torch.Tensor], *, scale: int) -> None:
        super().__init__()
        self.scale = scale
        self.register_buffer("mean", torch.tensor(SPAN_RGB_MEAN).view(1, 3, 1, 1), persistent=False)
        self.conv_1 = _conv(weights, "conv_1")
        self.blocks = nn.ModuleList(
            nn.ModuleList(_conv(weights, f"block_{block}.c{index}") for index in (1, 2, 3))
            for block in range(1, SPAN_BLOCKS + 1)
        )
        self.conv_2 = _conv(weights, "conv_2")
        self.conv_cat = _conv(weights, "conv_cat")
        self.upsampler = _conv(weights, "upsampler")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.conv_1((image - self.mean) * 255.0)
        hidden = features
        first_block = None
        for c1, c2, c3 in self.blocks:
            activated = F.silu(c1(hidden))
            out = c3(F.silu(c2(activated)))
            hidden = (out + hidden) * (torch.sigmoid(out) - 0.5)
            if first_block is None:
                first_block = hidden
        merged = self.conv_cat(torch.cat([features, self.conv_2(hidden), first_block, activated], 1))
        return F.pixel_shuffle(self.upsampler(merged), self.scale)


def load_span(path: Path, *, scale: int) -> Span:
    from safetensors.torch import load_file

    return Span(load_file(str(path)), scale=scale).eval()
