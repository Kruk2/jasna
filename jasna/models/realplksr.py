"""Inference-only RealPLKSR (Partial Large Kernel CNNs for Efficient Super-Resolution, Lee et
al. 2024), the neosr ``realplksr`` variant with GroupNorm and a pixel-shuffle output."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

REALPLKSR_NORM_GROUPS = 4


def _conv(weights: Mapping[str, torch.Tensor], name: str) -> nn.Conv2d:
    weight = weights[f"{name}.weight"]
    out_channels, in_channels, kernel, _ = weight.shape
    conv = nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2, device="meta")
    conv.weight = nn.Parameter(weight, requires_grad=False)
    conv.bias = nn.Parameter(weights[f"{name}.bias"], requires_grad=False)
    return conv


class _Block(nn.Module):
    def __init__(self, weights: Mapping[str, torch.Tensor], name: str) -> None:
        super().__init__()
        self.mix_in = _conv(weights, f"{name}.channel_mixer.0")
        self.mix_out = _conv(weights, f"{name}.channel_mixer.2")
        self.large_kernel = _conv(weights, f"{name}.lk.conv")
        self.attention = _conv(weights, f"{name}.attn.f.0")
        self.refine = _conv(weights, f"{name}.refine")
        dim = self.refine.out_channels
        self.norm = nn.GroupNorm(REALPLKSR_NORM_GROUPS, dim, device="meta")
        self.norm.weight = nn.Parameter(weights[f"{name}.norm.weight"], requires_grad=False)
        self.norm.bias = nn.Parameter(weights[f"{name}.norm.bias"], requires_grad=False)
        self.partial = self.large_kernel.in_channels

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        mixed = self.mix_out(F.mish(self.mix_in(features)))
        mixed = torch.cat([self.large_kernel(mixed[:, : self.partial]), mixed[:, self.partial :]], 1)
        mixed = mixed * torch.sigmoid(self.attention(mixed))
        return self.norm(self.refine(mixed)) + features


class RealPlksr(nn.Module):
    """RGB in ``[0, 1]`` (``[N, 3, H, W]``) to RGB at ``scale`` times the size."""

    def __init__(self, weights: Mapping[str, torch.Tensor], *, scale: int) -> None:
        super().__init__()
        self.scale = scale
        blocks = sorted({int(name.split(".")[1]) for name in weights if ".lk." in name})
        self.head = _conv(weights, "feats.0")
        self.blocks = nn.ModuleList(_Block(weights, f"feats.{index}") for index in blocks)
        # The training network has a dropout layer between the last block and the tail.
        self.tail = _conv(weights, f"feats.{blocks[-1] + 2}")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.head(image)
        for block in self.blocks:
            features = block(features)
        residual = image.repeat_interleave(self.scale**2, dim=1)
        return F.pixel_shuffle(self.tail(features) + residual, self.scale)


def load_realplksr(path: Path, *, scale: int) -> RealPlksr:
    from safetensors.torch import load_file

    return RealPlksr(load_file(str(path)), scale=scale).eval()
