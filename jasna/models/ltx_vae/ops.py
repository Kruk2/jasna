"""Shared VAE ops: patchify, per-channel latent statistics, PixelNorm."""

import torch
from torch import nn


def patchify(x: torch.Tensor, patch_size_hw: int) -> torch.Tensor:
    """Space-to-depth ``(B, C, F, H, W) -> (B, C*p*p, F, H/p, W/p)``; channel order ``(c, w-patch, h-patch)``."""
    b, c, f, h, w = x.shape
    p = patch_size_hw
    x = x.reshape(b, c, f, h // p, p, w // p, p).permute(0, 1, 6, 4, 2, 3, 5)
    return x.reshape(b, c * p * p, f, h // p, w // p)


def unpatchify(x: torch.Tensor, patch_size_hw: int) -> torch.Tensor:
    """Inverse of :func:`patchify`."""
    b, cpp, f, h, w = x.shape
    p = patch_size_hw
    x = x.reshape(b, cpp // (p * p), p, p, f, h, w).permute(0, 1, 4, 5, 3, 6, 2)
    return x.reshape(b, cpp // (p * p), f, h * p, w * p)


class PerChannelStatistics(nn.Module):
    """Dataset per-channel latent statistics used to (un)normalize latents."""

    def __init__(self, latent_channels: int):
        super().__init__()
        self.register_buffer("std-of-means", torch.ones(latent_channels))
        self.register_buffer("mean-of-means", torch.zeros(latent_channels))

    def un_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(x)) + self.get_buffer("mean-of-means").view(
            1, -1, 1, 1, 1
        ).to(x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(x)) / self.get_buffer("std-of-means").view(
            1, -1, 1, 1, 1
        ).to(x)


class PixelNorm(nn.Module):
    """Per-location RMS normalization over the channel dimension."""

    eps = 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_sq = torch.mean(x**2, dim=1, keepdim=True)
        rms = torch.sqrt(mean_sq + self.eps)
        return x / rms
