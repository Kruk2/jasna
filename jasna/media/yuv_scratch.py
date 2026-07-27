"""Reusable working buffers for the eager RGB↔YUV conversions.

NVIDIA converts with a fused kernel that writes straight into caller-owned
planes. Everywhere else the conversion is eager Torch math, running from decode
and encode worker threads while the restorer allocates and frees fp16 tensors.
Allocating conversion temporaries per frame let ROCm hand those blocks to the
restorer mid-conversion, which replaced whole frames with fp16 garbage (issue
#252), so the eager path builds its working set once and reuses it.
"""
from __future__ import annotations

import torch


class YuvScratch:
    """Float32 working set for one RGB→NV12/P010 frame size."""

    def __init__(self, height: int, width: int, device: torch.device):
        self.height = height
        self.width = width
        self.device = device
        self.yuv = torch.empty((3, height, width), dtype=torch.float32, device=device)
        self.chroma = torch.empty(
            (2, height // 2, width // 2), dtype=torch.float32, device=device
        )

    def matches(self, height: int, width: int, device: torch.device) -> bool:
        return (self.height, self.width, self.device) == (height, width, device)


def apply_matrix(rgb: torch.Tensor, rows, offsets, out: torch.Tensor) -> None:
    """out[i] = rows[i] · rgb + offsets[i], without BLAS or temporaries.

    Integer frames carry 0..255 codes, so the 1/255 normalization folds into the
    coefficients instead of materializing a second full-frame float buffer.
    """
    scale = 1.0 if rgb.dtype.is_floating_point else 1.0 / 255.0
    for plane, (red, green, blue) in enumerate(rows):
        torch.mul(rgb[0], red * scale, out=out[plane])
        out[plane].add_(rgb[1], alpha=green * scale)
        out[plane].add_(rgb[2], alpha=blue * scale)
        out[plane].add_(offsets[plane])


def average_quads(planes: torch.Tensor, out: torch.Tensor) -> None:
    """4:2:0 subsampling: mean of each 2x2 quad, in the kernel's summation order."""
    torch.add(planes[:, 0::2, 0::2], planes[:, 0::2, 1::2], out=out)
    out.add_(planes[:, 1::2, 0::2])
    out.add_(planes[:, 1::2, 1::2])
    out.mul_(0.25)


def interleave_chroma(subsampled: torch.Tensor, chroma: torch.Tensor) -> None:
    """Write (2, H/2, W/2) U and V into a (H/2, W) plane as alternating samples."""
    half_width = subsampled.shape[2]
    chroma.unflatten(-1, (half_width, 2)).copy_(subsampled.permute(1, 2, 0))
