from __future__ import annotations

from enum import Enum
import torch
import torch.nn.functional as F


class DenoiseStrength(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


_DENOISE_PARAMS: dict[DenoiseStrength, tuple[int, float, float]] = {
    DenoiseStrength.LOW: (5, 1.0, 0.04),
    DenoiseStrength.MEDIUM: (5, 1.5, 0.07),
    DenoiseStrength.HIGH: (5, 2.0, 0.09),
}


def spatial_denoise(
    frames: torch.Tensor, kernel_size: int, sigma_spatial: float, sigma_range: float
) -> torch.Tensor:
    """Spatial bilateral filter on GPU (batched over the T dimension).

    Each pixel is replaced by a weighted average of its spatial neighbours.
    Weights combine spatial proximity (Gaussian on distance) with intensity
    similarity (Gaussian on colour difference), so edges stay sharp while
    flat noisy areas are smoothed.  Operates per-frame — no temporal ghosting.

    Args:
        frames: [T, C, H, W] float tensor in [0, 1].
        kernel_size: Spatial window side (odd).
        sigma_spatial: Gaussian sigma for spatial distance.
        sigma_range: Gaussian sigma for intensity difference (in [0, 1]).
    """
    half = kernel_size // 2

    offsets = torch.arange(-half, half + 1, dtype=frames.dtype, device=frames.device)
    gy, gx = torch.meshgrid(offsets, offsets, indexing="ij")
    spatial_weights = torch.exp(-0.5 * (gx * gx + gy * gy) / (sigma_spatial ** 2))

    padded = F.pad(frames, (half, half, half, half), mode="reflect")

    range_scale = -0.5 / (sigma_range ** 2)
    H, W = frames.shape[2], frames.shape[3]

    result = torch.zeros_like(frames)
    weight_sum = torch.zeros(frames.shape[0], 1, H, W, dtype=frames.dtype, device=frames.device)

    for dy in range(kernel_size):
        for dx in range(kernel_size):
            neighbor = padded[:, :, dy:dy + H, dx:dx + W]
            diff_sq = (frames - neighbor).pow(2).mean(dim=1, keepdim=True)
            w = float(spatial_weights[dy, dx]) * torch.exp(diff_sq * range_scale)
            result.addcmul_(neighbor, w.expand_as(neighbor))
            weight_sum.add_(w)

    return result / weight_sum


def apply_denoise(frames: torch.Tensor, strength: DenoiseStrength) -> torch.Tensor:
    if strength is DenoiseStrength.NONE:
        return frames
    kernel_size, sigma_spatial, sigma_range = _DENOISE_PARAMS[strength]
    return spatial_denoise(frames, kernel_size, sigma_spatial, sigma_range)
