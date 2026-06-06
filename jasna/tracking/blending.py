from __future__ import annotations

import torch
import torch.nn.functional as F

_KERNEL_CACHE: dict[tuple[str, torch.dtype, int], tuple[torch.Tensor, torch.Tensor]] = {}

BLEND_DILATION_RATIO = 0.028
BLEND_FALLOFF_RATIO = 0.028


def _box_blur(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    # Separable box blur: a uniform KxK kernel = 1xK then Kx1, cost O(K^2) -> O(2K).
    cache_key = (str(x.device), x.dtype, kernel_size)
    kernels = _KERNEL_CACHE.get(cache_key)
    if kernels is None:
        kernel_h = torch.ones((1, 1, 1, kernel_size), device=x.device, dtype=x.dtype) / kernel_size
        kernel_v = torch.ones((1, 1, kernel_size, 1), device=x.device, dtype=x.dtype) / kernel_size
        kernels = (kernel_h, kernel_v)
        _KERNEL_CACHE[cache_key] = kernels
    kernel_h, kernel_v = kernels
    pad = kernel_size // 2
    x4d = F.pad(x.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode="reflect")
    x4d = F.conv2d(x4d, kernel_h)
    x4d = F.conv2d(x4d, kernel_v)
    return x4d.squeeze(0).squeeze(0)


def _make_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def create_blend_mask(crop_mask: torch.Tensor, frame_height: int) -> torch.Tensor:
    """Create blend mask from detection mask with dilation and falloff.

    Dilation ensures blend weight=1.0 extends past the mask edge to cover
    any adjacent mosaic blocks the detector missed.  Falloff creates a
    smooth transition entirely outside the mosaic area.
    Both are proportional to frame height (~30px each at 1080p).
    """
    mask = crop_mask.squeeze()
    blend_dtype = mask.dtype if mask.is_floating_point() else torch.get_default_dtype()

    dilation_px = max(3, round(frame_height * BLEND_DILATION_RATIO))
    falloff_px = max(3, round(frame_height * BLEND_FALLOFF_RATIO))

    dilate_k = _make_odd(dilation_px * 2 + 1)
    falloff_k = _make_odd(falloff_px * 2 + 1)

    blend = (mask > 0).to(dtype=blend_dtype)
    blend = _box_blur(blend, dilate_k)
    blend = (blend > 0.01).to(dtype=blend_dtype)
    blend = _box_blur(blend, falloff_k)

    return blend.clamp_(0.0, 1.0)

