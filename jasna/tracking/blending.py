from __future__ import annotations

import torch
import torch.nn.functional as F


def create_blend_mask(crop_mask: torch.Tensor, border_ratio: float = 0.05) -> torch.Tensor:
    """Create blend mask from detection mask with blurred border."""
    mask = crop_mask.squeeze()
    blend_dtype = mask.dtype if mask.is_floating_point() else torch.get_default_dtype()
    h, w = mask.shape
    h_inner = int(h * (1.0 - border_ratio))
    w_inner = int(w * (1.0 - border_ratio))
    h_outer = h - h_inner
    w_outer = w - w_inner
    border_size = min(h_outer, w_outer)

    if border_size < 5:
        return torch.ones_like(mask, dtype=blend_dtype)

    blur_size = border_size
    if blur_size % 2 == 0:
        blur_size += 1

    inner = torch.ones((h_inner, w_inner), device=mask.device, dtype=blend_dtype)
    pad_top = h_outer // 2
    pad_bottom = h_outer - pad_top
    pad_left = w_outer // 2
    pad_right = w_outer - pad_left
    blend = F.pad(inner, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)

    blend = torch.maximum((mask > 0).to(dtype=blend.dtype), blend)

    kernel = torch.ones((1, 1, blur_size, blur_size), device=blend.device, dtype=blend_dtype) / (blur_size**2)
    pad_size = blur_size // 2
    blend_4d = F.pad(blend.unsqueeze(0).unsqueeze(0), (pad_size, pad_size, pad_size, pad_size), mode="reflect")
    blend = F.conv2d(blend_4d, kernel).squeeze(0).squeeze(0)

    return blend

