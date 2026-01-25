import pytest
import torch

from jasna.tracking.blending import create_blend_mask


def test_create_blend_mask_small_border_returns_ones() -> None:
    crop_mask = torch.zeros((1, 4, 4), dtype=torch.bool)
    out = create_blend_mask(crop_mask)
    assert out.shape == (4, 4)
    assert out.dtype == torch.float32
    assert torch.all(out == 1.0)


def test_create_blend_mask_clamps_to_0_1_and_keeps_shape() -> None:
    crop_mask = torch.zeros((16, 16), dtype=torch.float32)
    crop_mask[0, 0] = 1.0
    out = create_blend_mask(crop_mask)
    assert out.shape == (16, 16)
    assert float(out.min()) >= -1e-6
    assert float(out.max()) <= 1.0 + 1e-6


def test_create_blend_mask_blur_path_keeps_all_ones() -> None:
    crop_mask = torch.ones((20, 20), dtype=torch.bool)
    out = create_blend_mask(crop_mask, border_ratio=0.5)
    assert out.shape == (20, 20)
    assert torch.allclose(out, torch.ones_like(out), atol=1e-6, rtol=0.0)


def test_create_blend_mask_blur_path_fades_towards_edges() -> None:
    crop_mask = torch.zeros((20, 20), dtype=torch.bool)
    out = create_blend_mask(crop_mask, border_ratio=0.5)
    assert out[10, 10] > out[0, 0]


@pytest.mark.parametrize("border_ratio", [0.0, 0.05, 0.2])
def test_create_blend_mask_accepts_common_border_ratios(border_ratio: float) -> None:
    crop_mask = torch.ones((12, 12), dtype=torch.bool)
    out = create_blend_mask(crop_mask, border_ratio=border_ratio)
    assert out.shape == (12, 12)

