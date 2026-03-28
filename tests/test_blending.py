import pytest
import torch

from jasna.tracking.blending import create_blend_mask


def test_create_blend_mask_all_ones_stays_ones() -> None:
    crop_mask = torch.ones((100, 100), dtype=torch.bool)
    out = create_blend_mask(crop_mask, frame_height=1080)
    assert out.shape == (100, 100)
    assert out.dtype == torch.float32
    assert torch.allclose(out, torch.ones_like(out), atol=1e-4)


def test_create_blend_mask_clamps_to_0_1_and_keeps_shape() -> None:
    crop_mask = torch.zeros((100, 100), dtype=torch.float32)
    crop_mask[40:60, 40:60] = 1.0
    out = create_blend_mask(crop_mask, frame_height=1080)
    assert out.shape == (100, 100)
    assert float(out.min()) >= -1e-6
    assert float(out.max()) <= 1.0 + 1e-6


def test_create_blend_mask_center_higher_than_corner() -> None:
    crop_mask = torch.zeros((200, 200), dtype=torch.bool)
    crop_mask[80:120, 80:120] = True
    out = create_blend_mask(crop_mask, frame_height=1080)
    assert out[100, 100] > out[0, 0]


def test_create_blend_mask_dilation_covers_past_mask_edge() -> None:
    crop_mask = torch.zeros((200, 200), dtype=torch.bool)
    crop_mask[80:120, 80:120] = True
    out = create_blend_mask(crop_mask, frame_height=1080)
    assert float(out[75, 100]) > 0.5


@pytest.mark.parametrize("frame_height", [720, 1080, 2160])
def test_create_blend_mask_scales_with_resolution(frame_height: int) -> None:
    crop_mask = torch.ones((200, 200), dtype=torch.bool)
    out = create_blend_mask(crop_mask, frame_height=frame_height)
    assert out.shape == (200, 200)

