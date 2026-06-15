import pytest
import torch
import torch.nn.functional as F

from jasna.tracking.blending import _box_blur, create_bbox_blend_mask, create_blend_mask


@pytest.mark.parametrize("kernel_size", [5, 61, 121])
def test_box_blur_matches_dense_reference(kernel_size: int) -> None:
    torch.manual_seed(0)
    x = torch.rand((128, 96))

    pad = kernel_size // 2
    kernel = torch.ones((1, 1, kernel_size, kernel_size)) / (kernel_size ** 2)
    x4d = F.pad(x.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode="reflect")
    dense = F.conv2d(x4d, kernel).squeeze(0).squeeze(0)

    assert torch.allclose(_box_blur(x, kernel_size, kernel_size), dense, atol=1e-4)


def test_create_blend_mask_all_ones_stays_ones() -> None:
    crop_mask = torch.ones((100, 100), dtype=torch.bool)
    out = create_blend_mask(crop_mask, frame_height=1080, scale_y=1.0, scale_x=1.0)
    assert out.shape == (100, 100)
    assert out.dtype == torch.float32
    assert torch.allclose(out, torch.ones_like(out), atol=1e-4)


def test_create_blend_mask_clamps_to_0_1_and_keeps_shape() -> None:
    crop_mask = torch.zeros((100, 100), dtype=torch.float32)
    crop_mask[40:60, 40:60] = 1.0
    out = create_blend_mask(crop_mask, frame_height=1080, scale_y=1.0, scale_x=1.0)
    assert out.shape == (100, 100)
    assert float(out.min()) >= -1e-6
    assert float(out.max()) <= 1.0 + 1e-6


def test_create_blend_mask_center_higher_than_corner() -> None:
    crop_mask = torch.zeros((200, 200), dtype=torch.bool)
    crop_mask[80:120, 80:120] = True
    out = create_blend_mask(crop_mask, frame_height=1080, scale_y=1.0, scale_x=1.0)
    assert out[100, 100] > out[0, 0]


def test_create_blend_mask_dilation_covers_past_mask_edge() -> None:
    crop_mask = torch.zeros((200, 200), dtype=torch.bool)
    crop_mask[80:120, 80:120] = True
    out = create_blend_mask(crop_mask, frame_height=1080, scale_y=1.0, scale_x=1.0)
    assert float(out[75, 100]) > 0.5


@pytest.mark.parametrize("frame_height", [720, 1080, 2160])
def test_create_blend_mask_scales_with_resolution(frame_height: int) -> None:
    crop_mask = torch.ones((200, 200), dtype=torch.bool)
    out = create_blend_mask(crop_mask, frame_height=frame_height, scale_y=1.0, scale_x=1.0)
    assert out.shape == (200, 200)


@pytest.mark.parametrize("kh,kw", [(5, 9), (61, 31)])
def test_box_blur_anisotropic_matches_dense(kh: int, kw: int) -> None:
    torch.manual_seed(0)
    x = torch.rand((128, 96))

    kernel = torch.ones((1, 1, kh, kw)) / (kh * kw)
    x4d = F.pad(x[None, None], (kw // 2, kw // 2, kh // 2, kh // 2), mode="reflect")
    dense = F.conv2d(x4d, kernel)[0, 0]

    assert torch.allclose(_box_blur(x, kh, kw), dense, atol=1e-4)


def _reference_fullres_blend_mask(
    mask_lr: torch.Tensor,
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
) -> torch.Tensor:
    """Old algorithm: nearest-upsample mask to bbox size at frame res, then blur there."""
    x1, y1, x2, y2 = bbox
    frame_h, frame_w = frame_shape
    hm, wm = mask_lr.shape
    y_idx = (torch.arange(y1, y2) * hm) // frame_h
    x_idx = (torch.arange(x1, x2) * wm) // frame_w
    crop_mask = mask_lr.float().index_select(0, y_idx).index_select(1, x_idx)
    return create_blend_mask(crop_mask, frame_height=frame_h, scale_y=1.0, scale_x=1.0)


def test_bbox_blend_mask_close_to_fullres_reference() -> None:
    yy, xx = torch.meshgrid(torch.arange(256), torch.arange(256), indexing="ij")
    mask = (((yy - 120.0) / 28.0) ** 2 + ((xx - 110.0) / 40.0) ** 2) <= 1.0
    bbox = (600, 320, 1150, 760)
    frame_shape = (1080, 1920)

    ref = _reference_fullres_blend_mask(mask, bbox, frame_shape)
    out = create_bbox_blend_mask(mask, bbox, frame_shape)

    assert out.shape == ref.shape
    diff = (out - ref).abs()
    assert float(diff.mean()) < 0.015
    assert float(diff.max()) < 0.12


def test_bbox_blend_mask_full_mask_is_ones() -> None:
    mask = torch.ones((256, 256), dtype=torch.bool)
    out = create_bbox_blend_mask(mask, (600, 320, 1150, 760), (1080, 1920))
    assert out.shape == (440, 550)
    assert torch.allclose(out, torch.ones_like(out), atol=1e-4)


def test_bbox_blend_mask_tiny_bbox_no_crash() -> None:
    mask = torch.ones((256, 256), dtype=torch.bool)
    out = create_bbox_blend_mask(mask, (1000, 500, 1008, 506), (2160, 3840))
    assert out.shape == (6, 8)
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_bbox_blend_mask_same_res_mask_matches_reference_exactly() -> None:
    """When mask res == frame res the low-res path degenerates to the old one."""
    mask = torch.zeros((64, 64), dtype=torch.bool)
    mask[20:40, 20:40] = True
    bbox = (10, 10, 60, 60)
    frame_shape = (64, 64)

    ref = _reference_fullres_blend_mask(mask, bbox, frame_shape)
    out = create_bbox_blend_mask(mask, bbox, frame_shape)
    assert torch.allclose(out, ref, atol=1e-5)

