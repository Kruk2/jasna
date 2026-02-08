import torch
import pytest

from jasna.mosaic.yolo import _mask_hw_for_frame


class _FakeYoloModel:
    """Minimal stand-in to test _get_empty_masks without loading a real model."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._empty_masks_cache: dict[tuple[int, int], torch.Tensor] = {}

    def _get_empty_masks(self, mask_h: int, mask_w: int) -> torch.Tensor:
        from jasna.mosaic.yolo import YoloMosaicDetectionModel
        return YoloMosaicDetectionModel._get_empty_masks(self, mask_h, mask_w)


def test_empty_masks_cache_returns_same_object() -> None:
    fake = _FakeYoloModel(torch.device("cpu"))
    mask_h, mask_w = _mask_hw_for_frame((1080, 1920))
    first = fake._get_empty_masks(mask_h, mask_w)
    second = fake._get_empty_masks(mask_h, mask_w)
    assert first is second
    assert first.shape == (0, mask_h, mask_w)
    assert first.dtype == torch.bool


def test_empty_masks_cache_different_sizes() -> None:
    fake = _FakeYoloModel(torch.device("cpu"))
    t1 = fake._get_empty_masks(128, 256)
    t2 = fake._get_empty_masks(64, 64)
    assert t1 is not t2
    assert t1.shape == (0, 128, 256)
    assert t2.shape == (0, 64, 64)
    assert fake._get_empty_masks(128, 256) is t1
    assert fake._get_empty_masks(64, 64) is t2


def test_fused_to_produces_correct_dtype() -> None:
    x = torch.randint(0, 256, (1, 3, 64, 64), dtype=torch.uint8)
    target_dtype = torch.float32
    result = x.to(device="cpu", dtype=target_dtype, non_blocking=False)
    assert result.dtype == target_dtype
    assert result.device == x.device


def test_inplace_div_matches_out_of_place() -> None:
    x1 = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8).float()
    x2 = x1.clone()
    expected = x2 / 255.0
    x1 /= 255.0
    assert torch.allclose(x1, expected)
