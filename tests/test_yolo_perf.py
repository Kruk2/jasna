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
