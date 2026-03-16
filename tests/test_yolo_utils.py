import torch
import pytest

from jasna.mosaic.yolo import _mask_hw_for_frame, YoloMosaicDetectionModel


class TestMaskHwForFrame:
    def test_square(self):
        mh, mw = _mask_hw_for_frame((256, 256))
        assert mh == mw == 256

    def test_landscape(self):
        mh, mw = _mask_hw_for_frame((1080, 1920))
        assert mw == 256
        assert mh == max(1, round(1080 * 256 / 1920))

    def test_portrait(self):
        mh, mw = _mask_hw_for_frame((1920, 1080))
        assert mh == 256
        assert mw == max(1, round(1080 * 256 / 1920))

    def test_small_frame(self):
        mh, mw = _mask_hw_for_frame((1, 1))
        assert mh >= 1
        assert mw >= 1

    def test_4k(self):
        mh, mw = _mask_hw_for_frame((2160, 3840))
        assert mw == 256
        assert mh == max(1, round(2160 * 256 / 3840))


class TestYoloGetEmptyMasks:
    def _make_model(self):
        with pytest.raises((FileNotFoundError, RuntimeError, Exception)):
            YoloMosaicDetectionModel(
                model_path="nonexistent.pt",
                batch_size=1,
                device=torch.device("cpu"),
            )

    def test_empty_masks_shape(self):
        model = object.__new__(YoloMosaicDetectionModel)
        model.device = torch.device("cpu")
        model._empty_masks_cache = {}

        masks = model._get_empty_masks(128, 256)
        assert masks.shape == (0, 128, 256)
        assert masks.dtype == torch.bool

    def test_empty_masks_cached(self):
        model = object.__new__(YoloMosaicDetectionModel)
        model.device = torch.device("cpu")
        model._empty_masks_cache = {}

        m1 = model._get_empty_masks(64, 64)
        m2 = model._get_empty_masks(64, 64)
        assert m1 is m2

    def test_different_sizes_not_shared(self):
        model = object.__new__(YoloMosaicDetectionModel)
        model.device = torch.device("cpu")
        model._empty_masks_cache = {}

        m1 = model._get_empty_masks(64, 64)
        m2 = model._get_empty_masks(128, 128)
        assert m1 is not m2
