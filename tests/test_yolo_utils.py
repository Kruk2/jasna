import torch
import pytest

from jasna.mosaic.yolo import _mask_hw_for_frame, YoloMosaicDetectionModel


@pytest.mark.parametrize(
    ("frame_hw", "mask_hw"),
    [
        ((256, 256), (256, 256)),
        ((1080, 1920), (144, 256)),
        ((1920, 1080), (256, 144)),
        ((2160, 3840), (144, 256)),
        ((1, 1), (256, 256)),
        ((1, 5000), (1, 256)),
    ],
)
def test_mask_hw_for_frame_fits_the_long_side_to_256(frame_hw, mask_hw):
    assert _mask_hw_for_frame(frame_hw) == mask_hw


class TestYoloGetEmptyMasks:
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
