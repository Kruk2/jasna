import pytest
import torch

from jasna.mosaic.rfdetr import _IMAGENET_MEAN, _IMAGENET_STD, RfDetrMosaicDetectionModel


class _Detector:
    """Exercises the normalization cache without building an engine."""

    _normalization = RfDetrMosaicDetectionModel._normalization

    def __init__(self):
        self._normalization_cache = {}


def test_normalization_constants_match_imagenet():
    detector = _Detector()
    x = torch.zeros(1, 3, 4, 4)

    mean, std = detector._normalization(x)

    assert mean.flatten().tolist() == pytest.approx(_IMAGENET_MEAN)
    assert std.flatten().tolist() == pytest.approx(_IMAGENET_STD)
    assert mean.shape == (3, 1, 1)


def test_normalization_tensors_are_reused_per_device_and_dtype():
    detector = _Detector()
    x = torch.zeros(1, 3, 4, 4)

    first = detector._normalization(x)
    second = detector._normalization(torch.zeros(2, 3, 8, 8))

    assert first[0] is second[0]
    assert first[1] is second[1]
    assert len(detector._normalization_cache) == 1


def test_a_different_dtype_gets_its_own_entry():
    detector = _Detector()

    full = detector._normalization(torch.zeros(1, 3, 4, 4))
    half = detector._normalization(torch.zeros(1, 3, 4, 4, dtype=torch.float16))

    assert full[0] is not half[0]
    assert half[0].dtype is torch.float16
    assert len(detector._normalization_cache) == 2
