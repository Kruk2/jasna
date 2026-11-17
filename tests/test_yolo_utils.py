import pytest

from jasna.mosaic.yolo import _mask_hw_for_frame


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
