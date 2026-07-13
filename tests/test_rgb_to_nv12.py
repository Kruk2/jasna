"""Unit tests for RGB→NV12 8-bit conversion (all matrices, both ranges)."""
import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace

from jasna.media.rgb_to_nv12 import (
    chw_rgb_to_nv12_bt601_full,
    chw_rgb_to_nv12_bt601_limited,
    chw_rgb_to_nv12_bt709_full,
    chw_rgb_to_nv12_bt709_limited,
    chw_rgb_to_nv12_bt2020_full,
    chw_rgb_to_nv12_bt2020_limited,
)
from jasna.media.yuv_to_rgb import YuvToRgbConverter

ALL_CONVERTERS = {
    (AvColorspace.ITU601, False): chw_rgb_to_nv12_bt601_limited,
    (AvColorspace.ITU601, True): chw_rgb_to_nv12_bt601_full,
    (AvColorspace.ITU709, False): chw_rgb_to_nv12_bt709_limited,
    (AvColorspace.ITU709, True): chw_rgb_to_nv12_bt709_full,
    (AvColorspace.BT2020, False): chw_rgb_to_nv12_bt2020_limited,
    (AvColorspace.BT2020, True): chw_rgb_to_nv12_bt2020_full,
}


def _uniform(r: float, g: float, b: float, h: int = 2, w: int = 2) -> torch.Tensor:
    img = torch.empty(3, h, w, dtype=torch.float32)
    img[0] = r
    img[1] = g
    img[2] = b
    return img


def test_bt601_limited_red_matches_reference_values():
    out = chw_rgb_to_nv12_bt601_limited(_uniform(1, 0, 0))
    # Y: 16 + 219*0.299 = 81.481 -> 81
    assert torch.equal(out[0:2], torch.full((2, 2), 81, dtype=torch.uint8))
    # U: 128 + 224*-0.168736 = 90.20 -> 90
    assert out[2, 0].item() == 90
    # V: 128 + 224*0.5 = 240
    assert out[2, 1].item() == 240


def test_bt709_limited_red_matches_reference_values():
    out = chw_rgb_to_nv12_bt709_limited(_uniform(1, 0, 0))
    # Y: 16 + 219*0.2126 = 62.56 -> 63
    assert torch.equal(out[0:2], torch.full((2, 2), 63, dtype=torch.uint8))


def test_bt601_limited_blue_matches_reference_values():
    out = chw_rgb_to_nv12_bt601_limited(_uniform(0, 0, 1))
    # Y: 16 + 219*0.114 = 40.97 -> 41; U: 128 + 224*0.5 = 240; V: 128 - 224*0.081312 -> 110
    assert torch.equal(out[0:2], torch.full((2, 2), 41, dtype=torch.uint8))
    assert out[2, 0].item() == 240
    assert out[2, 1].item() == 110


def test_limited_black_and_white_hit_code_range_bounds():
    black = chw_rgb_to_nv12_bt709_limited(_uniform(0, 0, 0))
    assert torch.equal(black[0:2], torch.full((2, 2), 16, dtype=torch.uint8))
    assert torch.equal(black[2], torch.full((2,), 128, dtype=torch.uint8))

    white = chw_rgb_to_nv12_bt709_limited(_uniform(1, 1, 1))
    assert torch.equal(white[0:2], torch.full((2, 2), 235, dtype=torch.uint8))
    assert torch.equal(white[2], torch.full((2,), 128, dtype=torch.uint8))


def test_full_range_black_and_white_span_0_to_255():
    black = chw_rgb_to_nv12_bt601_full(_uniform(0, 0, 0))
    assert torch.equal(black[0:2], torch.full((2, 2), 0, dtype=torch.uint8))
    assert torch.equal(black[2], torch.full((2,), 128, dtype=torch.uint8))

    white = chw_rgb_to_nv12_bt601_full(_uniform(1, 1, 1))
    assert torch.equal(white[0:2], torch.full((2, 2), 255, dtype=torch.uint8))
    assert torch.equal(white[2], torch.full((2,), 128, dtype=torch.uint8))


def test_full_range_red_saturates_v():
    out = chw_rgb_to_nv12_bt601_full(_uniform(1, 0, 0))
    # Y: 255*0.299 = 76.245 -> 76; U: 128 - 255*0.168736 -> 85; V: 128 + 255*0.5 clamps to 255
    assert torch.equal(out[0:2], torch.full((2, 2), 76, dtype=torch.uint8))
    assert out[2, 0].item() == 85
    assert out[2, 1].item() == 255


def test_bt601_and_bt709_differ_for_colored_input():
    red = _uniform(1, 0, 0)
    assert not torch.equal(chw_rgb_to_nv12_bt601_limited(red), chw_rgb_to_nv12_bt709_limited(red))


def test_uint8_input_is_normalized():
    red_u8 = torch.zeros(3, 2, 2, dtype=torch.uint8)
    red_u8[0] = 255
    from_u8 = chw_rgb_to_nv12_bt601_limited(red_u8)
    from_float = chw_rgb_to_nv12_bt601_limited(_uniform(1, 0, 0))
    assert torch.equal(from_u8, from_float)


def test_output_shape_dtype_and_contiguity():
    img = torch.rand(3, 4, 6, dtype=torch.float32)
    out = chw_rgb_to_nv12_bt601_limited(img)
    # Y plane (H rows) + interleaved UV (H/2 rows) = H + H/2 rows, W cols.
    assert out.shape == (4 + 2, 6)
    assert out.dtype == torch.uint8
    assert out.is_contiguous()


@pytest.mark.parametrize("h,w", [(3, 4), (4, 3), (5, 5)])
def test_odd_dimensions_rejected(h, w):
    with pytest.raises(ValueError, match="even dimensions"):
        chw_rgb_to_nv12_bt709_limited(torch.rand(3, h, w))


@pytest.mark.parametrize(("color_space", "full_range"), list(ALL_CONVERTERS))
def test_round_trip_against_yuv_to_rgb_converter(color_space, full_range):
    converter = ALL_CONVERTERS[(color_space, full_range)]
    torch.manual_seed(0)
    h, w = 16, 16
    # Smooth image: chroma subsampling averages neighbours, so keep 2x2 blocks flat.
    base = torch.rand(3, h // 2, w // 2, dtype=torch.float32)
    img = base.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)

    packed = converter(img)
    y = packed[:h]
    uv = packed[h:].reshape(h // 2, w // 2, 2)

    back = YuvToRgbConverter(
        h, w, color_space, full_range, False, torch.device("cpu")
    ).convert(y, uv)

    expected = img.mul(255).round().clamp(0, 255)
    assert (back.float() - expected).abs().max().item() <= 3.0
