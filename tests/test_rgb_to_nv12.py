"""Unit tests for RGB→NV12 8-bit conversion (all matrices, both ranges)."""
import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace

from jasna.media.rgb_to_yuv import RgbToYuvConverter
from jasna.media.yuv_to_rgb import YuvToRgbConverter

VARIANTS = {
    (AvColorspace.ITU601, False): "nv12_bt601_limited",
    (AvColorspace.ITU601, True): "nv12_bt601_full",
    (AvColorspace.ITU709, False): "nv12_bt709_limited",
    (AvColorspace.ITU709, True): "nv12_bt709_full",
    (AvColorspace.BT2020, False): "nv12_bt2020_limited",
    (AvColorspace.BT2020, True): "nv12_bt2020_full",
}


def _convert(variant: str, frame: torch.Tensor) -> torch.Tensor:
    return RgbToYuvConverter(variant, device=torch.device("cpu")).convert(frame)


def _uniform(r: int, g: int, b: int, h: int = 2, w: int = 2) -> torch.Tensor:
    img = torch.empty(3, h, w, dtype=torch.uint8)
    img[0] = r * 255
    img[1] = g * 255
    img[2] = b * 255
    return img


def test_bt601_limited_red_matches_reference_values():
    out = _convert("nv12_bt601_limited", _uniform(1, 0, 0))
    # Y: 16 + 219*0.299 = 81.481 -> 81
    assert torch.equal(out[0:2], torch.full((2, 2), 81, dtype=torch.uint8))
    # U: 128 + 224*-0.168736 = 90.20 -> 90
    assert out[2, 0].item() == 90
    # V: 128 + 224*0.5 = 240
    assert out[2, 1].item() == 240


def test_bt709_limited_red_matches_reference_values():
    out = _convert("nv12_bt709_limited", _uniform(1, 0, 0))
    # Y: 16 + 219*0.2126 = 62.56 -> 63
    assert torch.equal(out[0:2], torch.full((2, 2), 63, dtype=torch.uint8))


def test_bt601_limited_blue_matches_reference_values():
    out = _convert("nv12_bt601_limited", _uniform(0, 0, 1))
    # Y: 16 + 219*0.114 = 40.97 -> 41; U: 128 + 224*0.5 = 240; V: 128 - 224*0.081312 -> 110
    assert torch.equal(out[0:2], torch.full((2, 2), 41, dtype=torch.uint8))
    assert out[2, 0].item() == 240
    assert out[2, 1].item() == 110


def test_limited_black_and_white_hit_code_range_bounds():
    black = _convert("nv12_bt709_limited", _uniform(0, 0, 0))
    assert torch.equal(black[0:2], torch.full((2, 2), 16, dtype=torch.uint8))
    assert torch.equal(black[2], torch.full((2,), 128, dtype=torch.uint8))

    white = _convert("nv12_bt709_limited", _uniform(1, 1, 1))
    assert torch.equal(white[0:2], torch.full((2, 2), 235, dtype=torch.uint8))
    assert torch.equal(white[2], torch.full((2,), 128, dtype=torch.uint8))


def test_full_range_black_and_white_span_0_to_255():
    black = _convert("nv12_bt601_full", _uniform(0, 0, 0))
    assert torch.equal(black[0:2], torch.full((2, 2), 0, dtype=torch.uint8))
    assert torch.equal(black[2], torch.full((2,), 128, dtype=torch.uint8))

    white = _convert("nv12_bt601_full", _uniform(1, 1, 1))
    assert torch.equal(white[0:2], torch.full((2, 2), 255, dtype=torch.uint8))
    assert torch.equal(white[2], torch.full((2,), 128, dtype=torch.uint8))


def test_full_range_red_saturates_v():
    out = _convert("nv12_bt601_full", _uniform(1, 0, 0))
    # Y: 255*0.299 = 76.245 -> 76; U: 128 - 255*0.168736 -> 85; V: 128 + 255*0.5 clamps to 255
    assert torch.equal(out[0:2], torch.full((2, 2), 76, dtype=torch.uint8))
    assert out[2, 0].item() == 85
    assert out[2, 1].item() == 255


def test_bt601_and_bt709_differ_for_colored_input():
    red = _uniform(1, 0, 0)
    assert not torch.equal(_convert("nv12_bt601_limited", red), _convert("nv12_bt709_limited", red))


def test_output_shape_dtype_and_contiguity():
    img = torch.randint(0, 256, (3, 4, 6), dtype=torch.uint8)
    out = _convert("nv12_bt601_limited", img)
    # Y plane (H rows) + interleaved UV (H/2 rows) = H + H/2 rows, W cols.
    assert out.shape == (4 + 2, 6)
    assert out.dtype == torch.uint8
    assert out.is_contiguous()


@pytest.mark.parametrize("h,w", [(3, 4), (4, 3), (5, 5)])
def test_odd_dimensions_rejected(h, w):
    with pytest.raises(ValueError, match="even dimensions"):
        _convert("nv12_bt709_limited", torch.zeros(3, h, w, dtype=torch.uint8))


@pytest.mark.parametrize(("color_space", "full_range"), list(VARIANTS))
def test_round_trip_against_yuv_to_rgb_converter(color_space, full_range):
    torch.manual_seed(0)
    h, w = 16, 16
    # Smooth image: chroma subsampling averages neighbours, so keep 2x2 blocks flat.
    base = torch.randint(0, 256, (3, h // 2, w // 2), dtype=torch.uint8)
    img = base.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)

    packed = _convert(VARIANTS[(color_space, full_range)], img)
    y = packed[:h]
    uv = packed[h:].reshape(h // 2, w // 2, 2)

    back = YuvToRgbConverter(
        h, w, color_space, full_range, False, torch.device("cpu")
    ).convert(y, uv)

    assert (back.float() - img.float()).abs().max().item() <= 3.0
