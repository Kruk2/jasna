"""RGB to NV12 (8-bit) and P010 (10-bit) conversion on the Torch path, against the matrix formulas."""
import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace

from jasna.media.rgb_to_yuv import RgbToYuvConverter
from jasna.media.yuv_to_rgb import YuvToRgbConverter
from factories import yuv_to_rgb

NV12_VARIANTS = {
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


@pytest.mark.parametrize(
    ("variant", "rgb", "y", "u", "v"),
    [
        # Y = 16 + 219*0.299 = 81.48; U = 128 - 224*0.168736 = 90.2; V = 128 + 224*0.5
        ("nv12_bt601_limited", (1, 0, 0), 81, 90, 240),
        # Y = 16 + 219*0.2126 = 62.56; U = 128 - 224*0.114572 = 102.3
        ("nv12_bt709_limited", (1, 0, 0), 63, 102, 240),
        # Y = 16 + 219*0.114 = 40.97; V = 128 - 224*0.081312 = 109.8
        ("nv12_bt601_limited", (0, 0, 1), 41, 240, 110),
        ("nv12_bt709_limited", (0, 0, 0), 16, 128, 128),
        ("nv12_bt709_limited", (1, 1, 1), 235, 128, 128),
        ("nv12_bt601_full", (0, 0, 0), 0, 128, 128),
        ("nv12_bt601_full", (1, 1, 1), 255, 128, 128),
        # Y = 255*0.299 = 76.2; U = 128 - 255*0.168736 = 85; V = 128 + 255*0.5 clamps to 255
        ("nv12_bt601_full", (1, 0, 0), 76, 85, 255),
        # 10-bit codes in the high bits: Y = 64 + 876*0.299 = 325.9; U = 512 - 896*0.168736 = 360.8; V = 960
        ("p010_bt601_limited", (1, 0, 0), 326 << 6, 361 << 6, 960 << 6),
        # Y = 64 + 876*0.2126 = 250.2; U = 512 - 896*0.114572 = 409.3
        ("p010_bt709_limited", (1, 0, 0), 250 << 6, 409 << 6, 960 << 6),
    ],
)
def test_flat_colours_match_the_matrix_formulas(variant, rgb, y, u, v):
    out = _convert(variant, _uniform(*rgb)).to(torch.int32) & 0xFFFF

    assert torch.equal(out[0:2], torch.full((2, 2), y, dtype=torch.int32))
    assert out[2].tolist() == [u, v]


@pytest.mark.parametrize(("variant", "dtype"), [("nv12_bt601_limited", torch.uint8), ("p010_bt601_limited", torch.int16)])
def test_output_packs_y_and_interleaved_uv(variant, dtype):
    out = _convert(variant, torch.randint(0, 256, (3, 4, 6), dtype=torch.uint8))

    # Y plane (H rows) + interleaved UV (H/2 rows) = H + H/2 rows, W cols.
    assert out.shape == (4 + 2, 6)
    assert out.dtype == dtype
    assert out.is_contiguous()


@pytest.mark.parametrize("h,w", [(3, 4), (4, 3), (5, 5)])
def test_odd_dimensions_rejected(h, w):
    with pytest.raises(ValueError, match="even dimensions"):
        _convert("nv12_bt709_limited", torch.zeros(3, h, w, dtype=torch.uint8))


def test_rejects_an_unknown_variant():
    with pytest.raises(ValueError, match="Unknown RGB to YUV variant"):
        RgbToYuvConverter("nv12_bt709_studio", device=torch.device("cpu"))


@pytest.mark.parametrize(("color_space", "full_range"), list(NV12_VARIANTS))
def test_round_trip_against_yuv_to_rgb_converter(color_space, full_range):
    torch.manual_seed(0)
    h, w = 16, 16
    # Smooth image: chroma subsampling averages neighbours, so keep 2x2 blocks flat.
    base = torch.randint(0, 256, (3, h // 2, w // 2), dtype=torch.uint8)
    img = base.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)

    packed = _convert(NV12_VARIANTS[(color_space, full_range)], img)
    y = packed[:h]
    uv = packed[h:].reshape(h // 2, w // 2, 2)

    back = yuv_to_rgb(YuvToRgbConverter(h, w, color_space, full_range, False, torch.device("cpu")), y, uv)

    assert (back.float() - img.float()).abs().max().item() <= 3.0
