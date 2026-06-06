"""Unit tests for RGB→P010 limited-range conversion (BT.709 and BT.601)."""
import torch

from jasna.media.rgb_to_p010 import (
    chw_rgb_to_p010_bt601_limited,
    chw_rgb_to_p010_bt709_limited,
)


def _as_uint16(t: torch.Tensor) -> torch.Tensor:
    """P010 packs 10-bit samples in the high bits of a uint16; reinterpret the
    signed int16 output as unsigned for comparison."""
    return t.to(torch.int32) & 0xFFFF


def _uniform_red() -> torch.Tensor:
    # (3, 2, 2) float image, every pixel pure red.
    img = torch.zeros(3, 2, 2, dtype=torch.float32)
    img[0] = 1.0
    return img


def test_bt601_red_matches_reference_values():
    out = _as_uint16(chw_rgb_to_p010_bt601_limited(_uniform_red()))

    # Y plane (rows 0..1): 64 + 876*0.299 = 325.924 -> 326, <<6 = 20864
    assert torch.equal(out[0:2], torch.full((2, 2), 20864, dtype=torch.int32))
    # U: 512 + 896*-0.168736 = 360.81 -> 361, <<6 = 23104
    assert out[2, 0].item() == 23104
    # V: 512 + 896*0.5 = 960, <<6 = 61440
    assert out[2, 1].item() == 61440


def test_bt709_red_matches_reference_values():
    out = _as_uint16(chw_rgb_to_p010_bt709_limited(_uniform_red()))

    # Y: 64 + 876*0.2126 = 250.24 -> 250, <<6 = 16000
    assert torch.equal(out[0:2], torch.full((2, 2), 16000, dtype=torch.int32))


def test_bt601_and_bt709_differ_for_colored_input():
    red = _uniform_red()
    bt601 = chw_rgb_to_p010_bt601_limited(red)
    bt709 = chw_rgb_to_p010_bt709_limited(red)
    assert not torch.equal(bt601, bt709)


def test_uint8_input_is_normalized():
    red_u8 = torch.zeros(3, 2, 2, dtype=torch.uint8)
    red_u8[0] = 255
    from_u8 = chw_rgb_to_p010_bt601_limited(red_u8)
    from_float = chw_rgb_to_p010_bt601_limited(_uniform_red())
    assert torch.equal(from_u8, from_float)


def test_output_shape_packs_y_and_interleaved_uv():
    img = torch.rand(3, 4, 6, dtype=torch.float32)
    out = chw_rgb_to_p010_bt601_limited(img)
    # Y plane (H rows) + interleaved UV (H/2 rows) = H + H/2 rows, W cols.
    assert out.shape == (4 + 2, 6)
    assert out.dtype == torch.int16
