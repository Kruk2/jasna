"""Unit tests for NV12/P010 -> planar RGB uint8 conversion (BT.709/BT.601, limited/full range)."""
import numpy as np
import torch
from av.video.reformatter import Colorspace as AvColorspace

from jasna.media.rgb_to_p010 import (
    chw_rgb_to_p010_bt601_limited,
    chw_rgb_to_p010_bt709_limited,
)
from jasna.media.yuv_to_rgb import YuvToRgbConverter

CPU = torch.device("cpu")


def _converter(h=4, w=4, space=AvColorspace.ITU709, full_range=False, is_10bit=False):
    return YuvToRgbConverter(h, w, space, full_range, is_10bit, CPU)


def _uniform_planes(y_val, u_val, v_val, h=4, w=4, dtype=torch.uint8):
    y = torch.full((h, w), y_val, dtype=dtype)
    uv = torch.empty((h // 2, w // 2, 2), dtype=dtype)
    uv[:, :, 0] = u_val
    uv[:, :, 1] = v_val
    return y, uv


def test_limited_white_and_black_8bit():
    conv = _converter()
    y, uv = _uniform_planes(235, 128, 128)
    assert torch.equal(conv.convert(y, uv), torch.full((3, 4, 4), 255, dtype=torch.uint8))
    y, uv = _uniform_planes(16, 128, 128)
    assert torch.equal(conv.convert(y, uv), torch.zeros(3, 4, 4, dtype=torch.uint8))


def test_full_range_white_and_black_8bit():
    conv = _converter(full_range=True)
    y, uv = _uniform_planes(255, 128, 128)
    assert torch.equal(conv.convert(y, uv), torch.full((3, 4, 4), 255, dtype=torch.uint8))
    y, uv = _uniform_planes(0, 128, 128)
    assert torch.equal(conv.convert(y, uv), torch.zeros(3, 4, 4, dtype=torch.uint8))


def test_limited_white_10bit_dithered():
    conv = _converter(is_10bit=True)
    y, uv = _uniform_planes(940 << 6, 512 << 6, 512 << 6, dtype=torch.int32)
    out = conv.convert(y.to(torch.float32), uv.to(torch.float32))
    assert torch.equal(out, torch.full((3, 4, 4), 255, dtype=torch.uint8))


def test_bt601_and_bt709_differ_for_colored_input():
    y, uv = _uniform_planes(120, 90, 200)
    out709 = _converter().convert(y, uv)
    out601 = _converter(space=AvColorspace.ITU601).convert(y, uv)
    assert not torch.equal(out709, out601)


def _p010_to_planes(p010: torch.Tensor, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    as_u16 = p010.to(torch.int32) & 0xFFFF
    y = as_u16[:h].to(torch.float32)
    uv = as_u16[h:].reshape(h // 2, w // 2, 2).to(torch.float32)
    return y, uv


def test_round_trip_uniform_colors_bt709():
    for rgb in [(200, 30, 60), (10, 250, 128), (128, 128, 128), (255, 255, 255), (0, 0, 0)]:
        img = torch.empty(3, 8, 8, dtype=torch.uint8)
        for c, val in enumerate(rgb):
            img[c] = val
        y, uv = _p010_to_planes(chw_rgb_to_p010_bt709_limited(img), 8, 8)
        out = YuvToRgbConverter(8, 8, AvColorspace.ITU709, False, True, CPU).convert(y, uv)
        assert (out.to(torch.int16) - img.to(torch.int16)).abs().max() <= 2


def test_round_trip_uniform_colors_bt601():
    img = torch.empty(3, 8, 8, dtype=torch.uint8)
    for c, val in enumerate((60, 180, 240)):
        img[c] = val
    y, uv = _p010_to_planes(chw_rgb_to_p010_bt601_limited(img), 8, 8)
    out = YuvToRgbConverter(8, 8, AvColorspace.ITU601, False, True, CPU).convert(y, uv)
    assert (out.to(torch.int16) - img.to(torch.int16)).abs().max() <= 2


def test_dither_is_deterministic():
    conv = _converter(h=8, w=8, is_10bit=True)
    y = torch.randint(64 << 6, 940 << 6, (8, 8), dtype=torch.int32).to(torch.float32)
    uv = torch.randint(64 << 6, 960 << 6, (4, 4, 2), dtype=torch.int32).to(torch.float32)
    assert torch.equal(conv.convert(y, uv), conv.convert(y, uv))


def test_matches_swscale_reference_bt709_limited():
    import av

    rng = np.random.default_rng(7)
    h, w = 16, 16
    # random luma, uniform chroma: chroma upsampling filter differences vanish
    y_plane = rng.integers(16, 236, (h, w), dtype=np.uint8)
    uv_plane = np.empty((h // 2, w // 2, 2), dtype=np.uint8)
    uv_plane[:, :, 0] = 90
    uv_plane[:, :, 1] = 190

    nv12 = np.vstack([y_plane, uv_plane.reshape(h // 2, w)])
    frame = av.VideoFrame.from_ndarray(nv12, format="nv12")
    frame.colorspace = 1  # bt709
    frame.color_range = 1  # mpeg/limited
    reference = frame.to_ndarray(format="rgb24").astype(np.int16)  # (H, W, 3)

    out = _converter(h=h, w=w).convert(
        torch.from_numpy(y_plane), torch.from_numpy(uv_plane)
    )
    ours = out.permute(1, 2, 0).numpy().astype(np.int16)
    assert np.abs(ours - reference).max() <= 3
