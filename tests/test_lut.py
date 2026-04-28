from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from jasna.media.lut import CubeLut, GpuLutApplier, parse_cube_file, parse_cube_text


def _identity_3d_cube_text(size: int = 5) -> str:
    lines = [f"LUT_3D_SIZE {size}"]
    for b in range(size):
        for g in range(size):
            for r in range(size):
                rv = r / (size - 1)
                gv = g / (size - 1)
                bv = b / (size - 1)
                lines.append(f"{rv:.6f} {gv:.6f} {bv:.6f}")
    return "\n".join(lines)


def _invert_3d_cube_text(size: int = 5) -> str:
    lines = ["TITLE \"Invert\"", f"LUT_3D_SIZE {size}"]
    for b in range(size):
        for g in range(size):
            for r in range(size):
                rv = 1.0 - r / (size - 1)
                gv = 1.0 - g / (size - 1)
                bv = 1.0 - b / (size - 1)
                lines.append(f"{rv:.6f} {gv:.6f} {bv:.6f}")
    return "\n".join(lines)


def test_parse_cube_text_3d_identity() -> None:
    lut = parse_cube_text(_identity_3d_cube_text(5))
    assert lut.is_3d
    assert lut.size == 5
    assert lut.data.shape == (3, 5, 5, 5)
    assert lut.domain_min == (0.0, 0.0, 0.0)
    assert lut.domain_max == (1.0, 1.0, 1.0)


def test_parse_cube_text_skips_comments_and_title() -> None:
    text = "\n".join([
        "# header comment",
        "TITLE \"Some LUT\"",
        "LUT_3D_SIZE 2",
        "# another comment",
        "0 0 0",
        "1 0 0",
        "0 1 0",
        "1 1 0",
        "0 0 1",
        "1 0 1",
        "0 1 1",
        "1 1 1",
    ])
    lut = parse_cube_text(text)
    assert lut.size == 2 and lut.is_3d


def test_parse_cube_text_1d() -> None:
    text = "\n".join(["LUT_1D_SIZE 3", "0 0 0", "0.5 0.5 0.5", "1 1 1"])
    lut = parse_cube_text(text)
    assert not lut.is_3d
    assert lut.size == 3
    assert lut.data.shape == (3, 3)


def test_parse_cube_text_domain_min_max() -> None:
    text = "\n".join(
        ["LUT_3D_SIZE 2", "DOMAIN_MIN -0.1 -0.2 -0.3", "DOMAIN_MAX 1.1 1.2 1.3"]
        + ["0 0 0"] * 8
    )
    lut = parse_cube_text(text)
    assert lut.domain_min == (-0.1, -0.2, -0.3)
    assert lut.domain_max == pytest.approx((1.1, 1.2, 1.3))


def test_parse_cube_text_rejects_wrong_sample_count() -> None:
    text = "\n".join(["LUT_3D_SIZE 3", "0 0 0", "0 0 0"])
    with pytest.raises(ValueError):
        parse_cube_text(text)


def test_parse_cube_text_rejects_missing_size() -> None:
    with pytest.raises(ValueError):
        parse_cube_text("0 0 0\n1 1 1\n")


def test_parse_cube_file_reads_disk(tmp_path: Path) -> None:
    p = tmp_path / "id.cube"
    p.write_text(_identity_3d_cube_text(3), encoding="utf-8")
    lut = parse_cube_file(p)
    assert lut.size == 3 and lut.is_3d


# ---------------------------------------------------------------------------
# GPU apply tests (run on CPU when CUDA unavailable; grid_sample supports CPU).
# ---------------------------------------------------------------------------

def _device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def test_apply_identity_3d_lut_uint8_is_lossless() -> None:
    device = _device()
    lut = parse_cube_text(_identity_3d_cube_text(33))
    applier = GpuLutApplier(lut, device)

    torch.manual_seed(0)
    frame = torch.randint(0, 256, (3, 16, 24), dtype=torch.uint8)

    out = applier.apply(frame)
    assert out.dtype == torch.uint8
    assert out.shape == frame.shape
    # Identity LUT with size 33 (33 evenly spaced samples) recovers uint8 exactly
    # under trilinear interpolation.
    assert torch.equal(out.cpu(), frame)


def test_apply_invert_3d_lut() -> None:
    device = _device()
    lut = parse_cube_text(_invert_3d_cube_text(17))
    applier = GpuLutApplier(lut, device)

    frame = torch.tensor([
        [[0, 128, 255]],
        [[0, 128, 255]],
        [[0, 128, 255]],
    ], dtype=torch.uint8)

    out = applier.apply(frame).cpu()
    expected = (255 - frame.int()).to(torch.uint8)
    assert torch.allclose(out.float(), expected.float(), atol=1.0)


def test_apply_identity_1d_lut() -> None:
    device = _device()
    text = "\n".join(["LUT_1D_SIZE 256"] + [f"{v/255:.6f} {v/255:.6f} {v/255:.6f}" for v in range(256)])
    lut = parse_cube_text(text)
    applier = GpuLutApplier(lut, device)

    frame = torch.arange(256, dtype=torch.uint8).view(1, 1, 256).expand(3, 1, 256).contiguous()
    out = applier.apply(frame).cpu()
    assert torch.equal(out, frame)


def test_apply_rejects_wrong_shape() -> None:
    lut = parse_cube_text(_identity_3d_cube_text(3))
    applier = GpuLutApplier(lut, _device())
    with pytest.raises(ValueError):
        applier.apply(torch.zeros(1, 3, 4, 4, dtype=torch.uint8))


def test_apply_float_tensor_returns_float() -> None:
    lut = parse_cube_text(_identity_3d_cube_text(33))
    applier = GpuLutApplier(lut, _device())
    frame = torch.rand(3, 8, 8)
    out = applier.apply(frame).cpu()
    assert out.dtype == frame.dtype
    assert torch.allclose(out, frame, atol=1e-3)
