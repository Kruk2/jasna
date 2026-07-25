import logging
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jasna.media.cas import GpuCasSharpener, cas_weight_scale, sharpen_luma


def _sharpen(plane: torch.Tensor, strength: float, depth: int) -> torch.Tensor:
    return sharpen_luma(
        plane,
        weight_scale=cas_weight_scale(strength),
        headroom=float(2 << depth),
        peak=float((1 << depth) - 1),
    )


def _reference(plane: np.ndarray, strength: float, depth: int) -> np.ndarray:
    """Straight nine-tap transcription of the CAS formulation."""
    height, width = plane.shape
    padded = np.pad(plane.astype(np.float64), 1, mode="edge")
    a = padded[0:height, 0:width]
    b = padded[0:height, 1:width + 1]
    c = padded[0:height, 2:width + 2]
    d = padded[1:height + 1, 0:width]
    e = padded[1:height + 1, 1:width + 1]
    f = padded[1:height + 1, 2:width + 2]
    g = padded[2:height + 2, 0:width]
    h = padded[2:height + 2, 1:width + 1]
    i = padded[2:height + 2, 2:width + 2]

    min_cross = np.minimum.reduce([b, d, e, f, h])
    max_cross = np.maximum.reduce([b, d, e, f, h])
    minima = min_cross + np.minimum.reduce([a, b, c, d, e, f, g, h, i])
    maxima = np.maximum(max_cross + np.maximum.reduce([a, b, c, d, e, f, g, h, i]), 1.0)

    headroom = float(2 << depth)
    amp = np.sqrt(np.clip(np.minimum(minima, headroom - maxima) / maxima, 0.0, 1.0))
    weight = amp * cas_weight_scale(strength)
    denominator = 1.0 + 4.0 * weight
    delta = (b + d + f + h) - 4.0 * e
    safe = denominator > 0.0
    out = np.where(safe, e + delta * weight / np.where(safe, denominator, 1.0), e)
    return np.floor(np.clip(out, 0.0, float((1 << depth) - 1)))


def test_cas_weight_scale_matches_ffmpeg_curve() -> None:
    # FFmpeg maps strength onto -1 / lerp(16, 4, strength).
    assert cas_weight_scale(0.0) == pytest.approx(-1.0 / 16.0)
    assert cas_weight_scale(0.5) == pytest.approx(-1.0 / 10.0)
    assert cas_weight_scale(1.0) == pytest.approx(-1.0 / 4.0)


def test_cas_matches_hand_computed_reference() -> None:
    plane = torch.full((3, 3), 100.0)
    plane[1, 1] = 120.0

    # The centre pixel's neighbourhood is eight 100s and one 120, so
    # minima = 100 + 100 and maxima = 120 + 120.
    amp = math.sqrt(min(200.0, 512.0 - 240.0) / 240.0)
    weight = amp * cas_weight_scale(0.5)
    expected = math.floor((400.0 - 4.0 * 120.0) * weight / (1.0 + 4.0 * weight) + 120.0)

    assert expected == 131  # stays inside the 8-bit range, so no clamp is involved
    assert _sharpen(plane, 0.5, 8)[1, 1].item() == pytest.approx(expected)


@pytest.mark.parametrize("strength", [0.01, 0.5, 1.0])
@pytest.mark.parametrize("value", [0, 16, 100, 235])
def test_cas_preserves_a_constant_plane(value: int, strength: float) -> None:
    # A flat neighbourhood has no detail to sharpen, so every pixel including
    # the replicated borders must survive untouched. At full strength this is
    # also where the weight drives the denominator to zero.
    plane = torch.full((8, 8), float(value))
    out = _sharpen(plane, strength, 8)

    assert not out.isnan().any()
    assert torch.equal(out, plane)


@pytest.mark.parametrize("depth", [8, 10])
@pytest.mark.parametrize("strength", [0.1, 0.5, 1.0])
def test_cas_matches_naive_nine_tap_reference(depth: int, strength: float) -> None:
    peak = (1 << depth) - 1
    rng = np.random.default_rng(7)
    plane = rng.integers(0, peak + 1, size=(33, 47)).astype(np.float64)

    got = _sharpen(torch.from_numpy(plane).float(), strength, depth).numpy()

    assert np.abs(got - _reference(plane, strength, depth)).max() < 1e-3


@pytest.mark.parametrize("depth", [8, 10])
def test_cas_clamps_to_the_representable_range(depth: int) -> None:
    peak = (1 << depth) - 1
    plane = torch.zeros((16, 16))
    plane[:, ::2] = float(peak)

    out = _sharpen(plane, 1.0, depth)

    assert out.min().item() >= 0.0
    assert out.max().item() <= float(peak)


def test_cas_on_an_all_black_plane_is_not_nan() -> None:
    # Every extreme is zero here, so the amplitude divides by zero unguarded.
    out = _sharpen(torch.zeros((8, 8)), 0.5, 8)

    assert not out.isnan().any()
    assert torch.equal(out, torch.zeros((8, 8)))


def test_cas_bands_a_tall_plane_without_seams(monkeypatch) -> None:
    rng = np.random.default_rng(11)
    plane = torch.from_numpy(rng.integers(0, 256, size=(200, 64)).astype(np.float32))

    whole = _sharpen(plane, 0.5, 8)
    monkeypatch.setattr("jasna.media.cas._BAND_PIXELS", 64 * 7)
    banded = _sharpen(plane, 0.5, 8)

    assert torch.equal(whole, banded)


@pytest.mark.parametrize("strength", [0.0, -0.1, 1.5])
def test_sharpener_rejects_strengths_outside_the_open_unit_range(strength: float) -> None:
    with pytest.raises(ValueError, match="Sharpening strength must be in"):
        GpuCasSharpener(strength, ten_bit=False, device=torch.device("cpu"))


def test_sharpener_leaves_nv12_chroma_untouched() -> None:
    rng = np.random.default_rng(3)
    packed = torch.from_numpy(
        rng.integers(16, 236, size=(24, 16)).astype(np.uint8)
    )
    chroma_before = packed[16:].clone()

    GpuCasSharpener(0.5, ten_bit=False, device=torch.device("cpu")).apply_luma_(packed, 16)

    assert torch.equal(packed[16:], chroma_before)


def test_sharpener_round_trips_the_p010_sample_shift() -> None:
    # P010 keeps 10-bit codes shifted left by six; a flat plane must come back
    # bit-identical through the unpack/repack.
    codes = torch.full((6, 8), 640, dtype=torch.int32)
    packed = (codes << 6).to(torch.int16)
    packed = torch.cat([packed, torch.full((3, 8), 512 << 6, dtype=torch.int32).to(torch.int16)])
    chroma_before = packed[6:].clone()

    GpuCasSharpener(0.5, ten_bit=True, device=torch.device("cpu")).apply_luma_(packed, 6)

    assert torch.equal(packed[:6].view(torch.uint16).to(torch.int32) >> 6, codes)
    assert torch.equal(packed[6:], chroma_before)


def test_sharpener_sharpens_the_luma_plane() -> None:
    packed = torch.full((12, 8), 100, dtype=torch.uint8)
    packed[4, 4] = 200

    GpuCasSharpener(1.0, ten_bit=False, device=torch.device("cpu")).apply_luma_(packed, 8)

    assert packed[4, 4].item() > 200
    assert packed[4, 3].item() < 100


def test_kernel_failure_on_nvidia_warns_and_falls_back(caplog) -> None:
    sharpener = GpuCasSharpener(0.5, ten_bit=False, device=torch.device("cpu"))
    sharpener._kernel = SimpleNamespace(
        launch=lambda *args: (_ for _ in ()).throw(RuntimeError("no fatbin"))
    )
    packed = torch.full((12, 8), 100, dtype=torch.uint8)
    packed[4, 4] = 200

    with caplog.at_level(logging.WARNING, logger="jasna.media.cas"):
        sharpener.apply_luma_(packed, 8)

    assert "falling back to the much slower Torch implementation" in caplog.text
    assert sharpener._kernel is None
    assert packed[4, 4].item() > 200  # the fallback still sharpened the plane


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires an NVIDIA GPU"
)


@requires_cuda
@pytest.mark.parametrize("strength", [0.1, 0.5, 1.0])
def test_cuda_kernel_matches_the_torch_implementation_for_nv12(strength: float) -> None:
    rng = np.random.default_rng(5)
    luma = rng.integers(0, 256, size=(48, 64)).astype(np.uint8)
    packed = torch.from_numpy(np.vstack([luma, np.zeros((24, 64), np.uint8)]))

    on_gpu = packed.cuda()
    GpuCasSharpener(strength, ten_bit=False, device=torch.device("cuda:0")).apply_luma_(
        on_gpu, 48
    )
    GpuCasSharpener(strength, ten_bit=False, device=torch.device("cpu")).apply_luma_(
        packed, 48
    )

    assert torch.equal(on_gpu.cpu(), packed)


@requires_cuda
@pytest.mark.parametrize("strength", [0.1, 0.5, 1.0])
def test_cuda_kernel_matches_the_torch_implementation_for_p010(strength: float) -> None:
    rng = np.random.default_rng(6)
    luma = (rng.integers(0, 1024, size=(48, 64)).astype(np.int32) << 6).astype(np.int32)
    chroma = np.full((24, 64), 512 << 6, dtype=np.int32)
    packed = torch.from_numpy(np.vstack([luma, chroma]).astype(np.uint16)).view(torch.int16)

    on_gpu = packed.cuda()
    GpuCasSharpener(strength, ten_bit=True, device=torch.device("cuda:0")).apply_luma_(
        on_gpu, 48
    )
    GpuCasSharpener(strength, ten_bit=True, device=torch.device("cpu")).apply_luma_(
        packed, 48
    )

    assert torch.equal(on_gpu.cpu(), packed)
