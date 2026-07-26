from __future__ import annotations

import numpy as np
import pytest
import torch

from jasna.media.lut import GpuLutApplier, parse_cube_text

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _cube_3d(size: int, transform, domain: tuple[float, float] | None = None) -> str:
    lines = []
    if domain is not None:
        lines.append(f"DOMAIN_MIN {domain[0]} {domain[0]} {domain[0]}")
        lines.append(f"DOMAIN_MAX {domain[1]} {domain[1]} {domain[1]}")
    lines.append(f"LUT_3D_SIZE {size}")
    for b in range(size):
        for g in range(size):
            for r in range(size):
                values = transform(r / (size - 1), g / (size - 1), b / (size - 1))
                lines.append(" ".join(f"{v:.6f}" for v in values))
    return "\n".join(lines)


def _cube_1d(size: int, transform) -> str:
    lines = [f"LUT_1D_SIZE {size}"]
    for i in range(size):
        values = transform(i / (size - 1))
        lines.append(" ".join(f"{v:.6f}" for v in values))
    return "\n".join(lines)


CUBES = {
    "3d-identity": _cube_3d(17, lambda r, g, b: (r, g, b)),
    "3d-swap-rb": _cube_3d(17, lambda r, g, b: (b, g, r)),
    "3d-gamma": _cube_3d(33, lambda r, g, b: (r**0.45, g**0.45, b**0.45)),
    "3d-narrow-domain": _cube_3d(17, lambda r, g, b: (r, g, b), domain=(0.1, 0.9)),
    "3d-clipped": _cube_3d(9, lambda r, g, b: (min(1.0, r * 1.5), g, max(0.0, b - 0.2))),
    "1d-identity": _cube_1d(64, lambda v: (v, v, v)),
    "1d-lift-blacks": _cube_1d(64, lambda v: (0.1 + 0.9 * v, v**0.5, min(1.0, v * 1.2))),
    "1d-small": _cube_1d(2, lambda v: (v, 1.0 - v, 0.5)),
}


def _device() -> torch.device:
    return torch.device("cuda:0")


def _reference(applier: GpuLutApplier, frame: torch.Tensor) -> torch.Tensor:
    kernel, applier._kernel = applier._kernel, None
    try:
        return applier.apply(frame)
    finally:
        applier._kernel = kernel


def _exact(applier: GpuLutApplier, frame: torch.Tensor) -> torch.Tensor:
    """Trilinear/linear LUT application in float64, as ground truth.

    Both the kernel and the Torch path work in float32, where a sample can land
    either side of a .5 rounding boundary purely on accumulation order. Scoring
    both against float64 says which one is actually right, instead of pinning
    the kernel to whichever way Torch happened to round.
    """
    size = applier.lut.size
    values = frame.double().cpu().numpy() / 255.0
    minimum = applier._domain_min_flat.double().cpu().numpy()
    scale = applier._domain_scale_flat.double().cpu().numpy()
    normalized = ((values - minimum[:, None, None]) * scale[:, None, None]).clip(0.0, 1.0)

    if applier.lut.is_3d:
        cube = applier.lut.data.double().cpu().numpy().transpose(1, 2, 3, 0)  # (B, G, R, c)
        coordinates = normalized * (size - 1)
        base = np.floor(coordinates)
        fraction = coordinates - base
        out = np.zeros_like(coordinates)
        for corner in range(8):
            weight = np.ones(coordinates.shape[1:])
            index = []
            for axis in range(3):
                step = (corner >> axis) & 1
                weight = weight * (fraction[axis] if step else 1.0 - fraction[axis])
                index.append(np.clip(base[axis].astype(int) + step, 0, size - 1))
            out += cube[index[2], index[1], index[0]].transpose(2, 0, 1) * weight
        return torch.from_numpy(out)

    table = applier.lut.data.double().cpu().numpy()  # (N, 3)
    coordinates = normalized * (size - 1)
    low = np.clip(np.floor(coordinates).astype(int), 0, size - 1)
    high = np.clip(low + 1, 0, size - 1)
    fraction = np.clip(coordinates - low, 0.0, 1.0)
    out = np.stack([
        table[low[c], c] + (table[high[c], c] - table[low[c], c]) * fraction[c]
        for c in range(3)
    ])
    return torch.from_numpy(out)


def _codes(values: torch.Tensor) -> torch.Tensor:
    return (values * 255.0).round().clamp(0, 255).to(torch.int32)


@pytest.mark.parametrize("name", sorted(CUBES))
def test_kernel_is_at_least_as_accurate_as_the_torch_path(name):
    generator = torch.Generator(device="cuda").manual_seed(0)
    frame = torch.randint(
        0, 256, (3, 48, 80), generator=generator, device=_device(), dtype=torch.uint8
    )
    applier = GpuLutApplier(parse_cube_text(CUBES[name]), _device())
    assert applier._kernel is not None

    exact = _codes(_exact(applier, frame))
    kernel_error = (applier.apply(frame).int().cpu() - exact).abs()
    torch_error = (_reference(applier, frame).int().cpu() - exact).abs()

    assert kernel_error.max().item() <= 1
    # Measured on the dev box the kernel is marginally better than Torch on the
    # steep cube and identical on the rest; the slack only absorbs a different
    # FMA contraction on another architecture.
    assert kernel_error.sum().item() <= torch_error.sum().item() + exact.numel() // 1000


@pytest.mark.parametrize("name", sorted(n for n in CUBES if n.startswith("1d")))
def test_one_dimensional_lut_matches_the_torch_path_exactly(name):
    generator = torch.Generator(device="cuda").manual_seed(0)
    frame = torch.randint(
        0, 256, (3, 48, 80), generator=generator, device=_device(), dtype=torch.uint8
    )
    applier = GpuLutApplier(parse_cube_text(CUBES[name]), _device())

    assert torch.equal(applier.apply(frame), _reference(applier, frame))


@pytest.mark.parametrize("name", sorted(CUBES))
def test_kernel_matches_the_reference_on_extreme_codes(name):
    frame = torch.zeros((3, 4, 6), device=_device(), dtype=torch.uint8)
    frame[:, 0] = 0
    frame[:, 1] = 255
    frame[0, 2] = 255
    frame[:, 3] = 128
    applier = GpuLutApplier(parse_cube_text(CUBES[name]), _device())

    assert torch.equal(applier.apply(frame), _reference(applier, frame))


def test_identity_lut_is_lossless_through_the_kernel():
    generator = torch.Generator(device="cuda").manual_seed(1)
    frame = torch.randint(
        0, 256, (3, 16, 16), generator=generator, device=_device(), dtype=torch.uint8
    )
    applier = GpuLutApplier(parse_cube_text(CUBES["3d-identity"]), _device())

    assert torch.equal(applier.apply(frame), frame)


def test_kernel_output_is_a_fresh_contiguous_tensor():
    frame = torch.randint(0, 256, (3, 16, 16), device=_device(), dtype=torch.uint8)
    applier = GpuLutApplier(parse_cube_text(CUBES["3d-swap-rb"]), _device())

    out = applier.apply(frame)

    assert out.data_ptr() != frame.data_ptr()
    assert out.is_contiguous()
    assert out.dtype is torch.uint8


def test_float_input_still_uses_the_torch_path():
    frame = torch.rand((3, 8, 8), device=_device(), dtype=torch.float32)
    applier = GpuLutApplier(parse_cube_text(CUBES["3d-identity"]), _device())

    out = applier.apply(frame)

    assert out.dtype is torch.float32
