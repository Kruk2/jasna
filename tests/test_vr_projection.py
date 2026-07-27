"""Production VR180 region projectors (jasna/vr_projection.py).

Parity is pinned against scripts/vr_projection.py — the numpy reference the
FFmpeg v360 oracle (scripts/vrproj_ffmpeg_oracle.py) validated — so the torch
port cannot silently drift from the geometry the Phase-4 blind review rated.
"""
import math

import numpy as np
import pytest
import torch

import jasna.vr_projection as vr_projection
from jasna.crop_buffer import compute_enlarged_bbox
from jasna.vr_projection import (
    FisheyeProjector,
    GnomonicProjector,
    build_vr_projector,
)
from scripts import vr_projection as vpref

CPU = torch.device("cpu")


def _ref_forward_eye_uv(kind: str, bbox_uv, patch: int) -> np.ndarray:
    """scripts/-reference forward (patch->eye) uv map."""
    u1, v1, u2, v2 = bbox_uv
    tx = (np.arange(patch) + 0.5) / patch
    pyy, pxx = np.meshgrid(tx, tx, indexing="ij")
    if kind == "gnomonic":
        lon0, lat0, _hf, _vf, xmax, ymax = vpref.region_gnomonic_spec(bbox_uv)
        fov = 2 * math.degrees(math.atan(max(xmax, ymax)))
        uv, _ = vpref.v360_map("flat", patch, patch, h_fov=fov, v_fov=fov, yaw=lon0, pitch=lat0)
        return uv
    pu = np.concatenate([np.linspace(u1, u2, 33), np.full(33, u1), np.full(33, u2), np.linspace(u1, u2, 33)])
    pv = np.concatenate([np.full(33, v1), np.linspace(v1, v2, 33), np.linspace(v1, v2, 33), np.full(33, v2)])
    fuv, _ = vpref.xyz_to_fisheye_uv(vpref.hequirect_uv_to_xyz(pu, pv), 180, 180)
    cxx = (fuv[:, 0].min() + fuv[:, 0].max()) / 2
    cyy = (fuv[:, 1].min() + fuv[:, 1].max()) / 2
    half = max(np.ptp(fuv[:, 0]), np.ptp(fuv[:, 1])) / 2 * 1.06
    f1u, f2u, f1v, f2v = cxx - half, cxx + half, cyy - half, cyy + half
    fdir = vpref.fisheye_uv_to_xyz(f1u + (f2u - f1u) * pxx, f1v + (f2v - f1v) * pyy)
    uv, _ = vpref.xyz_to_hequirect_uv(fdir)
    return uv


@pytest.mark.parametrize("kind", ["gnomonic", "fisheye"])
def test_forward_grid_matches_numpy_reference(kind: str) -> None:
    eye_w, height, patch = 1600, 800, 96
    bbox_uv = (640 / eye_w, 300 / height, 900 / eye_w, 470 / height)
    local = (640, 300, 900, 470)
    proj = build_vr_projector(kind, eye_width=eye_w, height=height, device=CPU)
    got = proj._forward_eye_uv(local, patch, patch).numpy()
    ref = _ref_forward_eye_uv(kind, bbox_uv, patch)
    assert np.abs(got - ref).max() < 2e-4


@pytest.mark.parametrize("kind", ["gnomonic", "fisheye"])
def test_inverse_grid_matches_numpy_reference(kind: str) -> None:
    eye_w, height = 1600, 800
    x1, y1, x2, y2 = 640, 300, 900, 470
    rh, rw = y2 - y1, x2 - x1
    bbox_uv = (x1 / eye_w, y1 / height, x2 / eye_w, y2 / height)
    proj = build_vr_projector(kind, eye_width=eye_w, height=height, device=CPU)
    got = proj._inverse_patch_uv((x1, y1, x2, y2), rh, rw).numpy()

    ru = bbox_uv[0] + (bbox_uv[2] - bbox_uv[0]) * (np.arange(rw) + 0.5) / rw
    rv = bbox_uv[1] + (bbox_uv[3] - bbox_uv[1]) * (np.arange(rh) + 0.5) / rh
    rvv, ruu = np.meshgrid(rv, ru, indexing="ij")
    vec_raw = vpref.hequirect_uv_to_xyz(ruu, rvv)
    if kind == "gnomonic":
        lon0, lat0, _hf, _vf, xmax, ymax = vpref.region_gnomonic_spec(bbox_uv)
        fov = 2 * math.degrees(math.atan(max(xmax, ymax)))
        ref, _ = vpref.xyz_to_flat_uv(vpref.rotate_inv(vec_raw, lon0, lat0, 0.0), fov, fov)
    else:
        pu = np.concatenate([np.linspace(bbox_uv[0], bbox_uv[2], 33), np.full(33, bbox_uv[0]),
                             np.full(33, bbox_uv[2]), np.linspace(bbox_uv[0], bbox_uv[2], 33)])
        pv = np.concatenate([np.full(33, bbox_uv[1]), np.linspace(bbox_uv[1], bbox_uv[3], 33),
                             np.linspace(bbox_uv[1], bbox_uv[3], 33), np.full(33, bbox_uv[3])])
        fuv, _ = vpref.xyz_to_fisheye_uv(vpref.hequirect_uv_to_xyz(pu, pv), 180, 180)
        cxx = (fuv[:, 0].min() + fuv[:, 0].max()) / 2
        cyy = (fuv[:, 1].min() + fuv[:, 1].max()) / 2
        half = max(np.ptp(fuv[:, 0]), np.ptp(fuv[:, 1])) / 2 * 1.06
        f1u, f2u, f1v, f2v = cxx - half, cxx + half, cyy - half, cyy + half
        finv, _ = vpref.xyz_to_fisheye_uv(vec_raw, 180, 180)
        ref = np.stack(((finv[..., 0] - f1u) / (f2u - f1u), (finv[..., 1] - f1v) / (f2v - f1v)), -1)
    assert np.abs(got - ref).max() < 2e-4


@pytest.mark.parametrize("cls", [GnomonicProjector, FisheyeProjector])
def test_projected_crop_is_square_while_source_bbox_is_preserved(cls) -> None:
    proj = cls(eye_width=64, height=64, device=CPU)
    frame = torch.zeros((3, 64, 128), dtype=torch.float32)
    bbox = np.array([10.0, 12.0, 34.0, 40.0], dtype=np.float32)
    raw = proj.extract_region_crop(frame, bbox, 64, 128, x_bounds=(0, 64))
    x1, y1, x2, y2 = compute_enlarged_bbox(bbox, 64, 128, (0, 64))
    patch_size = max(y2 - y1, x2 - x1)
    assert raw.enlarged_bbox == (x1, y1, x2, y2)
    assert raw.crop_shape == (patch_size, patch_size)
    assert tuple(raw.crop.shape) == (3, patch_size, patch_size)


@pytest.mark.parametrize("cls", [GnomonicProjector, FisheyeProjector])
def test_project_region_matches_extract_crop(cls) -> None:
    # crop/inverse parameter identity: project_region reproduces the exact patch
    # extract_region_crop fed to the restorer (needed for the delta composite).
    proj = cls(eye_width=200, height=120, device=CPU)
    torch.manual_seed(0)
    frame = torch.randint(0, 255, (3, 120, 400), dtype=torch.float32)
    bbox = np.array([60.0, 30.0, 140.0, 90.0], dtype=np.float32)
    raw = proj.extract_region_crop(frame, bbox, 120, 400, x_bounds=(0, 200))
    projected = proj.project_region(frame, raw.enlarged_bbox)
    assert projected.shape == raw.crop.shape
    assert torch.allclose(projected, raw.crop.float(), atol=1e-4)


@pytest.mark.parametrize("cls", [GnomonicProjector, FisheyeProjector])
def test_project_region_matches_quantized_uint8_crop(cls) -> None:
    proj = cls(eye_width=200, height=120, device=CPU)
    torch.manual_seed(4)
    frame = torch.randint(0, 256, (3, 120, 400), dtype=torch.uint8)
    bbox = np.array([60.0, 30.0, 140.0, 90.0], dtype=np.float32)

    raw = proj.extract_region_crop(frame, bbox, 120, 400, x_bounds=(0, 200))
    projected = proj.project_region(frame, raw.enlarged_bbox)

    assert torch.equal(projected, raw.crop.float())


def _coordinate_ramp(height: int, width: int) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack((x, y, torch.zeros_like(x)))


def test_forward_sampling_uses_ffmpeg_source_pixel_coordinates(monkeypatch) -> None:
    projector = GnomonicProjector(eye_width=7, height=5, device=CPU)
    monkeypatch.setattr(
        projector,
        "_forward_eye_uv",
        lambda _bbox, _height, _width: torch.tensor([[[0.25, 0.75]]]),
    )
    frame = torch.cat((_coordinate_ramp(5, 7), torch.zeros((3, 5, 7))), dim=2)

    sampled = projector._forward_sample(frame, 0, (0, 0, 1, 1), 1, 1)

    assert torch.allclose(sampled[:, 0, 0], torch.tensor([1.5, 3.0, 0.0]))


def test_inverse_sampling_uses_ffmpeg_source_pixel_coordinates(monkeypatch) -> None:
    projector = GnomonicProjector(eye_width=7, height=5, device=CPU)
    monkeypatch.setattr(
        projector,
        "_inverse_patch_uv",
        lambda _bbox, _height, _width: torch.tensor([[[0.25, 0.75]]]),
    )

    sampled = projector.source_region_from_patch(
        _coordinate_ramp(5, 7),
        (0, 0, 1, 1),
    )

    assert torch.allclose(sampled[:, 0, 0], torch.tensor([1.5, 3.0, 0.0]))


@pytest.mark.parametrize("cls", [GnomonicProjector, FisheyeProjector])
def test_noop_delta_leaves_source_unchanged(cls) -> None:
    # When the restorer returns its input unchanged, the projected delta is 0 and
    # the source region must round-trip bit-for-bit (delta compositing invariant).
    proj = cls(eye_width=200, height=120, device=CPU)
    torch.manual_seed(1)
    frame = torch.randint(0, 255, (3, 120, 400), dtype=torch.float32)
    bbox = np.array([60.0, 30.0, 140.0, 90.0], dtype=np.float32)
    raw = proj.extract_region_crop(frame, bbox, 120, 400, x_bounds=(0, 200))
    original_projected = proj.project_region(frame, raw.enlarged_bbox)
    restored = original_projected.clone()  # no-op restoration
    source_delta = proj.source_region_from_patch(restored - original_projected, raw.enlarged_bbox)
    assert source_delta.abs().max().item() == 0.0


@pytest.mark.parametrize("cls", [GnomonicProjector, FisheyeProjector])
def test_extract_selects_the_correct_eye(cls) -> None:
    proj = cls(eye_width=64, height=64, device=CPU)
    frame = torch.zeros((3, 64, 128), dtype=torch.float32)
    frame[:, :, :64] = 30.0
    frame[:, :, 64:] = 200.0
    left = proj.extract_region_crop(
        frame, np.array([8.0, 8.0, 28.0, 28.0], dtype=np.float32), 64, 128, x_bounds=(0, 64)
    )
    right = proj.extract_region_crop(
        frame, np.array([72.0, 8.0, 92.0, 28.0], dtype=np.float32), 64, 128, x_bounds=(64, 128)
    )
    assert left.crop.max().item() <= 31.0
    assert right.crop.min().item() >= 199.0
    assert right.enlarged_bbox[0] >= 64


@pytest.mark.parametrize("cls", [GnomonicProjector, FisheyeProjector])
def test_uint8_frame_yields_uint8_crop(cls) -> None:
    proj = cls(eye_width=64, height=64, device=CPU)
    frame = torch.zeros((3, 64, 128), dtype=torch.uint8)
    frame[:, :, :64] = 120
    raw = proj.extract_region_crop(
        frame, np.array([8.0, 8.0, 40.0, 40.0], dtype=np.float32), 64, 128, x_bounds=(0, 64)
    )
    assert raw.crop.dtype == torch.uint8


@pytest.mark.parametrize("cls", [GnomonicProjector, FisheyeProjector])
def test_projector_rejects_degenerate_dimensions(cls) -> None:
    with pytest.raises(ValueError, match="Invalid eye dimensions"):
        cls(eye_width=0, height=64, device=CPU)


@pytest.mark.parametrize("cls", [GnomonicProjector, FisheyeProjector])
def test_projection_geometry_does_not_convert_tensors_to_python(
    cls,
    monkeypatch,
) -> None:
    builtin_float = float

    def checked_float(value):
        if isinstance(value, torch.Tensor):
            raise AssertionError("projection geometry synchronized a tensor")
        return builtin_float(value)

    monkeypatch.setattr(vr_projection, "float", checked_float, raising=False)
    proj = cls(eye_width=64, height=64, device=CPU)
    frame = torch.zeros((3, 64, 128), dtype=torch.float32)
    proj.extract_region_crop(
        frame,
        np.array([8.0, 12.0, 42.0, 36.0], dtype=np.float32),
        64,
        128,
        x_bounds=(0, 64),
    )


def test_factory_routes_projection_kinds() -> None:
    kw = dict(eye_width=64, height=64, device=CPU)
    assert build_vr_projector("raw", **kw) is None
    assert build_vr_projector("none", **kw) is None
    assert isinstance(build_vr_projector("fisheye", **kw), FisheyeProjector)
    assert isinstance(build_vr_projector("gnomonic", **kw), GnomonicProjector)


def test_factory_rejects_unknown_projection() -> None:
    with pytest.raises(ValueError, match="Unknown VR projection"):
        build_vr_projector(
            "cylindrical",
            eye_width=64,
            height=64,
            device=CPU,
        )


def test_axis_tensors_are_reused_across_calls():
    """Sampling axes depend only on their length, so they must be built once."""
    device = torch.device("cpu")
    vr_projection._AXIS_CACHE.clear()

    first = vr_projection._centers(64, device)
    second = vr_projection._centers(64, device)
    unit_first = vr_projection._unit_centers(64, device)
    unit_second = vr_projection._unit_centers(64, device)

    assert first is second
    assert unit_first is unit_second
    assert first is not unit_first
    assert len(vr_projection._AXIS_CACHE) == 2


def test_rotation_matrix_matches_the_composed_form():
    """The closed form must equal ry @ rx @ rz, which it replaced."""
    device = torch.device("cpu")
    for yaw, pitch, roll in ((0.0, 0.0, 0.0), (37.0, -12.0, 0.0), (-140.0, 61.0, 25.0)):
        a, b, c = yaw * vr_projection.DEG, -pitch * vr_projection.DEG, roll * vr_projection.DEG
        cy, sy = math.cos(a), math.sin(a)
        cp, sp = math.cos(b), math.sin(b)
        cr, sr = math.cos(c), math.sin(c)
        ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float32)
        rx = torch.tensor([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=torch.float32)
        rz = torch.tensor([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=torch.float32)

        composed = ry @ rx @ rz
        closed = vr_projection._rot_matrix(yaw, pitch, roll, device)

        assert torch.allclose(closed, composed, atol=1e-6)
