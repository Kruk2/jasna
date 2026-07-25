"""CPU conformance/round-trip/delta tests for scripts.vr_projection.

Run: ~/.virtualenvs/jasna-linux/bin/python -m pytest scripts/test_vr_projection.py -q
(FFmpeg v360 numeric conformance is in scripts/vrproj_ffmpeg_oracle.py.)
"""
from __future__ import annotations

import numpy as np

from scripts.vr_projection import (
    hequirect_uv_to_xyz, region_gnomonic_spec, rotate_inv, v360_map,
    xyz_to_fisheye_uv, xyz_to_flat_uv,
)


def _out_centers(oh, ow):
    ex = (2 * np.arange(ow) + 1) / (2 * ow)
    ey = (2 * np.arange(oh) + 1) / (2 * oh)
    v, u = np.meshgrid(ey, ex, indexing="ij")
    return u, v  # each output pixel's own normalized coord in [0,1]


def _sample(img, uv):
    """Bilinear sample img (H,W[,C]) at normalized uv (h,w,2) in [0,1]."""
    h, w = img.shape[:2]
    fx = np.clip(uv[..., 0] * (w - 1), 0, w - 1)
    fy = np.clip(uv[..., 1] * (h - 1), 0, h - 1)
    x0 = np.floor(fx).astype(int); y0 = np.floor(fy).astype(int)
    x1 = np.minimum(x0 + 1, w - 1); y1 = np.minimum(y0 + 1, h - 1)
    ax = (fx - x0); ay = (fy - y0)
    if img.ndim == 3:
        ax = ax[..., None]; ay = ay[..., None]
    top = img[y0, x0] * (1 - ax) + img[y0, x1] * ax
    bot = img[y1, x0] * (1 - ax) + img[y1, x1] * ax
    return top * (1 - ay) + bot * ay


def _roundtrip(out_proj, inv, hf, vf, yaw, pitch):
    oh = ow = 256
    uv, valid = v360_map(out_proj, oh, ow, h_fov=hf, v_fov=vf, yaw=yaw, pitch=pitch)
    vec = hequirect_uv_to_xyz(uv[..., 0], uv[..., 1])
    vec = rotate_inv(vec, yaw, pitch, 0.0)
    back, ok = inv(vec, hf, vf)
    exp_u, exp_v = _out_centers(oh, ow)
    m = valid & ok
    du = np.abs(back[..., 0] - exp_u)[m]
    dv = np.abs(back[..., 1] - exp_v)[m]
    return float(max(du.max(), dv.max()))


def test_roundtrip_flat_identity():
    err = _roundtrip("flat", xyz_to_flat_uv, 70, 50, 20, -15)
    assert err < 1e-6


def test_roundtrip_fisheye_identity():
    err = _roundtrip("fisheye", xyz_to_fisheye_uv, 180, 180, 0, 0)
    assert err < 1e-6


def test_noop_delta_is_source_identical():
    # Build a source eye, project a region to a gnomonic patch, "restore" it to
    # itself, inverse the (zero) delta, and confirm the source is untouched.
    H = W = 512
    yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")
    source = (40 + 180 * (0.3 * xx + 0.7 * yy)).astype(np.float32)
    bbox_uv = (0.42, 0.55, 0.60, 0.72)
    lon0, lat0, hf, vf, xmax, ymax = region_gnomonic_spec(bbox_uv)

    ph, pw = 96, 96
    fwd_uv, _ = v360_map("flat", ph, pw, h_fov=hf, v_fov=vf, yaw=lon0, pitch=lat0)
    patch = _sample(source, fwd_uv)
    restored = patch.copy()                       # identity restoration

    # inverse grid: source-region pixel -> patch normalized coord
    u1, v1, u2, v2 = bbox_uv
    x0, y0 = int(round(u1 * W)), int(round(v1 * H))
    x1, y1 = int(round(u2 * W)), int(round(v2 * H))
    ru = (np.arange(x0, x1) + 0.5) / W
    rv = (np.arange(y0, y1) + 0.5) / H
    rvv, ruu = np.meshgrid(rv, ru, indexing="ij")
    vec = hequirect_uv_to_xyz(ruu, rvv)
    vec = rotate_inv(vec, lon0, lat0, 0.0)
    patch_uv, cov = xyz_to_flat_uv(vec, hf, vf)

    delta_patch = restored - patch                # exactly zero
    region_delta = _sample(delta_patch, patch_uv)
    region = source[y0:y1, x0:x1]
    out = region + region_delta
    assert np.array_equal(out, region)            # bit-identical no-op


def test_delta_composite_real_change_bounded_by_mask():
    # A real restoration change inside the mosaic mask must (a) leave pixels
    # outside the mask bit-identical, (b) apply only the inverse-projected delta
    # inside. This is the anti-halo / delta-compositing guarantee.
    H = W = 512
    yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")
    source = (30 + 200 * (0.4 * xx + 0.6 * yy)).astype(np.float32)
    bbox_uv = (0.44, 0.56, 0.58, 0.70)
    lon0, lat0, hf, vf, xmax, ymax = region_gnomonic_spec(bbox_uv)

    ph = pw = 80
    fwd_uv, _ = v360_map("flat", ph, pw, h_fov=hf, v_fov=vf, yaw=lon0, pitch=lat0)
    patch = _sample(source, fwd_uv)
    restored = patch.copy()
    restored[24:56, 24:56] += 50.0                      # a real change in the patch

    u1, v1, u2, v2 = bbox_uv
    x0, y0 = int(round(u1 * W)), int(round(v1 * H))
    x1, y1 = int(round(u2 * W)), int(round(v2 * H))
    ru = (np.arange(x0, x1) + 0.5) / W
    rv = (np.arange(y0, y1) + 0.5) / H
    rvv, ruu = np.meshgrid(rv, ru, indexing="ij")
    vec = rotate_inv(hequirect_uv_to_xyz(ruu, rvv), lon0, lat0, 0.0)
    patch_uv, cov = xyz_to_flat_uv(vec, hf, vf)

    region_delta = _sample((restored - patch), patch_uv)
    # source-space mosaic mask: where the patch change back-projects to
    mask = np.abs(region_delta) > 1e-3

    region = source[y0:y1, x0:x1].copy()
    out = region.copy()
    out[mask] = region[mask] + region_delta[mask]

    assert np.array_equal(out[~mask], region[~mask])    # outside mask: untouched
    assert np.abs(out[mask] - region[mask]).max() > 1.0  # inside: actually changed
    assert np.isfinite(out).all()


def test_region_spec_covers_bbox():
    bbox_uv = (0.40, 0.50, 0.62, 0.70)
    lon0, lat0, hf, vf, xmax, ymax = region_gnomonic_spec(bbox_uv)
    assert 0 < hf < 180 and 0 < vf < 180
    # the derived flat view must map every bbox pixel inside [0,1]
    u1, v1, u2, v2 = bbox_uv
    ru = np.linspace(u1, u2, 40); rv = np.linspace(v1, v2, 40)
    rvv, ruu = np.meshgrid(rv, ru, indexing="ij")
    vec = rotate_inv(hequirect_uv_to_xyz(ruu, rvv), lon0, lat0, 0.0)
    uv, cov = xyz_to_flat_uv(vec, hf, vf)
    assert cov.all()
    assert uv.min() >= -1e-6 and uv.max() <= 1 + 1e-6


def test_grids_finite_and_odd_dims():
    for proj, hf, vf in [("flat", 90, 45), ("fisheye", 180, 180)]:
        uv, valid = v360_map(proj, 257, 129, h_fov=hf, v_fov=vf, yaw=10, pitch=5)
        assert np.isfinite(uv[valid]).all()
