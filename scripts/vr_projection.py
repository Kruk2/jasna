"""VR180 projection library — raw / fisheye / gnomonic, matched to FFmpeg v360.

Reference oracle: FFmpeg 8.1.2 `v360` (input `hequirect`, output `flat` /
`fisheye`). Conventions are validated empirically by scripts/vrproj_ffmpeg_oracle.py
against real `v360` runs; see that harness for the conformance thresholds.

All maps are OUTPUT->INPUT: for each output pixel we compute the normalized
[0,1] source coordinate to sample from the stored half-equirectangular eye.
FFmpeg converts that coordinate to an absolute source position with
``uv * (size - 1)``; Torch consumers must therefore use
``grid_sample(..., align_corners=True)``.
Everything here is numpy/float64 for the CPU oracle; the GPU/product port comes
later (Phase 5) once routing is decided.

Conventions (v360, verified by the oracle):
- pixel centers: p = (2*idx + 1)/N - 1  in (-1, 1)
- rotation order ypr (yaw, pitch, roll), degrees
- flat FOV default 90h/45v, fisheye FOV default 180/180 (diagonal overrides)
- image y is down; world y is up (note the sign flips)
"""
from __future__ import annotations

import math

import numpy as np

DEG = math.pi / 180.0


def _centers(n: int) -> np.ndarray:
    return (2.0 * np.arange(n, dtype=np.float64) + 1.0) / n - 1.0


def flat_to_xyz(h: int, w: int, h_fov: float, v_fov: float) -> np.ndarray:
    """Output rectilinear/gnomonic pixel -> 3D direction (h,w,3), normalized."""
    py, px = np.meshgrid(_centers(h), _centers(w), indexing="ij")
    rx = math.tan(0.5 * h_fov * DEG)
    ry = math.tan(0.5 * v_fov * DEG)
    x = px * rx
    y = -py * ry
    z = np.ones_like(x)
    vec = np.stack((x, y, z), axis=-1)
    return vec / np.linalg.norm(vec, axis=-1, keepdims=True)


def fisheye_to_xyz(h: int, w: int, h_fov: float, v_fov: float) -> np.ndarray:
    """Output equidistant-fisheye pixel -> 3D direction (h,w,3)."""
    py, px = np.meshgrid(_centers(h), _centers(w), indexing="ij")
    uf = px * (h_fov / 180.0)
    vf = py * (v_fov / 180.0)
    r = np.hypot(uf, vf)
    theta = r * math.pi * 0.5  # r=1 -> 90deg for 180 fov
    safe = np.where(r > 1e-12, r, 1.0)
    sin_t = np.sin(theta)
    x = sin_t * uf / safe
    y = -sin_t * vf / safe
    z = np.cos(theta)
    x = np.where(r > 1e-12, x, 0.0)
    y = np.where(r > 1e-12, y, 0.0)
    vec = np.stack((x, y, z), axis=-1)
    return vec / np.linalg.norm(vec, axis=-1, keepdims=True)


def _rot_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """ypr rotation matrix applied to direction vectors (world axes: x right,
    y up, z forward)."""
    # FFmpeg v360 pitch is positive-down relative to our world-up convention.
    a, b, c = yaw * DEG, -pitch * DEG, roll * DEG
    cy, sy = math.cos(a), math.sin(a)
    cp, sp = math.cos(b), math.sin(b)
    cr, sr = math.cos(c), math.sin(c)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])       # yaw about y
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])        # pitch about x
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])        # roll about z
    return Ry @ Rx @ Rz


def rotate(vec: np.ndarray, yaw: float, pitch: float, roll: float) -> np.ndarray:
    if yaw == 0.0 and pitch == 0.0 and roll == 0.0:
        return vec
    R = _rot_matrix(yaw, pitch, roll)
    return vec @ R.T


def xyz_to_hequirect_uv(vec: np.ndarray) -> np.ndarray:
    """3D direction -> input half-equirect normalized (u,v) in [0,1], plus a
    validity mask. hequirect spans longitude [-90,90], latitude [-90,90]."""
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    lon = np.arctan2(x, z)
    lat = np.arctan2(y, np.hypot(x, z))
    u = lon / math.pi + 0.5          # [-pi/2,pi/2] -> [0,1]
    v = 0.5 - lat / math.pi          # image y down: top=+lat
    uv = np.stack((u, v), axis=-1)
    valid = (np.abs(lon) <= math.pi / 2 + 1e-9) & (np.abs(lat) <= math.pi / 2 + 1e-9)
    return uv, valid


_OUT = {"flat": flat_to_xyz, "gnomonic": flat_to_xyz, "fisheye": fisheye_to_xyz}


def v360_map(
    out_proj: str, out_h: int, out_w: int, *,
    h_fov: float, v_fov: float, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0,
):
    """Return (uv, valid): output->input(hequirect) normalized map. uv is
    (out_h,out_w,2) in [0,1]; valid is the in-range mask."""
    vec = _OUT[out_proj](out_h, out_w, h_fov, v_fov)
    vec = rotate(vec, yaw, pitch, roll)
    return xyz_to_hequirect_uv(vec)


# --- inverse transforms (source hequirect -> output projection) ---

def hequirect_uv_to_xyz(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    lon = (u - 0.5) * math.pi
    lat = (0.5 - v) * math.pi
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)
    return np.stack((x, y, z), axis=-1)


def rotate_inv(vec: np.ndarray, yaw: float, pitch: float, roll: float) -> np.ndarray:
    if yaw == 0.0 and pitch == 0.0 and roll == 0.0:
        return vec
    return vec @ _rot_matrix(yaw, pitch, roll)  # inverse = R^T applied as vec @ R


def xyz_to_flat_uv(vec: np.ndarray, h_fov: float, v_fov: float):
    """Direction -> flat output normalized (u,v) in [0,1] + valid (in front, in FOV)."""
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    zc = np.where(np.abs(z) < 1e-12, 1e-12, z)
    px = (x / zc) / math.tan(0.5 * h_fov * DEG)
    py = -(y / zc) / math.tan(0.5 * v_fov * DEG)
    u = (px + 1.0) * 0.5
    v = (py + 1.0) * 0.5
    valid = (z > 0) & (np.abs(px) <= 1.0) & (np.abs(py) <= 1.0)
    return np.stack((u, v), axis=-1), valid


def xyz_to_fisheye_uv(vec: np.ndarray, h_fov: float, v_fov: float):
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    r = theta / (math.pi * 0.5)
    sin_t = np.sin(theta)
    safe = np.where(sin_t > 1e-12, sin_t, 1.0)
    uf = np.where(sin_t > 1e-12, x * r / safe, 0.0)
    vf = np.where(sin_t > 1e-12, -y * r / safe, 0.0)
    px = uf / (h_fov / 180.0)
    py = vf / (v_fov / 180.0)
    u = (px + 1.0) * 0.5
    v = (py + 1.0) * 0.5
    valid = (r <= 1.0) & (np.abs(px) <= 1.0) & (np.abs(py) <= 1.0)
    return np.stack((u, v), axis=-1), valid


def fisheye_uv_to_xyz(fu: np.ndarray, fv: np.ndarray) -> np.ndarray:
    """Fisheye output normalized (u,v) in [0,1] -> 3D direction (equidistant 180)."""
    px = fu * 2.0 - 1.0
    py = fv * 2.0 - 1.0
    r = np.hypot(px, py)
    theta = r * math.pi * 0.5
    sin_t = np.sin(theta)
    safe = np.where(r > 1e-12, r, 1.0)
    x = np.where(r > 1e-12, sin_t * px / safe, 0.0)
    y = np.where(r > 1e-12, -sin_t * py / safe, 0.0)
    z = np.cos(theta)
    return np.stack((x, y, z), axis=-1)


# --- region gnomonic: tangent centre + extents derived from the source region ---

def region_gnomonic_spec(bbox_uv: tuple[float, float, float, float]):
    """From a source-region bbox in hequirect normalized coords (u1,v1,u2,v2),
    derive (yaw, pitch, h_fov, v_fov) for a tangent-plane view whose FOV is the
    boundary's angular extent (square-pixel, aspect preserved via extents)."""
    u1, v1, u2, v2 = bbox_uv
    lon0 = ((u1 + u2) * 0.5 - 0.5) * 180.0
    lat0 = (0.5 - (v1 + v2) * 0.5) * 180.0
    # dense perimeter -> tangent-plane extents (not just 4 corners)
    us = np.concatenate([np.linspace(u1, u2, 33), np.full(33, u1), np.full(33, u2), np.linspace(u1, u2, 33)])
    vs = np.concatenate([np.full(33, v1), np.linspace(v1, v2, 33), np.linspace(v1, v2, 33), np.full(33, v2)])
    vec = hequirect_uv_to_xyz(us, vs)
    vec = rotate_inv(vec, lon0, lat0, 0.0)  # bring region centre to +z
    zc = np.where(np.abs(vec[:, 2]) < 1e-12, 1e-12, vec[:, 2])
    xmax = float(np.max(np.abs(vec[:, 0] / zc)))
    ymax = float(np.max(np.abs(vec[:, 1] / zc)))
    h_fov = 2.0 * math.degrees(math.atan(xmax))
    v_fov = 2.0 * math.degrees(math.atan(ymax))
    return lon0, lat0, h_fov, v_fov, xmax, ymax
