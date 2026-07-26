"""VR180 region projection for restoration conditioning (raw / fisheye / gnomonic).

Torch port of ``scripts/vr_projection.py``, matched to FFmpeg 8.1 ``v360``
(``hequirect`` input; ``flat`` / ``fisheye`` output) and validated against real
``v360`` runs by ``scripts/vrproj_ffmpeg_oracle.py``. ``tests/test_vr_projection.py``
pins this torch port to that numpy reference.

Detection, tracking, masks, and blend registration stay in source
half-equirectangular coordinates for every SBS mode. Only the mosaic-region crop
handed to the 2D restorer is reprojected here, and only the restoration *delta*
is projected back (see ``BlendBuffer``), so source pixels outside the blend mask
survive unchanged.

All maps are output->input: for each output pixel we compute the normalized
source coordinate to sample. FFmpeg maps that coordinate to ``uv * (size - 1)``,
which is PyTorch's ``align_corners=True`` convention. Grids are built on the
frame's device in float32.
"""
from __future__ import annotations

import logging
import math
from functools import lru_cache

import torch
import torch.nn.functional as F

from jasna.crop_buffer import RawCrop, compute_enlarged_bbox

log = logging.getLogger(__name__)

DEG = math.pi / 180.0
_PERIM = 33


# Sampling axes depend only on their length, and a video reuses a handful of
# patch and region sizes, so they are built once instead of per frame per
# region. Consumers only ever read them into new tensors.
_AXIS_CACHE: dict[tuple[str, int, torch.device], torch.Tensor] = {}


def _cached_axis(kind: str, n: int, device: torch.device, build) -> torch.Tensor:
    key = (kind, n, device)
    axis = _AXIS_CACHE.get(key)
    if axis is None:
        axis = build()
        _AXIS_CACHE[key] = axis
    return axis


def _centers(n: int, device: torch.device) -> torch.Tensor:
    """v360 pixel centers p = (2i+1)/n - 1 in (-1, 1)."""
    return _cached_axis(
        "v360",
        n,
        device,
        lambda: (2.0 * torch.arange(n, device=device, dtype=torch.float32) + 1.0) / n - 1.0,
    )


def _unit_centers(n: int, device: torch.device) -> torch.Tensor:
    """Pixel centers (i + 0.5) / n in (0, 1)."""
    return _cached_axis(
        "unit",
        n,
        device,
        lambda: (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n,
    )


def _flat_to_xyz(h: int, w: int, h_fov: float, v_fov: float, device) -> torch.Tensor:
    py, px = torch.meshgrid(_centers(h, device), _centers(w, device), indexing="ij")
    x = px * math.tan(0.5 * h_fov * DEG)
    y = -py * math.tan(0.5 * v_fov * DEG)
    z = torch.ones_like(x)
    vec = torch.stack((x, y, z), dim=-1)
    return vec / vec.norm(dim=-1, keepdim=True)


def _rot_matrix(yaw: float, pitch: float, roll: float, device) -> torch.Tensor:
    # v360 pitch is positive-down relative to our world-up convention.
    a, b, c = yaw * DEG, -pitch * DEG, roll * DEG
    cy, sy = math.cos(a), math.sin(a)
    cp, sp = math.cos(b), math.sin(b)
    cr, sr = math.cos(c), math.sin(c)
    # ry @ rx @ rz, multiplied out in double on the host: the tensor form cost
    # three host-to-device transfers and two 3x3 matmuls per region per frame,
    # which was a fifth of the whole grid build.
    rows = (
        (cy * cr + sy * sp * sr, -cy * sr + sy * sp * cr, sy * cp),
        (cp * sr, cp * cr, -sp),
        (-sy * cr + cy * sp * sr, sy * sr + cy * sp * cr, cy * cp),
    )
    return torch.tensor(rows, dtype=torch.float32, device=device)


def _rotate(vec: torch.Tensor, yaw: float, pitch: float, roll: float) -> torch.Tensor:
    if yaw == 0.0 and pitch == 0.0 and roll == 0.0:
        return vec
    return vec @ _rot_matrix(yaw, pitch, roll, vec.device).T


def _rotate_inv(vec: torch.Tensor, yaw: float, pitch: float, roll: float) -> torch.Tensor:
    if yaw == 0.0 and pitch == 0.0 and roll == 0.0:
        return vec
    return vec @ _rot_matrix(yaw, pitch, roll, vec.device)


def _xyz_to_hequirect_uv(vec: torch.Tensor) -> torch.Tensor:
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    lon = torch.atan2(x, z)
    lat = torch.atan2(y, torch.hypot(x, z))
    u = lon / math.pi + 0.5
    v = 0.5 - lat / math.pi
    return torch.stack((u, v), dim=-1)


def _hequirect_uv_to_xyz(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    lon = (u - 0.5) * math.pi
    lat = (0.5 - v) * math.pi
    x = torch.cos(lat) * torch.sin(lon)
    y = torch.sin(lat)
    z = torch.cos(lat) * torch.cos(lon)
    return torch.stack((x, y, z), dim=-1)


def _xyz_to_flat_uv(vec: torch.Tensor, h_fov: float, v_fov: float) -> torch.Tensor:
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    zc = torch.where(z.abs() < 1e-12, torch.full_like(z, 1e-12), z)
    px = (x / zc) / math.tan(0.5 * h_fov * DEG)
    py = -(y / zc) / math.tan(0.5 * v_fov * DEG)
    return torch.stack(((px + 1.0) * 0.5, (py + 1.0) * 0.5), dim=-1)


def _xyz_to_fisheye_uv(vec: torch.Tensor, h_fov: float, v_fov: float) -> torch.Tensor:
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    theta = torch.arccos(z.clamp(-1.0, 1.0))
    r = theta / (math.pi * 0.5)
    sin_t = torch.sin(theta)
    safe = torch.where(sin_t > 1e-12, sin_t, torch.ones_like(sin_t))
    uf = torch.where(sin_t > 1e-12, x * r / safe, torch.zeros_like(r))
    vf = torch.where(sin_t > 1e-12, -y * r / safe, torch.zeros_like(r))
    px = uf / (h_fov / 180.0)
    py = vf / (v_fov / 180.0)
    return torch.stack(((px + 1.0) * 0.5, (py + 1.0) * 0.5), dim=-1)


def _fisheye_uv_to_xyz(fu: torch.Tensor, fv: torch.Tensor) -> torch.Tensor:
    px = fu * 2.0 - 1.0
    py = fv * 2.0 - 1.0
    r = torch.hypot(px, py)
    theta = r * math.pi * 0.5
    sin_t = torch.sin(theta)
    safe = torch.where(r > 1e-12, r, torch.ones_like(r))
    x = torch.where(r > 1e-12, sin_t * px / safe, torch.zeros_like(r))
    y = torch.where(r > 1e-12, -sin_t * py / safe, torch.zeros_like(r))
    z = torch.cos(theta)
    return torch.stack((x, y, z), dim=-1)


def _perimeter_uv(
    u1: float,
    v1: float,
    u2: float,
    v2: float,
):
    for index in range(_PERIM):
        position = index / (_PERIM - 1)
        u = u1 + (u2 - u1) * position
        v = v1 + (v2 - v1) * position
        yield u, v1
        yield u1, v
        yield u2, v
        yield u, v2


def _hequirect_uv_to_xyz_scalar(
    u: float,
    v: float,
) -> tuple[float, float, float]:
    longitude = (u - 0.5) * math.pi
    latitude = (0.5 - v) * math.pi
    cos_latitude = math.cos(latitude)
    return (
        cos_latitude * math.sin(longitude),
        math.sin(latitude),
        cos_latitude * math.cos(longitude),
    )


def _rotate_inv_scalar(
    vector: tuple[float, float, float],
    yaw: float,
    pitch: float,
) -> tuple[float, float, float]:
    x, y, z = vector
    yaw_radians = yaw * DEG
    pitch_radians = pitch * DEG
    cos_yaw = math.cos(yaw_radians)
    sin_yaw = math.sin(yaw_radians)
    cos_pitch = math.cos(pitch_radians)
    sin_pitch = math.sin(pitch_radians)
    return (
        x * cos_yaw - z * sin_yaw,
        -x * sin_yaw * sin_pitch + y * cos_pitch - z * cos_yaw * sin_pitch,
        x * sin_yaw * cos_pitch + y * sin_pitch + z * cos_yaw * cos_pitch,
    )


@lru_cache(maxsize=4096)
def _region_gnomonic_spec(
    u1: float,
    v1: float,
    u2: float,
    v2: float,
) -> tuple[float, float, float]:
    lon0 = ((u1 + u2) * 0.5 - 0.5) * 180.0
    lat0 = (0.5 - (v1 + v2) * 0.5) * 180.0
    extent = 0.0
    for u, v in _perimeter_uv(u1, v1, u2, v2):
        x, y, z = _rotate_inv_scalar(
            _hequirect_uv_to_xyz_scalar(u, v),
            lon0,
            lat0,
        )
        z = z if abs(z) >= 1e-12 else 1e-12
        extent = max(extent, abs(x / z), abs(y / z))
    fov = 2.0 * math.degrees(math.atan(extent))
    return lon0, lat0, fov


@lru_cache(maxsize=4096)
def _fisheye_window(
    u1: float,
    v1: float,
    u2: float,
    v2: float,
    margin: float,
) -> tuple[float, float, float, float]:
    fisheye_u: list[float] = []
    fisheye_v: list[float] = []
    for u, v in _perimeter_uv(u1, v1, u2, v2):
        x, y, z = _hequirect_uv_to_xyz_scalar(u, v)
        theta = math.acos(max(-1.0, min(1.0, z)))
        radius = theta / (math.pi * 0.5)
        sin_theta = math.sin(theta)
        if sin_theta > 1e-12:
            fisheye_u.append((x * radius / sin_theta + 1.0) * 0.5)
            fisheye_v.append((-y * radius / sin_theta + 1.0) * 0.5)
        else:
            fisheye_u.append(0.5)
            fisheye_v.append(0.5)
    centre_u = (min(fisheye_u) + max(fisheye_u)) * 0.5
    centre_v = (min(fisheye_v) + max(fisheye_v)) * 0.5
    half = max(
        max(fisheye_u) - min(fisheye_u),
        max(fisheye_v) - min(fisheye_v),
    ) * 0.5 * margin
    return centre_u - half, centre_u + half, centre_v - half, centre_v + half


def _region_eye_uv_grid(
    u1: float, v1: float, u2: float, v2: float, rh: int, rw: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-source-pixel eye-uv meshgrid over the region (rh, rw)."""
    ru = u1 + (u2 - u1) * _unit_centers(rw, device)
    rv = v1 + (v2 - v1) * _unit_centers(rh, device)
    rvv, ruu = torch.meshgrid(rv, ru, indexing="ij")
    return ruu, rvv


class _RegionProjector:
    """Shared crop/uncrop machinery; subclasses supply the forward (patch->eye)
    and inverse (region->patch) uv maps for one mosaic region."""

    kind = "raw"

    def __init__(self, *, eye_width: int, height: int, device: torch.device) -> None:
        self.eye_width = int(eye_width)
        self.height = int(height)
        if self.eye_width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid eye dimensions {self.eye_width}x{self.height}")
        self.device = device
        log.info(
            "VR %s projection enabled (eye %dx%d)", self.kind, self.eye_width, self.height
        )

    def _eye_offset(self, enlarged_bbox: tuple[int, int, int, int]) -> int:
        centre_x = (int(enlarged_bbox[0]) + int(enlarged_bbox[2])) * 0.5
        return 0 if centre_x < self.eye_width else self.eye_width

    def _bbox_uv(self, local_bbox: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = local_bbox
        return (x1 / self.eye_width, y1 / self.height, x2 / self.eye_width, y2 / self.height)

    def _forward_eye_uv(self, local_bbox, patch_h: int, patch_w: int) -> torch.Tensor:
        raise NotImplementedError

    def _inverse_patch_uv(self, local_bbox, region_h: int, region_w: int) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _as_grid(uv: torch.Tensor) -> torch.Tensor:
        return (uv * 2.0 - 1.0).unsqueeze(0)

    @staticmethod
    def _patch_size(enlarged_bbox: tuple[int, int, int, int]) -> int:
        x1, y1, x2, y2 = enlarged_bbox
        return max(y2 - y1, x2 - x1)

    def _forward_sample(
        self, frame: torch.Tensor, offset: int, local_bbox, patch_h: int, patch_w: int
    ) -> torch.Tensor:
        eye = frame[:, :, offset : offset + self.eye_width]
        grid = self._as_grid(self._forward_eye_uv(local_bbox, patch_h, patch_w)).to(frame.device)
        return F.grid_sample(
            eye.unsqueeze(0).float(), grid,
            mode="bilinear", padding_mode="border", align_corners=True,
        )[0]

    @staticmethod
    def _match_source_precision(
        sampled: torch.Tensor,
        source_dtype: torch.dtype,
    ) -> torch.Tensor:
        if source_dtype == torch.uint8:
            return sampled.round().clamp(0, 255).to(torch.uint8)
        return sampled.to(source_dtype)

    @torch.inference_mode()
    def extract_region_crop(
        self,
        frame: torch.Tensor,
        bbox,
        frame_h: int,
        frame_w: int,
        *,
        x_bounds: tuple[int, int] | None = None,
    ) -> RawCrop:
        x1, y1, x2, y2 = compute_enlarged_bbox(bbox, frame_h, frame_w, x_bounds)
        offset = x_bounds[0] if x_bounds is not None else self._eye_offset((x1, y1, x2, y2))
        local = (x1 - offset, y1, x2 - offset, y2)
        patch_size = self._patch_size((x1, y1, x2, y2))
        sampled = self._forward_sample(frame, offset, local, patch_size, patch_size)
        crop = self._match_source_precision(sampled, frame.dtype)
        return RawCrop(
            crop=crop,
            enlarged_bbox=(x1, y1, x2, y2),
            crop_shape=(patch_size, patch_size),
        )

    @torch.inference_mode()
    def project_region(
        self, frame: torch.Tensor, enlarged_bbox: tuple[int, int, int, int]
    ) -> torch.Tensor:
        """Forward-project a full frame into this region's patch space (float),
        at the crop resolution ``extract_region_crop`` produced. Used to obtain
        the original patch for delta compositing."""
        x1, y1, x2, y2 = (int(v) for v in enlarged_bbox)
        offset = self._eye_offset((x1, y1, x2, y2))
        local = (x1 - offset, y1, x2 - offset, y2)
        patch_size = self._patch_size((x1, y1, x2, y2))
        sampled = self._forward_sample(frame, offset, local, patch_size, patch_size)
        return self._match_source_precision(sampled, frame.dtype).float()

    @torch.inference_mode()
    def source_region_from_patch(
        self, patch: torch.Tensor, enlarged_bbox: tuple[int, int, int, int]
    ) -> torch.Tensor:
        """Inverse-project a patch-space signal back to the source region. Uses
        zero padding so region pixels the projection does not cover contribute no
        delta (they keep the untouched source)."""
        x1, y1, x2, y2 = (int(v) for v in enlarged_bbox)
        offset = self._eye_offset((x1, y1, x2, y2))
        local = (x1 - offset, y1, x2 - offset, y2)
        grid = self._as_grid(self._inverse_patch_uv(local, y2 - y1, x2 - x1)).to(patch.device)
        return F.grid_sample(
            patch.unsqueeze(0).float(), grid,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )[0]


class GnomonicProjector(_RegionProjector):
    """Rectilinear (tangent-plane) view of the mosaic region: gives the 2D model a
    near-undistorted patch. Square FOV bounds the region's angular extent."""

    kind = "gnomonic"

    def _spec(self, local_bbox) -> tuple[float, float, float]:
        u1, v1, u2, v2 = self._bbox_uv(local_bbox)
        return _region_gnomonic_spec(u1, v1, u2, v2)

    def _forward_eye_uv(self, local_bbox, patch_h: int, patch_w: int) -> torch.Tensor:
        lon0, lat0, fov = self._spec(local_bbox)
        vec = _flat_to_xyz(patch_h, patch_w, fov, fov, self.device)
        vec = _rotate(vec, lon0, lat0, 0.0)
        return _xyz_to_hequirect_uv(vec)

    def _inverse_patch_uv(self, local_bbox, region_h: int, region_w: int) -> torch.Tensor:
        lon0, lat0, fov = self._spec(local_bbox)
        u1, v1, u2, v2 = self._bbox_uv(local_bbox)
        ruu, rvv = _region_eye_uv_grid(u1, v1, u2, v2, region_h, region_w, self.device)
        vec = _rotate_inv(_hequirect_uv_to_xyz(ruu, rvv), lon0, lat0, 0.0)
        return _xyz_to_flat_uv(vec, fov, fov)


class FisheyeProjector(_RegionProjector):
    """Equidistant-180 fisheye sub-window covering the mosaic region: the space
    several studios bake the mosaic in. Window is the region's disc bbox, squared
    with a small margin (matches the Phase-4 experiment)."""

    kind = "fisheye"
    _MARGIN = 1.06

    def _window(self, local_bbox) -> tuple[float, float, float, float]:
        u1, v1, u2, v2 = self._bbox_uv(local_bbox)
        return _fisheye_window(u1, v1, u2, v2, self._MARGIN)

    def _forward_eye_uv(self, local_bbox, patch_h: int, patch_w: int) -> torch.Tensor:
        f1u, f2u, f1v, f2v = self._window(local_bbox)
        py, px = torch.meshgrid(
            _unit_centers(patch_h, self.device),
            _unit_centers(patch_w, self.device),
            indexing="ij",
        )
        vec = _fisheye_uv_to_xyz(f1u + (f2u - f1u) * px, f1v + (f2v - f1v) * py)
        return _xyz_to_hequirect_uv(vec)

    def _inverse_patch_uv(self, local_bbox, region_h: int, region_w: int) -> torch.Tensor:
        f1u, f2u, f1v, f2v = self._window(local_bbox)
        u1, v1, u2, v2 = self._bbox_uv(local_bbox)
        ruu, rvv = _region_eye_uv_grid(u1, v1, u2, v2, region_h, region_w, self.device)
        fuv = _xyz_to_fisheye_uv(_hequirect_uv_to_xyz(ruu, rvv), 180.0, 180.0)
        pu = (fuv[..., 0] - f1u) / (f2u - f1u)
        pv = (fuv[..., 1] - f1v) / (f2v - f1v)
        return torch.stack((pu, pv), dim=-1)


_PROJECTORS = {"gnomonic": GnomonicProjector, "fisheye": FisheyeProjector}


def build_vr_projector(
    projection: str, *, eye_width: int, height: int, device: torch.device
) -> _RegionProjector | None:
    if projection in {"raw", "none"}:
        return None
    cls = _PROJECTORS.get(projection)
    if cls is None:
        raise ValueError(f"Unknown VR projection '{projection}'")
    return cls(eye_width=eye_width, height=height, device=device)
