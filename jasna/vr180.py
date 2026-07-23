from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from jasna.crop_buffer import RawCrop, compute_enlarged_bbox
from jasna.mosaic.detections import Detections

log = logging.getLogger(__name__)

VR_MODES = ("auto", "off", "sbs", "sbs-fisheye")
FISHEYE_STUDIO_TOKENS = frozenset({"FSVSS", "SAVR", "URVRSP", "CRVR", "PXVR"})
DIRECT_STUDIO_TOKENS = frozenset({"S1VR", "MDVR", "VRKM", "IPVR"})
_SBS_ASPECT_MIN = 1.90
_SBS_ASPECT_MAX = 2.10
_AUTO_SBS_MIN_HEIGHT = 1080


@dataclass(frozen=True)
class VrModeResolution:
    requested: str
    resolved: str
    reason: str
    display_aspect: float

    @property
    def is_sbs(self) -> bool:
        return self.resolved in {"sbs", "sbs-fisheye"}


def _studio_matches(path: Path, codes: frozenset[str]) -> list[str]:
    # Substring match: real releases glue the studio code to the number
    # (e.g. ``savr00327``), which a token split on separators would miss.
    stem = path.stem.upper()
    return sorted(code for code in codes if code in stem)


def _display_aspect(metadata) -> float:
    sar = metadata.sample_aspect_ratio
    return (
        float(metadata.video_width)
        * float(sar.numerator)
        / float(sar.denominator)
        / float(metadata.video_height)
    )


def _has_sbs_spatial_metadata(metadata) -> bool:
    stereo = str(getattr(metadata, "stereo_layout", "")).lower()
    projection = str(getattr(metadata, "spherical_projection", "")).lower()
    is_sbs = "side by side" in stereo or stereo in {
        "sbs",
        "left-right",
        "left_right",
    }
    is_equirectangular = "equirect" in projection
    return is_sbs and is_equirectangular


def resolve_vr_mode(
    requested: str,
    metadata,
    input_path: Path,
) -> VrModeResolution:
    requested = str(requested).strip().lower()
    if requested not in VR_MODES:
        raise ValueError(
            f"Unknown VR mode '{requested}'. Valid modes: {', '.join(VR_MODES)}"
        )

    width = int(metadata.video_width)
    height = int(metadata.video_height)
    aspect = _display_aspect(metadata)
    is_high_resolution_2_to_1 = (
        width == height * 2 and height > _AUTO_SBS_MIN_HEIGHT
    )
    if requested == "off":
        result = VrModeResolution(requested, "off", "explicit mode", aspect)
    elif requested != "auto":
        if width % 2:
            raise ValueError(
                f"VR SBS processing requires an even frame width, got {width}"
            )
        reason = "explicit mode"
        if not (_SBS_ASPECT_MIN <= aspect <= _SBS_ASPECT_MAX):
            reason += f"; unusual SBS display aspect {aspect:.3f}"
        result = VrModeResolution(requested, requested, reason, aspect)
    elif width % 2:
        result = VrModeResolution(
            requested,
            "off",
            f"odd frame width {width}",
            aspect,
        )
    elif (
        not is_high_resolution_2_to_1
        and not (_SBS_ASPECT_MIN <= aspect <= _SBS_ASPECT_MAX)
    ):
        result = VrModeResolution(
            requested,
            "off",
            f"display aspect {aspect:.3f} is outside the SBS gate",
            aspect,
        )
    else:
        fisheye_matches = _studio_matches(input_path, FISHEYE_STUDIO_TOKENS)
        direct_matches = _studio_matches(input_path, DIRECT_STUDIO_TOKENS)
        if fisheye_matches:
            token = fisheye_matches[0]
            result = VrModeResolution(
                requested,
                "sbs-fisheye",
                f"known fisheye-remap studio token {token}",
                aspect,
            )
        elif direct_matches:
            token = direct_matches[0]
            result = VrModeResolution(
                requested,
                "sbs",
                f"known direct-SBS studio token {token}",
                aspect,
            )
        elif _has_sbs_spatial_metadata(metadata):
            result = VrModeResolution(
                requested,
                "sbs",
                "side-by-side equirectangular spatial metadata",
                aspect,
            )
        elif is_high_resolution_2_to_1:
            result = VrModeResolution(
                requested,
                "sbs",
                f"2:1 frame above {_AUTO_SBS_MIN_HEIGHT}p",
                aspect,
            )
        else:
            result = VrModeResolution(
                requested,
                "off",
                "no trusted studio token or spatial metadata",
                aspect,
            )

    message = (
        "VR mode: requested=%s resolved=%s reason=%s"
        % (result.requested, result.resolved, result.reason)
    )
    if "unusual SBS" in result.reason:
        log.warning(message)
    else:
        log.info(message)
    return result


class SbsDetectionAdapter:
    def __init__(self, detector) -> None:
        self.detector = detector

    @staticmethod
    def _eye_width(frames: torch.Tensor) -> int:
        width = int(frames.shape[-1])
        if width % 2:
            raise ValueError(
                f"VR SBS processing requires an even frame width, got {width}"
            )
        return width // 2

    def __call__(
        self,
        frames: torch.Tensor,
        *,
        target_hw: tuple[int, int],
    ) -> Detections:
        eye_width = self._eye_width(frames)
        target_h, target_w = map(int, target_hw)
        if target_w % 2:
            raise ValueError(
                f"VR SBS processing requires an even target width, got {target_w}"
            )
        target_eye_width = target_w // 2
        left = self.detector(
            frames[:, :, :, :eye_width],
            target_hw=(target_h, target_eye_width),
        )
        right = self.detector(
            frames[:, :, :, eye_width:],
            target_hw=(target_h, target_eye_width),
        )

        boxes: list[np.ndarray] = []
        masks: list[torch.Tensor] = []
        offset = np.array(
            [target_eye_width, 0, target_eye_width, 0],
            dtype=np.float32,
        )
        for left_boxes, right_boxes, left_masks, right_masks in zip(
            left.boxes_xyxy,
            right.boxes_xyxy,
            left.masks,
            right.masks,
        ):
            if left_masks.shape[-2:] != right_masks.shape[-2:]:
                raise RuntimeError(
                    "Per-eye detector masks have mismatched shapes: "
                    f"{tuple(left_masks.shape)} vs {tuple(right_masks.shape)}"
                )
            mask_width = int(left_masks.shape[-1])
            boxes.append(
                np.concatenate(
                    (left_boxes, right_boxes + offset),
                    axis=0,
                ).astype(np.float32, copy=False)
            )
            masks.append(
                torch.cat(
                    (
                        F.pad(left_masks, (0, mask_width)),
                        F.pad(right_masks, (mask_width, 0)),
                    ),
                    dim=0,
                )
            )
        return Detections(boxes_xyxy=boxes, masks=masks)

    def scan_scores_masks(
        self,
        frames: torch.Tensor,
        *,
        mask_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eye_width = self._eye_width(frames)
        mask_h, mask_w = map(int, mask_hw)
        left_mask_w = mask_w // 2
        right_mask_w = mask_w - left_mask_w
        left_scores, left_masks = self.detector.scan_scores_masks(
            frames[:, :, :, :eye_width],
            mask_hw=(mask_h, left_mask_w),
        )
        right_scores, right_masks = self.detector.scan_scores_masks(
            frames[:, :, :, eye_width:],
            mask_hw=(mask_h, right_mask_w),
        )
        return torch.maximum(left_scores, right_scores), torch.cat(
            (left_masks, right_masks),
            dim=-1,
        )

    def close(self) -> None:
        if hasattr(self.detector, "close"):
            self.detector.close()


class GnomonicProjector:
    """Per-mosaic-region rectilinear (gnomonic) projection for VR180 SBS eyes.

    Each eye is treated as an equirectangular hemisphere: a pixel column maps to
    longitude in [-pi/2, pi/2] and a row to latitude in [-pi/2, pi/2] (the same
    mapping the whole-eye reprojection used, and consistent with the SBS
    detector split). For one tracked region a tangent plane is placed at the
    region's centre direction and the equirectangular content is resampled onto
    it, giving the 2D-trained restoration model a near-undistorted flat patch.
    The restored patch is resampled back onto the equirectangular region before
    blending. Both grids are pure functions of the region's enlarged bbox and
    the eye dimensions, so the blend side rebuilds the inverse without carrying
    extra per-frame state.
    """

    def __init__(
        self,
        *,
        eye_width: int,
        height: int,
        device: torch.device,
    ) -> None:
        self.eye_width = int(eye_width)
        self.height = int(height)
        if self.eye_width <= 0 or self.height <= 0:
            raise ValueError(
                f"Invalid eye dimensions {self.eye_width}x{self.height}"
            )
        self.device = device
        log.info(
            "VR per-region flat projection enabled (eye %dx%d)",
            self.eye_width,
            self.height,
        )

    def _eye_offset(self, enlarged_bbox: tuple[int, int, int, int]) -> int:
        centre_x = (int(enlarged_bbox[0]) + int(enlarged_bbox[2])) * 0.5
        return 0 if centre_x < self.eye_width else self.eye_width

    @staticmethod
    def _forward_gnomonic_scalar(
        lon: float, lat: float, lon0: float, lat0: float
    ) -> tuple[float, float]:
        cos_c = (
            math.sin(lat0) * math.sin(lat)
            + math.cos(lat0) * math.cos(lat) * math.cos(lon - lon0)
        )
        cos_c = max(cos_c, 1e-6)
        x = math.cos(lat) * math.sin(lon - lon0) / cos_c
        y = (
            math.cos(lat0) * math.sin(lat)
            - math.sin(lat0) * math.cos(lat) * math.cos(lon - lon0)
        ) / cos_c
        return x, y

    def _region_geometry(
        self, local_bbox: tuple[int, int, int, int]
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = (float(v) for v in local_bbox)
        lon1 = ((x1 + 0.5) / self.eye_width - 0.5) * math.pi
        lon2 = ((x2 - 0.5) / self.eye_width - 0.5) * math.pi
        lat1 = ((y1 + 0.5) / self.height - 0.5) * math.pi
        lat2 = ((y2 - 0.5) / self.height - 0.5) * math.pi
        lon0 = 0.5 * (lon1 + lon2)
        lat0 = 0.5 * (lat1 + lat2)
        x_extent = 1e-6
        y_extent = 1e-6
        for lon in (lon1, lon2):
            for lat in (lat1, lat2):
                x, y = self._forward_gnomonic_scalar(lon, lat, lon0, lat0)
                x_extent = max(x_extent, abs(x))
                y_extent = max(y_extent, abs(y))
        return lon0, lat0, x_extent, y_extent

    def _forward_grid(
        self, local_bbox: tuple[int, int, int, int], patch_h: int, patch_w: int
    ) -> torch.Tensor:
        """Grid mapping each flat-patch pixel to a normalized eye coordinate."""
        lon0, lat0, x_extent, y_extent = self._region_geometry(local_bbox)
        ty, tx = torch.meshgrid(
            (torch.arange(patch_h, dtype=torch.float64) + 0.5) / patch_h * 2.0 - 1.0,
            (torch.arange(patch_w, dtype=torch.float64) + 0.5) / patch_w * 2.0 - 1.0,
            indexing="ij",
        )
        plane_x = tx * x_extent
        plane_y = ty * y_extent
        radius = torch.sqrt(plane_x.square() + plane_y.square())
        c = torch.atan(radius)
        sin_c = torch.sin(c)
        cos_c = torch.cos(c)
        safe_radius = torch.where(radius > 1e-12, radius, torch.ones_like(radius))
        latitude = torch.asin(
            (
                cos_c * math.sin(lat0)
                + plane_y * sin_c * math.cos(lat0) / safe_radius
            ).clamp(-1.0, 1.0)
        )
        longitude = lon0 + torch.atan2(
            plane_x * sin_c,
            safe_radius * math.cos(lat0) * cos_c - plane_y * math.sin(lat0) * sin_c,
        )
        at_centre = radius <= 1e-12
        latitude = torch.where(at_centre, torch.full_like(latitude, lat0), latitude)
        longitude = torch.where(at_centre, torch.full_like(longitude, lon0), longitude)
        grid_x = (longitude / math.pi + 0.5) * 2.0 - 1.0
        grid_y = (latitude / math.pi + 0.5) * 2.0 - 1.0
        return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).float()

    def _inverse_grid(
        self, local_bbox: tuple[int, int, int, int], region_h: int, region_w: int
    ) -> torch.Tensor:
        """Grid mapping each eye-region pixel back to a normalized patch coord."""
        x1, y1, _, _ = (float(v) for v in local_bbox)
        lon0, lat0, x_extent, y_extent = self._region_geometry(local_bbox)
        eye_y, eye_x = torch.meshgrid(
            torch.arange(region_h, dtype=torch.float64) + y1 + 0.5,
            torch.arange(region_w, dtype=torch.float64) + x1 + 0.5,
            indexing="ij",
        )
        longitude = (eye_x / self.eye_width - 0.5) * math.pi
        latitude = (eye_y / self.height - 0.5) * math.pi
        cos_c = (
            math.sin(lat0) * torch.sin(latitude)
            + math.cos(lat0) * torch.cos(latitude) * torch.cos(longitude - lon0)
        ).clamp_min(1e-6)
        plane_x = torch.cos(latitude) * torch.sin(longitude - lon0) / cos_c
        plane_y = (
            math.cos(lat0) * torch.sin(latitude)
            - math.sin(lat0) * torch.cos(latitude) * torch.cos(longitude - lon0)
        ) / cos_c
        grid_x = plane_x / x_extent
        grid_y = plane_y / y_extent
        return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).float()

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
        offset = self._eye_offset((x1, y1, x2, y2))
        local = (x1 - offset, y1, x2 - offset, y2)
        patch_h = y2 - y1
        patch_w = x2 - x1
        eye = frame[:, :, offset : offset + self.eye_width]
        grid = self._forward_grid(local, patch_h, patch_w).to(frame.device)
        sampled = F.grid_sample(
            eye.unsqueeze(0).float(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )[0]
        if frame.dtype == torch.uint8:
            crop = sampled.round().clamp(0, 255).to(torch.uint8)
        else:
            crop = sampled.to(frame.dtype)
        return RawCrop(
            crop=crop,
            enlarged_bbox=(x1, y1, x2, y2),
            crop_shape=(patch_h, patch_w),
        )

    @torch.inference_mode()
    def source_region_from_patch(
        self,
        patch: torch.Tensor,
        enlarged_bbox: tuple[int, int, int, int],
    ) -> torch.Tensor:
        x1, y1, x2, y2 = (int(v) for v in enlarged_bbox)
        offset = self._eye_offset((x1, y1, x2, y2))
        local = (x1 - offset, y1, x2 - offset, y2)
        grid = self._inverse_grid(local, y2 - y1, x2 - x1).to(patch.device)
        return F.grid_sample(
            patch.unsqueeze(0).float(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )[0]
