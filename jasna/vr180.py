from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from jasna.mosaic.detections import Detections

log = logging.getLogger(__name__)

VR_MODES = ("auto", "off", "sbs", "sbs-fisheye")
FISHEYE_STUDIO_TOKENS = frozenset({"FSVSS", "SAVR", "URVRSP", "CRVR", "PXVR"})
DIRECT_STUDIO_TOKENS = frozenset({"S1VR", "MDVR", "VRKM", "IPVR"})
PROJECTION_KINDS = ("raw", "fisheye", "gnomonic")
PROJECTION_CHOICES = ("auto", *PROJECTION_KINDS)
_SBS_ASPECT_MIN = 1.90
_SBS_ASPECT_MAX = 2.10
_AUTO_SBS_MIN_HEIGHT = 1080

STUDIO_PROJECTION: dict[str, str] = {
    "ATVR": "raw",
    "CJVR": "raw",
    "IPVR": "raw",
    "KAVR": "raw",
    "NHVR": "fisheye",
    "PXVR": "fisheye",
    "TMAVR": "fisheye",
    "VRPRD": "gnomonic",
}


@dataclass(frozen=True)
class VrModeResolution:
    requested: str
    resolved: str
    reason: str
    display_aspect: float
    projection: str

    @property
    def is_sbs(self) -> bool:
        return self.resolved == "sbs"


def _studio_matches(path: Path, codes: frozenset[str]) -> list[str]:
    # Substring match: real releases glue the studio code to the number
    # (e.g. ``savr00327``), which a token split on separators would miss.
    stem = path.stem.upper()
    return sorted(code for code in codes if code in stem)


def studio_code(name: str) -> str:
    stem = re.sub(r"^\[[^\]]*\]\s*", "", name).upper()
    m = re.match(r"^([0-9]?[A-Z]{2,7})", stem)
    return m.group(1) if m else ""


def _normalize_projection(value: str) -> str:
    projection = str(value).strip().lower()
    if projection not in PROJECTION_CHOICES:
        raise ValueError(
            f"Unknown VR projection '{projection}'. "
            f"Valid projections: {', '.join(PROJECTION_CHOICES)}"
        )
    return projection


def resolve_projection(input_path: Path, requested: str = "auto") -> str:
    projection = _normalize_projection(requested)
    if projection != "auto":
        return projection
    routed = STUDIO_PROJECTION.get(studio_code(input_path.stem))
    if routed:
        return routed
    if _studio_matches(input_path, FISHEYE_STUDIO_TOKENS):
        return "fisheye"
    if _studio_matches(input_path, DIRECT_STUDIO_TOKENS):
        return "raw"
    return "raw"


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
    *,
    projection: str = "auto",
) -> VrModeResolution:
    requested = str(requested).strip().lower()
    if requested not in VR_MODES:
        raise ValueError(
            f"Unknown VR mode '{requested}'. Valid modes: {', '.join(VR_MODES)}"
        )
    projection = _normalize_projection(projection)

    width = int(metadata.video_width)
    height = int(metadata.video_height)
    aspect = _display_aspect(metadata)
    is_high_resolution_2_to_1 = (
        width == height * 2 and height > _AUTO_SBS_MIN_HEIGHT
    )
    if requested == "off":
        resolved, reason = "off", "explicit mode"
    elif requested != "auto":
        if width % 2:
            raise ValueError(
                f"VR SBS processing requires an even frame width, got {width}"
            )
        resolved, reason = "sbs", "explicit mode"
        if not (_SBS_ASPECT_MIN <= aspect <= _SBS_ASPECT_MAX):
            reason += f"; unusual SBS display aspect {aspect:.3f}"
    elif width % 2:
        resolved, reason = "off", f"odd frame width {width}"
    elif (
        not is_high_resolution_2_to_1
        and not (_SBS_ASPECT_MIN <= aspect <= _SBS_ASPECT_MAX)
    ):
        resolved, reason = "off", f"display aspect {aspect:.3f} is outside the SBS gate"
    else:
        fisheye_matches = _studio_matches(input_path, FISHEYE_STUDIO_TOKENS)
        direct_matches = _studio_matches(input_path, DIRECT_STUDIO_TOKENS)
        routed_code = studio_code(input_path.stem)
        if fisheye_matches:
            resolved, reason = "sbs", f"known fisheye-remap studio token {fisheye_matches[0]}"
        elif direct_matches:
            resolved, reason = "sbs", f"known direct-SBS studio token {direct_matches[0]}"
        elif routed_code in STUDIO_PROJECTION:
            resolved, reason = "sbs", f"routed VR studio {routed_code}"
        elif _has_sbs_spatial_metadata(metadata):
            resolved, reason = "sbs", "side-by-side equirectangular spatial metadata"
        elif is_high_resolution_2_to_1:
            resolved, reason = "sbs", f"2:1 frame above {_AUTO_SBS_MIN_HEIGHT}p"
        else:
            resolved, reason = "off", "no trusted studio token or spatial metadata"

    if resolved != "sbs":
        resolved_projection = "none"
    elif projection != "auto":
        resolved_projection = projection
    elif requested == "sbs-fisheye":
        resolved_projection = "fisheye"
    else:
        resolved_projection = resolve_projection(input_path)

    result = VrModeResolution(
        requested,
        resolved,
        reason,
        aspect,
        resolved_projection,
    )
    message = (
        "VR mode: requested=%s resolved=%s projection=%s reason=%s"
        % (result.requested, result.resolved, result.projection, result.reason)
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
