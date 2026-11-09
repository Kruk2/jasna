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

VR_MODES = ("auto", "off", "sbs", "sbs-fisheye", "tb")
FISHEYE_NAME_TOKENS = frozenset({"FSVSS", "SAVR", "URVRSP", "CRVR", "PXVR"})
DIRECT_VR_NAME_TOKENS = frozenset({"S1VR", "MDVR", "VRKM", "IPVR"})
PROJECTION_KINDS = ("raw", "fisheye", "gnomonic")
PROJECTION_CHOICES = ("auto", *PROJECTION_KINDS)
_SBS_ASPECT_MIN = 1.90
_SBS_ASPECT_MAX = 2.10
_AUTO_MIN_EYE_SIZE = 1080

NAME_PREFIX_PROJECTION: dict[str, str] = {
    "ATVR": "raw",
    "CCVR": "fisheye",
    "CJVR": "raw",
    "DSVR": "fisheye",
    "IPVR": "raw",
    "JPSVR": "fisheye",
    "KAVR": "raw",
    "KBVR": "fisheye",
    "KMVR": "fisheye",
    "MAXVR": "fisheye",
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
    def is_stereo(self) -> bool:
        return self.resolved != "off"

    def layout(self, metadata) -> StereoLayout | None:
        """Where the eyes sit in this video's frames; None for flat video."""
        if not self.is_stereo:
            return None
        return StereoLayout(self.resolved, int(metadata.video_width), int(metadata.video_height))


@dataclass(frozen=True)
class StereoLayout:
    """Where the two eyes of a stereo VR frame sit: left|right side by side (``sbs``) or
    left above right (``tb``). Boxes are (x1, y1, x2, y2) frame pixels."""

    kind: str
    frame_w: int
    frame_h: int

    def __post_init__(self) -> None:
        if self.kind not in ("sbs", "tb"):
            raise ValueError(f"Unknown stereo layout '{self.kind}'")
        split = self.frame_h if self.top_bottom else self.frame_w
        if split <= 0 or split % 2:
            side = "height" if self.top_bottom else "width"
            raise ValueError(f"VR {self.kind.upper()} processing requires an even frame {side}, got {split}")

    @property
    def top_bottom(self) -> bool:
        return self.kind == "tb"

    @property
    def eye_w(self) -> int:
        return self.frame_w if self.top_bottom else self.frame_w // 2

    @property
    def eye_h(self) -> int:
        return self.frame_h // 2 if self.top_bottom else self.frame_h

    @property
    def right_offset(self) -> tuple[int, int]:
        """Frame position of the right eye's top-left corner."""
        return (0, self.eye_h) if self.top_bottom else (self.eye_w, 0)

    def eye_of(self, bbox) -> tuple[int, int, int, int]:
        """The eye holding the centre of ``bbox``."""
        dx, dy = self.right_offset
        if self.top_bottom:
            right = (float(bbox[1]) + float(bbox[3])) * 0.5 >= dy
        else:
            right = (float(bbox[0]) + float(bbox[2])) * 0.5 >= dx
        x0, y0 = (dx, dy) if right else (0, 0)
        return x0, y0, x0 + self.eye_w, y0 + self.eye_h


def left_eye(image, stereo: str):
    """The left eye of a (..., H, W) frame or mask; ``stereo`` is ``VrModeResolution.resolved``."""
    if stereo == "sbs":
        return image[..., : image.shape[-1] // 2]
    if stereo == "tb":
        return image[..., : image.shape[-2] // 2, :]
    return image


def _name_token_matches(path: Path, codes: frozenset[str]) -> list[str]:
    # Substring match: file names often glue the prefix to the number
    # (e.g. ``savr00001``), which a token split on separators would miss.
    stem = path.stem.upper()
    return sorted(code for code in codes if code in stem)


def name_prefix(name: str) -> str:
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
    routed = NAME_PREFIX_PROJECTION.get(name_prefix(input_path.stem))
    if routed:
        return routed
    if _name_token_matches(input_path, FISHEYE_NAME_TOKENS):
        return "fisheye"
    return "raw"


def _display_aspect(metadata) -> float:
    sar = metadata.sample_aspect_ratio
    return (
        float(metadata.video_width)
        * float(sar.numerator)
        / float(sar.denominator)
        / float(metadata.video_height)
    )


_SPATIAL_STEREO = {
    "sbs": ("side by side", {"sbs", "left-right", "left_right"}),
    "tb": ("top and bottom", {"tb", "top-bottom", "top_bottom"}),
}


def _has_spatial_metadata(metadata, layout: str) -> bool:
    stereo = str(getattr(metadata, "stereo_layout", "")).lower()
    projection = str(getattr(metadata, "spherical_projection", "")).lower()
    phrase, names = _SPATIAL_STEREO[layout]
    return (phrase in stereo or stereo in names) and "equirect" in projection


def _auto_layout(metadata, input_path: Path, width: int, height: int, aspect: float) -> tuple[str, str]:
    """Auto mode: VR when the frame has a two-eye shape (2:1 side by side, 1:2 top/bottom)
    and a trusted sign says so."""
    sbs_exact = width == height * 2 and height > _AUTO_MIN_EYE_SIZE
    tb_exact = height == width * 2 and width > _AUTO_MIN_EYE_SIZE
    if sbs_exact or _SBS_ASPECT_MIN <= aspect <= _SBS_ASPECT_MAX:
        layout, exact, split = "sbs", sbs_exact, width
    elif tb_exact or _SBS_ASPECT_MIN <= 1.0 / aspect <= _SBS_ASPECT_MAX:
        layout, exact, split = "tb", tb_exact, height
    else:
        return "off", f"display aspect {aspect:.3f} is outside the VR gate"
    if split % 2:
        return "off", f"odd frame {'width' if layout == 'sbs' else 'height'} {split}"

    fisheye_matches = _name_token_matches(input_path, FISHEYE_NAME_TOKENS)
    direct_matches = _name_token_matches(input_path, DIRECT_VR_NAME_TOKENS)
    routed_prefix = name_prefix(input_path.stem)
    if fisheye_matches:
        return layout, f"known fisheye-remap name token {fisheye_matches[0]}"
    if direct_matches:
        return layout, f"known direct VR name token {direct_matches[0]}"
    if routed_prefix in NAME_PREFIX_PROJECTION:
        return layout, f"routed VR name prefix {routed_prefix}"
    if _has_spatial_metadata(metadata, layout):
        return layout, f"{layout.upper()} equirectangular spatial metadata"
    if exact:
        shape = "2:1" if layout == "sbs" else "1:2"
        return layout, f"{shape} frame above {_AUTO_MIN_EYE_SIZE}p"
    return "off", "no trusted name token or spatial metadata"


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
    if requested == "off":
        resolved, reason = "off", "explicit mode"
    elif requested != "auto":
        resolved = "tb" if requested == "tb" else "sbs"
        StereoLayout(resolved, width, height)
        eye_aspect = aspect / 2.0 if resolved == "sbs" else aspect * 2.0
        reason = "explicit mode"
        if not (_SBS_ASPECT_MIN / 2.0 <= eye_aspect <= _SBS_ASPECT_MAX / 2.0):
            reason += f"; unusual {resolved.upper()} display aspect {aspect:.3f}"
    else:
        resolved, reason = _auto_layout(metadata, input_path, width, height, aspect)

    if resolved == "off":
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
    if "unusual" in result.reason:
        log.warning(message)
    else:
        log.info(message)
    return result


def stereo_detector(detector, stereo: StereoLayout | None):
    """``detector`` run per eye for stereo VR; itself for flat video."""
    return StereoDetectionAdapter(detector, stereo.kind) if stereo is not None else detector


class StereoDetectionAdapter:
    """Runs the detector on each eye of a stereo frame and returns frame-space detections."""

    def __init__(self, detector, kind: str) -> None:
        self.detector = detector
        self.kind = kind

    def _split(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stereo = StereoLayout(self.kind, int(frames.shape[-1]), int(frames.shape[-2]))
        dx, dy = stereo.right_offset
        return frames[..., : stereo.eye_h, : stereo.eye_w], frames[..., dy:, dx:]

    def __call__(
        self,
        frames: torch.Tensor,
        *,
        target_hw: tuple[int, int],
    ) -> Detections:
        left_frames, right_frames = self._split(frames)
        target_h, target_w = map(int, target_hw)
        target = StereoLayout(self.kind, target_w, target_h)
        eye_hw = (target.eye_h, target.eye_w)
        left = self.detector(left_frames, target_hw=eye_hw)
        right = self.detector(right_frames, target_hw=eye_hw)

        dx, dy = target.right_offset
        offset = np.array([dx, dy, dx, dy], dtype=np.float32)
        boxes: list[np.ndarray] = []
        masks: list[torch.Tensor] = []
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
            mask_h, mask_w = left_masks.shape[-2:]
            pad_left, pad_right = (
                ((0, 0, 0, mask_h), (0, 0, mask_h, 0))
                if target.top_bottom
                else ((0, mask_w), (mask_w, 0))
            )
            boxes.append(
                np.concatenate(
                    (left_boxes, right_boxes + offset),
                    axis=0,
                ).astype(np.float32, copy=False)
            )
            masks.append(
                torch.cat(
                    (F.pad(left_masks, pad_left), F.pad(right_masks, pad_right)),
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
        left_frames, right_frames = self._split(frames)
        mask_h, mask_w = map(int, mask_hw)
        if self.kind == "tb":
            left_hw, right_hw, dim = (mask_h // 2, mask_w), (mask_h - mask_h // 2, mask_w), -2
        else:
            left_hw, right_hw, dim = (mask_h, mask_w // 2), (mask_h, mask_w - mask_w // 2), -1
        left_scores, left_masks = self.detector.scan_scores_masks(left_frames, mask_hw=left_hw)
        right_scores, right_masks = self.detector.scan_scores_masks(right_frames, mask_hw=right_hw)
        return torch.maximum(left_scores, right_scores), torch.cat((left_masks, right_masks), dim=dim)

    def close(self) -> None:
        if hasattr(self.detector, "close"):
            self.detector.close()
