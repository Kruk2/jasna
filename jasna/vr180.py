from __future__ import annotations

import logging
import math
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

    @property
    def uses_fisheye(self) -> bool:
        return self.resolved == "sbs-fisheye"


def _filename_tokens(path: Path) -> set[str]:
    return {
        token
        for token in re.split(r"[^A-Z0-9]+", path.stem.upper())
        if token
    }


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
        tokens = _filename_tokens(input_path)
        fisheye_matches = sorted(tokens & FISHEYE_STUDIO_TOKENS)
        direct_matches = sorted(tokens & DIRECT_STUDIO_TOKENS)
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


class FisheyeProjector:
    def __init__(
        self,
        *,
        eye_width: int,
        height: int,
        device: torch.device,
        fov_degrees: float = 180.0,
    ) -> None:
        self.eye_width = int(eye_width)
        self.height = int(height)
        if self.eye_width <= 0 or self.height <= 0:
            raise ValueError(
                f"Invalid eye dimensions {self.eye_width}x{self.height}"
            )
        self.device = device
        self.forward_grid = self._build_forward_grid(
            self.eye_width,
            self.height,
            fov_degrees,
        ).to(device)
        self.inverse_grid = self._build_inverse_grid(
            self.eye_width,
            self.height,
            fov_degrees,
        ).to(device)

    @staticmethod
    def _pixel_centers(length: int) -> torch.Tensor:
        return (torch.arange(length, dtype=torch.float64) + 0.5) / length

    @classmethod
    def _build_forward_grid(
        cls,
        width: int,
        height: int,
        fov_degrees: float,
    ) -> torch.Tensor:
        half_fov = math.radians(float(fov_degrees)) * 0.5
        output_y, output_x = torch.meshgrid(
            cls._pixel_centers(height),
            cls._pixel_centers(width),
            indexing="ij",
        )
        fisheye_x = output_x * 2.0 - 1.0
        fisheye_y = output_y * 2.0 - 1.0
        radius = torch.sqrt(fisheye_x.square() + fisheye_y.square())
        theta = radius * half_fov
        phi = torch.atan2(fisheye_y, fisheye_x)
        sin_theta = torch.sin(theta)
        direction_x = sin_theta * torch.cos(phi)
        direction_y = sin_theta * torch.sin(phi)
        direction_z = torch.cos(theta)
        longitude = torch.atan2(direction_x, direction_z)
        latitude = torch.asin(direction_y.clamp(-1.0, 1.0))
        source_x = longitude / math.pi + 0.5
        source_y = latitude / math.pi + 0.5
        grid_x = source_x * 2.0 - 1.0
        grid_y = source_y * 2.0 - 1.0
        outside = radius > 1.0
        grid_x = torch.where(outside, torch.full_like(grid_x, 2.0), grid_x)
        grid_y = torch.where(outside, torch.full_like(grid_y, 2.0), grid_y)
        return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).float()

    @classmethod
    def _build_inverse_grid(
        cls,
        width: int,
        height: int,
        fov_degrees: float,
    ) -> torch.Tensor:
        half_fov = math.radians(float(fov_degrees)) * 0.5
        output_y, output_x = torch.meshgrid(
            cls._pixel_centers(height),
            cls._pixel_centers(width),
            indexing="ij",
        )
        longitude = (output_x - 0.5) * math.pi
        latitude = (output_y - 0.5) * math.pi
        direction_x = torch.cos(latitude) * torch.sin(longitude)
        direction_y = torch.sin(latitude)
        direction_z = torch.cos(latitude) * torch.cos(longitude)
        theta = torch.acos(direction_z.clamp(-1.0, 1.0))
        phi = torch.atan2(direction_y, direction_x)
        radius = theta / half_fov
        fisheye_x = radius * torch.cos(phi)
        fisheye_y = radius * torch.sin(phi)
        return torch.stack((fisheye_x, fisheye_y), dim=-1).unsqueeze(0).float()

    def _validate_eye(self, eye: torch.Tensor) -> tuple[torch.Tensor, bool]:
        single = eye.ndim == 3
        if single:
            eye = eye.unsqueeze(0)
        if eye.ndim != 4 or eye.shape[1] != 3:
            raise ValueError(
                f"Expected (3,H,W) or (N,3,H,W), got {tuple(eye.shape)}"
            )
        if eye.shape[-2:] != (self.height, self.eye_width):
            raise ValueError(
                f"Eye size {tuple(eye.shape[-2:])} does not match "
                f"{(self.height, self.eye_width)}"
            )
        return eye, single

    def _sample_eye(
        self,
        eye: torch.Tensor,
        grid: torch.Tensor,
        *,
        preserve_dtype: bool,
    ) -> torch.Tensor:
        eye, single = self._validate_eye(eye)
        output_dtype = eye.dtype
        output = torch.empty(
            eye.shape,
            dtype=torch.float32 if not preserve_dtype else output_dtype,
            device=eye.device,
        )
        for index in range(eye.shape[0]):
            sampled = F.grid_sample(
                eye[index : index + 1].float(),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )[0]
            if preserve_dtype and not output_dtype.is_floating_point:
                sampled = sampled.round().clamp(0, 255).to(output_dtype)
            elif preserve_dtype:
                sampled = sampled.to(output_dtype)
            output[index] = sampled
        return output[0] if single else output

    def _validate_sbs(self, frames: torch.Tensor) -> tuple[torch.Tensor, bool]:
        single = frames.ndim == 3
        if single:
            frames = frames.unsqueeze(0)
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(
                f"Expected (3,H,W) or (N,3,H,W), got {tuple(frames.shape)}"
            )
        expected = (self.height, self.eye_width * 2)
        if frames.shape[-2:] != expected:
            raise ValueError(
                f"SBS size {tuple(frames.shape[-2:])} does not match {expected}"
            )
        return frames, single

    @torch.inference_mode()
    def forward_sbs(self, frames: torch.Tensor) -> torch.Tensor:
        frames, single = self._validate_sbs(frames)
        output = torch.empty_like(frames)
        for index in range(frames.shape[0]):
            for eye_index in range(2):
                start = eye_index * self.eye_width
                end = start + self.eye_width
                output[index, :, :, start:end] = self._sample_eye(
                    frames[index, :, :, start:end],
                    self.forward_grid,
                    preserve_dtype=True,
                )
        return output[0] if single else output

    @torch.inference_mode()
    def restore_delta_to_source(
        self,
        source_sbs: torch.Tensor,
        projected_source_sbs: torch.Tensor,
        blended_fisheye_sbs: torch.Tensor,
    ) -> torch.Tensor:
        source_sbs, single = self._validate_sbs(source_sbs)
        projected_source_sbs, _ = self._validate_sbs(projected_source_sbs)
        blended_fisheye_sbs, _ = self._validate_sbs(blended_fisheye_sbs)
        if (
            source_sbs.shape != projected_source_sbs.shape
            or source_sbs.shape != blended_fisheye_sbs.shape
        ):
            raise ValueError("Source, projected source, and blended frames must match")

        output = source_sbs.float()
        for index in range(source_sbs.shape[0]):
            for eye_index in range(2):
                start = eye_index * self.eye_width
                end = start + self.eye_width
                delta = (
                    blended_fisheye_sbs[index, :, :, start:end].float()
                    - projected_source_sbs[index, :, :, start:end].float()
                )
                source_delta = self._sample_eye(
                    delta,
                    self.inverse_grid,
                    preserve_dtype=False,
                )
                output[index, :, :, start:end].add_(source_delta)
        output = output.round_().clamp_(0, 255).to(source_sbs.dtype)
        return output[0] if single else output

    @torch.inference_mode()
    def inverse_mask_sbs(self, masks: torch.Tensor) -> torch.Tensor:
        single = masks.ndim == 2
        if single:
            masks = masks.unsqueeze(0)
        if masks.ndim != 3:
            raise ValueError(
                f"Expected (H,W) or (N,H,W), got {tuple(masks.shape)}"
            )
        expected = (self.height, self.eye_width * 2)
        if masks.shape[-2:] != expected:
            raise ValueError(
                f"SBS mask size {tuple(masks.shape[-2:])} does not match {expected}"
            )
        output = torch.empty_like(masks, dtype=torch.bool)
        for index in range(masks.shape[0]):
            for eye_index in range(2):
                start = eye_index * self.eye_width
                end = start + self.eye_width
                sampled = F.grid_sample(
                    masks[index : index + 1, :, start:end]
                    .unsqueeze(1)
                    .float(),
                    self.inverse_grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                output[index, :, start:end] = sampled[0, 0] > 0.5
        return output[0] if single else output
