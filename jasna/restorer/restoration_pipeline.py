from __future__ import annotations

import logging
import math

import torch

from jasna.crop_buffer import (
    RawCrop,
    prepare_crops_for_restoration,
)
from jasna.pipeline_items import PrimaryRestoreResult, SecondaryRestoreResult
from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
from jasna.restorer.denoise import DenoiseStep, DenoiseStrength, apply_denoise, apply_denoise_u8
from jasna.restorer.secondary_restorer import SecondaryRestorer
from jasna.tracking.clip_tracker import TrackedClip

logger = logging.getLogger(__name__)


class RestorationPipeline:
    def __init__(
        self,
        restorer: BasicvsrppMosaicRestorer,
        *,
        secondary_restorer: SecondaryRestorer | None = None,
        denoise_strength: DenoiseStrength = DenoiseStrength.NONE,
        denoise_step: DenoiseStep = DenoiseStep.AFTER_PRIMARY,
    ) -> None:
        self.restorer = restorer
        self.secondary_restorer = secondary_restorer
        self._denoise_strength = denoise_strength
        self._denoise_step = denoise_step
        logger.info(
            "RestorationPipeline: secondary=%s denoise=%s denoise_step=%s",
            secondary_restorer.name if secondary_restorer else "none",
            denoise_strength.name,
            denoise_step.name,
        )

    @property
    def secondary_num_workers(self) -> int:
        if self.secondary_restorer is not None:
            return self.secondary_restorer.num_workers
        return 1

    @property
    def secondary_prefers_cpu_input(self) -> bool:
        if self.secondary_restorer is not None:
            return bool(getattr(self.secondary_restorer, "prefers_cpu_input", False))
        return False

    def _apply_denoise(self, frames: torch.Tensor) -> torch.Tensor:
        return apply_denoise(frames, self._denoise_strength)

    def _prepare_from_raw_crops(
        self,
        raw_crops: list[RawCrop],
    ) -> tuple[
        list[torch.Tensor],
        list[tuple[int, int, int, int]],
        list[tuple[int, int]],
        list[tuple[int, int]],
        list[tuple[int, int]],
    ]:
        resized_crops, pad_offsets, resize_shapes = prepare_crops_for_restoration(raw_crops)
        enlarged_bboxes = [c.enlarged_bbox for c in raw_crops]
        crop_shapes = [c.crop_shape for c in raw_crops]
        return resized_crops, enlarged_bboxes, crop_shapes, pad_offsets, resize_shapes

    def _run_secondary(self, primary_raw: torch.Tensor, keep_start: int, keep_end: int) -> list[torch.Tensor]:
        if self.secondary_restorer is not None:
            result = self.secondary_restorer.restore(primary_raw, keep_start=keep_start, keep_end=keep_end)
            if isinstance(result, torch.Tensor):
                restored_frames = list(result.unbind(0)) if result.dim() > 3 else [result]
            else:
                restored_frames = result
        else:
            kept = primary_raw[keep_start:keep_end]
            restored_frames = list(kept.clamp(0, 1).mul(255.0).round().clamp(0, 255).to(dtype=torch.uint8).unbind(0))

        if self._denoise_step is DenoiseStep.AFTER_SECONDARY:
            batch_u8 = torch.stack(restored_frames, dim=0)
            batch_u8 = apply_denoise_u8(batch_u8, self._denoise_strength)
            restored_frames = list(batch_u8.unbind(0))

        return restored_frames

    def prepare_and_run_primary(
        self,
        clip: TrackedClip,
        raw_crops: list[RawCrop],
        frame_shape: tuple[int, int],
        keep_start: int,
        keep_end: int,
        crossfade_weights: dict[int, float] | None,
    ) -> PrimaryRestoreResult:
        resized_crops, enlarged_bboxes, crop_shapes, pad_offsets, resize_shapes = self._prepare_from_raw_crops(raw_crops)
        primary_raw = self.restorer.raw_process(resized_crops)
        if self._denoise_step is DenoiseStep.AFTER_PRIMARY:
            primary_raw = self._apply_denoise(primary_raw)

        return PrimaryRestoreResult(
            track_id=clip.track_id,
            start_frame=clip.start_frame,
            frame_count=len(raw_crops),
            frame_shape=frame_shape,
            frame_device=raw_crops[0].crop.device,
            masks=clip.masks,
            primary_raw=primary_raw,
            keep_start=keep_start,
            keep_end=keep_end,
            crossfade_weights=crossfade_weights,
            enlarged_bboxes=enlarged_bboxes,
            crop_shapes=crop_shapes,
            pad_offsets=pad_offsets,
            resize_shapes=resize_shapes,
        )

    def build_secondary_result(
        self,
        pr: PrimaryRestoreResult,
        restored_frames: list[torch.Tensor],
    ) -> SecondaryRestoreResult:
        if self._denoise_step is DenoiseStep.AFTER_SECONDARY:
            batch_u8 = torch.stack(restored_frames, dim=0)
            batch_u8 = apply_denoise_u8(batch_u8, self._denoise_strength)
            restored_frames = list(batch_u8.unbind(0))

        ks = max(0, pr.keep_start)
        ke = min(pr.frame_count, pr.keep_end)
        kept_count = ke - ks

        return SecondaryRestoreResult(
            track_id=pr.track_id,
            start_frame=pr.start_frame,
            frame_count=pr.frame_count,
            frame_shape=pr.frame_shape,
            frame_device=pr.frame_device,
            masks=pr.masks[ks:ke],
            restored_frames=restored_frames,
            keep_start=0,
            keep_end=kept_count,
            crossfade_weights=pr.crossfade_weights,
            enlarged_bboxes=pr.enlarged_bboxes[ks:ke],
            crop_shapes=pr.crop_shapes[ks:ke],
            pad_offsets=pr.pad_offsets[ks:ke],
            resize_shapes=pr.resize_shapes[ks:ke],
            clip_keep_offset=ks,
        )
