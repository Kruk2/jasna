from __future__ import annotations

import logging

import torch

from jasna.crop_buffer import (
    RawCrop,
    prepare_crops_for_restoration,
)
from jasna.pipeline_items import PrimaryRestoreResult, SecondaryRestoreResult
from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
from jasna.restorer.denoise import DenoiseStrength, apply_denoise
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
    ) -> None:
        self.restorer = restorer
        self.secondary_restorer = secondary_restorer
        self._denoise_strength = denoise_strength
        logger.info(
            "RestorationPipeline: secondary=%s denoise=%s",
            secondary_restorer.name if secondary_restorer else "none",
            denoise_strength.name,
        )

    @property
    def secondary_num_workers(self) -> int:
        if self.secondary_restorer is not None:
            return self.secondary_restorer.num_workers
        return 1

    @property
    def secondary_prefers_cpu_input(self) -> bool:
        if self.secondary_restorer is not None:
            return self.secondary_restorer.prefers_cpu_input
        return False

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
        resized_crops, pad_offsets, resize_shapes = prepare_crops_for_restoration(
            raw_crops, self.restorer.device, self.restorer.input_dtype
        )
        enlarged_bboxes = [c.enlarged_bbox for c in raw_crops]
        crop_shapes = [c.crop_shape for c in raw_crops]
        return resized_crops, enlarged_bboxes, crop_shapes, pad_offsets, resize_shapes

    def _run_secondary(self, primary_raw: torch.Tensor, keep_start: int, keep_end: int) -> list[torch.Tensor]:
        if self.secondary_restorer is not None:
            restored_frames = self.secondary_restorer.restore(primary_raw, keep_start=keep_start, keep_end=keep_end)
        else:
            kept = primary_raw[keep_start:keep_end]
            restored_frames = list(kept.clamp(0, 1).mul(255.0).round().clamp(0, 255).to(dtype=torch.uint8).unbind(0))

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
        primary_raw = apply_denoise(self.restorer.raw_process(resized_crops), self._denoise_strength)

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
