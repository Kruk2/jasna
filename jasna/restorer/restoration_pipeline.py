from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING
import numpy as np
import torch
import torch.nn.functional as F

from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
from jasna.restorer.denoise import DenoiseStep, DenoiseStrength, apply_denoise, apply_denoise_u8
from jasna.restorer.restored_clip import RestoredClip
from jasna.restorer.secondary_restorer import SecondaryRestorer
from jasna.tracking.clip_tracker import TrackedClip

if TYPE_CHECKING:
    from jasna.tracking.frame_buffer import FrameBuffer

logger = logging.getLogger(__name__)

RESTORATION_SIZE = 256
BORDER_RATIO = 0.06
MIN_BORDER = 20
MAX_EXPANSION_FACTOR = 1.0


def _torch_pad_reflect(image: torch.Tensor, paddings: tuple[int, int, int, int]) -> torch.Tensor:
    paddings_arr = np.array(paddings, dtype=int)
    while np.any(paddings_arr):
        image_limits = np.repeat(np.array(image.shape[::-1][:len(paddings_arr)//2]), 2) - 1
        possible_paddings = np.minimum(paddings_arr, image_limits)
        image = F.pad(image, tuple(possible_paddings), mode='reflect')
        paddings_arr = paddings_arr - possible_paddings
    return image


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

    def _apply_denoise(self, frames: torch.Tensor) -> torch.Tensor:
        return apply_denoise(frames, self._denoise_strength)

    def _prepare_clip_inputs(
        self,
        clip: TrackedClip,
        frames: list[torch.Tensor],
    ) -> tuple[
        list[torch.Tensor],
        list[tuple[int, int, int, int]],
        list[tuple[int, int]],
        list[tuple[int, int]],
        list[tuple[int, int]],
    ]:
        crops: list[torch.Tensor] = []
        enlarged_bboxes: list[tuple[int, int, int, int]] = []
        crop_shapes: list[tuple[int, int]] = []

        for i, frame in enumerate(frames):
            _, frame_h, frame_w = frame.shape
            bbox = clip.bboxes[i]
            x1 = int(np.floor(bbox[0]))
            y1 = int(np.floor(bbox[1]))
            x2 = int(np.ceil(bbox[2]))
            y2 = int(np.ceil(bbox[3]))
            x1 = max(0, min(x1, frame_w))
            y1 = max(0, min(y1, frame_h))
            x2 = max(0, min(x2, frame_w))
            y2 = max(0, min(y2, frame_h))

            x1_exp, y1_exp, x2_exp, y2_exp = self._expand_bbox(x1, y1, x2, y2, frame_h, frame_w)
            enlarged_bboxes.append((x1_exp, y1_exp, x2_exp, y2_exp))

            crop = frame[:, y1_exp:y2_exp, x1_exp:x2_exp]
            crop_shapes.append((int(crop.shape[1]), int(crop.shape[2])))
            crops.append(crop)

        max_h = max(s[0] for s in crop_shapes)
        max_w = max(s[1] for s in crop_shapes)

        scale_h = RESTORATION_SIZE / max_h
        scale_w = RESTORATION_SIZE / max_w
        if scale_h > 1.0 and scale_w > 1.0:
            scale_h = scale_w = 1.0

        resized_crops: list[torch.Tensor] = []
        resize_shapes: list[tuple[int, int]] = []
        pad_offsets: list[tuple[int, int]] = []

        for crop, (crop_h, crop_w) in zip(crops, crop_shapes):
            new_h = int(crop_h * scale_h)
            new_w = int(crop_w * scale_w)
            resize_shapes.append((new_h, new_w))

            resized = F.interpolate(
                crop.unsqueeze(0).float(),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            pad_top = (RESTORATION_SIZE - new_h) // 2
            pad_left = (RESTORATION_SIZE - new_w) // 2
            pad_bottom = RESTORATION_SIZE - new_h - pad_top
            pad_right = RESTORATION_SIZE - new_w - pad_left
            pad_offsets.append((pad_left, pad_top))

            padded = _torch_pad_reflect(resized, (pad_left, pad_right, pad_top, pad_bottom))
            resized_crops.append(padded.to(crop.dtype).permute(1, 2, 0))

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

    def _scale_offsets(self, frame_u8: torch.Tensor, pad_offset_256: tuple[int, int], resize_shape_256: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        out_h = int(frame_u8.shape[1])
        out_w = int(frame_u8.shape[2])
        pl, pt = pad_offset_256
        rh, rw = resize_shape_256
        x0 = int(round(pl * out_w / RESTORATION_SIZE))
        x1 = int(round((pl + rw) * out_w / RESTORATION_SIZE))
        y0 = int(round(pt * out_h / RESTORATION_SIZE))
        y1 = int(round((pt + rh) * out_h / RESTORATION_SIZE))
        return (x0, y0), (y1 - y0, x1 - x0)

    def restore_and_blend_clip(
        self,
        clip: TrackedClip,
        frames: list[torch.Tensor],
        *,
        keep_start: int,
        keep_end: int,
        frame_buffer: FrameBuffer,
        crossfade_weights: dict[int, float] | None = None,
    ) -> None:
        t = len(frames)
        ks = max(0, keep_start)
        ke = min(t, keep_end)

        for i, frame_idx in enumerate(clip.frame_indices()):
            if not (ks <= i < ke):
                pending = frame_buffer.frames.get(frame_idx)
                if pending is not None:
                    pending.pending_clips.discard(clip.track_id)

        if ks >= ke:
            return

        resized_crops, enlarged_bboxes, crop_shapes, pad_offsets, resize_shapes = self._prepare_clip_inputs(clip, frames)
        primary_raw = self.restorer.raw_process(resized_crops)
        if self._denoise_step is DenoiseStep.AFTER_PRIMARY:
            primary_raw = self._apply_denoise(primary_raw)

        frame_h, frame_w = frames[0].shape[1], frames[0].shape[2]
        restored_frames = self._run_secondary(primary_raw, ks, ke)

        for local_i, i in enumerate(range(ks, ke)):
            frame_idx = clip.start_frame + i
            track_id = clip.track_id
            if not frame_buffer.needs_blend(frame_idx=frame_idx, track_id=track_id):
                continue

            frame_u8 = restored_frames[local_i]
            pad_offset, resize_shape = self._scale_offsets(frame_u8, pad_offsets[i], resize_shapes[i])
            cw = crossfade_weights.get(i, 1.0) if crossfade_weights else 1.0

            frame_buffer.blend_restored_frame(
                frame_idx=frame_idx,
                track_id=track_id,
                restored=frame_u8,
                mask_lr=clip.masks[i],
                frame_shape=(frame_h, frame_w),
                enlarged_bbox=enlarged_bboxes[i],
                crop_shape=crop_shapes[i],
                pad_offset=pad_offset,
                resize_shape=resize_shape,
                crossfade_weight=cw,
            )

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def restore_clip(
        self,
        clip: TrackedClip,
        frames: list[torch.Tensor],
        *,
        keep_start: int,
        keep_end: int,
    ) -> RestoredClip:
        resized_crops, enlarged_bboxes, crop_shapes, pad_offsets, resize_shapes = self._prepare_clip_inputs(clip, frames)
        primary_raw = self.restorer.raw_process(resized_crops)
        if self._denoise_step is DenoiseStep.AFTER_PRIMARY:
            primary_raw = self._apply_denoise(primary_raw)

        restored_frames = self._run_secondary(primary_raw, int(keep_start), int(keep_end))

        scaled_pad_offsets: list[tuple[int, int]] = []
        scaled_resize_shapes: list[tuple[int, int]] = []
        for frame_u8, po, rs in zip(restored_frames, pad_offsets, resize_shapes):
            pad_offset, resize_shape = self._scale_offsets(frame_u8, po, rs)
            scaled_pad_offsets.append(pad_offset)
            scaled_resize_shapes.append(resize_shape)

        _, frame_h, frame_w = frames[0].shape
        return RestoredClip(
            restored_frames=restored_frames,
            masks=clip.masks,
            frame_shape=(frame_h, frame_w),
            enlarged_bboxes=enlarged_bboxes,
            crop_shapes=crop_shapes,
            pad_offsets=scaled_pad_offsets,
            resize_shapes=scaled_resize_shapes,
        )

    def _expand_bbox(
        self, x1: int, y1: int, x2: int, y2: int, frame_h: int, frame_w: int
    ) -> tuple[int, int, int, int]:
        w = x2 - x1
        h = y2 - y1

        border = max(MIN_BORDER, int(max(w, h) * BORDER_RATIO)) if BORDER_RATIO > 0.0 else 0
        x1_exp = max(0, x1 - border)
        y1_exp = max(0, y1 - border)
        x2_exp = min(frame_w, x2 + border)
        y2_exp = min(frame_h, y2 + border)

        w = x2_exp - x1_exp
        h = y2_exp - y1_exp
        down_scale_factor = min(RESTORATION_SIZE / w, RESTORATION_SIZE / h) if w > 0 and h > 0 else 1.0
        if down_scale_factor > 1.0:
            down_scale_factor = 1.0

        missing_w = int((RESTORATION_SIZE - (w * down_scale_factor)) / down_scale_factor) if down_scale_factor > 0 else 0
        missing_h = int((RESTORATION_SIZE - (h * down_scale_factor)) / down_scale_factor) if down_scale_factor > 0 else 0

        available_w_l = x1_exp
        available_w_r = frame_w - x2_exp
        available_h_t = y1_exp
        available_h_b = frame_h - y2_exp

        budget_w = int(MAX_EXPANSION_FACTOR * w)
        budget_h = int(MAX_EXPANSION_FACTOR * h)

        expand_w_lr = min(available_w_l, available_w_r, missing_w // 2, budget_w)
        expand_w_l = min(available_w_l - expand_w_lr, missing_w - expand_w_lr * 2, budget_w - expand_w_lr)
        expand_w_r = min(
            available_w_r - expand_w_lr,
            missing_w - expand_w_lr * 2 - expand_w_l,
            budget_w - expand_w_lr - expand_w_l,
        )

        expand_h_tb = min(available_h_t, available_h_b, missing_h // 2, budget_h)
        expand_h_t = min(available_h_t - expand_h_tb, missing_h - expand_h_tb * 2, budget_h - expand_h_tb)
        expand_h_b = min(
            available_h_b - expand_h_tb,
            missing_h - expand_h_tb * 2 - expand_h_t,
            budget_h - expand_h_tb - expand_h_t,
        )

        x1_exp = x1_exp - math.floor(expand_w_lr / 2) - expand_w_l
        x2_exp = x2_exp + math.ceil(expand_w_lr / 2) + expand_w_r
        y1_exp = y1_exp - math.floor(expand_h_tb / 2) - expand_h_t
        y2_exp = y2_exp + math.ceil(expand_h_tb / 2) + expand_h_b

        x1_exp = max(0, min(int(x1_exp), frame_w))
        x2_exp = max(0, min(int(x2_exp), frame_w))
        y1_exp = max(0, min(int(y1_exp), frame_h))
        y2_exp = max(0, min(int(y2_exp), frame_h))

        return x1_exp, y1_exp, x2_exp, y2_exp
