from __future__ import annotations

import collections
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING
import numpy as np
import torch
import torch.nn.functional as F

from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
from jasna.restorer.restored_clip import RestoredClip
from jasna.restorer.secondary_restorer import SecondaryRestorer
from jasna.tracking.clip_tracker import TrackedClip

if TYPE_CHECKING:
    from jasna.tracking.frame_buffer import FrameBuffer

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


@dataclass(frozen=True)
class _TvaiPendingBlend:
    frame_idx: int
    track_id: int
    mask_lr: torch.Tensor
    frame_shape: tuple[int, int]
    enlarged_bbox: tuple[int, int, int, int]
    crop_shape: tuple[int, int]
    pad_offset: tuple[int, int]
    resize_shape: tuple[int, int]

class RestorationPipeline:
    def __init__(
        self,
        restorer: BasicvsrppMosaicRestorer,
        *,
        secondary_restorer: SecondaryRestorer | None = None,
    ) -> None:
        self.restorer = restorer
        self.secondary_restorer = secondary_restorer
        self._tvai_pending: collections.deque[_TvaiPendingBlend] = collections.deque()

    def _is_tvai(self) -> bool:
        return bool(self.secondary_restorer is not None and getattr(self.secondary_restorer, "name", None) == "tvai")

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

    def restore_and_blend_clip(
        self,
        clip: TrackedClip,
        frames: list[torch.Tensor],
        *,
        keep_start: int,
        keep_end: int,
        frame_buffer: FrameBuffer,
    ) -> None:
        if self._is_tvai():
            self._restore_and_blend_clip_tvai(
                clip,
                frames,
                keep_start=int(keep_start),
                keep_end=int(keep_end),
                frame_buffer=frame_buffer,
            )
            return

        restored_clip = self.restore_clip(clip, frames, keep_start=int(keep_start), keep_end=int(keep_end))
        frame_buffer.blend_clip(clip, restored_clip, keep_start=int(keep_start), keep_end=int(keep_end))

    def flush_secondary(self, *, frame_buffer: FrameBuffer) -> None:
        if not self._is_tvai():
            return
        if self.secondary_restorer is None:
            return

        out = self.secondary_restorer.flush()
        self._consume_tvai_outputs(out, frame_buffer=frame_buffer)
        if self._tvai_pending:
            raise RuntimeError(f"TVAI flush did not produce all pending frames (pending={len(self._tvai_pending)})")

    def restore_clip(
        self,
        clip: TrackedClip,
        frames: list[torch.Tensor],
        *,
        keep_start: int,
        keep_end: int,
    ) -> RestoredClip:
        """
        clip: TrackedClip with bbox/mask info
        frames: list of (C, H, W) original frames, GPU
        Returns: RestoredClip with restored frames and metadata for blending
        """
        resized_crops, enlarged_bboxes, crop_shapes, pad_offsets, resize_shapes = self._prepare_clip_inputs(clip, frames)

        if self.secondary_restorer is None:
            restored_hwc = self.restorer.restore(resized_crops)  # list[(256,256,C)] uint8
            restored_frames = [r.permute(2, 0, 1) for r in restored_hwc]
        else:
            if self._is_tvai():
                raise RuntimeError("TVAI secondary must be executed via restore_and_blend_clip()/flush_secondary()")

            primary_raw = self.restorer.raw_process(resized_crops)  # (T,C,256,256) float in [0, 1]
            secondary_out = self.secondary_restorer.restore(primary_raw, keep_start=int(keep_start), keep_end=int(keep_end))
            if isinstance(secondary_out, list):
                restored_frames = secondary_out
            else:
                restored_frames = list(torch.unbind(secondary_out, 0))

            scaled_pad_offsets: list[tuple[int, int]] = []
            scaled_resize_shapes: list[tuple[int, int]] = []
            for i, frame_u8 in enumerate(restored_frames):
                out_h = int(frame_u8.shape[1])
                out_w = int(frame_u8.shape[2])
                pad_left, pad_top = pad_offsets[i]
                resize_h, resize_w = resize_shapes[i]

                x0 = int(round((pad_left * out_w) / RESTORATION_SIZE))
                x1 = int(round(((pad_left + resize_w) * out_w) / RESTORATION_SIZE))
                y0 = int(round((pad_top * out_h) / RESTORATION_SIZE))
                y1 = int(round(((pad_top + resize_h) * out_h) / RESTORATION_SIZE))

                scaled_pad_offsets.append((x0, y0))
                scaled_resize_shapes.append((y1 - y0, x1 - x0))

            pad_offsets = scaled_pad_offsets
            resize_shapes = scaled_resize_shapes

        _, frame_h, frame_w = frames[0].shape
        return RestoredClip(
            restored_frames=restored_frames,
            masks=clip.masks,
            frame_shape=(frame_h, frame_w),
            enlarged_bboxes=enlarged_bboxes,
            crop_shapes=crop_shapes,
            pad_offsets=pad_offsets,
            resize_shapes=resize_shapes,
        )

    def _consume_tvai_outputs(self, out: torch.Tensor | list[torch.Tensor], *, frame_buffer: FrameBuffer) -> None:
        if isinstance(out, list):
            frames = out
        else:
            frames = list(torch.unbind(out, 0))

        for restored in frames:
            if not self._tvai_pending:
                continue
            item = self._tvai_pending.popleft()
            frame_buffer.blend_restored_frame(
                frame_idx=item.frame_idx,
                track_id=item.track_id,
                restored=restored,
                mask_lr=item.mask_lr,
                frame_shape=item.frame_shape,
                enlarged_bbox=item.enlarged_bbox,
                crop_shape=item.crop_shape,
                pad_offset=item.pad_offset,
                resize_shape=item.resize_shape,
            )

    def _restore_and_blend_clip_tvai(
        self,
        clip: TrackedClip,
        frames: list[torch.Tensor],
        *,
        keep_start: int,
        keep_end: int,
        frame_buffer: FrameBuffer,
    ) -> None:
        if self.secondary_restorer is None:
            raise RuntimeError("TVAI secondary restorer is missing")

        t = int(len(frames))
        ks = max(0, int(keep_start))
        ke = min(t, int(keep_end))

        for i, frame_idx in enumerate(clip.frame_indices()):
            if not (ks <= i < ke):
                pending = frame_buffer.frames.get(int(frame_idx))
                if pending is not None:
                    pending.pending_clips.discard(int(clip.track_id))

        if ks >= ke:
            return

        resized_crops, enlarged_bboxes, crop_shapes, pad_offsets, resize_shapes = self._prepare_clip_inputs(clip, frames)
        resized_crops_kept = resized_crops[ks:ke]
        primary_raw = self.restorer.raw_process(resized_crops_kept)  # (T,C,256,256) float in [0, 1]

        out_w = int(getattr(self.secondary_restorer, "out_w", 0))
        out_h = int(getattr(self.secondary_restorer, "out_h", 0))
        if out_w <= 0 or out_h <= 0:
            raise RuntimeError("TVAI secondary restorer is missing out_w/out_h")

        frame_h, frame_w = frames[0].shape[1], frames[0].shape[2]
        for i in range(ks, ke):
            pad_left, pad_top = pad_offsets[i]
            resize_h, resize_w = resize_shapes[i]

            x0 = int(round((pad_left * out_w) / RESTORATION_SIZE))
            x1 = int(round(((pad_left + resize_w) * out_w) / RESTORATION_SIZE))
            y0 = int(round((pad_top * out_h) / RESTORATION_SIZE))
            y1 = int(round(((pad_top + resize_h) * out_h) / RESTORATION_SIZE))

            self._tvai_pending.append(
                _TvaiPendingBlend(
                    frame_idx=int(clip.start_frame) + int(i),
                    track_id=int(clip.track_id),
                    mask_lr=clip.masks[i],
                    frame_shape=(int(frame_h), int(frame_w)),
                    enlarged_bbox=enlarged_bboxes[i],
                    crop_shape=crop_shapes[i],
                    pad_offset=(x0, y0),
                    resize_shape=(y1 - y0, x1 - x0),
                )
            )

        out = self.secondary_restorer.restore(primary_raw, keep_start=0, keep_end=int(primary_raw.shape[0]))
        self._consume_tvai_outputs(out, frame_buffer=frame_buffer)

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

