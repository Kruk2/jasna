from __future__ import annotations

import itertools
import logging
import threading
from collections.abc import Callable

import torch
import torch.nn.functional as F

from jasna.crop_buffer import scale_offsets
from jasna.pipeline_items import SecondaryRestoreResult
from jasna.tracking.blending import create_blend_mask

_log = logging.getLogger(__name__)


class BlendBuffer:
    def __init__(
        self,
        device: torch.device,
        blend_mask_fn: Callable[[torch.Tensor], torch.Tensor] = create_blend_mask,
    ):
        self.device = device
        self.blend_mask_fn = blend_mask_fn
        self._lock = threading.Lock()
        self.pending_map: dict[int, set[int]] = {}
        self._results: dict[int, SecondaryRestoreResult] = {}
        self._result_last_frame: dict[int, int] = {}

    def register_frame(self, frame_idx: int, pending_track_ids: set[int]) -> None:
        if pending_track_ids:
            with self._lock:
                self.pending_map[frame_idx] = pending_track_ids.copy()

    def add_pending_clip(self, frame_indices: list[int], track_id: int) -> None:
        with self._lock:
            for frame_idx in frame_indices:
                pending = self.pending_map.get(frame_idx)
                if pending is None:
                    continue
                pending.add(track_id)

    def remove_pending_clip(self, frame_indices: list[int], track_id: int) -> None:
        with self._lock:
            for frame_idx in frame_indices:
                pending = self.pending_map.get(frame_idx)
                if pending is None:
                    continue
                pending.discard(track_id)

    def add_result(self, sr: SecondaryRestoreResult) -> None:
        clip_offset = sr.clip_keep_offset
        kept_count = sr.keep_end
        start = sr.start_frame

        with self._lock:
            for i in itertools.chain(range(clip_offset), range(clip_offset + kept_count, sr.frame_count)):
                pending = self.pending_map.get(start + i)
                if pending is not None:
                    pending.discard(sr.track_id)

            self._results[sr.track_id] = sr
            last_frame = start + clip_offset + kept_count - 1
            self._result_last_frame[sr.track_id] = last_frame

    def is_frame_ready(self, frame_idx: int) -> bool:
        with self._lock:
            pending = self.pending_map.get(frame_idx)
            if not pending:
                return True
            return all(tid in self._results for tid in pending)

    def blend_frame(self, frame_idx: int, original_frame: torch.Tensor) -> torch.Tensor:
        with self._lock:
            pending = self.pending_map.pop(frame_idx, None)
            if not pending:
                return original_frame
            results_snapshot = [
                (track_id, self._results.get(track_id))
                for track_id in pending
            ]

        blended = original_frame.clone()
        device = original_frame.device

        for track_id, sr in results_snapshot:
            if sr is None:
                continue
            self._apply_blend(blended, original_frame, frame_idx, track_id, sr, device)

        with self._lock:
            for track_id, sr in results_snapshot:
                if sr is not None and self._result_last_frame.get(track_id) == frame_idx:
                    del self._results[track_id]
                    del self._result_last_frame[track_id]

        return blended

    def _apply_blend(
        self,
        blended: torch.Tensor,
        original: torch.Tensor,
        frame_idx: int,
        track_id: int,
        sr: SecondaryRestoreResult,
        device: torch.device,
    ) -> None:
        clip_offset = sr.clip_keep_offset
        local_i = frame_idx - sr.start_frame - clip_offset

        if local_i < 0 or local_i >= sr.keep_end:
            return

        frame_u8 = sr.restored_frames[local_i].to(device)
        pad_offset, resize_shape = scale_offsets(frame_u8, sr.pad_offsets[local_i], sr.resize_shapes[local_i])
        i_clip = clip_offset + local_i
        cw = sr.crossfade_weights.get(i_clip, 1.0) if sr.crossfade_weights else 1.0

        x1, y1, x2, y2 = sr.enlarged_bboxes[local_i]
        crop_h, crop_w = sr.crop_shapes[local_i]
        pad_left, pad_top = pad_offset
        resize_h, resize_w = resize_shape

        mask_lr = sr.masks[local_i].to(device)

        unpadded = frame_u8[:, pad_top:pad_top + resize_h, pad_left:pad_left + resize_w]
        resized_back = F.interpolate(
            unpadded.unsqueeze(0).float(),
            size=(crop_h, crop_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        frame_h, frame_w = sr.frame_shape
        hm, wm = mask_lr.shape
        y_idx = (torch.arange(y1, y2, device=device) * hm) // frame_h
        x_idx = (torch.arange(x1, x2, device=device) * wm) // frame_w
        crop_mask = mask_lr.float().index_select(0, y_idx).index_select(1, x_idx)
        blend_mask = self.blend_mask_fn(crop_mask)

        if cw < 1.0:
            blend_mask = blend_mask * cw
            original_crop = original[:, y1:y2, x1:x2].float()
            delta = (resized_back - original_crop) * blend_mask.unsqueeze(0)
            current = blended[:, y1:y2, x1:x2].float()
            current.add_(delta).round_().clamp_(0, 255)
            blended[:, y1:y2, x1:x2] = current.to(blended.dtype)
        else:
            original_crop = blended[:, y1:y2, x1:x2].float()
            original_crop.lerp_(resized_back, blend_mask.unsqueeze(0)).round_().clamp_(0, 255)
            blended[:, y1:y2, x1:x2] = original_crop.to(blended.dtype)
