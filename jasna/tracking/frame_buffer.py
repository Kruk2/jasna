from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from jasna.restorer.restored_clip import RestoredClip
from jasna.tracking.clip_tracker import TrackedClip
from jasna.tracking.blending import create_blend_mask

@dataclass
class PendingFrame:
    frame_idx: int
    pts: int
    frame: torch.Tensor
    blended_frame: torch.Tensor
    pending_clips: set[int] = field(default_factory=set)


class FrameBuffer:
    def __init__(
        self,
        device: torch.device,
        *,
        blend_mask_fn: Callable[[torch.Tensor], torch.Tensor] = create_blend_mask,
    ):
        self.device = device
        self.frames: dict[int, PendingFrame] = {}
        self.next_encode_idx: int = 0
        self.blend_mask_fn = blend_mask_fn

    def add_frame(
        self, frame_idx: int, pts: int, frame: torch.Tensor, clip_track_ids: set[int]
    ) -> None:
        self.frames[frame_idx] = PendingFrame(
            frame_idx=frame_idx,
            pts=pts,
            frame=frame,
            pending_clips=clip_track_ids.copy(),
            blended_frame=frame,
        )

    def get_frame(self, frame_idx: int) -> torch.Tensor | None:
        pending = self.frames.get(frame_idx)
        return pending.frame if pending else None

    def add_pending_clip(self, frame_indices: list[int], track_id: int) -> None:
        for frame_idx in frame_indices:
            pending = self.frames.get(frame_idx)
            if pending is None:
                continue
            pending.pending_clips.add(track_id)

    def remove_pending_clip(self, frame_indices: list[int], track_id: int) -> None:
        for frame_idx in frame_indices:
            pending = self.frames.get(frame_idx)
            if pending is None:
                continue
            pending.pending_clips.discard(track_id)

    def blend_clip(
        self,
        clip: TrackedClip,
        restored_clip: RestoredClip,
        *,
        keep_start: int,
        keep_end: int,
    ) -> None:
        for i, frame_idx in enumerate(clip.frame_indices()):
            if frame_idx not in self.frames:
                continue

            pending = self.frames[frame_idx]
            if clip.track_id not in pending.pending_clips:
                continue

            if not (keep_start <= i < keep_end):
                pending.pending_clips.discard(clip.track_id)
                continue

            restored = restored_clip.restored_frames[i]
            pad_left, pad_top = restored_clip.pad_offsets[i]
            resize_h, resize_w = restored_clip.resize_shapes[i]
            crop_h, crop_w = restored_clip.crop_shapes[i]
            x1, y1, x2, y2 = restored_clip.enlarged_bboxes[i]

            if pending.blended_frame is pending.frame:
                pending.blended_frame = pending.frame.clone()
            blended = pending.blended_frame

            unpadded = restored[:, pad_top:pad_top + resize_h, pad_left:pad_left + resize_w]

            resized_back = F.interpolate(
                unpadded.unsqueeze(0).float(),
                size=(crop_h, crop_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            frame_h, frame_w = restored_clip.frame_shape
            mask_lr = restored_clip.masks[i].float()  # (Hm, Wm)
            hm, wm = mask_lr.shape
            y_idx = (torch.arange(y1, y2, device=mask_lr.device) * hm) // frame_h
            x_idx = (torch.arange(x1, x2, device=mask_lr.device) * wm) // frame_w
            crop_mask = mask_lr.index_select(0, y_idx).index_select(1, x_idx)

            blend_mask = self.blend_mask_fn(crop_mask)

            original_crop = blended[:, y1:y2, x1:x2].float()

            blended_crop = original_crop + (resized_back - original_crop) * blend_mask.unsqueeze(0)
            blended[:, y1:y2, x1:x2] = blended_crop.round().clamp(0, 255).to(blended.dtype)

            pending.pending_clips.discard(clip.track_id)

    def blend_restored_frame(
        self,
        *,
        frame_idx: int,
        track_id: int,
        restored: torch.Tensor,
        mask_lr: torch.Tensor,
        frame_shape: tuple[int, int],
        enlarged_bbox: tuple[int, int, int, int],
        crop_shape: tuple[int, int],
        pad_offset: tuple[int, int],
        resize_shape: tuple[int, int],
        crossfade_weight: float = 1.0,
    ) -> None:
        pending = self.frames.get(int(frame_idx))
        if pending is None:
            return
        if int(track_id) not in pending.pending_clips:
            return

        x1, y1, x2, y2 = enlarged_bbox
        crop_h, crop_w = crop_shape
        pad_left, pad_top = pad_offset
        resize_h, resize_w = resize_shape

        if pending.blended_frame is pending.frame:
            pending.blended_frame = pending.frame.clone()
        blended = pending.blended_frame

        unpadded = restored[:, pad_top:pad_top + resize_h, pad_left:pad_left + resize_w]
        resized_back = F.interpolate(
            unpadded.unsqueeze(0).float(),
            size=(crop_h, crop_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        frame_h, frame_w = frame_shape
        hm, wm = mask_lr.shape
        y_idx = (torch.arange(y1, y2, device=mask_lr.device) * hm) // frame_h
        x_idx = (torch.arange(x1, x2, device=mask_lr.device) * wm) // frame_w
        crop_mask = mask_lr.float().index_select(0, y_idx).index_select(1, x_idx)
        blend_mask = self.blend_mask_fn(crop_mask)

        if crossfade_weight < 1.0:
            blend_mask = blend_mask * crossfade_weight
            original_crop = pending.frame[:, y1:y2, x1:x2].float()
            delta = (resized_back - original_crop) * blend_mask.unsqueeze(0)
            current = blended[:, y1:y2, x1:x2].float()
            blended[:, y1:y2, x1:x2] = (current + delta).round().clamp(0, 255).to(blended.dtype)
        else:
            original_crop = blended[:, y1:y2, x1:x2].float()
            blended_crop = original_crop + (resized_back - original_crop) * blend_mask.unsqueeze(0)
            blended[:, y1:y2, x1:x2] = blended_crop.round().clamp(0, 255).to(blended.dtype)

        pending.pending_clips.discard(int(track_id))

    def get_ready_frames(self) -> list[tuple[int, torch.Tensor, int]]:
        ready: list[tuple[int, torch.Tensor, int]] = []

        while self.next_encode_idx in self.frames:
            pending = self.frames[self.next_encode_idx]
            if pending.pending_clips:
                break
            ready.append((pending.frame_idx, pending.blended_frame, pending.pts))
            del self.frames[self.next_encode_idx]
            self.next_encode_idx += 1

        return ready

    def flush(self) -> list[tuple[int, torch.Tensor, int]]:
        remaining: list[tuple[int, torch.Tensor, int]] = []
        for frame_idx in sorted(self.frames.keys()):
            pending = self.frames[frame_idx]
            remaining.append((pending.frame_idx, pending.blended_frame, pending.pts))
        self.frames.clear()
        return remaining
