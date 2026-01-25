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
        compute_dtype: torch.dtype,
        *,
        blend_mask_fn: Callable[[torch.Tensor], torch.Tensor] = create_blend_mask,
    ):
        self.device = device
        self.frames: dict[int, PendingFrame] = {}
        self.next_encode_idx: int = 0
        self.blend_mask_fn = blend_mask_fn
        self.compute_dtype = compute_dtype

    def add_frame(
        self, frame_idx: int, pts: int, frame: torch.Tensor, clip_track_ids: set[int]
    ) -> None:
        blended = frame.clone() if clip_track_ids else frame
        self.frames[frame_idx] = PendingFrame(
            frame_idx=frame_idx,
            pts=pts,
            frame=frame,
            pending_clips=clip_track_ids.copy(),
            blended_frame=blended,
        )

    def get_frame(self, frame_idx: int) -> torch.Tensor | None:
        pending = self.frames.get(frame_idx)
        return pending.frame if pending else None

    def blend_clip(self, clip: TrackedClip, restored_clip: RestoredClip) -> None:
        for i, frame_idx in enumerate(clip.frame_indices()):
            if frame_idx not in self.frames:
                continue

            pending = self.frames[frame_idx]
            if clip.track_id not in pending.pending_clips:
                continue

            restored = restored_clip.restored_frames[i]
            pad_left, pad_top = restored_clip.pad_offsets[i]
            resize_h, resize_w = restored_clip.resize_shapes[i]
            crop_h, crop_w = restored_clip.crop_shapes[i]
            x1, y1, x2, y2 = restored_clip.enlarged_bboxes[i]

            blended = pending.blended_frame

            unpadded = restored[:, pad_top:pad_top + resize_h, pad_left:pad_left + resize_w]

            resized_back = F.interpolate(
                unpadded.unsqueeze(0).to(dtype=self.compute_dtype),
                size=(crop_h, crop_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            padded_mask = restored_clip.padded_masks[i]
            unpadded_mask = padded_mask[pad_top:pad_top + resize_h, pad_left:pad_left + resize_w].to(dtype=self.compute_dtype)
            crop_mask = F.interpolate(
                unpadded_mask.unsqueeze(0).unsqueeze(0),
                size=(crop_h, crop_w),
                mode="nearest",
            ).squeeze(0).squeeze(0)

            blend_mask = self.blend_mask_fn(crop_mask)

            original_crop = blended[:, y1:y2, x1:x2].to(dtype=self.compute_dtype)

            blended_crop = original_crop + (resized_back - original_crop) * blend_mask.unsqueeze(0)
            blended[:, y1:y2, x1:x2] = blended_crop.round().clamp(0, 255).to(blended.dtype)

            pending.pending_clips.discard(clip.track_id)

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
