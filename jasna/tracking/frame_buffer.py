from __future__ import annotations

from dataclasses import dataclass, field

import torch

from jasna.tracking.clip_tracker import TrackedClip


@dataclass
class PendingFrame:
    frame_idx: int
    pts: int
    frame: torch.Tensor
    pending_clips: set[int] = field(default_factory=set)
    blended_frame: torch.Tensor | None = None


class FrameBuffer:
    def __init__(self, device: torch.device):
        self.device = device
        self.frames: dict[int, PendingFrame] = {}
        self.next_encode_idx: int = 0

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

    def blend_clip(self, clip: TrackedClip, restored_regions: list[torch.Tensor]) -> None:
        """
        Blend restored regions from a clip onto the pending frames.
        
        restored_regions: list of (C, H_crop, W_crop) restored tensors, one per frame in clip
        """
        for i, frame_idx in enumerate(clip.frame_indices()):
            if frame_idx not in self.frames:
                continue

            pending = self.frames[frame_idx]
            if clip.track_id not in pending.pending_clips:
                continue

            bbox = clip.bboxes[i].astype(int)
            mask = clip.masks[i]
            restored = restored_regions[i]

            x1 = max(0, bbox[0])
            y1 = max(0, bbox[1])
            x2 = bbox[2]
            y2 = bbox[3]

            crop_h = y2 - y1
            crop_w = x2 - x1
            restored_resized = restored[:, :crop_h, :crop_w]

            mask_crop = mask[y1:y2, x1:x2]
            actual_h, actual_w = mask_crop.shape
            if actual_h < crop_h or actual_w < crop_w:
                crop_h, crop_w = actual_h, actual_w
                restored_resized = restored_resized[:, :crop_h, :crop_w]

            blended = pending.blended_frame
            blended[:, y1:y1 + crop_h, x1:x1 + crop_w] = torch.where(
                mask_crop.unsqueeze(0),
                restored_resized,
                blended[:, y1:y1 + crop_h, x1:x1 + crop_w],
            )

            pending.pending_clips.discard(clip.track_id)

    def get_ready_frames(self) -> list[tuple[int, torch.Tensor, int]]:
        """
        Get frames that are ready to encode (all clips blended) in order.
        Returns list of (frame_idx, frame, pts) tuples.
        Removes them from the buffer.
        """
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
        """
        Flush all remaining frames in order, regardless of pending clips.
        Used at end of video.
        """
        remaining: list[tuple[int, torch.Tensor, int]] = []
        for frame_idx in sorted(self.frames.keys()):
            pending = self.frames[frame_idx]
            remaining.append((pending.frame_idx, pending.blended_frame, pending.pts))
        self.frames.clear()
        return remaining
