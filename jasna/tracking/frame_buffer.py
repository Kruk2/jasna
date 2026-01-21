from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from jasna.tracking.clip_tracker import TrackedClip
from jasna.restorer.restoration_pipeline import RestoredClip


def create_blend_mask(crop_mask: torch.Tensor, border_ratio: float = 0.05) -> torch.Tensor:
    """Create blend mask from detection mask with blurred border."""
    mask = crop_mask.squeeze()
    h, w = mask.shape
    h_inner = int(h * (1.0 - border_ratio))
    w_inner = int(w * (1.0 - border_ratio))
    h_outer = h - h_inner
    w_outer = w - w_inner
    border_size = min(h_outer, w_outer)

    if border_size < 5:
        return torch.ones_like(mask, dtype=torch.float32)

    blur_size = border_size
    if blur_size % 2 == 0:
        blur_size += 1

    inner = torch.ones((h_inner, w_inner), device=mask.device, dtype=torch.float32)
    pad_top = h_outer // 2
    pad_bottom = h_outer - pad_top
    pad_left = w_outer // 2
    pad_right = w_outer - pad_left
    blend = F.pad(inner, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)

    blend = torch.maximum((mask > 0).float(), blend)

    kernel = torch.ones((1, 1, blur_size, blur_size), device=blend.device, dtype=torch.float32) / (blur_size ** 2)
    pad_size = blur_size // 2
    blend_4d = F.pad(blend.unsqueeze(0).unsqueeze(0), (pad_size, pad_size, pad_size, pad_size), mode='replicate')
    blend = F.conv2d(blend_4d, kernel).squeeze(0).squeeze(0)

    return blend.clamp(0, 1)


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

            unpadded = restored[:, pad_top:pad_top + resize_h, pad_left:pad_left + resize_w]

            resized_back = F.interpolate(
                unpadded.unsqueeze(0).float(),
                size=(crop_h, crop_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            # Upscale mask from model resolution directly to crop size
            mask = restored_clip.masks[i].float().unsqueeze(0).unsqueeze(0)  # (1, 1, Hm, Wm)
            crop_mask = F.interpolate(mask, size=(crop_h, crop_w), mode='nearest').squeeze()

            blend_mask = create_blend_mask(crop_mask)

            blended = pending.blended_frame
            original_crop = blended[:, y1:y2, x1:x2].float()

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
