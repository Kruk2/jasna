from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from jasna.tracking.clip_tracker import TrackedClip


def create_blend_mask(h: int, w: int, device: torch.device, border_ratio: float = 0.05) -> torch.Tensor:
    h_inner, w_inner = int(h * (1.0 - border_ratio)), int(w * (1.0 - border_ratio))
    h_outer, w_outer = h - h_inner, w - w_inner
    border_size = min(h_outer, w_outer)
    
    if border_size < 5:
        return torch.ones((h, w), device=device, dtype=torch.float32)
    
    blur_size = border_size
    if blur_size % 2 == 0:
        blur_size += 1
    
    inner = torch.ones((h_inner, w_inner), device=device, dtype=torch.float32)
    
    pad_top = h_outer // 2
    pad_bottom = h_outer - pad_top
    pad_left = w_outer // 2
    pad_right = w_outer - pad_left
    
    blend = F.pad(inner, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    
    kernel = torch.ones((1, 1, blur_size, blur_size), device=device, dtype=torch.float32) / (blur_size ** 2)
    pad_size = blur_size // 2
    blend_4d = F.pad(blend.unsqueeze(0).unsqueeze(0), (pad_size, pad_size, pad_size, pad_size), mode='replicate')
    blend = F.conv2d(blend_4d, kernel).squeeze(0).squeeze(0)
    
    return blend


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

    def blend_clip(
        self, clip: TrackedClip, restored_regions: list[torch.Tensor], frame_hw: tuple[int, int]
    ) -> None:
        frame_h, frame_w = frame_hw
        
        for i, frame_idx in enumerate(clip.frame_indices()):
            if frame_idx not in self.frames:
                continue

            pending = self.frames[frame_idx]
            if clip.track_id not in pending.pending_clips:
                continue

            bbox = clip.bboxes[i].astype(int)
            restored = restored_regions[i]

            x1 = max(0, bbox[0])
            y1 = max(0, bbox[1])
            x2 = bbox[2]
            y2 = bbox[3]

            crop_h = y2 - y1
            crop_w = x2 - x1

            if y2 > frame_h or x2 > frame_w:
                crop_h = min(crop_h, frame_h - y1)
                crop_w = min(crop_w, frame_w - x1)

            restored_resized = F.interpolate(
                restored.unsqueeze(0).float(),
                size=(crop_h, crop_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            blend_mask = create_blend_mask(crop_h, crop_w, restored.device)

            blended = pending.blended_frame
            original_crop = blended[:, y1:y1 + crop_h, x1:x1 + crop_w].float()
            
            blended_crop = (restored_resized - original_crop) * blend_mask.unsqueeze(0) + original_crop
            blended[:, y1:y1 + crop_h, x1:x1 + crop_w] = blended_crop.round().clamp(0, 255).to(blended.dtype)

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
