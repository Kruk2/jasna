from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from jasna.restorer.restored_clip import RestoredClip
from jasna.tracking.clip_tracker import TrackedClip
from jasna.tracking.blending import create_blend_mask

from jasna.tensor_utils import to_device as _to_device

_log = logging.getLogger(__name__)

@dataclass
class PendingFrame:
    frame_idx: int
    pts: int
    frame: torch.Tensor
    blended_frame: torch.Tensor
    pending_clips: set[int] = field(default_factory=set)
    device_lock: threading.Lock = field(default_factory=threading.Lock)


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
        self._gpu_pinned: set[int] = set()

    def _ensure_on_device(self, pending: PendingFrame) -> None:
        self._gpu_pinned.add(pending.frame_idx)
        if pending.frame.device != self.device:
            gpu_frame = _to_device(pending.frame, self.device)
            if pending.blended_frame is pending.frame:
                pending.frame = gpu_frame
                pending.blended_frame = gpu_frame
            else:
                pending.blended_frame = _to_device(pending.blended_frame, self.device)
                pending.frame = gpu_frame

    def offload_gpu_frames(self, bytes_to_free: int) -> int:
        count = 0
        freed = 0
        for idx in list(self.frames):
            if freed >= bytes_to_free:
                break
            if idx in self._gpu_pinned:
                continue
            pending = self.frames.get(idx)
            if pending is None:
                continue
            with pending.device_lock:
                if idx in self._gpu_pinned:
                    continue
                if pending.frame.device.type == "cpu":
                    continue
                frame_bytes = pending.frame.nelement() * pending.frame.element_size()
                cpu_frame = pending.frame.cpu()
                if pending.blended_frame is pending.frame:
                    pending.frame = cpu_frame
                    pending.blended_frame = cpu_frame
                    freed += frame_bytes
                else:
                    freed += frame_bytes + pending.blended_frame.nelement() * pending.blended_frame.element_size()
                    pending.blended_frame = pending.blended_frame.cpu()
                    pending.frame = cpu_frame
                count += 1
        if count > 0:
            _log.debug("[fb] offloaded %d gpu frames to cpu (freed ~%.1f MiB)", count, freed / (1024 ** 2))
        return count

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

    def needs_blend(self, *, frame_idx: int, track_id: int) -> bool:
        pending = self.frames.get(int(frame_idx))
        if pending is None:
            return False
        return int(track_id) in pending.pending_clips

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

            with pending.device_lock:
                self._ensure_on_device(pending)
                if pending.blended_frame is pending.frame:
                    pending.blended_frame = pending.frame.clone()
                blended = pending.blended_frame

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
            # Always unpin after blend; get_ready_frames re-pins when encoding.
            # This lets the offload thread reclaim VRAM from frames that are
            # done blending but stuck behind a not-yet-ready head frame.
            self._gpu_pinned.discard(frame_idx)

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

        with pending.device_lock:
            self._ensure_on_device(pending)
            if pending.blended_frame is pending.frame:
                pending.blended_frame = pending.frame.clone()
            blended = pending.blended_frame
            device = pending.frame.device

        x1, y1, x2, y2 = enlarged_bbox
        crop_h, crop_w = crop_shape
        pad_left, pad_top = pad_offset
        resize_h, resize_w = resize_shape

        restored = restored.to(device)
        mask_lr = mask_lr.to(device)

        unpadded = restored[:, pad_top:pad_top + resize_h, pad_left:pad_left + resize_w]
        resized_back = F.interpolate(
            unpadded.unsqueeze(0).float(),
            size=(crop_h, crop_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        frame_h, frame_w = frame_shape
        hm, wm = mask_lr.shape
        y_idx = (torch.arange(y1, y2, device=device) * hm) // frame_h
        x_idx = (torch.arange(x1, x2, device=device) * wm) // frame_w
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
        self._gpu_pinned.discard(int(frame_idx))

    def get_ready_frames(self) -> Iterator[tuple[int, torch.Tensor, int]]:
        while self.next_encode_idx in self.frames:
            pending = self.frames[self.next_encode_idx]
            if pending.pending_clips:
                break
            with pending.device_lock:
                self._ensure_on_device(pending)
                result = (pending.frame_idx, pending.blended_frame, pending.pts)
            del self.frames[self.next_encode_idx]
            self._gpu_pinned.discard(self.next_encode_idx)
            self.next_encode_idx += 1
            yield result

    def flush(self) -> Iterator[tuple[int, torch.Tensor, int]]:
        for frame_idx in sorted(self.frames.keys()):
            pending = self.frames.pop(frame_idx)
            with pending.device_lock:
                self._ensure_on_device(pending)
                result = (pending.frame_idx, pending.blended_frame, pending.pts)
            self._gpu_pinned.discard(frame_idx)
            yield result
