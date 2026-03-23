from __future__ import annotations

from dataclasses import dataclass

import torch

from jasna.crop_buffer import RawCrop
from jasna.tracking.clip_tracker import TrackedClip


_SENTINEL = object()


@dataclass(frozen=True)
class FrameMeta:
    frame_idx: int
    pts: int


@dataclass
class ClipRestoreItem:
    clip: TrackedClip
    raw_crops: list[RawCrop]
    frame_shape: tuple[int, int]
    keep_start: int
    keep_end: int
    crossfade_weights: dict[int, float] | None


@dataclass
class _RestoreResultBase:
    track_id: int
    start_frame: int
    frame_count: int
    frame_shape: tuple[int, int]
    frame_device: torch.device
    masks: list[torch.Tensor]
    keep_start: int
    keep_end: int
    crossfade_weights: dict[int, float] | None
    enlarged_bboxes: list[tuple[int, int, int, int]]
    crop_shapes: list[tuple[int, int]]
    pad_offsets: list[tuple[int, int]]
    resize_shapes: list[tuple[int, int]]


@dataclass
class PrimaryRestoreResult(_RestoreResultBase):
    primary_raw: torch.Tensor


@dataclass
class SecondaryRestoreResult(_RestoreResultBase):
    restored_frames: list[torch.Tensor]
    clip_keep_offset: int = 0


@dataclass
class SecondaryLoopStats:
    starvation_flushes: int = 0
    starvation_seconds: float = 0.0
    pusher_stall_seconds: float = 0.0
    clips_pushed: int = 0
    clips_popped: int = 0
