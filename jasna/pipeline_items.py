from __future__ import annotations

from dataclasses import dataclass

import torch

from jasna.tracking.clip_tracker import TrackedClip


_SENTINEL = object()
_SECONDARY_FLUSH = object()


@dataclass
class ClipRestoreItem:
    clip: TrackedClip
    frames: list[torch.Tensor]
    keep_start: int
    keep_end: int
    crossfade_weights: dict[int, float] | None


@dataclass
class _RestoreResultBase:
    clip: TrackedClip
    frame_count: int
    frame_shape: tuple[int, int]
    frame_device: torch.device
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
