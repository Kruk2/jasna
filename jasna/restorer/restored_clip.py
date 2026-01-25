from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RestoredClip:
    restored_frames: list[torch.Tensor]  # each (C, 256, 256), GPU
    masks: list[torch.Tensor]  # each (Hm, Wm) bool, GPU (model resolution)
    padded_masks: list[torch.Tensor]  # each (256, 256) bool, GPU (aligned with restored_frames)
    frame_shape: tuple[int, int]  # (H, W) original frame shape
    enlarged_bboxes: list[tuple[int, int, int, int]]  # each (x1, y1, x2, y2) after expansion
    crop_shapes: list[tuple[int, int]]  # each (H, W) original crop shape before resize
    pad_offsets: list[tuple[int, int]]  # each (pad_left, pad_top)
    resize_shapes: list[tuple[int, int]]  # each (H, W) shape after resize, before padding

