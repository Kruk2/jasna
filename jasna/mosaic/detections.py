from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Detections:
    scores: torch.Tensor  # (B, K) on CPU
    boxes_xyxy: torch.Tensor  # (B, K, 4) in pixels, original frame space, on CPU
    masks: torch.Tensor  # (B, K, H, W) bool, original frame space, on GPU

