from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Detections:
    boxes_xyxy: list[np.ndarray]  # len=B, each (N_i, 4) xyxy in pixels, CPU
    masks: list[torch.Tensor]  # len=B, each (N_i, Hm, Wm) bool, GPU

