from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class SecondaryRestorer(Protocol):
    name: str
    supports_temporal_overlap: bool

    def restore(self, frames_256: torch.Tensor, *, keep_start: int, keep_end: int) -> torch.Tensor:
        """
        Args:
            frames_256: (T, C, 256, 256) uint8 tensor (primary restorer output)
            keep_start/keep_end: indices in [0, T] that will be kept for blending/encoding
        Returns:
            (T, C, H, W) uint8 tensor. (H, W) can be any resolution but should be consistent for the clip.
        """


@dataclass(frozen=True)
class Swin2srSecondaryRestorer:
    name: str = "swin2sr"
    supports_temporal_overlap: bool = True

    def restore(self, frames_256: torch.Tensor, *, keep_start: int, keep_end: int) -> torch.Tensor:
        return frames_256

