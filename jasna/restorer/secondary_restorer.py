from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class SecondaryRestorer(Protocol):
    name: str

    def restore(self, frames_256: torch.Tensor, *, keep_start: int, keep_end: int) -> list[torch.Tensor]:
        """
        Args:
            frames_256: (T, C, 256, 256) tensor, float [0, 1]
            keep_start/keep_end: indices in [0, T] selecting the frames to return
        Returns:
            List of T' tensors each (C, H, W) uint8, where T' = keep_end - keep_start
        """
