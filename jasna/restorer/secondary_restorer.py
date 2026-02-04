from __future__ import annotations

from typing import Protocol

import torch


class SecondaryRestorer(Protocol):
    name: str

    def restore(self, frames_256: torch.Tensor, *, keep_start: int, keep_end: int) -> torch.Tensor | list[torch.Tensor]:
        """
        Args:
            frames_256: (T, C, 256, 256) tensor (primary restorer output; typically float in [0, 1])
            keep_start/keep_end: indices in [0, T] that will be kept for blending/encoding
        Returns:
            Either (T, C, H, W) uint8 tensor or list of T tensors each (C, H, W) uint8.
            (H, W) can be any resolution but should be consistent for the clip.
        """
