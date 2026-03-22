from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class SecondaryRestorer(Protocol):
    name: str

    @property
    def num_workers(self) -> int:
        return 1

    def restore(self, frames_256: torch.Tensor, *, keep_start: int, keep_end: int) -> list[torch.Tensor]:
        """
        Args:
            frames_256: (T, C, 256, 256) tensor, float [0, 1]
            keep_start/keep_end: indices in [0, T] selecting the frames to return
        Returns:
            List of T' tensors each (C, H, W) uint8, where T' = keep_end - keep_start
        """


@runtime_checkable
class AsyncSecondaryRestorer(Protocol):
    name: str

    @property
    def num_workers(self) -> int:
        return 1

    def push_clip(self, frames_256: torch.Tensor, *, keep_start: int, keep_end: int) -> int:
        """Push a clip for async processing. Returns a sequence number."""
        ...

    def pop_completed(self) -> list[tuple[int, list[np.ndarray]]]:
        """Return list of (seq, frames_hwc_uint8) for completed clips."""
        ...

    @property
    def has_pending(self) -> bool:
        """True if any clips are still being processed."""
        ...

    def flush_pending(self, target_seqs: set[int] | None = None) -> None:
        """Push filler frames to unstick latency-buffered output.

        If *target_seqs* is given, only flush workers that hold one of
        those clip sequences.  Otherwise flush all workers with pending clips.
        """
        ...

    def flush_all(self) -> None:
        """Close stdin, drain all remaining output, restart workers."""
        ...

    def close(self) -> None:
        ...

    @staticmethod
    def _to_tensors(frames_np: list[np.ndarray]) -> torch.Tensor:
        ...
