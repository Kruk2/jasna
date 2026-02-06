from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
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


@runtime_checkable
class StreamingSecondaryCompleted(Protocol):
    meta: object

    def to_frame_u8(self, device: torch.device) -> torch.Tensor:
        """Return (C, H, W) uint8 tensor on `device`."""


@runtime_checkable
class StreamingSecondaryRestorer(Protocol):
    name: str

    def submit(self, frames_256: torch.Tensor, *, keep_start: int, keep_end: int, meta: list[object], track_id: int = 0) -> None:
        """Submit work (may complete immediately or later)."""

    def drain_completed(self, *, limit: int | None = None) -> list[StreamingSecondaryCompleted]:
        """Drain completed items (any order)."""

    def flush(self, *, timeout_s: float = 300.0) -> None:
        """Finish all pending work and make remaining outputs drainable."""

    def flush_track(self, track_id: int) -> None:
        """Flush a specific track by pushing padding frames to force outputs."""

    def transfer_track(self, old_track_id: int, new_track_id: int) -> None:
        """Transfer worker mapping from old track to continuation without flushing."""
