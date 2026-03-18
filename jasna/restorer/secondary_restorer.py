from __future__ import annotations

from typing import Protocol, runtime_checkable

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

    def push_clip(self, frames_256: torch.Tensor, keep_start: int, keep_end: int) -> int: ...
    def pop_completed(self) -> list[tuple[int, list[torch.Tensor]]]: ...
    def flush_all(self) -> None: ...
    def close(self) -> None: ...


class SecondaryRestorerAdapter:
    """Wraps a sync SecondaryRestorer to provide the AsyncSecondaryRestorer interface."""

    def __init__(self, restorer: SecondaryRestorer) -> None:
        self._restorer = restorer
        self._next_seq = 0
        self._completed: list[tuple[int, list[torch.Tensor]]] = []

    @property
    def name(self) -> str:
        return self._restorer.name

    @property
    def num_workers(self) -> int:
        return self._restorer.num_workers

    def push_clip(self, frames_256: torch.Tensor, keep_start: int, keep_end: int) -> int:
        seq = self._next_seq
        self._next_seq += 1
        result = self._restorer.restore(frames_256, keep_start=keep_start, keep_end=keep_end)
        self._completed.append((seq, result))
        return seq

    def pop_completed(self) -> list[tuple[int, list[torch.Tensor]]]:
        out = self._completed
        self._completed = []
        return out

    def flush_all(self) -> None:
        pass

    def close(self) -> None:
        if hasattr(self._restorer, "close"):
            self._restorer.close()

