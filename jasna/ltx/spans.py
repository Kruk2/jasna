"""What an LTX run restores: segments (pts ranges with their seeds), the decoded spans holding them and
their frames, batched per segment."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Protocol

import torch

from jasna.ltx.inference import Cancelled
from jasna.session_config import DecodeBackendName

FrameBatches = Iterator[tuple[torch.Tensor, list[int]]]


FrameSource = Callable[[], FrameBatches]


WHOLE_VIDEO_PTS = (-(1 << 63), (1 << 63) - 1)


class FrameWriter(Protocol):
    """``finish`` once every frame is written; ``close`` (also after ``finish``) releases it."""

    def write(self, frame: torch.Tensor, pts: int) -> None: ...

    def finish(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class LtxSegment:
    """The frames with ``start_pts <= pts < end_pts``; window ``i`` draws noise from ``seed + i``."""

    start_pts: int
    end_pts: int
    seed: int

    def contains(self, pts: int) -> bool:
        return self.start_pts <= pts < self.end_pts


@dataclass(frozen=True)
class FrameNumbers:
    """A frame's 0-based number in the whole input from its pts, whatever stretch was decoded."""

    start_pts: int
    time_base: Fraction
    fps: float

    @classmethod
    def of_video(cls, metadata) -> FrameNumbers:
        return cls(metadata.start_pts, metadata.time_base, metadata.mean_fps)

    def frame_number(self, pts: int) -> int:
        return round((int(pts) - self.start_pts) * self.time_base * self.fps)


@dataclass(frozen=True)
class LtxSpan:
    """``frames`` decodes a stretch of video holding ``segments``; every frame goes to the
    writer ``open_writer`` makes when the span is composed, a frame outside the segments
    unchanged and without the LUT."""

    frames: FrameSource
    segments: tuple[LtxSegment, ...]
    open_writer: Callable[[], FrameWriter]


def video_frames(
    path: Path,
    metadata,
    *,
    batch_size: int,
    device: torch.device,
    frame_stride: int,
    decode_backend: DecodeBackendName,
    seek_ts: float | None,
    end_pts: int | None,
) -> FrameSource:
    """Decoded batches of ``path`` from the keyframe at ``seek_ts`` (the start when None),
    stopping before ``end_pts`` (the end when None)."""
    from jasna.media.video_decoder import VideoReader

    def frames() -> FrameBatches:
        with VideoReader(
            str(path), batch_size, device, metadata, frame_stride=frame_stride, decode_backend=decode_backend
        ) as reader:
            for batch, pts in reader.frames(seek_ts):
                if end_pts is None:
                    yield batch, pts
                    continue
                keep = next((i for i, p in enumerate(pts) if int(p) >= end_pts), len(pts))
                if keep:
                    yield batch[:keep], pts[:keep]
                if keep < len(pts):
                    return

    return frames


def check_cancel(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise Cancelled()


def segment_batches(
    batches: FrameBatches, segments: Sequence[LtxSegment], batch_size: int
) -> Iterator[tuple[int | None, torch.Tensor, list[int]]]:
    """``(segment index, frames, pts)`` in decode order. A segment's frames come in batches
    of ``batch_size`` counted from its first frame (the last may be short), so detection
    sees the same batches wherever decoding started; frames outside every segment come as
    decoded with index None."""
    owner: int | None = None
    pending: list[tuple[torch.Tensor, list[int]]] = []

    def full_batches(flush: bool) -> Iterator[tuple[int | None, torch.Tensor, list[int]]]:
        count = sum(len(pts) for _, pts in pending)
        if count == 0 or (count < batch_size and not flush):
            return
        if len(pending) == 1:
            frames, pts = pending[0]
        else:
            frames = torch.cat([chunk for chunk, _ in pending])
            pts = [p for _, chunk_pts in pending for p in chunk_pts]
        pending.clear()
        whole = len(pts) if flush else len(pts) - len(pts) % batch_size
        for start in range(0, whole, batch_size):
            yield owner, frames[start : start + batch_size], pts[start : start + batch_size]
        if whole < len(pts):
            pending.append((frames[whole:], pts[whole:]))

    for frames, pts in batches:
        owners = [next((i for i, s in enumerate(segments) if s.contains(int(p))), None) for p in pts]
        start = 0
        while start < len(pts):
            stop = start + 1
            while stop < len(pts) and owners[stop] == owners[start]:
                stop += 1
            if owners[start] != owner:
                yield from full_batches(flush=True)
                owner = owners[start]
            if owner is None:
                yield None, frames[start:stop], list(pts[start:stop])
            else:
                pending.append((frames[start:stop], list(pts[start:stop])))
                yield from full_batches(flush=False)
            start = stop
    yield from full_batches(flush=True)
