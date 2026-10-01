#!/usr/bin/env python3
"""Measure the decoded-frame backlog needed by a hypothetical shared reader.

This probe does not replace either product reader.  It observes when the
DecodeDetect reader yields each PTS and when BlendEncode asks its independent
reader for the same PTS.  The difference is a conservative upper bound for a
reference-counted shared decoded-frame ring under the current scheduler.
"""

from __future__ import annotations

import argparse
import atexit
from collections import deque
import json
from pathlib import Path
import statistics
import sys
import threading
import time
from typing import Callable


class CapacityAudit:
    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._pending: dict[int, tuple[float, int]] = {}
        self._order: deque[int] = deque()
        self.produced = 0
        self.consumed = 0
        self.current_bytes = 0
        self.max_frames = 0
        self.max_bytes = 0
        self.frame_bytes: int | None = None
        self.residency_seconds: list[float] = []

    def produce(self, pts_values: list[int], frame_bytes: int) -> None:
        now = self._clock()
        frame_bytes = int(frame_bytes)
        if frame_bytes <= 0:
            raise ValueError("frame_bytes must be positive")
        with self._lock:
            if self.frame_bytes not in (None, frame_bytes):
                raise RuntimeError(
                    f"decoded frame size changed: {self.frame_bytes} -> {frame_bytes}"
                )
            self.frame_bytes = frame_bytes
            for raw_pts in pts_values:
                pts = int(raw_pts)
                if pts in self._pending:
                    raise RuntimeError(f"duplicate produced PTS {pts}")
                self._pending[pts] = (now, frame_bytes)
                self._order.append(pts)
                self.produced += 1
                self.current_bytes += frame_bytes
            self.max_frames = max(self.max_frames, len(self._pending))
            self.max_bytes = max(self.max_bytes, self.current_bytes)

    def consume(self, pts: int) -> None:
        now = self._clock()
        pts = int(pts)
        with self._lock:
            if not self._order or self._order[0] != pts:
                first = self._order[0] if self._order else None
                raise RuntimeError(
                    f"shared-reader order mismatch: consume={pts}, oldest={first}"
                )
            self._order.popleft()
            produced_at, size = self._pending.pop(pts)
            self.consumed += 1
            self.current_bytes -= size
            self.residency_seconds.append(now - produced_at)

    def report(self) -> dict[str, object]:
        with self._lock:
            values = sorted(self.residency_seconds)

            def percentile(fraction: float) -> float | None:
                if not values:
                    return None
                index = round((len(values) - 1) * fraction)
                return float(values[index])

            return {
                "produced_frames": self.produced,
                "consumed_frames": self.consumed,
                "pending_frames": len(self._pending),
                "oldest_pending_pts": self._order[0] if self._order else None,
                "frame_bytes": self.frame_bytes,
                "max_outstanding_frames": self.max_frames,
                "max_outstanding_rgb_bytes": self.max_bytes,
                "residency_seconds": {
                    "median": statistics.median(values) if values else None,
                    "p95": percentile(0.95),
                    "max": max(values) if values else None,
                },
            }


def value_after(arguments: list[str], option: str) -> str:
    matches = [index for index, value in enumerate(arguments) if value == option]
    if len(matches) != 1 or matches[0] + 1 >= len(arguments):
        raise ValueError(f"expected exactly one {option} VALUE pair")
    return arguments[matches[0] + 1]


def install_audit(audit: CapacityAudit, destination: Path) -> None:
    from jasna.media.video_decoder import NvidiaVideoReader
    from jasna import pipeline_threads

    original_frames = NvidiaVideoReader.frames
    original_read_exact = pipeline_threads._PtsAlignedFrameReader.read_exact

    def frames(self, *args, **kwargs):
        iterator = original_frames(self, *args, **kwargs)
        for batch, pts_values in iterator:
            if threading.current_thread().name == "DecodeDetect":
                per_frame_bytes = int(batch[0].numel() * batch[0].element_size())
                audit.produce([int(value) for value in pts_values], per_frame_bytes)
            yield batch, pts_values

    def read_exact(self, expected_pts: int):
        frame = original_read_exact(self, expected_pts)
        audit.consume(int(expected_pts))
        return frame

    def save() -> None:
        destination.write_text(
            json.dumps(audit.report(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    NvidiaVideoReader.frames = frames
    pipeline_threads._PtsAlignedFrameReader.read_exact = read_exact
    atexit.register(save)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("jasna_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.jasna_args[:1] == ["--"]:
        args.jasna_args.pop(0)
    if not args.jasna_args:
        parser.error("missing Jasna CLI arguments after --")
    return args


def main() -> None:
    args = parse_args()
    output = Path(value_after(args.jasna_args, "--output")).resolve()
    audit = CapacityAudit()
    install_audit(audit, output.parent / "single-decode-capacity.json")
    sys.argv = ["jasna", *args.jasna_args]
    from jasna.main import main as jasna_main

    started = time.monotonic()
    try:
        jasna_main()
    finally:
        print(
            "SINGLE_DECODE_CAPACITY_PROBE "
            f"wall_seconds={time.monotonic() - started:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
