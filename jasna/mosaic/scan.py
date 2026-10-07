"""Vendor-neutral helpers for scanning a whole video with the detection model.

The scan samples the video at a fixed frame stride, runs the configured detector
on every sampled frame and keeps the best score per sample. Turning those scores
into time ranges lets the pipeline process only the parts that actually contain
mosaics (see :mod:`jasna.mosaic.auto_segments`); the GUI segment editor uses the
same helpers.

Nothing here touches the GUI or the accelerator, so both the GUI worker and the
command line can share it.
"""

from __future__ import annotations

import bisect
import os
from dataclasses import dataclass

from jasna.segments import SegmentRange, normalize_segments

SCAN_SCORE_FLOOR = 0.05
SCAN_MASK_HW = (90, 160)
SCAN_VRAM_RESERVE_BYTES = 750 * 1024**2
SCAN_SPILL_CHUNK_BYTES = 64 * 1024**2


@dataclass(frozen=True)
class MosaicScanResult:
    """Per-sample detection scores and low-res masks, on CPU after the scan.

    Sample ``i`` was taken at ``times[i]`` seconds. ``scores`` holds the best
    detection score per sample (0.0 when nothing was detected), ``masks`` a
    uint8 [N, H, W] tensor of merged detection masks downscaled to
    ``mask_size``. ``completed_until`` is the last scanned timestamp; earlier
    than ``duration`` when the scan was stopped.
    """

    times: tuple[float, ...]
    scores: tuple[float, ...]
    masks: object
    stride: float
    duration: float
    completed_until: float

    def sample_at(self, seconds: float, *, tolerance: float):
        if not self.times:
            return None
        position = bisect.bisect_left(self.times, float(seconds))
        candidates = {
            max(0, position - 1),
            min(len(self.times) - 1, position),
        }
        index = min(candidates, key=lambda candidate: abs(self.times[candidate] - seconds))
        if abs(self.times[index] - seconds) > float(tolerance):
            return None
        return self.times[index], self.scores[index], self.masks[index]


def scan_sample_stride(fps: float, *, seconds: float = 1.0) -> int:
    """Frame stride for one detection sample roughly every ``seconds``."""

    return max(1, round(float(fps) * float(seconds)))


SCAN_PARALLEL_DECODERS = 2
SCAN_PARALLEL_MIN_PIXELS = 3840 * 2160
SCAN_PARALLEL_MIN_DURATION = 10.0
SCAN_DECODERS_ENV = "JASNA_SCAN_DECODERS"


def scan_decoder_count(
    video_width: int,
    video_height: int,
    duration: float,
    *,
    override: int | None = None,
) -> int:
    """Parallel decoders for a scan.

    A scan decodes every frame regardless of the sampling stride, so 4K+ material
    is decode-bound and is split across decoders when the GPU has more than one
    decode engine. Smaller resolutions are detection- or loop-overhead-bound and
    stay on a single decoder.

    AMD used to be pinned to one decoder here on the assumption that duplicated
    decode sessions were unsafe. That was never observed: each decoder is its own
    container/session, and a scan of 4K material on RDNA3 runs correctly with two
    of them, so both vendors now follow the same rule. ``JASNA_SCAN_DECODERS``
    (or ``override``) forces a specific count for anyone who does hit a driver
    limit, which is also what the tests use.

    Note that more decoders only pay off where decoding is the bottleneck: a 45 s
    4K clip scanned on an RX 7900 XT took 12.7 s with one decoder and 13.2 s with
    two, so the win there is correctness/consistency, not speed.
    """

    if override is not None:
        return max(1, int(override))
    from_env = os.environ.get(SCAN_DECODERS_ENV, "").strip()
    if from_env:
        try:
            return max(1, int(from_env))
        except ValueError:
            pass
    if duration < SCAN_PARALLEL_MIN_DURATION:
        return 1
    if video_width * video_height < SCAN_PARALLEL_MIN_PIXELS:
        return 1
    return SCAN_PARALLEL_DECODERS


def segment_sample_indices(
    times: list[float], start: float, end: float, *, is_last: bool
) -> list[int]:
    """Indices of samples a segment owns: ``start <= t < end`` (last segment
    keeps everything from ``start``)."""

    return [i for i, t in enumerate(times) if t >= start and (is_last or t < end)]


def segments_from_scores(
    times: tuple[float, ...] | list[float],
    scores: tuple[float, ...] | list[float],
    *,
    threshold: float,
    stride: float,
    duration: float,
    pad: float | None = None,
) -> tuple[SegmentRange, ...]:
    """Merge above-threshold samples into padded, normalized time ranges."""

    if len(times) != len(scores):
        raise ValueError("times and scores must have the same length")
    stride = float(stride)
    if stride <= 0:
        raise ValueError("stride must be greater than zero")
    if pad is None:
        pad = stride / 2
    hits = []
    for seconds, score in zip(times, scores):
        if score < threshold:
            continue
        start = max(0.0, float(seconds) - pad)
        end = min(float(duration), float(seconds) + stride + pad)
        if end > start:
            hits.append(SegmentRange(start, end))
    return normalize_segments(hits, duration=duration)


def covered_seconds(segments: tuple[SegmentRange, ...]) -> float:
    """Total duration covered by ``segments`` (they are normalized/disjoint)."""

    return float(sum(segment.duration for segment in segments))
