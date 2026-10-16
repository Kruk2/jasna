"""Find the parts of a video that need restoring, before the main pass.

A full pass runs the detector on every frame, whether or not the frame contains a
mosaic. When mosaics cover only part of the video that work is wasted: a strided
scan (one detector sample per second by default) locates the mosaicked regions far
more cheaply, and the pipeline then renders those spans only while stream-copying
everything else. A scan that finds nothing turns the whole job into a copy.

The scan is deliberately coarse — it is a proposal generator, not a detector pass
— so every hit is padded by half the sampling stride and neighbouring hits are
merged (`jasna.mosaic.scan.segments_from_scores`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from jasna.mosaic.scan import (
    SCAN_MASK_HW,
    covered_seconds,
    scan_sample_stride,
    segments_from_scores,
)
from jasna.segments import SegmentRange, format_segments

logger = logging.getLogger(__name__)

AUTO_SEGMENT_STRIDE_SECONDS = 1.0
# Above this coverage the scan has found mosaics (almost) everywhere, so segmenting
# would add a scan pass without saving anything: a normal full pass is cheaper.
AUTO_SEGMENT_COVERAGE_LIMIT = 0.98


@dataclass(frozen=True)
class VideoScoreScan:
    """Detector scores for one sample every ``stride`` seconds of the video."""

    times: tuple[float, ...]
    scores: tuple[float, ...]
    stride: float
    duration: float

    @property
    def samples(self) -> int:
        return len(self.times)


@dataclass(frozen=True)
class AutoSegmentPlan:
    segments: tuple[SegmentRange, ...]
    scan: VideoScoreScan
    threshold: float
    coverage_limit: float = AUTO_SEGMENT_COVERAGE_LIMIT

    @property
    def coverage(self) -> float:
        """Fraction of the video the detected segments would render."""

        return min(1.0, covered_seconds(self.segments) / max(self.scan.duration, 1e-6))

    @property
    def worth_segmenting(self) -> bool:
        """False when rendering the segments would cost as much as a full pass."""

        return self.coverage < self.coverage_limit

    def describe(self) -> str:
        if not self.segments:
            return (
                f"scan: {self.scan.samples} samples, no mosaic detected above "
                f"{self.threshold:g} — the video will be copied without re-encoding"
            )
        return (
            f"scan: {self.scan.samples} samples, {len(self.segments)} region(s) covering "
            f"{self.coverage * 100:.1f}% of the video ({format_segments(self.segments[:6])}"
            f"{'…' if len(self.segments) > 6 else ''})"
        )


def scan_video_scores(
    input_path: str | Path,
    metadata,
    detector,
    *,
    device,
    batch_size: int = 4,
    stride_seconds: float = AUTO_SEGMENT_STRIDE_SECONDS,
    on_progress: Callable[[float, float], None] | None = None,
) -> VideoScoreScan:
    """Best per-sample detection score for every ``stride_seconds`` of the video.

    ``detector`` must provide ``scan_scores_masks`` (YOLO, RF-DETR and the VR SBS
    adapter all do). Frames are decoded with the hardware decoder and only the
    sampled ones reach the detector.
    """

    from jasna.media.video_decoder import VideoReader

    fps = float(metadata.video_fps)
    frame_stride = scan_sample_stride(fps, seconds=stride_seconds)
    stride = frame_stride / fps if fps > 0 else float(stride_seconds)
    duration = float(metadata.duration)
    time_base = float(metadata.time_base)
    reader_batch = max(1, int(batch_size))

    times: list[float] = []
    scores: list[float] = []
    with VideoReader(
        str(input_path), reader_batch, device, metadata, frame_stride=frame_stride
    ) as reader:
        start_pts = reader.start_pts
        for batch, pts_list in reader.frames():
            sample_times = [max(0.0, (pts - start_pts) * time_base) for pts in pts_list]
            batch_scores = detector.scan_scores_masks(batch, mask_hw=SCAN_MASK_HW)[0]
            for seconds, score in zip(
                sample_times, batch_scores.detach().float().cpu().tolist()
            ):
                times.append(float(seconds))
                scores.append(float(score))
            if on_progress is not None and times:
                on_progress(times[-1], duration)
    if len(times) != len(scores):
        raise RuntimeError(
            f"scan produced {len(times)} sample times but {len(scores)} scores"
        )
    return VideoScoreScan(tuple(times), tuple(scores), stride, duration)


def plan_auto_segments(
    scan: VideoScoreScan,
    *,
    threshold: float,
    coverage_limit: float = AUTO_SEGMENT_COVERAGE_LIMIT,
) -> AutoSegmentPlan:
    """Turn scan scores into padded time ranges, merged and sorted."""

    segments = segments_from_scores(
        scan.times,
        scan.scores,
        threshold=float(threshold),
        stride=scan.stride,
        duration=scan.duration,
    )
    return AutoSegmentPlan(
        segments=segments,
        scan=scan,
        threshold=float(threshold),
        coverage_limit=float(coverage_limit),
    )
