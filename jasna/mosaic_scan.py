"""Whole-video mosaic sampling shared by the segment editor's scan and the CLI's LTX auto split.

A scan samples one frame every ``stride`` seconds and keeps, per sample, the best
detection score and a low-res mask of every detection. The LTX auto split turns those
into ranges: LTX where a mosaic is at least a given share of the frame height,
BasicVSR++ for the rest.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from jasna.segments import SegmentRange, SegmentRestoration, normalize_segments

SCAN_MASK_HW = (90, 160)


def scan_sample_stride(fps: float, *, seconds: float = 1.0) -> int:
    """Frame stride for one detection sample roughly every ``seconds``."""

    return max(1, round(float(fps) * float(seconds)))


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


@dataclass(frozen=True)
class MosaicBox:
    """One mosaic of a scan mask: its box as fractions of the frame and its size, the
    longest side as a fraction of the frame height."""

    left: float
    top: float
    width: float
    height: float
    size: float


def mosaic_boxes(mask: np.ndarray, *, aspect: float) -> list[MosaicBox]:
    """The mosaics of one scan mask (uint8 [H, W] covering a frame ``aspect`` wide per unit of height)."""
    import cv2

    mask_h, mask_w = mask.shape
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.ascontiguousarray(mask, dtype=np.uint8))
    return [
        MosaicBox(x / mask_w, y / mask_h, w / mask_w, h / mask_h, max(w / mask_w * aspect, h / mask_h))
        for x, y, w, h, _ in stats[1:]
    ]


def mosaic_sizes(masks: np.ndarray, *, frame_width: int, frame_height: int) -> tuple[float, ...]:
    """Per scan mask (uint8 [N, H, W] covering the whole frame), the size of its biggest
    mosaic; 0.0 when it holds none."""
    aspect = frame_width / frame_height
    return tuple(max((box.size for box in mosaic_boxes(mask, aspect=aspect)), default=0.0) for mask in masks)


def ltx_auto_segments(
    times: tuple[float, ...] | list[float],
    scores: tuple[float, ...] | list[float],
    sizes: tuple[float, ...] | list[float],
    *,
    threshold: float,
    ltx_min_size: float,
    stride: float,
    duration: float,
    keyframes: tuple[float, ...],
    ltx_seed: int,
) -> tuple[SegmentRange, ...]:
    """The scan's mosaic ranges, each with its model: LTX across every keyframe interval
    that holds a mosaic at least ``ltx_min_size`` frame heights big, BasicVSR++ elsewhere.
    Models change only at ``keyframes`` (seconds, ascending), so each render span keeps one."""

    detected = segments_from_scores(times, scores, threshold=threshold, stride=stride, duration=duration)
    large = segments_from_scores(
        times,
        [score if size >= ltx_min_size else 0.0 for score, size in zip(scores, sizes, strict=True)],
        threshold=threshold,
        stride=stride,
        duration=duration,
    )
    ltx_spans = [
        (_keyframe_at_or_before(keyframes, span.start), _keyframe_at_or_after(keyframes, span.end, duration))
        for span in large
    ]
    ltx = SegmentRestoration("ltx", ltx_seed)
    basic = [replace(span, restoration=SegmentRestoration("basicvsrpp", None)) for span in detected]
    ltx_pieces = [
        SegmentRange(max(span.start, start), min(span.end, end), ltx)
        for span in detected
        for start, end in ltx_spans
        if max(span.start, start) < min(span.end, end)
    ]
    return normalize_segments(basic + ltx_pieces, duration=duration)


def _keyframe_at_or_before(keyframes: tuple[float, ...], seconds: float) -> float:
    index = bisect.bisect_right(keyframes, seconds) - 1
    return keyframes[index] if index >= 0 else 0.0


def _keyframe_at_or_after(keyframes: tuple[float, ...], seconds: float, duration: float) -> float:
    index = bisect.bisect_left(keyframes, seconds)
    return keyframes[index] if index < len(keyframes) else duration


@dataclass(frozen=True)
class MosaicSamples:
    """One scan: sample times (seconds), best detection score and biggest mosaic size per sample."""

    times: tuple[float, ...]
    scores: tuple[float, ...]
    sizes: tuple[float, ...]
    stride: float


def scan_mosaic_sizes(
    detector,
    path: Path,
    metadata,
    *,
    batch_size: int,
    device,
    disable_progress: bool,
    on_progress: Callable[[float], None],
) -> MosaicSamples:
    """Sample the video once a second with ``detector`` for the LTX auto split of a job
    (the editor's scan runs in ``gui.mosaic_scan``); ``on_progress`` gets the fraction done."""
    import torch
    from tqdm import tqdm

    from jasna.media.video_decoder import VideoReader

    frame_stride = scan_sample_stride(metadata.video_fps)
    time_base = float(metadata.time_base)
    times: list[float] = []
    scores: list[float] = []
    sizes: list[float] = []
    total = max(1, int(metadata.num_frames) // frame_stride)
    reader = VideoReader(str(path), batch_size, device, metadata, frame_stride=frame_stride)
    with reader, tqdm(total=total, desc="Mosaic scan", unit="sample", disable=disable_progress) as bar:
        for batch, pts_list in reader.frames():
            count = len(pts_list)
            if batch.shape[0] < batch_size:
                batch = torch.cat((batch, batch[-1:].expand(batch_size - batch.shape[0], -1, -1, -1)))
            batch_scores, batch_masks = detector.scan_scores_masks(batch, mask_hw=SCAN_MASK_HW)
            times.extend(max(0.0, (pts - reader.start_pts) * time_base) for pts in pts_list)
            scores.extend(batch_scores[:count].tolist())
            sizes.extend(
                mosaic_sizes(
                    batch_masks[:count].to(torch.uint8).cpu().numpy(),
                    frame_width=int(metadata.video_width),
                    frame_height=int(metadata.video_height),
                )
            )
            bar.update(count)
            on_progress(min(1.0, len(times) / total))
    return MosaicSamples(tuple(times), tuple(scores), tuple(sizes), frame_stride / float(metadata.video_fps))
