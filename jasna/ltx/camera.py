"""The LTX restoration crop camera: a smoothed follow crop with a slowly zooming side.

A track gets one camera; its centre follows the Lada crop, Gaussian-smoothed, and is
clamped so the mosaic support always stays inside. The crop side follows the raw Lada
side through a rate-limited envelope (at most ``ZOOM_PER_WINDOW`` per window).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

Box = tuple[float, float, float, float]

PLAN_CANVAS = 512
SMOOTH_SIGMA_FRAMES = 12.0
WINDOW_FRAMES = 121
ZOOM_PER_WINDOW = 1.25
ZOOM_LOG_STEP = math.log(ZOOM_PER_WINDOW) / (WINDOW_FRAMES - 1)
LADA_BORDER_RATIO = 0.06
LADA_MIN_BORDER = 20


@dataclass(frozen=True)
class CropFrame:
    """One source-frame crop letterboxed into a square model canvas."""

    crop_box: tuple[int, int, int, int]  # xyxy half-open, source px
    resize_hw: tuple[int, int]
    pad: tuple[int, int, int, int]  # top, bottom, left, right

    @property
    def canvas(self) -> int:
        return self.resize_hw[0] + self.pad[0] + self.pad[1]

    @property
    def side(self) -> int:
        x1, y1, x2, y2 = self.crop_box
        return max(x2 - x1, y2 - y1)


def letterbox(crop_box: tuple[int, int, int, int], canvas: int) -> CropFrame:
    x1, y1, x2, y2 = crop_box
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError(f"crop must be a positive rectangle: {crop_box}")
    scale = min(canvas / width, canvas / height)
    resize_w = min(canvas, max(1, int(round(width * scale))))
    resize_h = min(canvas, max(1, int(round(height * scale))))
    pad_left = (canvas - resize_w) // 2
    pad_top = (canvas - resize_h) // 2
    return CropFrame(
        crop_box=crop_box,
        resize_hw=(resize_h, resize_w),
        pad=(pad_top, canvas - resize_h - pad_top, pad_left, canvas - resize_w - pad_left),
    )


def _lada_crop_box(box: Box, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(int(math.floor(box[0])), frame_w))
    y1 = max(0, min(int(math.floor(box[1])), frame_h))
    x2 = max(0, min(int(math.ceil(box[2])), frame_w))
    y2 = max(0, min(int(math.ceil(box[3])), frame_h))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"degenerate mosaic box {box}")

    border = max(LADA_MIN_BORDER, int(max(x2 - x1, y2 - y1) * LADA_BORDER_RATIO))
    x1, y1 = max(0, x1 - border), max(0, y1 - border)
    x2, y2 = min(frame_w, x2 + border), min(frame_h, y2 + border)

    width, height = x2 - x1, y2 - y1
    down_scale = min(PLAN_CANVAS / width, PLAN_CANVAS / height, 1.0)
    missing_w = int((PLAN_CANVAS - width * down_scale) / down_scale)
    missing_h = int((PLAN_CANVAS - height * down_scale) / down_scale)

    def expand(available_low: int, available_high: int, missing: int, budget: int) -> tuple[int, int, int]:
        both = min(available_low, available_high, missing // 2, budget)
        low = min(available_low - both, missing - 2 * both, budget - both)
        high = min(available_high - both, missing - 2 * both - low, budget - both - low)
        return both, low, high

    lr, left, right = expand(x1, frame_w - x2, missing_w, width)
    tb, top, bottom = expand(y1, frame_h - y2, missing_h, height)
    return (
        x1 - (math.floor(lr / 2) + left),
        y1 - (math.floor(tb / 2) + top),
        x2 + math.ceil(lr / 2) + right,
        y2 + math.ceil(tb / 2) + bottom,
    )


def lada_crop_boxes(boxes: Sequence[Box], frame_w: int, frame_h: int) -> np.ndarray:
    return np.asarray([_lada_crop_box(box, frame_w, frame_h) for box in boxes], dtype=np.int64)


def gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    """Zero-phase Gaussian smoothing with edge replication."""
    if sigma <= 0 or len(values) < 2:
        return values.astype(np.float64, copy=True)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(offsets / sigma))
    kernel /= kernel.sum()
    padded = np.pad(values.astype(np.float64), (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def zoom_crop_sides(raw_sides: np.ndarray, max_log_step: float = ZOOM_LOG_STEP) -> np.ndarray:
    """Smooth per-frame sides, never below any frame's raw side, zooming at most
    ``exp(max_log_step)`` per frame (log-space running max, rate limit, Gaussian)."""
    values = np.log(np.asarray(raw_sides, dtype=np.float64))
    radius = max(1, int(math.ceil(3.0 * SMOOTH_SIGMA_FRAMES)))
    padded = np.pad(values, (radius, radius), mode="edge")
    envelope = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1).max(axis=1)
    for index in range(1, len(envelope)):
        envelope[index] = max(envelope[index], envelope[index - 1] - max_log_step)
    for index in range(len(envelope) - 2, -1, -1):
        envelope[index] = max(envelope[index], envelope[index + 1] - max_log_step)
    return np.ceil(np.exp(gaussian_smooth(envelope, SMOOTH_SIGMA_FRAMES)) - 1e-6).astype(np.int64)


def _feasible_interval(lower: float, upper: float, side: int, extent: int, margin: float) -> tuple[int, int]:
    minimum = max(0, min(extent, int(math.ceil(upper + margin))) - side)
    maximum = min(extent - side, max(0, int(math.floor(lower - margin))))
    if minimum <= maximum:
        return minimum, maximum
    minimum = max(0, int(math.ceil(upper)) - side)
    maximum = min(extent - side, int(math.floor(lower)))
    if minimum > maximum:
        raise ValueError("mosaic support has no valid crop placement")
    return minimum, maximum


def follow_camera(
    boxes: Sequence[Box], sides: np.ndarray, *, frame_w: int, frame_h: int, margin: int
) -> list[CropFrame]:
    """Place a crop of ``sides[i]`` on every frame, centred on the smoothed Lada crop and
    clamped so the mosaic box (plus ``margin`` where it fits) stays inside."""
    raw = lada_crop_boxes(boxes, frame_w, frame_h)
    support = np.asarray(boxes, dtype=np.float64)
    widths = np.minimum(sides, frame_w).astype(int)
    heights = np.minimum(sides, frame_h).astype(int)
    if ((support[:, 2] - support[:, 0]) > widths).any() or ((support[:, 3] - support[:, 1]) > heights).any():
        raise ValueError("mosaic support does not fit inside the crop")
    target_x = gaussian_smooth((raw[:, 0] + raw[:, 2]) / 2.0, SMOOTH_SIGMA_FRAMES)
    target_y = gaussian_smooth((raw[:, 1] + raw[:, 3]) / 2.0, SMOOTH_SIGMA_FRAMES)
    frames = []
    for index, (x1, y1, x2, y2) in enumerate(support):
        width, height = int(widths[index]), int(heights[index])
        min_left, max_left = _feasible_interval(x1, x2, width, frame_w, margin)
        min_top, max_top = _feasible_interval(y1, y2, height, frame_h, margin)
        left = min(max(int(round(target_x[index] - width / 2)), min_left), max_left)
        top = min(max(int(round(target_y[index] - height / 2)), min_top), max_top)
        frames.append(letterbox((left, top, left + width, top + height), PLAN_CANVAS))
    return frames


def _edge_gap(crop: tuple[int, int, int, int], box: Box, frame_w: int, frame_h: int) -> float:
    """Distance from the mosaic box to the nearest crop edge that is not the frame border."""
    x1, y1, x2, y2 = crop
    return float(
        min(
            box[0] - x1 if x1 > 0 else np.inf,
            box[1] - y1 if y1 > 0 else np.inf,
            x2 - box[2] if x2 < frame_w else np.inf,
            y2 - box[3] if y2 < frame_h else np.inf,
        )
    )


def track_camera(boxes: Sequence[Box], *, frame_w: int, frame_h: int, feather: int) -> list[CropFrame]:
    """The zooming track camera, widened by a feather per side (with a feather margin)
    when the tight one brings the mask within a feather of an inner crop edge."""
    raw = lada_crop_boxes(boxes, frame_w, frame_h)
    sides = zoom_crop_sides(np.maximum(raw[:, 2] - raw[:, 0], raw[:, 3] - raw[:, 1]))
    tight = follow_camera(boxes, sides, frame_w=frame_w, frame_h=frame_h, margin=0)
    if all(_edge_gap(crop.crop_box, box, frame_w, frame_h) >= feather for crop, box in zip(tight, boxes)):
        return tight
    return follow_camera(boxes, sides + 2 * feather, frame_w=frame_w, frame_h=frame_h, margin=feather)
