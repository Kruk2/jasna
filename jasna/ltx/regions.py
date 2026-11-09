"""Mosaic detections as LTX planning regions: the model box plus the outline of its mask.

Each low-res mask is bilinearly upsampled to the frame and thresholded at 0.5, dilated
by an ellipse of ``EXPAND_PIXELS``, and its largest external contour becomes the region
polygon. The box is the model box truncated to whole pixels. A batch's masks are traced
together: one host sync per ``UPSAMPLE_BYTES`` of upsampled masks and one per
``DILATE_PIXELS`` of cropped ones, instead of several per detection (each sync and small
kernel launch costs far more on Windows).
"""

from __future__ import annotations

import itertools

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from jasna.ltx.plan import Region
from jasna.mosaic.detections import Detections
from jasna.tracking.blending import dilate_ellipse

EXPAND_PIXELS = 20
UPSAMPLE_BYTES = 256 << 20
"""Most float frame-sized masks upsampled in one call (32 at 1080p, 2 at 8K)."""
DILATE_PIXELS = 1 << 22
"""Most crop pixels, each crop padded to the largest, dilated in one call."""

CropBox = tuple[int, int, int, int]


def _crop_box(first_row: int, last_row: int, first_col: int, last_col: int, frame_h: int, frame_w: int) -> CropBox:
    """(y0, y1, x0, x1) around a mask's set pixels, wide enough to hold their dilation."""
    return (
        max(0, first_row - EXPAND_PIXELS - 1),
        min(frame_h, last_row + EXPAND_PIXELS + 2),
        max(0, first_col - EXPAND_PIXELS - 1),
        min(frame_w, last_col + EXPAND_PIXELS + 2),
    )


def _first_and_last(lines: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Index of the first and of the last True in each bool row."""
    codes = lines.to(torch.uint8)
    return codes.argmax(1), lines.shape[1] - 1 - codes.flip(1).argmax(1)


def _dilation_groups(crops: dict[int, CropBox]) -> list[list[int]]:
    """Detections whose crops, padded to the group's largest, stay within ``DILATE_PIXELS``."""
    groups: list[list[int]] = [[]]
    height = width = 0
    for index, (y0, y1, x0, x1) in crops.items():
        grown_h, grown_w = max(height, y1 - y0), max(width, x1 - x0)
        if groups[-1] and (len(groups[-1]) + 1) * grown_h * grown_w > DILATE_PIXELS:
            groups.append([])
            grown_h, grown_w = y1 - y0, x1 - x0
        groups[-1].append(index)
        height, width = grown_h, grown_w
    return groups


def _region(box: np.ndarray, dilated: np.ndarray, x0: int, y0: int) -> Region | None:
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 3:
        return None
    x_min, y_min, x_max, y_max = (float(int(v)) for v in box.tolist())
    return Region(
        box=(x_min, y_min, x_max, y_max),
        polygon=largest[:, 0, :] + np.array((x0, y0), dtype=np.int32),
    )


def _trace(boxes: list[np.ndarray], masks: torch.Tensor, frame_h: int, frame_w: int) -> list[Region | None]:
    full = F.interpolate(masks[:, None].float(), size=(frame_h, frame_w), mode="bilinear", align_corners=False)[:, 0] > 0.5
    rows, cols = full.any(dim=2), full.any(dim=1)
    extents = torch.stack((rows.any(dim=1).long(), *_first_and_last(rows), *_first_and_last(cols)), dim=1).tolist()
    crops = {i: _crop_box(*extent[1:], frame_h, frame_w) for i, extent in enumerate(extents) if extent[0]}
    regions: list[Region | None] = [None] * len(boxes)
    for group in _dilation_groups(crops) if crops else []:
        sizes = [(crops[i][1] - crops[i][0], crops[i][3] - crops[i][2]) for i in group]
        padded = torch.zeros(
            (len(group), max(h for h, _ in sizes), max(w for _, w in sizes)), dtype=torch.bool, device=full.device
        )
        for slot, i in enumerate(group):
            y0, y1, x0, x1 = crops[i]
            padded[slot, : y1 - y0, : x1 - x0] = full[i, y0:y1, x0:x1]
        dilated = dilate_ellipse(padded, EXPAND_PIXELS).to(torch.uint8).mul_(255).cpu().numpy()
        for slot, (i, (height, width)) in enumerate(zip(group, sizes)):
            y0, _, x0, _ = crops[i]
            regions[i] = _region(boxes[i], np.ascontiguousarray(dilated[slot, :height, :width]), x0, y0)
    return regions


def regions_per_frame(detections: Detections, frame_h: int, frame_w: int) -> list[list[Region]]:
    boxes = [box for frame_boxes in detections.boxes_xyxy for box in frame_boxes]
    if not boxes:
        return [[] for _ in detections.boxes_xyxy]
    masks = torch.cat(list(detections.masks))
    per_call = max(1, UPSAMPLE_BYTES // (4 * frame_h * frame_w))
    traced = iter([
        region
        for start in range(0, len(boxes), per_call)
        for region in _trace(boxes[start : start + per_call], masks[start : start + per_call], frame_h, frame_w)
    ])
    return [
        [region for region in itertools.islice(traced, len(frame_boxes)) if region is not None]
        for frame_boxes in detections.boxes_xyxy
    ]
