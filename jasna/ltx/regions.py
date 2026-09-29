"""Mosaic detections as LTX planning regions: the model box plus the outline of its mask.

Each low-res mask is bilinearly upsampled to the frame and thresholded at 0.5, dilated
by an ellipse of ``EXPAND_PIXELS``, and its largest external contour becomes the region
polygon. The box is the model box truncated to whole pixels.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from jasna.ltx.plan import Region
from jasna.mosaic.detections import Detections
from jasna.tracking.blending import dilate_ellipse

EXPAND_PIXELS = 20


def trace_region(box: np.ndarray, mask: torch.Tensor, frame_h: int, frame_w: int) -> Region | None:
    full = F.interpolate(mask[None, None].float(), size=(frame_h, frame_w), mode="bilinear", align_corners=False)[0, 0] > 0.5
    rows = torch.nonzero(full.any(dim=1)).flatten()
    if not len(rows):
        return None
    cols = torch.nonzero(full.any(dim=0)).flatten()
    y0 = max(0, int(rows[0]) - EXPAND_PIXELS - 1)
    y1 = min(frame_h, int(rows[-1]) + EXPAND_PIXELS + 2)
    x0 = max(0, int(cols[0]) - EXPAND_PIXELS - 1)
    x1 = min(frame_w, int(cols[-1]) + EXPAND_PIXELS + 2)
    dilated = dilate_ellipse(full[y0:y1, x0:x1], EXPAND_PIXELS).to(torch.uint8).mul_(255).cpu().numpy()
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 3:
        return None
    x_min, y_min, x_max, y_max = (float(int(v)) for v in box.tolist())
    return Region(
        box=(x_min, y_min, x_max, y_max),
        polygon=[[float(x + x0), float(y + y0)] for x, y in largest[:, 0, :]],
    )


def regions_per_frame(detections: Detections, frame_h: int, frame_w: int) -> list[list[Region]]:
    out = []
    for boxes, masks in zip(detections.boxes_xyxy, detections.masks):
        traced = (trace_region(box, mask, frame_h, frame_w) for box, mask in zip(boxes, masks))
        out.append([region for region in traced if region is not None])
    return out
