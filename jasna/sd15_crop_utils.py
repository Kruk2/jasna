"""SD 1.5 crop geometry ported verbatim from jav-restoration-v2
(`restoration_v2/dataset/crop_utils.py`).

This is the crop/paste pipeline the SD 1.5 model was trained and tuned with:
a square crop centered on the detection (sized by ``min_mosaic_fraction``),
resized so the long side is ``target_size`` and reflect-padded bottom+right,
with ``paste_back`` inverting that exact transform. Kept separate from jasna's
own letterbox crop (``crop_buffer``) so the two can be compared.
"""
from __future__ import annotations

import cv2
import numpy as np

BORDER_FRACTION = 0.06
MIN_BORDER = 20
MIN_MOSAIC_FRACTION = 0.75
CROP_MIN_MOSAIC_FRACTIONS = {256: 0.92, 384: 0.92, 480: 0.92, 512: 0.30, 768: 0.75, 1024: 0.19}


def reflect_pad_to_size(arr: np.ndarray, target_size: int) -> np.ndarray:
    """Reflect-pad a 2D or HxWxC array (bottom + right) up to a target square."""
    h, w = arr.shape[:2]
    ndim_pad = [(0, 0)] if arr.ndim == 3 else []

    if h != w:
        side = max(h, w)
        while h < side or w < side:
            ph = min(side - h, max(h - 1, 1))
            pw = min(side - w, max(w - 1, 1))
            arr = np.pad(arr, [(0, ph), (0, pw)] + ndim_pad, mode="reflect")
            h, w = arr.shape[:2]

    while h < target_size:
        step = min(target_size - h, h - 1)
        arr = np.pad(arr, [(0, step), (0, step)] + ndim_pad, mode="reflect")
        h, w = arr.shape[:2]

    return arr


def compute_crop_bbox(bbox, frame_w: int, frame_h: int, crop_size: int,
                      min_mosaic_fraction: float | None = None) -> tuple[int, int, int, int]:
    if min_mosaic_fraction is None:
        min_mosaic_fraction = CROP_MIN_MOSAIC_FRACTIONS.get(crop_size, MIN_MOSAIC_FRACTION)
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    bw, bh = x2 - x1, y2 - y1
    det_side = max(bw, bh)

    border = max(MIN_BORDER, int(det_side * BORDER_FRACTION))
    side_with_border = det_side + 2 * border

    max_allowed = int(det_side / min_mosaic_fraction)
    target_side = min(max_allowed, crop_size)
    target_side = max(target_side, side_with_border)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    c_x1 = int(cx - target_side / 2)
    c_y1 = int(cy - target_side / 2)
    c_x2 = c_x1 + target_side
    c_y2 = c_y1 + target_side

    if c_x1 < 0:
        c_x2 = min(c_x2 - c_x1, frame_w)
        c_x1 = 0
    if c_y1 < 0:
        c_y2 = min(c_y2 - c_y1, frame_h)
        c_y1 = 0
    if c_x2 > frame_w:
        c_x1 = max(c_x1 - (c_x2 - frame_w), 0)
        c_x2 = frame_w
    if c_y2 > frame_h:
        c_y1 = max(c_y1 - (c_y2 - frame_h), 0)
        c_y2 = frame_h

    return c_x1, c_y1, c_x2, c_y2


def crop_and_resize_np(img: np.ndarray, crop_bbox: tuple[int, int, int, int],
                       target_size: int) -> tuple[np.ndarray, int, int]:
    cx1, cy1, cx2, cy2 = crop_bbox
    crop = img[cy1:cy2, cx1:cx2]
    ch, cw = crop.shape[:2]
    max_side = max(ch, cw)

    scale = target_size / max_side if max_side != target_size else 1.0
    if scale != 1.0:
        new_h = round(ch * scale)
        new_w = round(cw * scale)
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
        crop = cv2.resize(crop, (new_w, new_h), interpolation=interp)

    h, w = crop.shape[:2]
    if h != target_size or w != target_size:
        crop = reflect_pad_to_size(crop, target_size)

    return crop, ch, cw


def paste_back(canvas: np.ndarray, restored: np.ndarray,
               crop_bbox: tuple[int, int, int, int], content_h: int, content_w: int,
               mask: np.ndarray | None = None,
               feather_radius: int = 0,
               feather_outward: bool = True) -> None:
    """Paste ``restored`` content into ``canvas`` at ``crop_bbox`` (inverse of
    ``crop_and_resize_np``). With ``feather_radius > 0`` and a binary ``mask`` the
    seam is softened via an outward distance-transform alpha ramp."""
    rh, rw = restored.shape[:2]
    side = max(content_h, content_w)
    th = round(content_h / side * rh)
    tw = round(content_w / side * rw)
    content = restored[:th, :tw]
    resized = cv2.resize(content, (content_w, content_h), interpolation=cv2.INTER_LANCZOS4)
    cx1, cy1, _, _ = crop_bbox
    if mask is None:
        canvas[cy1:cy1 + content_h, cx1:cx1 + content_w] = resized
        return

    mask_roi = mask[cy1:cy1 + content_h, cx1:cx1 + content_w]
    roi = canvas[cy1:cy1 + content_h, cx1:cx1 + content_w]

    if feather_radius <= 0:
        roi[mask_roi > 127] = resized[mask_roi > 127]
        return

    binary = (mask_roi > 127).astype(np.uint8)
    if not binary.any():
        return

    if feather_outward:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (feather_radius * 2 + 1, feather_radius * 2 + 1))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        dist_outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 3)
        alpha_ring = 1.0 - np.clip(dist_outside / float(feather_radius), 0.0, 1.0)
        alpha = np.where(binary > 0, 1.0, alpha_ring)
        alpha = np.where(dilated > 0, alpha, 0.0).astype(np.float32)
    else:
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        alpha = np.clip(dist / float(feather_radius), 0.0, 1.0).astype(np.float32)

    alpha = alpha[:, :, np.newaxis]
    blended = alpha * resized.astype(np.float32) + (1.0 - alpha) * roi.astype(np.float32)
    roi[:] = blended.clip(0, 255).astype(np.uint8)
