"""Crops into the model canvas and the composite of restored windows back into frames.

Frames are ``[3, H, W]`` uint8 RGB tensors. A window paints only its own track's
feathered mask and never a pixel inside another track's mask; outside every mask the
source stays bit-exact. A track in a handover (presence below 1) also paints over the
other tracks' masks and takes that share of its feathered area from them.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from jasna.ltx.camera import CropFrame
from jasna.ltx.plan import Polygon
from jasna.tracking.blending import dilate_ellipse

COLOUR_MATCH_STRENGTH = 0.75


def _reflect101_index(size: int, before: int, after: int, device: torch.device) -> torch.Tensor:
    index = torch.arange(-before, size + after, device=device)
    period = 2 * (size - 1) if size > 1 else 1
    index = index.remainder(period)
    return torch.where(index >= size, period - index, index)


def crop_to_canvas(frame: torch.Tensor, crop: CropFrame) -> torch.Tensor:
    """Crop, bilinear-resize and reflect-101 pad one frame onto the model canvas."""
    x1, y1, x2, y2 = crop.crop_box
    content = frame[:, y1:y2, x1:x2]
    if content.shape[1:] != crop.resize_hw:
        resized = F.interpolate(content[None].float(), size=crop.resize_hw, mode="bilinear", align_corners=False)
        content = resized[0].round_().clamp_(0, 255).to(torch.uint8)
    top, bottom, left, right = crop.pad
    if top or bottom or left or right:
        rows = _reflect101_index(content.shape[1], top, bottom, frame.device)
        cols = _reflect101_index(content.shape[2], left, right, frame.device)
        content = content[:, rows][:, :, cols]
    return content


def _area_weights(out_size: int, in_size: int, device: torch.device) -> torch.Tensor:
    """``[out, in]`` pixel-area resampling weights (OpenCV INTER_AREA for downscaling)."""
    scale = in_size / out_size
    starts = torch.arange(out_size, dtype=torch.float64) * scale
    edges = torch.arange(in_size + 1, dtype=torch.float64)
    lo = torch.maximum(starts[:, None], edges[None, :-1])
    hi = torch.minimum(starts[:, None] + scale, edges[None, 1:])
    return ((hi - lo).clamp(min=0) / scale).to(device=device, dtype=torch.float32)


def canvas_to_source(canvas: torch.Tensor, crop: CropFrame) -> torch.Tensor:
    """The restored canvas mapped back to its source crop box, float ``[3, h, w]`` in 0..255
    rounded to whole grey levels (bicubic when enlarging, pixel area when shrinking)."""
    top, _, left, _ = crop.pad
    resize_h, resize_w = crop.resize_hw
    content = canvas[:, top : top + resize_h, left : left + resize_w].float()
    x1, y1, x2, y2 = crop.crop_box
    height, width = y2 - y1, x2 - x1
    if max(height, width) > max(resize_h, resize_w):
        out = F.interpolate(content[None], size=(height, width), mode="bicubic", align_corners=False)[0]
    else:
        rows = _area_weights(height, resize_h, canvas.device)
        cols = _area_weights(width, resize_w, canvas.device)
        out = torch.einsum("hy,cyx,wx->chw", rows, content, cols)
    return out.round_().clamp_(0, 255)


@functools.cache
def _gaussian(radius: int, device: torch.device) -> torch.Tensor:
    kernel = cv2.getGaussianKernel(2 * radius + 1, 0.0, cv2.CV_32F)[:, 0]
    return torch.from_numpy(kernel).to(device)


def feather_alpha(mask: torch.Tensor, feather: int) -> torch.Tensor:
    """Dilate a bool ``[H, W]`` mask by an ellipse of ``feather`` px and Gaussian-blur it
    (kernel ``2*feather+1``, reflect-101 border) into a [0, 1] alpha."""
    dilated = dilate_ellipse(mask, feather)[None, None]
    kernel = _gaussian(feather, mask.device)
    x = F.pad(dilated.float(), (feather, feather, 0, 0), mode="reflect")
    x = F.conv2d(x, kernel.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, feather, feather), mode="reflect")
    return F.conv2d(x, kernel.view(1, 1, -1, 1))[0, 0]


def crop_edge_ramp(box: tuple[int, int, int, int], shape: tuple[int, int], feather: int, device: torch.device) -> torch.Tensor:
    """1 inside the crop, falling to 0 over ``feather`` px at every crop edge that is not the
    region border, 0 outside."""
    height, width = shape
    x1, y1, x2, y2 = box

    def axis(low: int, high: int, size: int) -> torch.Tensor:
        centre = torch.arange(size, dtype=torch.float32, device=device) + 0.5
        ramp = ((centre > low) & (centre < high)).float()
        if low > 0:
            ramp = torch.minimum(ramp, (centre - low) / max(feather, 1))
        if high < size:
            ramp = torch.minimum(ramp, (high - centre) / max(feather, 1))
        return ramp.clamp(0.0, 1.0)

    return torch.minimum(axis(y1, y2, height)[:, None], axis(x1, x2, width)[None, :])


def polygon_mask(polygons: Sequence[Polygon], shape: tuple[int, int], offset: tuple[int, int], device: torch.device) -> torch.Tensor:
    """Bool ``[h, w]`` union of ``polygons`` inside the region at ``offset`` (x, y)."""
    mask = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        if len(polygon) >= 3:
            contour = np.asarray(polygon, dtype=np.float32).round().astype(np.int32) - np.asarray(offset, dtype=np.int32)
            # One fillPoly call per polygon: a single call uses the even-odd rule and punches holes where polygons overlap.
            cv2.fillPoly(mask, [contour], 255)
    return torch.from_numpy(mask).to(device) > 0


def masked_median(difference: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-channel median of a whole-valued ``[C, h, w]`` difference in -255..255 over the bool
    ``[h, w]`` mask, the mean of the two central values for an even count (as
    ``torch.quantile(q=0.5)``), 0 for an empty mask; a histogram, so no host sync."""
    channels = difference.shape[0]
    channel_offset = torch.arange(channels, device=difference.device)[:, None, None] * 511.0
    values = torch.where(mask, difference + 255.0 + channel_offset, -1.0)
    counts = torch.histc(values, bins=channels * 511, min=-0.5, max=channels * 511 - 0.5)
    cumulative = counts.view(channels, 511).long().cumsum(1)
    count = cumulative[:, -1:]
    central = torch.searchsorted(cumulative, torch.cat([(count - 1) // 2, count // 2], dim=1), right=True)
    return torch.where(count[:, 0] > 0, central.sum(1) * 0.5 - 255.0, 0.0)


@dataclass(frozen=True)
class Candidate:
    """One window's restored canvas for a frame, with its crop, crossfade weight, and the
    id, polygons and handover presence of the track it restores."""

    canvas: torch.Tensor
    crop: CropFrame
    weight: float
    track: int
    polygons: Sequence[Polygon]
    presence: float


def composite_frame(source: torch.Tensor, candidates: Sequence[Candidate], *, feather: int) -> torch.Tensor:
    live = [c for c in candidates if c.weight > 0.0 and c.presence > 0.0]
    if not live:
        return source
    _, height, width = source.shape
    reach = 2 * feather + 1
    x0 = max(min(c.crop.crop_box[0] for c in live) - reach, 0)
    y0 = max(min(c.crop.crop_box[1] for c in live) - reach, 0)
    x3 = min(max(c.crop.crop_box[2] for c in live) + reach, width)
    y3 = min(max(c.crop.crop_box[3] for c in live) + reach, height)
    shape = (y3 - y0, x3 - x0)
    device = source.device
    tracks = {c.track: c for c in live}
    rois = {track: polygon_mask(c.polygons, shape, (x0, y0), device) for track, c in tracks.items()}
    feathered = {track: feather_alpha(roi, feather) for track, roi in rois.items()}
    if len(rois) == 1:
        (union,), (union_feathered,) = rois.values(), feathered.values()
    else:
        union = torch.stack(list(rois.values())).any(dim=0)
        union_feathered = feather_alpha(union, feather)
    outside = union_feathered <= 0.0
    region = source[:, y0:y3, x0:x3].float()
    layers = []
    for candidate in live:
        roi = rois[candidate.track]
        x1, y1, x2, y2 = candidate.crop.crop_box
        x1, y1, x2, y2 = x1 - x0, y1 - y0, x2 - x0, y2 - y0
        ramp = crop_edge_ramp((x1, y1, x2, y2), shape, feather, device)
        covered = ramp > 0.0
        restored_crop = canvas_to_source(candidate.canvas, candidate.crop)
        restored = region.clone()
        restored[:, y1:y2, x1:x2] = restored_crop
        edge = torch.where(roi, covered.float(), ramp)
        alpha = feathered[candidate.track] * edge * candidate.weight
        if candidate.presence == 1.0:
            alpha = alpha * (roi | ~union).float()
        ring = (covered & outside)[y1:y2, x1:x2]
        offset = masked_median(region[:, y1:y2, x1:x2] - restored_crop, ring) * COLOUR_MATCH_STRENGTH
        layers.append((candidate, restored + offset[:, None, None], alpha))
    handing = {track: c.presence for track, c in tracks.items() if c.presence < 1.0}
    shares = {
        track: sum(alpha for c, _, alpha in layers if c.track == track).clamp(max=1.0) * presence
        for track, presence in handing.items()
    }
    accumulator = torch.zeros_like(region)
    total = torch.zeros(shape, dtype=torch.float32, device=device)
    for candidate, restored, alpha in layers:
        if candidate.track in handing:
            alpha = alpha * candidate.presence
        else:
            for share in shares.values():
                alpha = alpha * (1.0 - share)
        accumulator += restored * alpha
        total += alpha
    over = total > 1.0
    accumulator = torch.where(over, accumulator / total, accumulator)
    total = torch.where(over, torch.ones_like(total), total)
    blended = accumulator + region * (1.0 - total)
    out = source.clone()
    out[:, y0:y3, x0:x3] = blended.round_().clamp_(0, 255).to(torch.uint8)
    return out
