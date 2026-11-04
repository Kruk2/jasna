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
from jasna.vr_projection import RegionProjector

COLOUR_MATCH_STRENGTH = 0.75


def _reflect101_index(size: int, before: int, after: int, device: torch.device) -> torch.Tensor:
    index = torch.arange(-before, size + after, device=device)
    period = 2 * (size - 1) if size > 1 else 1
    index = index.remainder(period)
    return torch.where(index >= size, period - index, index)


def crop_to_canvas(frame: torch.Tensor, crop: CropFrame, projector: RegionProjector | None) -> torch.Tensor:
    """Crop, bilinear-resize and reflect-101 pad one frame onto the model canvas. A VR
    ``projector`` instead fills the whole canvas with its view of the crop box."""
    if projector is not None:
        return projector.project_region(frame, crop.crop_box, crop.canvas).to(torch.uint8)
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
    starts = torch.arange(out_size, dtype=torch.float64, device=device) * scale
    edges = torch.arange(in_size + 1, dtype=torch.float64, device=device)
    lo = torch.maximum(starts[:, None], edges[None, :-1])
    hi = torch.minimum(starts[:, None] + scale, edges[None, 1:])
    return ((hi - lo).clamp(min=0) / scale).float()


def canvas_to_source(
    canvas: torch.Tensor, crop: CropFrame, source: torch.Tensor, projector: RegionProjector | None
) -> torch.Tensor:
    """The restored canvas mapped back to its source crop box, float ``[3, h, w]`` in 0..255
    rounded to whole grey levels (bicubic when enlarging, pixel area when shrinking). With a
    VR ``projector`` only the restoration's change to the ``source`` frame's view maps back,
    so the crop keeps its source pixels wherever the canvas kept the view."""
    if projector is not None:
        x1, y1, x2, y2 = crop.crop_box
        change = canvas.float() - projector.project_region(source, crop.crop_box, canvas.shape[-1])
        restored = source[:, y1:y2, x1:x2].float() + projector.source_region_from_patch(change, crop.crop_box)
        return restored.round_().clamp_(0, 255)
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


def feather_alpha(mask: torch.Tensor, feather: int, box: tuple[int, int, int, int]) -> torch.Tensor:
    """Dilate a bool ``[H, W]`` mask by an ellipse of ``feather`` px and Gaussian-blur it
    (kernel ``2*feather+1``, reflect-101 border) into a [0, 1] alpha. ``box`` (x1, y1, x2, y2)
    holds every set pixel; the alpha is 0 farther than ``2*feather`` px from it, so only the box
    grown by ``2*feather+1`` px is computed (the extra px keeps the reflect-101 border of the
    cut-out in the zeros)."""
    x1, y1, x2, y2 = _grown(box, 2 * feather + 1, mask.shape)
    alpha = torch.zeros(mask.shape, dtype=torch.float32, device=mask.device)
    alpha[y1:y2, x1:x2] = _dilate_and_blur(mask[y1:y2, x1:x2], feather)
    return alpha


def _grown(box: tuple[int, int, int, int], reach: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = shape
    x1, y1, x2, y2 = box
    return max(x1 - reach, 0), max(y1 - reach, 0), min(x2 + reach, width), min(y2 + reach, height)


def _dilate_and_blur(mask: torch.Tensor, feather: int) -> torch.Tensor:
    dilated = dilate_ellipse(mask, feather)[None, None]
    kernel = _gaussian(feather, mask.device)
    x = F.pad(dilated.float(), (feather, feather, 0, 0), mode="reflect")
    x = F.conv2d(x, kernel.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, feather, feather), mode="reflect")
    return F.conv2d(x, kernel.view(1, 1, -1, 1))[0, 0]


def crop_edge_ramp(
    box: tuple[int, int, int, int],
    shape: tuple[int, int],
    feather: int,
    window: tuple[int, int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """1 inside the crop, falling to 0 over ``feather`` px at every crop edge that is not the
    region border, 0 outside; only its ``window`` (x1, y1, x2, y2) of the region."""
    height, width = shape
    x1, y1, x2, y2 = box
    wx1, wy1, wx2, wy2 = window

    def axis(low: int, high: int, size: int) -> torch.Tensor:
        centre = torch.arange(size, dtype=torch.float32, device=device) + 0.5
        ramp = ((centre > low) & (centre < high)).float()
        if low > 0:
            ramp = torch.minimum(ramp, (centre - low) / max(feather, 1))
        if high < size:
            ramp = torch.minimum(ramp, (high - centre) / max(feather, 1))
        return ramp.clamp(0.0, 1.0)

    return torch.minimum(axis(y1, y2, height)[wy1:wy2, None], axis(x1, x2, width)[None, wx1:wx2])


def _contours(polygons: Sequence[Polygon], offset: tuple[int, int]) -> list[np.ndarray]:
    origin = np.asarray(offset, dtype=np.int32)
    return [np.asarray(p, dtype=np.float32).round().astype(np.int32) - origin for p in polygons if len(p) >= 3]


def polygon_box(polygons: Sequence[Polygon], shape: tuple[int, int], offset: tuple[int, int]) -> tuple[int, int, int, int]:
    """(x1, y1, x2, y2) inside the ``[h, w]`` region at ``offset`` holding every pixel of
    ``polygons``; known on the host, so cropping by it needs no GPU sync."""
    height, width = shape
    contours = _contours(polygons, offset)
    if not contours:
        return 0, 0, 0, 0
    points = np.concatenate(contours)
    x1, y1 = np.clip(points.min(axis=0), 0, (width, height)).tolist()
    x2, y2 = np.clip(points.max(axis=0) + 1, 0, (width, height)).tolist()
    return x1, y1, x2, y2


def polygon_mask(polygons: Sequence[Polygon], shape: tuple[int, int], offset: tuple[int, int], device: torch.device) -> torch.Tensor:
    """Bool ``[h, w]`` union of ``polygons`` inside the region at ``offset`` (x, y); only their
    ``polygon_box`` is filled on the host and uploaded."""
    x1, y1, x2, y2 = polygon_box(polygons, shape, offset)
    patch = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    for contour in _contours(polygons, (offset[0] + x1, offset[1] + y1)):
        # One fillPoly call per polygon: a single call uses the even-odd rule and punches holes where polygons overlap.
        cv2.fillPoly(patch, [contour], 255)
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    # Non-blocking: the driver stages a small pageable source before returning, without waiting for queued kernels.
    mask[y1:y2, x1:x2] = torch.from_numpy(patch).to(device, non_blocking=True) > 0
    return mask


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


def composite_frame(
    source: torch.Tensor, candidates: Sequence[Candidate], *, feather: int, projector: RegionProjector | None
) -> torch.Tensor:
    live = [c for c in candidates if c.weight > 0.0 and c.presence > 0.0]
    if not live:
        return source
    _, height, width = source.shape
    reach = 2 * feather + 1
    x1s, y1s, x2s, y2s = zip(*(c.crop.crop_box for c in live))
    x0, y0, x3, y3 = _grown((min(x1s), min(y1s), max(x2s), max(y2s)), reach, (height, width))
    shape = (y3 - y0, x3 - x0)
    device = source.device
    tracks = {c.track: c for c in live}
    rois = {track: polygon_mask(c.polygons, shape, (x0, y0), device) for track, c in tracks.items()}
    boxes = {track: polygon_box(c.polygons, shape, (x0, y0)) for track, c in tracks.items()}
    feathered = {track: feather_alpha(rois[track], feather, boxes[track]) for track in tracks}
    union = torch.stack(list(rois.values())).any(dim=0)
    # Outside the feather of the union exactly where outside every track's: the dilation and the
    # positive blur kernel of a union are the union of each track's.
    outside = torch.stack([alpha <= 0.0 for alpha in feathered.values()]).all(dim=0)
    region = source[:, y0:y3, x0:x3].float()
    layers = []
    for candidate in live:
        roi = rois[candidate.track]
        x1, y1, x2, y2 = candidate.crop.crop_box
        x1, y1, x2, y2 = x1 - x0, y1 - y0, x2 - x0, y2 - y0
        # Alpha is 0 outside the crop and outside the track's feathered box, so a layer paints only their overlap.
        fx1, fy1, fx2, fy2 = _grown(boxes[candidate.track], reach, shape)
        px1, py1 = max(x1, fx1), max(y1, fy1)
        px2, py2 = max(min(x2, fx2), px1), max(min(y2, fy2), py1)
        ramp = crop_edge_ramp((x1, y1, x2, y2), shape, feather, (px1, py1, px2, py2), device)
        covered = ramp > 0.0
        restored_crop = canvas_to_source(candidate.canvas, candidate.crop, source, projector)
        roi = roi[py1:py2, px1:px2]
        edge = torch.where(roi, covered.float(), ramp)
        alpha = feathered[candidate.track][py1:py2, px1:px2] * edge * candidate.weight
        if candidate.presence == 1.0:
            alpha = alpha * (roi | ~union[py1:py2, px1:px2]).float()
        # The ramp is positive on the whole crop, so the colour ring is all of its feather-free pixels.
        ring = outside[y1:y2, x1:x2]
        offset = masked_median(region[:, y1:y2, x1:x2] - restored_crop, ring) * COLOUR_MATCH_STRENGTH
        restored = restored_crop[:, py1 - y1 : py2 - y1, px1 - x1 : px2 - x1] + offset[:, None, None]
        layers.append((candidate, (px1, py1, px2, py2), restored, alpha))
    handing = {track: c.presence for track, c in tracks.items() if c.presence < 1.0}
    shares = {}
    for track, presence in handing.items():
        share = torch.zeros(shape, dtype=torch.float32, device=device)
        for candidate, (px1, py1, px2, py2), _, alpha in layers:
            if candidate.track == track:
                share[py1:py2, px1:px2] += alpha
        shares[track] = share.clamp(max=1.0) * presence
    bx1, by1 = min(box[0] for _, box, _, _ in layers), min(box[1] for _, box, _, _ in layers)
    bx2, by2 = max(box[2] for _, box, _, _ in layers), max(box[3] for _, box, _, _ in layers)
    accumulator = torch.zeros((3, by2 - by1, bx2 - bx1), dtype=torch.float32, device=device)
    total = torch.zeros((by2 - by1, bx2 - bx1), dtype=torch.float32, device=device)
    for candidate, (px1, py1, px2, py2), restored, alpha in layers:
        if candidate.track in handing:
            alpha = alpha * candidate.presence
        else:
            for share in shares.values():
                alpha = alpha * (1.0 - share[py1:py2, px1:px2])
        accumulator[:, py1 - by1 : py2 - by1, px1 - bx1 : px2 - bx1] += restored * alpha
        total[py1 - by1 : py2 - by1, px1 - bx1 : px2 - bx1] += alpha
    over = total > 1.0
    accumulator = torch.where(over, accumulator / total, accumulator)
    total = torch.where(over, torch.ones_like(total), total)
    blended = accumulator + region[:, by1:by2, bx1:bx2] * (1.0 - total)
    out = source.clone()
    out[:, y0 + by1 : y0 + by2, x0 + bx1 : x0 + bx2] = blended.round_().clamp_(0, 255).to(torch.uint8)
    return out
