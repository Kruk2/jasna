"""Crops into the model canvas and the composite of restored windows back into frames.

Frames are ``[3, H, W]`` uint8 RGB tensors. A window paints only its own track's
feathered mask and never a pixel inside another track's mask; outside every mask the
source stays bit-exact. A track in a handover (presence below 1) also paints over the
other tracks' masks and takes that share of its feathered area from them.
"""

from __future__ import annotations

import ctypes
import functools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from jasna.accelerator import AcceleratorVendor, vendor_for_device
from jasna.ltx.camera import CropFrame
from jasna.ltx.plan import Polygon
from jasna.media.cuda_kernel import Kernel, grid_size, has_precompiled_kernels
from jasna.tracking.blending import dilate_ellipse, ellipse_row_reach
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


@functools.lru_cache(maxsize=64)
def _area_weights(out_size: int, in_size: int, device: torch.device) -> torch.Tensor:
    """``[out, in]`` pixel-area resampling weights (OpenCV INTER_AREA for downscaling), computed on
    the host in float64 (dividing by the scale as CUDA divides by a scalar: times its reciprocal)
    and uploaded once; kept per size, as a crop keeps its size over many frames (at most 768x768
    floats each)."""
    scale = in_size / out_size
    starts = np.arange(out_size, dtype=np.float64) * scale
    edges = np.arange(in_size + 1, dtype=np.float64)
    lo = np.maximum(starts[:, None], edges[None, :-1])
    hi = np.minimum(starts[:, None] + scale, edges[None, 1:])
    weights = (np.maximum(hi - lo, 0.0) * (1.0 / scale)).astype(np.float32)
    # Non-blocking: the driver stages a small pageable source before returning, without waiting for queued kernels.
    return torch.from_numpy(weights).to(device, non_blocking=True)


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
    return _resampled_crop(canvas, crop).round_().clamp_(0, 255)


def _resampled_crop(canvas: torch.Tensor, crop: CropFrame) -> torch.Tensor:
    """The canvas content resampled to the crop box, float ``[3, h, w]`` before rounding."""
    top, _, left, _ = crop.pad
    resize_h, resize_w = crop.resize_hw
    content = canvas[:, top : top + resize_h, left : left + resize_w].float()
    x1, y1, x2, y2 = crop.crop_box
    height, width = y2 - y1, x2 - x1
    if max(height, width) > max(resize_h, resize_w):
        return F.interpolate(content[None], size=(height, width), mode="bicubic", align_corners=False)[0]
    rows = _area_weights(height, resize_h, canvas.device)
    cols = _area_weights(width, resize_w, canvas.device)
    return torch.einsum("hy,cyx,wx->chw", rows, content, cols)


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


@functools.cache
def _gaussian_band(radius: int, size: int, device: torch.device) -> torch.Tensor:
    """``[size + 2*radius, size]`` matrix with the Gaussian at rows ``j..j + 2*radius`` of column ``j``."""
    offsets = torch.arange(size + 2 * radius, device=device)[:, None] - torch.arange(size, device=device)[None, :]
    taps = _gaussian(radius, device)[offsets.clamp(0, 2 * radius)]
    return torch.where((offsets >= 0) & (offsets <= 2 * radius), taps, 0.0)


def _blur_band(radius: int, length: int, device: torch.device) -> torch.Tensor:
    """Blurs a line of ``length`` px reflect-padded by ``radius``: the top-left block of a band
    cached per power of two."""
    size = 1 << (length - 1).bit_length()
    return _gaussian_band(radius, size, device)[: length + 2 * radius, :length]


def _dilate_and_blur(mask: torch.Tensor, feather: int) -> torch.Tensor:
    # Matrix products, not conv2d: the box size changes every frame, and MIOpen benchmarks
    # every new conv shape (~50 ms each on an R9700) before it runs it.
    dilated = dilate_ellipse(mask, feather)[None, None].float()
    height, width = mask.shape
    rows = F.pad(dilated, (feather, feather, 0, 0), mode="reflect")[0, 0] @ _blur_band(feather, width, mask.device)
    columns = F.pad(rows[None, None], (0, 0, feather, feather), mode="reflect")[0, 0]
    return _blur_band(feather, height, mask.device).T @ columns


def crop_edge_ramp(
    box: tuple[int, int, int, int],
    shape: tuple[int, int],
    feather: int,
    window: tuple[int, int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """1 inside the crop, falling to 0 over ``feather`` px at every crop edge that is not the
    region border, 0 outside; only its ``window`` (x1, y1, x2, y2) of the region. The two axes
    are computed on the host in float32 and uploaded in one copy (a frame's composite is short
    kernels, each waiting for the GPU), dividing as CUDA divides by a scalar: times the float32
    reciprocal."""
    height, width = shape
    x1, y1, x2, y2 = box
    wx1, wy1, wx2, wy2 = window
    inverse = np.float32(1.0) / np.float32(max(feather, 1))

    def axis(low: int, high: int, size: int, start: int, stop: int) -> np.ndarray:
        centre = np.arange(start, stop, dtype=np.float32) + np.float32(0.5)
        ramp = ((centre > low) & (centre < high)).astype(np.float32)
        if low > 0:
            ramp = np.minimum(ramp, (centre - np.float32(low)) * inverse)
        if high < size:
            ramp = np.minimum(ramp, (np.float32(high) - centre) * inverse)
        return np.clip(ramp, np.float32(0.0), np.float32(1.0))

    rows, cols = axis(y1, y2, height, wy1, wy2), axis(x1, x2, width, wx1, wx2)
    # Non-blocking: the driver stages a small pageable source before returning, without waiting for queued kernels.
    both = torch.from_numpy(np.concatenate((rows, cols))).to(device, non_blocking=True)
    return torch.minimum(both[: len(rows), None], both[len(rows) :][None, :])


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


def polygon_patch(
    polygons: Sequence[Polygon], shape: tuple[int, int], offset: tuple[int, int]
) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """The ``polygon_box`` of ``polygons`` inside the region at ``offset`` (x, y) and their union
    filled on the host over it, uint8 0 or 255."""
    x1, y1, x2, y2 = polygon_box(polygons, shape, offset)
    patch = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    for contour in _contours(polygons, (offset[0] + x1, offset[1] + y1)):
        # One fillPoly call per polygon: a single call uses the even-odd rule and punches holes where polygons overlap.
        cv2.fillPoly(patch, [contour], 255)
    return (x1, y1, x2, y2), patch


def polygon_mask(polygons: Sequence[Polygon], shape: tuple[int, int], offset: tuple[int, int], device: torch.device) -> torch.Tensor:
    """Bool ``[h, w]`` union of ``polygons`` inside the region at ``offset`` (x, y); only their
    ``polygon_box`` is filled on the host and uploaded."""
    (x1, y1, x2, y2), patch = polygon_patch(polygons, shape, offset)
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
    if projector is None and _fits_compose_kernels(source, live, feather):
        return composite_on_kernels(source, live, feather=feather)
    return composite_on_torch(source, live, feather=feather, projector=projector)


def _compositing_region(source: torch.Tensor, live: Sequence[Candidate], feather: int) -> tuple[int, int, int, int]:
    """(x0, y0, x3, y3): every live crop grown by the feather reach, inside the frame."""
    _, height, width = source.shape
    x1s, y1s, x2s, y2s = zip(*(c.crop.crop_box for c in live))
    return _grown((min(x1s), min(y1s), max(x2s), max(y2s)), 2 * feather + 1, (height, width))


def composite_on_torch(
    source: torch.Tensor, live: Sequence[Candidate], *, feather: int, projector: RegionProjector | None
) -> torch.Tensor:
    """The composite of ``live`` candidates (weight and presence above 0) in Torch ops: the
    reference, and the path of every frame the compose kernels do not take."""
    reach = 2 * feather + 1
    x0, y0, x3, y3 = _compositing_region(source, live, feather)
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
        x1, y1, x2, y2 = _region_box(candidate.crop.crop_box, (x0, y0))
        px1, py1, px2, py2 = _paint_box((x1, y1, x2, y2), _grown(boxes[candidate.track], reach, shape))
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


def _region_box(box: tuple[int, int, int, int], origin: tuple[int, int]) -> tuple[int, int, int, int]:
    x0, y0 = origin
    x1, y1, x2, y2 = box
    return x1 - x0, y1 - y0, x2 - x0, y2 - y0


def _paint_box(crop: tuple[int, int, int, int], feathered: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """The overlap of a layer's crop and its track's feathered box, where its alpha can be above 0."""
    x1, y1, x2, y2 = crop
    fx1, fy1, fx2, fy2 = feathered
    px1, py1 = max(x1, fx1), max(y1, fy1)
    return px1, py1, max(min(x2, fx2), px1), max(min(y2, fy2), py1)


COMPOSE_KERNEL_VENDORS = (AcceleratorVendor.NVIDIA,)
"""Vendors whose frames composite on ``ltx_compose`` kernels (where ``has_precompiled_kernels``).
AMD's build is bit-identical too but keeps the Torch path: there the window decoder bounds the
compose pass, and the kernels gained ~1 % on an RX 7900 XTX."""
_KERNEL_MODULE = "ltx_compose"
_HEADER_SLOTS, _TRACK_SLOTS, _LAYER_SLOTS = 20, 12, 16
_MEDIAN_BINS = 511
_TILE = 32
_THREADS = (32, 8, 1)
_DEVICE_ALIGNMENT = 512
_MAX_KERNEL_FEATHER = 254


class _ComposeKernels(NamedTuple):
    histograms: Kernel
    offsets: Kernel
    blend: Kernel


@functools.cache
def _compose_kernels() -> _ComposeKernels:
    table = (ctypes.c_void_p,)
    return _ComposeKernels(
        Kernel(_KERNEL_MODULE, "colour_ring_histograms", table),
        Kernel(_KERNEL_MODULE, "colour_offsets", table),
        Kernel(_KERNEL_MODULE, "blend_layers", table),
    )


@functools.cache
def _dilate_kernel(radius: int) -> Kernel:
    """Per feather radius: its shared memory holds ``TILE + 2*radius`` rows of column distances."""
    return Kernel(
        _KERNEL_MODULE, "dilate_feather_masks", (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int),
        shared_memory_bytes=(_TILE + 2 * radius) * (_TILE + 1),
    )


def _fits_compose_kernels(source: torch.Tensor, live: Sequence[Candidate], feather: int) -> bool:
    """Whether ``composite_on_kernels`` takes a frame: a GPU of ``COMPOSE_KERNEL_VENDORS`` with the
    precompiled kernels, a frame with unit column stride, tracks at full
    presence, and a feather the Torch path can reflect-pad (also at the region edge) and the
    kernel's byte distances can hold."""
    x0, y0, x3, y3 = _compositing_region(source, live, feather)
    return (
        vendor_for_device(source.device) in COMPOSE_KERNEL_VENDORS
        and has_precompiled_kernels(source.device)
        and source.stride(-1) == 1
        and all(c.presence == 1.0 for c in live)
        and feather <= _MAX_KERNEL_FEATHER
        and min(y3 - y0, x3 - x0) > 2 * feather + 1
    )


def _aligned(size: int) -> int:
    return -(-size // _DEVICE_ALIGNMENT) * _DEVICE_ALIGNMENT


def _float_view(workspace: torch.Tensor, offset: int, shape: tuple[int, int]) -> torch.Tensor:
    return workspace[offset : offset + 4 * shape[0] * shape[1]].view(torch.float32).view(shape)


def composite_on_kernels(source: torch.Tensor, live: Sequence[Candidate], *, feather: int) -> torch.Tensor:
    """``composite_on_torch`` for plain crops of tracks at full presence, bit-identical: the
    dilation, colour match and blend run as four ``ltx_compose`` kernels over one workspace whose
    geometry table, zeroed histograms and mask patches arrive in one upload; the feather blur
    stays the Torch path's two matrix products. A frame's compositing is bound by launches, each
    waiting behind the window decoder's queued kernels (~150 Torch launches become ~18)."""
    device = source.device
    x0, y0, x3, y3 = _compositing_region(source, live, feather)
    origin, shape = (x0, y0), (y3 - y0, x3 - x0)
    height, width = shape
    patches = {c.track: polygon_patch(c.polygons, shape, origin) for c in live}
    reach = 2 * feather + 1
    track_ids = list(patches)
    feather_boxes = [_grown(patches[track][0], reach, shape) for track in track_ids]
    crops = [_region_box(c.crop.crop_box, origin) for c in live]
    paints = [_paint_box(crop, feather_boxes[track_ids.index(c.track)]) for c, crop in zip(live, crops)]
    blend = (
        min(p[0] for p in paints), min(p[1] for p in paints), max(p[2] for p in paints), max(p[3] for p in paints)
    )
    resampled = [_resampled_crop(c.canvas, c.crop) for c in live]
    out = source.clone()

    table_slots = _HEADER_SLOTS + _TRACK_SLOTS * len(track_ids) + _LAYER_SLOTS * len(live)
    histogram_bytes = 4 * 3 * _MEDIAN_BINS * len(live)
    upload = [_aligned(8 * table_slots), _aligned(histogram_bytes)] + [_aligned(patches[t][1].nbytes) for t in track_ids]
    feather_shapes = [(y2 - y1, x2 - x1) for x1, y1, x2, y2 in feather_boxes]
    written = (
        [_aligned(4 * 3 * len(live))]
        + [_aligned(4 * h * (w + 2 * feather)) for h, w in feather_shapes]
        + [_aligned(4 * h * w) for h, w in feather_shapes]
    )
    starts = np.cumsum([0] + upload + written).tolist()
    workspace = torch.empty(starts[-1], dtype=torch.uint8, device=device)
    base = workspace.data_ptr()
    histogram_at, patch_at = starts[1], starts[2 : 2 + len(track_ids)]
    offsets_at = starts[2 + len(track_ids)]
    dilated_at = starts[3 + len(track_ids) : 3 + 2 * len(track_ids)]
    alpha_at = starts[3 + 2 * len(track_ids) : 3 + 3 * len(track_ids)]

    host = np.zeros(starts[len(upload)], dtype=np.uint8)
    table = host[: 8 * table_slots].view(np.int64)
    inverse = np.float32(1.0) / np.float32(max(feather, 1))
    table[:_HEADER_SLOTS] = (
        len(track_ids), len(live), origin[0], origin[1], height, width,
        source.data_ptr(), source.stride(0), source.stride(1),
        out.data_ptr(), out.stride(0), out.stride(1),
        inverse.view(np.int32), base + histogram_at, base + offsets_at, 0, *blend,
    )
    for index, track in enumerate(track_ids):
        box, patch = patches[track]
        slots = _HEADER_SLOTS + _TRACK_SLOTS * index
        table[slots : slots + 11] = (
            base + patch_at[index], *box, *feather_boxes[index], base + alpha_at[index], base + dilated_at[index]
        )
        host[patch_at[index] : patch_at[index] + patch.nbytes] = patch.reshape(-1)
    for index, (candidate, crop, paint, crop_pixels) in enumerate(zip(live, crops, paints, resampled)):
        slots = _HEADER_SLOTS + _TRACK_SLOTS * len(track_ids) + _LAYER_SLOTS * index
        table[slots : slots + 14] = (
            track_ids.index(candidate.track), *crop, *paint, crop_pixels.data_ptr(), *crop_pixels.stride(),
            np.float32(candidate.weight).view(np.int32),
        )
    # Non-blocking: the driver stages a small pageable source before returning, without waiting for queued kernels.
    workspace[: len(host)].copy_(torch.from_numpy(host), non_blocking=True)

    kernels = _compose_kernels()
    stream = torch.cuda.current_stream(device).cuda_stream
    _dilate_kernel(feather).launch(
        (
            grid_size(max(w for _, w in feather_shapes), _TILE),
            grid_size(max(h for h, _ in feather_shapes), _TILE),
            len(track_ids),
        ),
        _THREADS, (base, ellipse_row_reach(feather, device).data_ptr(), feather), stream,
    )
    for index, (h, w) in enumerate(feather_shapes):
        dilated = _float_view(workspace, dilated_at[index], (h, w + 2 * feather))
        rows = dilated @ _blur_band(feather, w, device)
        columns = F.pad(rows[None, None], (0, 0, feather, feather), mode="reflect")[0, 0]
        torch.mm(_blur_band(feather, h, device).T, columns, out=_float_view(workspace, alpha_at[index], (h, w)))
    crop_shapes = [(y2 - y1, x2 - x1) for x1, y1, x2, y2 in crops]
    kernels.histograms.launch(
        (grid_size(max(w for _, w in crop_shapes), _TILE), grid_size(max(h for h, _ in crop_shapes), _TILE), len(live)),
        _THREADS, (base,), stream,
    )
    kernels.offsets.launch((len(live), 1, 1), (32, 1, 1), (base,), stream)
    blend_width, blend_height = blend[2] - blend[0], blend[3] - blend[1]
    if blend_width and blend_height:
        kernels.blend.launch(
            (grid_size(blend_width, _TILE), grid_size(blend_height, _TILE), 1), _THREADS, (base,), stream
        )
    return out
