"""DiffVAE tiling helpers: schedule, pad/crop/size-floor, blend utilities.

Subset of ``ltx_core/model/video_vae/diffusion_tiling.py`` for the ``chunked_eager`` mode.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import List, Literal, Sequence, Tuple

import torch

from jasna.models.ltx_vae.tiling import (
    VIDEO_SCALE_FACTORS,
    DimensionInterval,
    DimensionSizeConfig,
    SpatioTemporalScaleFactors,
    Tile,
    TileSizeConfig,
    compute_trapezoidal_mask_1d,
    split_by_size,
    untiled_mask_1d,
)

ResizeAxisMode = Literal["repeat_last", "symmetric"]

# Peak-activation heuristic (bytes):
#   stage-4 input feature (resident for the whole decode): s4_t x s4_h x s4_w x stage4_channels x bf16
#   accumulator: H x W x (2 x tile_t) x out_channels x fp16
#   stage-5: stage5_tokens x stage5_channels x bf16 x coef (NA working-set multiplicity)
_STAGE5_MEM_COEF: float = 5
_ELEMENT_SIZE: int = 2
_MIN_MODEL_BYTES_FLOOR: int = 1 << 30
_BUDGET_SAFETY_BYTES: int = 1 << 30


def stage4_feature_bytes(
    *,
    height: int,
    width: int,
    num_frames: int,
    upsample_strides: Sequence[Tuple[int, int, int]],
    stage4_channels: int,
    natten_trailing_pad_latent_frames: int,
) -> int:
    """Resident stages-1-3 output size (full volume, including the trailing latent pad)."""
    latent_frames = (num_frames - 1) // VIDEO_SCALE_FACTORS.time + 1
    s4_t, s4_h, s4_w = stage4_thw_from_latent(
        upsample_strides[:3],
        latent_frames + natten_trailing_pad_latent_frames,
        height // VIDEO_SCALE_FACTORS.height,
        width // VIDEO_SCALE_FACTORS.width,
        drop_leading_frame=True,
    )
    return s4_t * s4_h * s4_w * stage4_channels * _ELEMENT_SIZE


def recommended_decode_tiling_config(
    *,
    tile_halos: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
    pixel_scale: SpatioTemporalScaleFactors,
    min_tile_size_s4: Tuple[int, int, int],
    patch_size: int,
    height: int,
    width: int,
    num_frames: int,
    free_bytes: int,
    stage5_channels: int,
    stage4_channels: int,
    upsample_strides: Sequence[Tuple[int, int, int]],
    model_bytes: int,
    natten_trailing_pad_latent_frames: int,
    out_channels: int,
) -> TileSizeConfig:
    """Pick decode tiling from stage-4/5 halos and the activation budget.

    Enumerates legal tile sizes, drops those whose peak-bytes estimate exceeds
    ``free - max(model, 1 GiB) - 1 GiB - stage4_feature``, and picks the one with the
    least overlap recompute (then the largest volume).
    """
    overlap_t, overlap_hw = recommended_pixel_overlaps(tile_halos, pixel_scale)

    ft, fh, fw = pixel_scale.time, pixel_scale.height, pixel_scale.width
    step_t = math.lcm(ft, VIDEO_SCALE_FACTORS.time)
    step_h = math.lcm(fh, VIDEO_SCALE_FACTORS.height)
    step_w = math.lcm(fw, VIDEO_SCALE_FACTORS.width)
    # ``2 * overlap`` so left+right ramps fit (else masks are not complementary).
    min_t_px = _round_up(max(2 * ft, 2 * overlap_t, _round_up(min_tile_size_s4[0] * ft, ft), 16), step_t)
    min_h_px = _round_up(max(2 * fh, 2 * overlap_hw, _round_up(min_tile_size_s4[1] * fh, fh), 64), step_h)
    min_w_px = _round_up(max(2 * fw, 2 * overlap_hw, _round_up(min_tile_size_s4[2] * fw, fw), 64), step_w)

    model_cost = max(model_bytes, _MIN_MODEL_BYTES_FLOOR)
    s4_feat_bytes = stage4_feature_bytes(
        height=height,
        width=width,
        num_frames=num_frames,
        upsample_strides=upsample_strides,
        stage4_channels=stage4_channels,
        natten_trailing_pad_latent_frames=natten_trailing_pad_latent_frames,
    )
    usable = max(0, free_bytes - model_cost - _BUDGET_SAFETY_BYTES - s4_feat_bytes)
    s5_bytes_per_token = max(1.0, float(stage5_channels) * float(_ELEMENT_SIZE) * _STAGE5_MEM_COEF)
    acc_bytes_per_pixel = out_channels * _ELEMENT_SIZE

    t_cands = _axis_candidates(num_frames, overlap_t, min_t_px, step_t)
    h_cands = _axis_candidates(height, overlap_hw, min_h_px, step_h)
    w_cands = _axis_candidates(width, overlap_hw, min_w_px, step_w)

    scored: list[tuple[float, int, int, int, int, int]] = []
    for tile_t, n_t in t_cands:
        # Current group buffer + still-live emit/stub during temporal handoff.
        acc_bytes = 2 * tile_t * height * width * acc_bytes_per_pixel
        if acc_bytes >= usable:
            continue
        max_s5_tokens = int((usable - acc_bytes) // s5_bytes_per_token)
        for tile_h, n_h in h_cands:
            for tile_w, n_w in w_cands:
                if stage5_tokens_for_pixel_tile(tile_t, tile_h, tile_w, patch_size=patch_size) > max_s5_tokens:
                    continue
                waste = (n_t * n_h * n_w * tile_t * tile_h * tile_w) / max(1, num_frames * height * width)
                scored.append((waste, -tile_t * tile_h * tile_w, n_t * n_h * n_w, tile_t, tile_h, tile_w))

    if not scored:
        raise ValueError(
            "Cannot fit a DiffVAE decode tile under the memory budget: "
            f"min tile ~{min_t_px}f x {min_h_px}x{min_w_px}px "
            f"(overlaps T={overlap_t}, HW={overlap_hw}), "
            f"coef={_STAGE5_MEM_COEF}, stage5_channels={stage5_channels}, "
            f"stage4_feature_bytes={s4_feat_bytes}, usable_bytes={usable}. "
            "Reduce resolution or free GPU memory."
        )

    scored.sort()
    _waste, _vol, _ntiles, tile_t, tile_h, tile_w = scored[0]
    return TileSizeConfig(
        frames=DimensionSizeConfig(tile_size=tile_t, overlap=overlap_t),
        height=DimensionSizeConfig(tile_size=tile_h, overlap=overlap_hw),
        width=DimensionSizeConfig(tile_size=tile_w, overlap=overlap_hw),
    )


def prepare_tile_schedule(
    stage4_shape_bcthw: torch.Size,
    tiling_config: TileSizeConfig,
    *,
    upsample3_stride: Tuple[int, int, int],
    patch_size: int,
    min_tile_size: Tuple[int, int, int],
) -> List[Tile]:
    """Build pixel-blend tiles whose ``in_coords`` land on the stage-4 input grid.

    Temporal intervals propagate through the pixel-shuffle hop (``drop_leading_frame``
    only on the origin tile); ramps are symmetric so masks stay complementary.
    """
    pixel_scale = stage4_to_pixel_scale_factors(upsample3_stride, patch_size)
    split_sizes = tiling_config.split_sizes(pixel_scale)

    def axis_specs(axis: int, *, propagate_causal: bool, apply_patch: bool) -> list[tuple[slice, slice, torch.Tensor]]:
        tile, overlap = split_sizes[axis]
        stride_component = upsample3_stride[axis]
        specs = []
        for iv in split_by_size(stage4_shape_bcthw[2 + axis], tile, overlap, min_tile_size[axis]):
            stage5 = _propagate_interval_through_upsample_hops(iv, [stride_component], propagate_causal)
            pixel = _propagate_interval_through_upsample_hops(stage5, [patch_size], False) if apply_patch else stage5
            mask_pixel = compute_trapezoidal_mask_1d(
                pixel.end - pixel.start, pixel.left_ramp, pixel.right_ramp, left_starts_from_0=False
            )
            specs.append((slice(iv.start, iv.end), slice(pixel.start, pixel.end), mask_pixel))
        return specs

    t_specs = axis_specs(0, propagate_causal=True, apply_patch=False)
    h_specs = axis_specs(1, propagate_causal=False, apply_patch=True)
    w_specs = axis_specs(2, propagate_causal=False, apply_patch=True)

    tiles: List[Tile] = []
    for t_spec, h_spec, w_spec in itertools.product(t_specs, h_specs, w_specs):
        t_s4, t_px, t_mask = t_spec
        h_s4, h_px, h_mask = h_spec
        w_s4, w_px, w_mask = w_spec
        tiles.append(
            Tile(
                in_coords=(slice(None), t_s4, h_s4, w_s4, slice(None)),
                out_coords=(slice(None), slice(None), t_px, h_px, w_px),
                masks_1d=(untiled_mask_1d(), untiled_mask_1d(), t_mask, h_mask, w_mask),
            )
        )
    return tiles


def slice_stage4_tile(
    feat_s4: torch.Tensor,
    tile: Tile,
    *,
    content_frames: int,
) -> tuple[torch.Tensor, bool, bool, tuple[int, int, int]]:
    """Slice a stage-4 feature tile, extending trailing tiles to include ghost frames."""
    is_origin = tile.in_coords[1].start in (0, None)
    _, stop, _ = tile.in_coords[1].indices(content_frames)
    pad_trailing = stop == content_frames
    _b, t_coord, h_coord, w_coord, _c = tile.in_coords
    t0, t1, _ = t_coord.indices(content_frames)
    h0, h1, _ = h_coord.indices(feat_s4.shape[2])
    w0, w1, _ = w_coord.indices(feat_s4.shape[3])
    content_thw = (t1 - t0, h1 - h0, w1 - w0)
    if pad_trailing:
        t1 = feat_s4.shape[1]
    feat_tile = feat_s4[:, t0:t1, h_coord, w_coord, :]
    return feat_tile, is_origin, pad_trailing, content_thw


@dataclass(frozen=True)
class AxisPad:
    """How many elements were added (pad) or removed (crop) on each side of one axis."""

    before: int
    after: int


def resize_axis(x: torch.Tensor, dim: int, size: int, *, mode: ResizeAxisMode) -> tuple[torch.Tensor, AxisPad]:
    """Pad or crop axis ``dim`` to ``size``.

    Pad: ``repeat_last`` appends copies of the last slice; ``symmetric`` edge-replicates
    both ends (leftover goes to the end). Crop: ``repeat_last`` drops from the end;
    ``symmetric`` drops from both ends with the same split.
    """
    length = x.shape[dim]
    if length == size:
        return x, AxisPad(0, 0)

    if length < size:
        need = size - length
        if mode == "repeat_last":
            last = x.narrow(dim, length - 1, 1)
            expand_shape = list(x.shape)
            expand_shape[dim] = need
            return torch.cat([x, last.expand(expand_shape)], dim=dim), AxisPad(0, need)

        before = need // 2
        after = need - before
        first = x.narrow(dim, 0, 1)
        last = x.narrow(dim, length - 1, 1)
        parts: list[torch.Tensor] = []
        if before:
            expand_shape = list(x.shape)
            expand_shape[dim] = before
            parts.append(first.expand(expand_shape))
        parts.append(x)
        if after:
            expand_shape = list(x.shape)
            expand_shape[dim] = after
            parts.append(last.expand(expand_shape))
        return torch.cat(parts, dim=dim), AxisPad(before, after)

    need = length - size
    if mode == "repeat_last":
        return x.narrow(dim, 0, size).contiguous(), AxisPad(0, need)

    before = need // 2
    after = need - before
    return x.narrow(dim, before, size).contiguous(), AxisPad(before, after)


def ensure_min_latent_shape(
    latent: torch.Tensor,
    min_tile_sizes: Tuple[int, int, int],
) -> tuple[torch.Tensor, tuple[AxisPad, AxisPad, AxisPad]]:
    """Pad latent ``(B, C, T, H, W)`` up to ``min_tile_sizes`` if needed."""
    min_t, min_h, min_w = min_tile_sizes
    t_pad = AxisPad(0, 0)
    h_pad = AxisPad(0, 0)
    w_pad = AxisPad(0, 0)
    x = latent
    if x.shape[2] < min_t:
        x, t_pad = resize_axis(x, 2, min_t, mode="repeat_last")
    if x.shape[3] < min_h:
        x, h_pad = resize_axis(x, 3, min_h, mode="symmetric")
    if x.shape[4] < min_w:
        x, w_pad = resize_axis(x, 4, min_w, mode="symmetric")
    return x, (t_pad, h_pad, w_pad)


def scale_axis_pad(pad: AxisPad, scale: int) -> AxisPad:
    return AxisPad(pad.before * scale, pad.after * scale)


def crop_pixels_to_content(
    pixels: torch.Tensor,
    frames: int,
    height: int,
    width: int,
    *,
    h_pad: AxisPad | None = None,
    w_pad: AxisPad | None = None,
    spatial_scale: Tuple[int, int] = (1, 1),
) -> torch.Tensor:
    """Crop padded decode output ``(B, C, F, H, W)`` back to the content shape.

    Temporal pad is always trailing. Spatial size-floor pads must pass the recorded
    ``h_pad`` / ``w_pad`` (latent units) plus ``spatial_scale``; otherwise the crop is centered.
    """
    x, _ = resize_axis(pixels, 2, frames, mode="repeat_last")
    scale_h, scale_w = spatial_scale
    if h_pad is not None:
        x = x.narrow(3, scale_axis_pad(h_pad, scale_h).before, height).contiguous()
    else:
        x, _ = resize_axis(x, 3, height, mode="symmetric")
    if w_pad is not None:
        x = x.narrow(4, scale_axis_pad(w_pad, scale_w).before, width).contiguous()
    else:
        x, _ = resize_axis(x, 4, width, mode="symmetric")
    return x


def stage5_pixel_shape_from_stage4(
    stage4_t: int,
    stage4_h: int,
    stage4_w: int,
    *,
    upsample_stride: Tuple[int, int, int],
    patch_size: int,
    stage5_kernel_t: int,
    drop_leading_frame: bool,
    pad_trailing: bool,
) -> tuple[int, int, int]:
    """Pixel ``(F, H, W)`` for a stage-4-input extent (one remaining NA hop + patch)."""
    st, sh, sw = upsample_stride
    frames = stage4_t * st - 1 if drop_leading_frame and st == 2 else stage4_t * st
    if pad_trailing:
        frames = max(frames, stage5_kernel_t)
    return frames, stage4_h * sh * patch_size, stage4_w * sw * patch_size


def pad_trailing_latent_for_natten_border(latent: torch.Tensor, n_frames: int) -> torch.Tensor:
    """Replicate the last latent frame ``n_frames`` times for the NA last-frame border."""
    if n_frames <= 0:
        return latent
    padded, _ = resize_axis(latent, 2, latent.shape[2] + n_frames, mode="repeat_last")
    return padded


def crop_trailing_context_natten_pad(
    context: torch.Tensor,
    *,
    n_latent_frames: int,
    time_scale: int,
    stage5_kernel_t: int,
) -> torch.Tensor:
    """Crop the ghosting appendix before stage 5, leaving at least ``stage5_kernel_t``."""
    if n_latent_frames <= 0:
        return context
    ghost = n_latent_frames * time_scale
    content_t = max(context.shape[1] - ghost, 1)
    keep = min(context.shape[1], max(content_t, stage5_kernel_t))
    cropped, _ = resize_axis(context, 1, keep, mode="repeat_last")
    return cropped


def weight_floor(dtype: torch.dtype) -> float:
    """Smallest divisor that safely guards ``buffer / weights`` in ``dtype``."""
    return max(1e-8, torch.finfo(dtype).tiny)


def stage4_thw_from_latent(
    upsample_strides: Sequence[Tuple[int, int, int]],
    latent_t: int,
    latent_h: int,
    latent_w: int,
    *,
    drop_leading_frame: bool,
) -> Tuple[int, int, int]:
    """Stage-4 input ``(T, H, W)`` after the first three upsample hops."""
    t, h, w = latent_t, latent_h, latent_w
    for st, sh, sw in upsample_strides[:3]:
        t, h, w = t * st, h * sh, w * sw
        if st == 2 and drop_leading_frame:
            t -= 1
    return t, h, w


def stage4_to_pixel_scale_factors(upsample_stride: Tuple[int, int, int], patch_size: int) -> SpatioTemporalScaleFactors:
    """Pixel/frame units per stage-4-input cell (last NA hop + unpatchify)."""
    st, sh, sw = upsample_stride
    return SpatioTemporalScaleFactors(time=st, height=sh * patch_size, width=sw * patch_size)


def compute_tile_min_size(
    stage4_kernel: Tuple[int, int, int],
    stage5_kernel: Tuple[int, int, int],
    upsample3_stride: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """Min stage-4-input ``(T, H, W)`` so stages 4 and 5 each see ``>= kernel``."""
    return tuple(max(stage4_kernel[a], -(-stage5_kernel[a] // upsample3_stride[a])) for a in range(3))


def compute_tile_halos(
    stage4_kernel: Tuple[int, int, int],
    stage4_depth: int,
    stage5_kernel: Tuple[int, int, int],
    stage5_depth: int,
    upsample3_stride: Tuple[int, int, int],
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """One-sided halos in stage-4-input units for stages 4 and 5."""
    halo4 = tuple(stage4_depth * (stage4_kernel[a] // 2) for a in range(3))
    halo5 = tuple(-(-(stage5_depth * (stage5_kernel[a] // 2)) // upsample3_stride[a]) for a in range(3))
    return halo4, halo5  # type: ignore[return-value]


def all_stages_min_tile_size(
    stage_kernels: Sequence[Tuple[int, int, int]],
    upsamples: Sequence[Tuple[Tuple[int, int, int], int]],
    stage5_kernel: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """Per-axis latent-grid floor so every stage's NA sees dims ``>= kernel_size``."""
    cumulative = [(1, 1, 1)]
    t, h, w = 1, 1, 1
    for stride, _ in upsamples:
        t, h, w = t * stride[0], h * stride[1], w * stride[2]
        cumulative.append((t, h, w))
    mins = [1, 1, 1]
    for stage_i in range(len(upsamples)):
        for axis in range(3):
            mins[axis] = max(mins[axis], -(-stage_kernels[stage_i][axis] // cumulative[stage_i][axis]))
    for axis in range(3):
        mins[axis] = max(mins[axis], -(-stage5_kernel[axis] // cumulative[len(upsamples)][axis]))
    return (mins[0], mins[1], mins[2])


def pixel_tile_shape(full_shape: tuple[int, ...], out_coords: tuple[slice, ...]) -> tuple[int, ...]:
    return tuple(len(range(*coord.indices(size))) for size, coord in zip(full_shape, out_coords, strict=True))


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def recommended_pixel_overlaps(
    tile_halos: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
    pixel_scale: SpatioTemporalScaleFactors,
) -> Tuple[int, int]:
    """Stage-4/5-safe ``(temporal_overlap_frames, spatial_overlap_pixels)``."""

    def dominant(axis: int) -> int:
        return max(tile_halos[i][axis] for i in range(len(tile_halos)))

    overlap_t = _round_up(dominant(0) * pixel_scale.time, 8)
    halo_hw = max(dominant(1), dominant(2))
    overlap_hw = _round_up(halo_hw * pixel_scale.height, 32)
    return overlap_t, overlap_hw


def stage5_tokens_for_pixel_tile(tile_frames: int, tile_height: int, tile_width: int, *, patch_size: int) -> int:
    """Pre-unpatchify stage-5 token count for a pixel-space tile."""
    return tile_frames * max(1, tile_height // patch_size) * max(1, tile_width // patch_size)


def _axis_candidates(length: int, overlap: int, min_size: int, multiple: int) -> list[tuple[int, int]]:
    """``(tile_size, num_tiles)`` for every legal size on ``multiple``'s grid."""
    out: list[tuple[int, int]] = []
    max_size = max(_round_up(length, multiple), min_size)
    for size in range(min_size, max_size + multiple, multiple):
        if size <= overlap:
            continue
        out.append((size, len(split_by_size(length, size, overlap, None))))
    return out


def _propagate_interval_through_upsample_hops(
    interval: DimensionInterval,
    strides: Sequence[int],
    causal: bool,
) -> DimensionInterval:
    """Forward-propagate one interval through upsample hops on one axis.

    Multiply by ``stride``; on the causal temporal axis with ``stride == 2`` drop the
    duplicate frame (``end -= 1``; non-origin tiles also ``start -= 1``).
    """
    x = interval
    for stride in strides:
        start = x.start * stride
        end = x.end * stride
        if causal and stride == 2:
            end -= 1
            if x.start != 0:
                start -= 1
        x = DimensionInterval(start=start, end=end, left_ramp=x.left_ramp * stride, right_ramp=x.right_ramp * stride)
    return x
