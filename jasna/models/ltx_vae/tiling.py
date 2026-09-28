"""Video-grid tiling primitives (subset of ``ltx_core/tiling.py`` and ``ltx_core/types.py``)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple, Sequence

import torch


class SpatioTemporalScaleFactors(NamedTuple):
    """Spatiotemporal downscaling between decoded video space and the VAE latent grid."""

    time: int
    height: int
    width: int

    @classmethod
    def default(cls) -> SpatioTemporalScaleFactors:
        return cls(time=8, height=32, width=32)

    @classmethod
    def from_blocks(cls, blocks: list, patch_size: int) -> SpatioTemporalScaleFactors:
        """Each ``compress_*`` block halves its target axes; patchify adds ``patch_size`` spatially."""
        spatial_steps = 0
        temporal_steps = 0
        for block_name, _ in blocks:
            if block_name.startswith(("compress_space", "compress_all")):
                spatial_steps += 1
            if block_name.startswith(("compress_time", "compress_all")):
                temporal_steps += 1
        spatial = patch_size * (2**spatial_steps)
        return cls(time=2**temporal_steps, height=spatial, width=spatial)


VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()


class VideoLatentShape(NamedTuple):
    """``(batch, channels, frames, height, width)`` of a video latent."""

    batch: int
    channels: int
    frames: int
    height: int
    width: int

    def to_torch_shape(self) -> torch.Size:
        return torch.Size([self.batch, self.channels, self.frames, self.height, self.width])

    @staticmethod
    def from_torch_shape(shape: torch.Size) -> VideoLatentShape:
        return VideoLatentShape(batch=shape[0], channels=shape[1], frames=shape[2], height=shape[3], width=shape[4])

    def upscale(self, scale_factors: SpatioTemporalScaleFactors) -> VideoLatentShape:
        return self._replace(
            channels=3,
            frames=(self.frames - 1) * scale_factors.time + 1,
            height=self.height * scale_factors.height,
            width=self.width * scale_factors.width,
        )


def compute_trapezoidal_mask_1d(
    length: int,
    ramp_left: int,
    ramp_right: int,
    left_starts_from_0: bool,
) -> torch.Tensor:
    """1D blending mask with linear fade-in / fade-out ramps, values in [0, 1]."""
    if length <= 0:
        raise ValueError("Mask length must be positive.")

    ramp_left = max(0, min(ramp_left, length))
    ramp_right = max(0, min(ramp_right, length))

    mask = torch.ones(length)

    if ramp_left > 0:
        interval_length = ramp_left + 1 if left_starts_from_0 else ramp_left + 2
        fade_in = torch.linspace(0.0, 1.0, interval_length)[:-1]
        if not left_starts_from_0:
            fade_in = fade_in[1:]
        mask[:ramp_left] *= fade_in

    if ramp_right > 0:
        fade_out = torch.linspace(1.0, 0.0, steps=ramp_right + 2)[1:-1]
        mask[-ramp_right:] *= fade_out

    return mask.clamp_(0, 1)


@dataclass(frozen=True)
class DimensionInterval:
    start: int
    end: int
    left_ramp: int
    right_ramp: int


def untiled_mask_1d() -> torch.Tensor:
    """Length-1 ones that broadcast over an untiled axis."""
    return torch.ones(1)


def _grow_last_tile_to_min(intervals: list[DimensionInterval], min_tile_size: int) -> list[DimensionInterval]:
    """Grow a short last tile left to ``min_tile_size``; widen penultimate ``right_ramp``."""
    if len(intervals) <= 1:
        return list(intervals)
    last = intervals[-1]
    if last.end - last.start >= min_tile_size:
        return list(intervals)
    new_start = last.end - min_tile_size
    prev = intervals[-2]
    new_overlap = prev.end - new_start
    return [
        *intervals[:-2],
        replace(prev, right_ramp=new_overlap),
        replace(last, start=new_start, left_ramp=new_overlap),
    ]


def split_by_size(dimension_size: int, size: int, overlap: int, min_tile_size: int | None) -> list[DimensionInterval]:
    """Split ``dimension_size`` into tiles of ``size`` sharing ``overlap`` elements.

    The last tile may be shorter; with ``min_tile_size`` it is grown leftward instead.
    """
    if (min_tile_size is not None and dimension_size < min_tile_size) or dimension_size <= size:
        return [DimensionInterval(start=0, end=dimension_size, left_ramp=0, right_ramp=0)]
    amount = (dimension_size + size - 2 * overlap - 1) // (size - overlap)
    intervals = [
        DimensionInterval(start=0, end=size, left_ramp=0, right_ramp=overlap),
        *(
            DimensionInterval(
                start=i * (size - overlap),
                end=i * (size - overlap) + size,
                left_ramp=overlap,
                right_ramp=overlap,
            )
            for i in range(1, amount - 1)
        ),
        DimensionInterval(start=(amount - 1) * (size - overlap), end=dimension_size, left_ramp=overlap, right_ramp=0),
    ]
    if min_tile_size is not None:
        intervals = _grow_last_tile_to_min(intervals, min_tile_size)
    return intervals


class Tile(NamedTuple):
    """``in_coords`` cut the tile from the input, ``out_coords`` place its output; ``masks_1d`` blend it."""

    in_coords: tuple[slice, ...]
    out_coords: tuple[slice, ...]
    masks_1d: tuple[torch.Tensor, ...]


def scale_by_masks_1d(x: torch.Tensor, masks_1d: Sequence[torch.Tensor]) -> torch.Tensor:
    """Multiply ``x`` by separable 1d masks with broadcasting (one mask per axis)."""
    out = x
    for axis, mask in enumerate(masks_1d):
        view_shape = [1] * x.ndim
        view_shape[axis] = -1
        out = out * mask.reshape(*view_shape)
    return out


def masks_are_complementary(tiles: Sequence[Tile], full_shape: Sequence[int]) -> bool:
    """Whether per-axis 1d blend masks partition unity, so blending needs no weight buffer."""
    for axis, length in enumerate(full_shape):
        acc = torch.zeros(length, dtype=torch.float32, device="cpu")
        seen: set[tuple[int | None, int | None]] = set()
        for tile in tiles:
            sl = tile.out_coords[axis]
            key = (sl.start, sl.stop)
            if key in seen:
                continue
            seen.add(key)
            acc[sl] += tile.masks_1d[axis].detach().float().cpu()
        if not torch.allclose(acc, torch.ones(length, dtype=torch.float32), atol=1e-5, rtol=0.0):
            return False
    return True


def group_tiles_by_temporal_slice(tiles: list[Tile]) -> list[list[Tile]]:
    """Group consecutive tiles sharing a temporal ``out_coords`` slice (temporal axis varies slowest)."""
    groups = []
    current_slice = tiles[0].out_coords[2]
    current_group = []
    for tile in tiles:
        tile_slice = tile.out_coords[2]
        if tile_slice == current_slice:
            current_group.append(tile)
        else:
            groups.append(current_group)
            current_slice = tile_slice
            current_group = [tile]
    groups.append(current_group)
    return groups


@dataclass(frozen=True)
class DimensionSizeConfig:
    """Tile size and overlap for one video axis, in pixel / frame units."""

    tile_size: int
    overlap: int


@dataclass(frozen=True)
class TileSizeConfig:
    """Size-based tiling layout for a ``(F, H, W)`` video."""

    frames: DimensionSizeConfig
    height: DimensionSizeConfig
    width: DimensionSizeConfig

    def split_sizes(
        self, scale_factors: SpatioTemporalScaleFactors
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        """``(tile, overlap)`` per ``(T, H, W)`` axis on the grid scaled down by ``scale_factors``."""

        def axis(cfg: DimensionSizeConfig, factor: int) -> tuple[int, int]:
            overlap = cfg.overlap // factor
            return max(max(2, overlap + 1), cfg.tile_size // factor), overlap

        return (
            axis(self.frames, scale_factors.time),
            axis(self.height, scale_factors.height),
            axis(self.width, scale_factors.width),
        )
