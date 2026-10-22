"""Tile configs and launch geometry of jasna's neighborhood-attention kernel, without Triton: shared by
the JIT launcher (``na_triton``) and the ahead-of-time AMD launcher (``na_hip``)."""

import math
from typing import NamedTuple

import torch


class NaConfig(NamedTuple):
    tile: tuple[int, int, int]
    block_n: int
    num_warps: int
    num_stages: int
    box_rows: bool


# Fastest in scripts/benchmark_na3d.py, stage 5 (11x11x11 windows) and stages 1-4 (3x5x5, 3x7x7):
# on an RTX 5090 the flattened box walk, on an R9700 (gfx1201) one box row per KV chunk.
LARGE_WINDOW = NaConfig((4, 4, 4), 32, 2, 3, False)
SMALL_WINDOW = NaConfig((2, 4, 4), 32, 2, 2, False)
AMD_LARGE_WINDOW = NaConfig((4, 2, 4), 16, 2, 1, True)
AMD_SMALL_WINDOW = NaConfig((1, 4, 4), 16, 1, 1, True)
CONFIGS = (LARGE_WINDOW, SMALL_WINDOW, AMD_LARGE_WINDOW, AMD_SMALL_WINDOW)
LARGE_WINDOW_MIN_VOLUME = 512


def config_for(kernel_size: tuple[int, int, int], amd: bool) -> NaConfig:
    large = math.prod(kernel_size) >= LARGE_WINDOW_MIN_VOLUME
    if amd:
        return AMD_LARGE_WINDOW if large else AMD_SMALL_WINDOW
    return LARGE_WINDOW if large else SMALL_WINDOW


class NaLaunch(NamedTuple):
    grid: tuple[int, int]
    scalars: tuple[int, ...]
    """The kernel's int arguments after its four pointers, in order."""
    kernel: tuple[int, int, int]
    """The window clamped to the grid, the kernel's ``KERNEL_T/H/W``."""


def na_launch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: tuple[int, int, int],
    query_w_start: int,
    tile: tuple[int, int, int],
    block_n: int,
    box_rows: bool,
) -> NaLaunch:
    """Validated launch geometry for contiguous ``(B, T, H, W, NH, HD)`` K/V and the queries of W
    columns ``[query_w_start, query_w_start + q.shape[3])``."""
    batch, t, h, w, num_heads, _head_dim = k.shape
    query_w_len = q.shape[3]
    if v.shape != k.shape or q.shape[:3] != k.shape[:3] or q.shape[4:] != k.shape[4:]:
        raise ValueError(f"q {tuple(q.shape)} must match k/v {tuple(k.shape)} in all but W, and k must match v")
    if not 0 <= query_w_start <= w - query_w_len:
        raise ValueError(f"query columns [{query_w_start}, {query_w_start + query_w_len}) exceed W={w}")
    if math.prod(k.shape[1:]) >= 2**31:
        raise ValueError(f"one batch item of {tuple(k.shape)} overflows int32 offsets")
    kernel = tuple(min(window, length) for window, length in zip(kernel_size, (t, h, w), strict=True))
    if box_rows and tile[2] + kernel[2] - 1 > block_n:
        raise ValueError(f"a box row of tile width {tile[2]} and window {kernel[2]} exceeds BLOCK_N={block_n}")
    tiles_t, tiles_h, tiles_w = (-(-length // size) for length, size in zip((t, h, query_w_len), tile, strict=True))
    scalars = (t, h, w, query_w_start, query_w_len, num_heads, tiles_h, tiles_w, *q.stride()[:4], *k.stride()[:4])
    return NaLaunch((tiles_t * tiles_h * tiles_w, batch * num_heads), scalars, kernel)
