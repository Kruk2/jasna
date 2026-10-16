"""3D neighborhood attention forward in Triton, same semantics as ``eager_na.na3d``.

Each program owns a TILE_T x TILE_H x TILE_W block of queries of one (batch, head). Its KV
range is the union of those queries' windows ("dynamic KV tiling", GNA arXiv 2504.16922): per
axis from the window start of the tile's first query to the window end of its last, at most
tile + kernel - 1 positions. The box is walked flattened (w fastest) in BLOCK_N-token chunks
with an FA2 online softmax, each score masked to its own query's window.

Queries may cover only W columns ``[query_w_start, query_w_start + Wq)`` of the K/V grid, so a
W slab with halo columns attends from its core columns alone.
"""

import math

import torch
import triton
import triton.language as tl

LOG2E = tl.constexpr(1.4426950408889634)
MASKED_SCORE = tl.constexpr(-1.0e30)


def tile_config(tile: tuple[int, int, int], block_n: int, num_warps: int, num_stages: int) -> triton.Config:
    tile_t, tile_h, tile_w = tile
    return triton.Config(
        {"TILE_T": tile_t, "TILE_H": tile_h, "TILE_W": tile_w, "BLOCK_N": block_n},
        num_warps=num_warps,
        num_stages=num_stages,
    )


# Fastest on an RTX 5090 in scripts/benchmark_na3d.py: stage 5 (11x11x11 windows) and stages 1-4 (3x5x5, 3x7x7).
LARGE_WINDOW_CONFIG = tile_config((4, 4, 4), 32, 2, 3)
SMALL_WINDOW_CONFIG = tile_config((2, 4, 4), 32, 2, 2)
LARGE_WINDOW_MIN_VOLUME = 512


def config_for(kernel_size: tuple[int, int, int]) -> triton.Config:
    return LARGE_WINDOW_CONFIG if math.prod(kernel_size) >= LARGE_WINDOW_MIN_VOLUME else SMALL_WINDOW_CONFIG


@triton.jit
def _window_start(i, length, KERNEL: tl.constexpr):
    return tl.minimum(tl.maximum(i - KERNEL // 2, 0), length - KERNEL)


@triton.jit
def _in_window(key, start, KERNEL: tl.constexpr):
    """``start <= key < start + KERNEL`` for every (query row, key) pair, as one unsigned compare."""
    return (key[None, :] - start[:, None]).to(tl.uint32, bitcast=True) < KERNEL


@triton.jit
def _na3d_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    T, H, W, query_w_start, query_w_len, num_heads, tiles_h, tiles_w,
    q_stride_b, q_stride_t, q_stride_h, q_stride_w, kv_stride_b, kv_stride_t, kv_stride_h, kv_stride_w,
    KERNEL_T: tl.constexpr, KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr, HEAD_DIM: tl.constexpr,
    TILE_T: tl.constexpr, TILE_H: tl.constexpr, TILE_W: tl.constexpr, BLOCK_N: tl.constexpr,
):
    BLOCK_M: tl.constexpr = TILE_T * TILE_H * TILE_W
    tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    t0 = tile // (tiles_h * tiles_w) * TILE_T
    h0 = tile // tiles_w % tiles_h * TILE_H
    local_w0 = tile % tiles_w * TILE_W
    w0 = query_w_start + local_w0

    rows = tl.arange(0, BLOCK_M)
    qt = t0 + rows // (TILE_H * TILE_W)
    qh = h0 + rows // TILE_W % TILE_H
    local_qw = local_w0 + rows % TILE_W
    q_valid = (qt < T) & (qh < H) & (local_qw < query_w_len)
    # Rows past the grid edge repeat the last position so their softmax stays finite; they are not stored.
    qt = tl.minimum(qt, T - 1)
    qh = tl.minimum(qh, H - 1)
    local_qw = tl.minimum(local_qw, query_w_len - 1)
    last_w = query_w_start + query_w_len - 1

    box_t = _window_start(t0, T, KERNEL_T)
    box_h = _window_start(h0, H, KERNEL_H)
    box_w = _window_start(w0, W, KERNEL_W)
    extent_t = _window_start(tl.minimum(t0 + TILE_T - 1, T - 1), T, KERNEL_T) + KERNEL_T - box_t
    extent_h = _window_start(tl.minimum(h0 + TILE_H - 1, H - 1), H, KERNEL_H) + KERNEL_H - box_h
    extent_w = _window_start(tl.minimum(w0 + TILE_W - 1, last_w), W, KERNEL_W) + KERNEL_W - box_w
    extent_hw = extent_h * extent_w
    box_size = extent_t * extent_hw
    # Each row's window start, relative to the box.
    win_t = _window_start(qt, T, KERNEL_T) - box_t
    win_h = _window_start(qh, H, KERNEL_H) - box_h
    win_w = _window_start(query_w_start + local_qw, W, KERNEL_W) - box_w

    batch = (batch_head // num_heads).to(tl.int64)
    head_offset = batch_head % num_heads * HEAD_DIM
    q_ptr += batch * q_stride_b + head_offset
    o_ptr += batch * q_stride_b + head_offset
    k_ptr += batch * kv_stride_b + head_offset
    v_ptr += batch * kv_stride_b + head_offset
    dims = tl.arange(0, HEAD_DIM)
    q_offsets = (qt * q_stride_t + qh * q_stride_h + local_qw * q_stride_w)[:, None] + dims[None, :]
    q = tl.load(q_ptr + q_offsets)

    row_max = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    for chunk_start in range(0, box_size, BLOCK_N):
        n = chunk_start + tl.arange(0, BLOCK_N)
        n_valid = n < box_size
        n = tl.minimum(n, box_size - 1)
        kt = n // extent_hw
        rest = n - kt * extent_hw
        kh = rest // extent_w
        kw = rest - kh * extent_w
        kv_offsets = ((box_t + kt) * kv_stride_t + (box_h + kh) * kv_stride_h + (box_w + kw) * kv_stride_w)
        kv_offsets = kv_offsets[:, None] + dims[None, :]
        k = tl.load(k_ptr + kv_offsets)
        v = tl.load(v_ptr + kv_offsets)

        visible = (
            n_valid[None, :]
            & _in_window(kt, win_t, KERNEL_T)
            & _in_window(kh, win_h, KERNEL_H)
            & _in_window(kw, win_w, KERNEL_W)
        )
        scores = tl.where(visible, tl.dot(q, tl.trans(k)) * LOG2E, MASKED_SCORE)
        new_max = tl.maximum(row_max, tl.max(scores, 1))
        rescale = tl.math.exp2(row_max - new_max)
        p = tl.math.exp2(scores - new_max[:, None])
        row_sum = row_sum * rescale + tl.sum(p, 1)
        acc = tl.dot(p.to(v.dtype), v, acc * rescale[:, None])
        row_max = new_max

    out = acc / row_sum[:, None]
    tl.store(o_ptr + q_offsets, out.to(o_ptr.dtype.element_ty), mask=q_valid[:, None])


def na3d_with_config(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: tuple[int, int, int],
    query_w_start: int,
    config: triton.Config,
) -> torch.Tensor:
    batch, t, h, w, num_heads, head_dim = k.shape
    query_w_len = q.shape[3]
    if v.shape != k.shape or q.shape[:3] != k.shape[:3] or q.shape[4:] != k.shape[4:]:
        raise ValueError(f"q {tuple(q.shape)} must match k/v {tuple(k.shape)} in all but W, and k must match v")
    if not 0 <= query_w_start <= w - query_w_len:
        raise ValueError(f"query columns [{query_w_start}, {query_w_start + query_w_len}) exceed W={w}")
    if math.prod(k.shape[1:]) >= 2**31:
        raise ValueError(f"one batch item of {tuple(k.shape)} overflows int32 offsets")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    kernel_t, kernel_h, kernel_w = (min(kernel, length) for kernel, length in zip(kernel_size, (t, h, w), strict=True))
    tile_t, tile_h, tile_w = config.kwargs["TILE_T"], config.kwargs["TILE_H"], config.kwargs["TILE_W"]
    tiles_t, tiles_h, tiles_w = triton.cdiv(t, tile_t), triton.cdiv(h, tile_h), triton.cdiv(query_w_len, tile_w)
    out = torch.empty_like(q)
    _na3d_fwd_kernel[(tiles_t * tiles_h * tiles_w, batch * num_heads)](
        q, k, v, out,
        t, h, w, query_w_start, query_w_len, num_heads, tiles_h, tiles_w,
        *q.stride()[:4], *k.stride()[:4],
        KERNEL_T=kernel_t, KERNEL_H=kernel_h, KERNEL_W=kernel_w, HEAD_DIM=head_dim,
        num_warps=config.num_warps, num_stages=config.num_stages, **config.kwargs,
    )
    return out


@torch.library.custom_op("jasna::na3d", mutates_args=())
def na3d(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kernel_size: list[int], query_w_start: int
) -> torch.Tensor:
    """3D neighborhood attention over ``(B, T, H, W, NH, HD)`` K/V; Q must already be scaled.

    ``q`` holds the queries of W columns ``[query_w_start, query_w_start + q.shape[3])``.
    """
    kernel = tuple(kernel_size)
    return na3d_with_config(q, k, v, kernel, query_w_start, config_for(kernel))


@na3d.register_fake
def _na3d_fake(q, k, v, kernel_size, query_w_start):
    return torch.empty_like(q)
