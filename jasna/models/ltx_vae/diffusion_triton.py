"""Triton source of the decoder's fused kernels: NVIDIA JIT-compiles them with the per-GPU ``NVIDIA_LAUNCH``
configs, ``scripts/build_ltx_vae_hipfbs.py`` builds them ahead of time for AMD (``diffusion_hip`` launches them).
The torch code they replace stays the reference.

- ``_swiglu_gate_up_kernel`` + ``_swiglu_down_kernel``: ``transformer.modulated_swiglu`` plus the residual add as two
  GEMMs, the norm and modulation in the first's prologue and the SwiGLU in its epilogue, the residual add in the
  second's epilogue; the hidden activation round-trips through L2. 1.3x the compiled torch chunk on an RTX 5090,
  where one kernel keeping the hidden activation in registers is slower than torch.
- ``_proj_residual_kernel``: a W slab's attention output projection, its bias and residual add in the epilogue
  straight into the slab's columns of the block input (0.6x the GEMM plus strided add).
- ``_context_add_kernel``: a diffusion block's folded stage-4 context projection, its pixel shuffle and add in the
  epilogue straight into the block input (0.6x the compiled GEMM, shuffle and add).
- ``_slab_qkv_kernel``: a W slab's Q/K/V projections with Q/K RMSNorm and absolute RoPE (``_attn_on_w_slab``),
  gathering and modulating the slab's rows from the block input once per row tile, one head at a time.

Projection outputs are rounded to the input dtype as the torch GEMMs store them.
"""

from typing import NamedTuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

EPS = tl.constexpr(1e-6)


@triton.jit
def _swiglu_gate_up_kernel(
    x_ptr, norm_ptr, scale_ptr, shift_ptr, gate_ptr, up_ptr, h_ptr, rows,
    DIM: tl.constexpr, HIDDEN: tl.constexpr, BLOCK_ROWS: tl.constexpr, BLOCK_HIDDEN: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    dtype = x_ptr.dtype.element_ty
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = row < rows
    squares = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
    for k in tl.static_range(0, DIM, BLOCK_K):
        col = k + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + row[:, None] * DIM + col[None, :], mask=row_mask[:, None], other=0.0).to(tl.float32)
        squares += tl.sum(x * x, 1)
    inv_rms = libdevice.rsqrt(squares / DIM + EPS)
    hidden = tl.program_id(1) * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
    gate = tl.zeros((BLOCK_ROWS, BLOCK_HIDDEN), dtype=tl.float32)
    up = tl.zeros((BLOCK_ROWS, BLOCK_HIDDEN), dtype=tl.float32)
    for k in range(0, DIM, BLOCK_K):
        col = k + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + row[:, None] * DIM + col[None, :], mask=row_mask[:, None], other=0.0).to(tl.float32)
        y = (x * inv_rms[:, None] * tl.load(norm_ptr + col).to(tl.float32)[None, :]
             * (1 + tl.load(scale_ptr + col).to(tl.float32))[None, :] + tl.load(shift_ptr + col).to(tl.float32)[None, :])
        w_in = hidden[None, :] * DIM + col[:, None]
        gate = tl.dot(y.to(dtype), tl.load(gate_ptr + w_in), gate)
        up = tl.dot(y.to(dtype), tl.load(up_ptr + w_in), up)
    gate = gate.to(dtype).to(tl.float32)
    up = up.to(dtype).to(tl.float32)
    tl.store(h_ptr + row[:, None] * HIDDEN + hidden[None, :], (gate * tl.sigmoid(gate) * up).to(dtype),
             mask=row_mask[:, None])


@triton.jit
def _swiglu_down_kernel(
    h_ptr, down_ptr, x_ptr, rows,
    DIM: tl.constexpr, HIDDEN: tl.constexpr, BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    dtype = x_ptr.dtype.element_ty
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = row < rows
    col = tl.program_id(1) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    acc = tl.zeros((BLOCK_ROWS, BLOCK_COLS), dtype=tl.float32)
    for k in range(0, HIDDEN, BLOCK_K):
        hidden = k + tl.arange(0, BLOCK_K)
        h = tl.load(h_ptr + row[:, None] * HIDDEN + hidden[None, :], mask=row_mask[:, None], other=0.0)
        acc = tl.dot(h, tl.load(down_ptr + col[None, :] * HIDDEN + hidden[:, None]), acc)
    offsets = row[:, None] * DIM + col[None, :]
    x = tl.load(x_ptr + offsets, mask=row_mask[:, None], other=0.0).to(tl.float32)
    tl.store(x_ptr + offsets, (x + acc.to(dtype).to(tl.float32)).to(dtype), mask=row_mask[:, None])


@triton.jit
def _proj_residual_kernel(
    a_ptr, w_ptr, b_ptr, x_ptr, rows, core_len, x_line_stride, core_start,
    DIM: tl.constexpr, BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Row r of the (lines, core_len, DIM) NA output lands at W column core_start + r % core_len of x line r // core_len.
    dtype = x_ptr.dtype.element_ty
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = row < rows
    col = tl.program_id(1) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    acc = tl.zeros((BLOCK_ROWS, BLOCK_COLS), dtype=tl.float32)
    for k in range(0, DIM, BLOCK_K):
        channel = k + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + row[:, None] * DIM + channel[None, :], mask=row_mask[:, None], other=0.0)
        acc = tl.dot(a, tl.load(w_ptr + col[None, :] * DIM + channel[:, None]), acc)
    out = (acc + tl.load(b_ptr + col).to(tl.float32)[None, :]).to(dtype).to(tl.float32)
    x_row = (row // core_len).to(tl.int64) * x_line_stride + (core_start + row % core_len) * DIM
    offsets = x_row[:, None] + col[None, :]
    x = tl.load(x_ptr + offsets, mask=row_mask[:, None], other=0.0).to(tl.float32)
    tl.store(x_ptr + offsets, (x + out).to(dtype), mask=row_mask[:, None])


@triton.jit
def _context_add_kernel(
    f_ptr, w_ptr, b_ptr, x_ptr, rows, f_t, f_h, f_w, f_stride_b, f_stride_t, f_stride_h, f_stride_w,
    x_t, x_h, x_w, x_w_start, x_w_len, drop_frames,
    IN_DIM: tl.constexpr, DIM: tl.constexpr, STRIDE_T: tl.constexpr, STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Output column n of a program is channel n % DIM at pixel-shuffle position n // DIM (one position per program),
    # folded weight row channel * positions + position; row r is stage-4 voxel (b, t, h, w) of the feature chunk.
    dtype = x_ptr.dtype.element_ty
    positions: tl.constexpr = STRIDE_T * STRIDE_H * STRIDE_W
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = row < rows
    fw = row % f_w
    fh = row // f_w % f_h
    ft = row // (f_w * f_h) % f_t
    fb = row // (f_w * f_h * f_t)
    f_row = fb.to(tl.int64) * f_stride_b + ft * f_stride_t + fh * f_stride_h + fw * f_stride_w
    position = tl.program_id(1) * BLOCK_COLS // DIM
    channel = tl.program_id(1) * BLOCK_COLS % DIM + tl.arange(0, BLOCK_COLS)
    w_row = channel * positions + position
    acc = tl.zeros((BLOCK_ROWS, BLOCK_COLS), dtype=tl.float32)
    for k in range(0, IN_DIM, BLOCK_K):
        kk = k + tl.arange(0, BLOCK_K)
        a = tl.load(f_ptr + f_row[:, None] + kk[None, :], mask=row_mask[:, None], other=0.0)
        acc = tl.dot(a, tl.load(w_ptr + w_row[None, :] * IN_DIM + kk[:, None]), acc)
    out = (acc + tl.load(b_ptr + w_row).to(tl.float32)[None, :]).to(dtype).to(tl.float32)
    xt = ft * STRIDE_T + position // (STRIDE_H * STRIDE_W) - drop_frames
    xh = fh * STRIDE_H + position // STRIDE_W % STRIDE_H
    xw = fw * STRIDE_W + position % STRIDE_W
    keep = row_mask & (xt >= 0) & (xw < x_w_len)
    x_row = ((fb.to(tl.int64) * x_t + xt) * x_h + xh) * x_w + x_w_start + xw
    offsets = x_row[:, None] * DIM + channel[None, :]
    x = tl.load(x_ptr + offsets, mask=keep[:, None], other=0.0).to(tl.float32)
    tl.store(x_ptr + offsets, (x + out).to(dtype), mask=keep[:, None])


class NvidiaLaunch(NamedTuple):
    """Per-program (rows, columns, K step, warps, stages) of the MLP's gate/up and down kernels per width, of the
    attention projection and of the context inject, and (rows, warps) of the slab Q/K/V per width."""

    swiglu: dict[int, tuple[tuple[int, int, int, int, int], tuple[int, int, int, int, int]]]
    proj_residual: tuple[int, int, int, int, int]
    context_add: tuple[int, int, int, int, int]
    qkv: dict[int, tuple[int, int]]


NVIDIA_LAUNCH = {
    (8, 6): NvidiaLaunch(  # RTX 3090
        swiglu={256: ((64, 128, 32, 4, 3), (64, 128, 64, 8, 3)), 512: ((64, 128, 64, 4, 2), (64, 256, 32, 4, 2)),
                1024: ((64, 256, 32, 8, 3), (64, 256, 32, 4, 3))},
        proj_residual=(64, 256, 32, 4, 3), context_add=(64, 256, 64, 4, 2), qkv={256: (32, 4), 512: (64, 4)}),
    (8, 9): NvidiaLaunch(  # RTX 4090
        swiglu={256: ((64, 128, 32, 4, 3), (128, 128, 64, 4, 2)), 512: ((64, 128, 32, 4, 3), (128, 256, 64, 8, 3)),
                1024: ((64, 128, 64, 4, 2), (128, 256, 64, 8, 3))},
        proj_residual=(64, 256, 64, 4, 3), context_add=(64, 256, 64, 8, 3), qkv={256: (32, 4), 512: (64, 4)}),
    (12, 0): NvidiaLaunch(  # RTX 5090
        swiglu={256: ((64, 128, 32, 8, 3), (64, 64, 64, 4, 3)), 512: ((64, 128, 32, 4, 3), (128, 64, 64, 4, 2)),
                1024: ((64, 128, 32, 8, 2), (64, 128, 32, 4, 2))},
        proj_residual=(64, 256, 32, 4, 4), context_add=(64, 128, 32, 8, 4), qkv={256: (128, 8), 512: (64, 4)}),
}
"""Fastest of 32-128 rows x 64-256 columns x 32-64 K x 4-8 warps x 1-4 stages at the decode's shapes (Q/K/V: 16-128
rows x 2-8 warps on the real decode calls) on each GPU."""
NVIDIA_DEFAULT_LAUNCH = NvidiaLaunch(
    swiglu={256: ((64, 128, 32, 4, 3), (128, 128, 64, 4, 2)), 512: ((64, 128, 32, 4, 3), (64, 128, 64, 4, 1)),
            1024: ((64, 128, 64, 4, 2), (64, 256, 64, 4, 3))},
    proj_residual=(64, 256, 32, 4, 3), context_add=(64, 256, 64, 4, 2), qkv={256: (32, 4), 512: (64, 4)})
"""Other GPUs: per kernel the config with the least worst-case slowdown on those three (at most 1.15x)."""


def nvidia_launch(device: torch.device) -> NvidiaLaunch:
    return NVIDIA_LAUNCH.get(torch.cuda.get_device_capability(device), NVIDIA_DEFAULT_LAUNCH)


def proj_residual_(
    x: torch.Tensor, out: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, core_start: int
) -> None:
    """In-place ``x[:, :, :, core_start:core_start + core_len] += F.linear(out, weight, bias)`` for contiguous
    ``(B, T, H, W, dim)`` ``x`` and ``(B, T, H, core_len, dim)`` ``out``: the projection rounds to x's dtype
    before the add, as the torch GEMM stores it."""
    batch, t, h, w, dim = x.shape
    core_len = out.shape[3]
    rows = batch * t * h * core_len
    block_rows, block_cols, block_k, warps, stages = nvidia_launch(x.device).proj_residual
    block_cols, block_k = min(block_cols, dim), min(block_k, dim)
    _proj_residual_kernel[(triton.cdiv(rows, block_rows), dim // block_cols)](
        out, weight, bias, x, rows, core_len, w * dim, core_start, DIM=dim, BLOCK_ROWS=block_rows,
        BLOCK_COLS=block_cols, BLOCK_K=block_k, num_warps=warps, num_stages=stages,
    )


def context_add_(
    x: torch.Tensor, w_start: int, w_len: int, feat: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
    stride: tuple[int, int, int], drop_leading_frame: bool,
) -> None:
    """In-place ``transformer.add_folded_context`` of the stage-4 W chunk ``feat`` ``(B, T, H, W, in_dim)``
    (channels contiguous) into W columns ``[w_start, w_start + w_len)`` of contiguous ``x`` ``(B, T', H', W', dim)``,
    the folded projection and pixel shuffle in one kernel."""
    if not x.is_contiguous() or feat.stride(4) != 1:
        raise ValueError("context_add_ takes a contiguous x and a channels-contiguous feat")
    batch, t, h, w, in_dim = feat.shape
    dim = x.shape[4]
    rows = batch * t * h * w
    positions = stride[0] * stride[1] * stride[2]
    block_rows, block_cols, block_k, warps, stages = nvidia_launch(x.device).context_add
    block_cols, block_k = min(block_cols, dim), min(block_k, in_dim)
    _context_add_kernel[(triton.cdiv(rows, block_rows), positions * dim // block_cols)](
        feat, weight, bias, x, rows, t, h, w, *feat.stride()[:4], x.shape[1], x.shape[2], x.shape[3], w_start, w_len,
        int(stride[0] == 2 and drop_leading_frame), IN_DIM=in_dim, DIM=dim, STRIDE_T=stride[0], STRIDE_H=stride[1],
        STRIDE_W=stride[2], BLOCK_ROWS=block_rows, BLOCK_COLS=block_cols, BLOCK_K=block_k, num_warps=warps,
        num_stages=stages,
    )


def residual_swiglu_(
    x: torch.Tensor, norm_weight: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,
    w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor,
) -> None:
    """In-place ``x += w_down(silu(gate) * up)`` of ``modulate(rms_norm(x))`` for contiguous ``(rows, dim)`` ``x``,
    as ``diffusion_hip.residual_swiglu_``, through the two NVIDIA kernels."""
    rows, dim = x.shape
    hidden = w_gate.shape[0]
    h = x.new_empty(rows, hidden)
    gate_up, down = nvidia_launch(x.device).swiglu[dim]
    block_rows, block_hidden, block_k, warps, stages = gate_up
    _swiglu_gate_up_kernel[(triton.cdiv(rows, block_rows), hidden // block_hidden)](
        x, norm_weight, scale, shift, w_gate, w_up, h, rows, DIM=dim, HIDDEN=hidden, BLOCK_ROWS=block_rows,
        BLOCK_HIDDEN=block_hidden, BLOCK_K=block_k, num_warps=warps, num_stages=stages,
    )
    block_rows, block_cols, block_k, warps, stages = down
    _swiglu_down_kernel[(triton.cdiv(rows, block_rows), dim // block_cols)](
        h, w_down, x, rows, DIM=dim, HIDDEN=hidden, BLOCK_ROWS=block_rows, BLOCK_COLS=block_cols,
        BLOCK_K=block_k, num_warps=warps, num_stages=stages,
    )


@triton.jit
def _slab_qkv_kernel(
    x_ptr, halo_ptr, col_ptr, norm_ptr, scale_ptr, shift_ptr,
    wq_ptr, bq_ptr, wk_ptr, bk_ptr, wv_ptr, bv_ptr, qn_ptr, kn_ptr, inv_ptr, wpos_ptr, q_ptr, k_ptr, v_ptr,
    rows, H, W, EXT, halo, core, q_scale,
    DIM: tl.constexpr, HEAD_DIM: tl.constexpr, PAIRS_T: tl.constexpr, PAIRS_H: tl.constexpr, BLOCK_ROWS: tl.constexpr,
    MODULATE: tl.constexpr,
):
    # Rows of the (B, T, H, EXT, DIM) slab, W fastest: its first ``halo`` columns come from the
    # (B, T, H, halo, DIM) ``halo_ptr``, the rest from the (B, T, H, W, DIM) ``x_ptr`` at W ``col_ptr[w - halo]``.
    # MODULATE: the slab is ``rms_norm(x) * (1 + scale) + shift``. RoPE pair i of every head rotates by the
    # row's T position for i < PAIRS_T, its H position for the next PAIRS_H pairs, else w_pos[w].
    dtype = x_ptr.dtype.element_ty
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = row < rows
    col = tl.arange(0, DIM)
    w = row % EXT
    line = (row // EXT).to(tl.int64)
    in_halo = w < halo
    source_w = tl.load(col_ptr + (w - halo), mask=row_mask & ~in_halo, other=0)
    y = tl.load(x_ptr + (line * W + source_w)[:, None] * DIM + col[None, :],
                mask=(row_mask & ~in_halo)[:, None], other=0.0)
    y += tl.load(halo_ptr + (line * halo + w)[:, None] * DIM + col[None, :], mask=(row_mask & in_halo)[:, None], other=0.0)
    if MODULATE:
        yf = y.to(tl.float32)
        yf = yf * libdevice.rsqrt(tl.sum(yf * yf, 1) / DIM + EPS)[:, None] * tl.load(norm_ptr + col).to(tl.float32)[None, :]
        y = (yf * (1 + tl.load(scale_ptr + col).to(tl.float32))[None, :]
             + tl.load(shift_ptr + col).to(tl.float32)[None, :]).to(dtype)
    h = (row // EXT) % H
    t = row // (EXT * H)
    pair = tl.arange(0, HEAD_DIM // 2)
    position = tl.where(pair[None, :] < PAIRS_T, t[:, None].to(tl.float32),
                        tl.where(pair[None, :] < PAIRS_T + PAIRS_H, h[:, None].to(tl.float32),
                                 tl.load(wpos_ptr + w, mask=row_mask, other=0.0)[:, None]))
    angle = position * tl.load(inv_ptr + pair)[None, :]
    cos = libdevice.cos(angle)
    sin = libdevice.sin(angle)
    head_col = tl.arange(0, HEAD_DIM)
    q_mask = row_mask & (w >= halo) & (w < halo + core)
    q_row = (row // EXT) * core + (w - halo)
    for head in tl.static_range(DIM // HEAD_DIM):
        out_col = head * HEAD_DIM + head_col
        for which in tl.static_range(3):
            if which == 0:
                w_ptr, b_ptr = wq_ptr, bq_ptr
            elif which == 1:
                w_ptr, b_ptr = wk_ptr, bk_ptr
            else:
                w_ptr, b_ptr = wv_ptr, bv_ptr
            acc = tl.dot(y, tl.load(w_ptr + out_col[None, :] * DIM + col[:, None]))
            x = (acc + tl.load(b_ptr + out_col).to(tl.float32)[None, :]).to(dtype).to(tl.float32)
            if which == 2:
                tl.store(v_ptr + row[:, None] * DIM + out_col[None, :], x.to(dtype), mask=row_mask[:, None])
            else:
                norm = tl.load((qn_ptr if which == 0 else kn_ptr) + head_col).to(tl.float32)
                x = x * libdevice.rsqrt(tl.sum(x * x, 1) / HEAD_DIM + EPS)[:, None] * norm[None, :]
                if which == 0:
                    x = x * q_scale
                even, odd = tl.split(tl.reshape(x, (BLOCK_ROWS, HEAD_DIM // 2, 2)))
                rotated = tl.reshape(tl.join(even * cos - odd * sin, even * sin + odd * cos), (BLOCK_ROWS, HEAD_DIM))
                if which == 0:
                    tl.store(q_ptr + q_row[:, None] * DIM + out_col[None, :], rotated.to(dtype), mask=q_mask[:, None])
                else:
                    tl.store(k_ptr + row[:, None] * DIM + out_col[None, :], rotated.to(dtype), mask=row_mask[:, None])


@torch.library.custom_op("jasna::slab_qkv", mutates_args=())
def slab_qkv(
    x: torch.Tensor,
    left_halo: torch.Tensor,
    columns: torch.Tensor,
    modulation: list[torch.Tensor],
    weights: list[torch.Tensor],
    inv_freqs: torch.Tensor,
    w_pos: torch.Tensor,
    rope_pairs: list[int],
    core_len: int,
    halo: int,
    q_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``diffusion_hip.slab_qkv`` through the Triton JIT, any width and head dim."""
    batch, t, h, w, dim = x.shape
    ext = halo + columns.numel()
    head_dim = weights[6].shape[0]
    q = x.new_empty(batch, t, h, core_len, dim // head_dim, head_dim)
    k = x.new_empty(batch, t, h, ext, dim // head_dim, head_dim)
    v = torch.empty_like(k)
    rows = batch * t * h * ext
    norm, scale, shift = modulation or (x, x, x)
    block_rows, warps = nvidia_launch(x.device).qkv[dim]
    _slab_qkv_kernel[(triton.cdiv(rows, block_rows),)](
        x.contiguous(), left_halo.contiguous(), columns.contiguous(), norm, scale, shift,
        *weights, inv_freqs.float().contiguous(), w_pos.float().contiguous(), q, k, v,
        rows, h, w, ext, halo, core_len, q_scale,
        DIM=dim, HEAD_DIM=head_dim, PAIRS_T=rope_pairs[0], PAIRS_H=rope_pairs[1], BLOCK_ROWS=block_rows,
        MODULATE=bool(modulation), num_warps=warps, num_stages=1,
    )
    return q, k, v


@slab_qkv.register_fake
def _slab_qkv_fake(x, left_halo, columns, modulation, weights, inv_freqs, w_pos, rope_pairs, core_len, halo, q_scale):
    batch, t, h, _w, dim = x.shape
    head_dim = weights[6].shape[0]
    k = x.new_empty(batch, t, h, halo + columns.shape[0], dim // head_dim, head_dim)
    return x.new_empty(batch, t, h, core_len, dim // head_dim, head_dim), k, torch.empty_like(k)
