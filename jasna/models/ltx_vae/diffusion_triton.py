"""Triton source of the decoder's fused diffusion-block kernels, which ``scripts/build_ltx_vae_hipfbs.py``
builds ahead of time for AMD (``diffusion_hip`` launches them); NVIDIA keeps the compiled torch code they
replace, which stays the reference.

- ``_residual_swiglu_kernel``: ``transformer.modulated_swiglu`` plus the residual add. Each program keeps
  its rows' ``y`` and output accumulator in registers and walks the hidden width a few columns at a time,
  so the hidden activation never reaches memory (1.5x the compiled torch chunk on an R9700).
- ``_slab_qkv_kernel``: a W slab's Q/K/V projections with Q/K RMSNorm and absolute RoPE (``_attn_on_w_slab``),
  gathering and modulating the slab's rows from the block input once per row tile, one head at a time.

Projection outputs are rounded to the input dtype as the torch GEMMs store them.
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice

EPS = tl.constexpr(1e-6)


@triton.jit
def _residual_swiglu_kernel(
    x_ptr, norm_ptr, scale_ptr, shift_ptr, gate_ptr, up_ptr, down_ptr, rows,
    DIM: tl.constexpr, HIDDEN: tl.constexpr, BLOCK_ROWS: tl.constexpr, BLOCK_HIDDEN: tl.constexpr,
):
    dtype = x_ptr.dtype.element_ty
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = row < rows
    col = tl.arange(0, DIM)
    offsets = row[:, None] * DIM + col[None, :]
    x = tl.load(x_ptr + offsets, mask=row_mask[:, None], other=0.0).to(tl.float32)
    y = x * libdevice.rsqrt(tl.sum(x * x, 1) / DIM + EPS)[:, None] * tl.load(norm_ptr + col).to(tl.float32)[None, :]
    y = (y * (1 + tl.load(scale_ptr + col).to(tl.float32))[None, :] + tl.load(shift_ptr + col).to(tl.float32)[None, :])
    y = y.to(dtype)
    hidden = tl.arange(0, BLOCK_HIDDEN)
    acc = tl.zeros((BLOCK_ROWS, DIM), dtype=tl.float32)
    for start in range(0, HIDDEN, BLOCK_HIDDEN):
        w_in = (start + hidden)[None, :] * DIM + col[:, None]
        gate = tl.dot(y, tl.load(gate_ptr + w_in)).to(dtype).to(tl.float32)
        up = tl.dot(y, tl.load(up_ptr + w_in)).to(dtype).to(tl.float32)
        product = (gate * tl.sigmoid(gate) * up).to(dtype)
        acc = tl.dot(product, tl.load(down_ptr + col[None, :] * HIDDEN + (start + hidden)[:, None]), acc)
    out = x + acc.to(dtype).to(tl.float32)
    tl.store(x_ptr + offsets, out.to(dtype), mask=row_mask[:, None])


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
