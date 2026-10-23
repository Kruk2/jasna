"""Fused Triton kernels of the decoder's diffusion blocks, used on AMD (``transformer.enable_fused_kernels``);
NVIDIA keeps the compiled torch code they replace, which stays the reference.

- ``residual_swiglu_``: ``transformer.modulated_swiglu`` plus the residual add, ``x += w_down(silu(gate) * up)``
  with ``gate, up = w_gate(y), w_up(y)`` and ``y = rms_norm(x) * (1 + scale) + shift``. Each program keeps
  its rows' ``y`` and output accumulator in registers and walks the hidden width a few columns at a time,
  so the hidden activation never reaches memory (1.5x the compiled torch chunk on an R9700).
- ``slab_qkv``: a W slab's Q/K/V projections with Q/K RMSNorm and absolute RoPE (``_attn_on_w_slab``),
  gathering and modulating the slab's rows from the block input once per row tile, one head at a time.

Projection outputs are rounded to the input dtype as the torch GEMMs store them.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from jasna.models.ltx_vae import diffusion_hip as aot

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


def residual_swiglu_(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> None:
    """In place on contiguous ``x`` ``(rows, dim)``: one row each of ``scale``/``shift`` ``(dim,)``,
    ``w_gate``/``w_up`` ``(hidden, dim)``, ``w_down`` ``(dim, hidden)``."""
    rows, dim = x.shape
    _residual_swiglu_kernel[(triton.cdiv(rows, aot.SWIGLU_BLOCK_ROWS),)](
        x, norm_weight, scale, shift, w_gate, w_up, w_down, rows,
        DIM=dim, HIDDEN=w_gate.shape[0], BLOCK_ROWS=aot.SWIGLU_BLOCK_ROWS, BLOCK_HIDDEN=aot.SWIGLU_BLOCK_HIDDEN,
        num_warps=aot.SWIGLU_WARPS, num_stages=1,
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
    """Q of the core columns ``[halo, halo + core_len)`` (scaled by ``q_scale``) and K/V of every column of
    the W slab ``(B, T, H, halo + len(columns), DIM)`` made of ``left_halo`` ``(B, T, H, halo, DIM)`` then
    ``x``'s ``(B, T, H, W, DIM)`` W ``columns``, as ``(B, T, H, W, heads, head_dim)``. ``modulation``: empty,
    or the norm weight, scale and shift ``(DIM,)`` the slab is modulated with (``rms_norm(slab) * (1 + scale)
    + shift``). ``weights``: to_q weight, bias, to_k weight, bias, to_v weight, bias, q_norm weight, k_norm
    weight; ``inv_freqs`` the (T, H, W) RoPE frequencies, ``rope_pairs`` the T and H pair counts, ``w_pos``
    each slab column's W."""
    batch, t, h, w, dim = x.shape
    ext = halo + columns.numel()
    head_dim = weights[6].shape[0]
    q = x.new_empty(batch, t, h, core_len, dim // head_dim, head_dim)
    k = x.new_empty(batch, t, h, ext, dim // head_dim, head_dim)
    v = torch.empty_like(k)
    rows = batch * t * h * ext
    norm, scale, shift = modulation or (x, x, x)
    _slab_qkv_kernel[(triton.cdiv(rows, aot.QKV_BLOCK_ROWS),)](
        x.contiguous(), left_halo.contiguous(), columns.contiguous(), norm, scale, shift,
        *weights, inv_freqs.float().contiguous(), w_pos.float().contiguous(), q, k, v,
        rows, h, w, ext, halo, core_len, q_scale,
        DIM=dim, HEAD_DIM=head_dim, PAIRS_T=rope_pairs[0], PAIRS_H=rope_pairs[1], BLOCK_ROWS=aot.QKV_BLOCK_ROWS,
        MODULATE=bool(modulation), num_warps=aot.QKV_WARPS, num_stages=1,
    )
    return q, k, v


@slab_qkv.register_fake
def _slab_qkv_fake(x, left_halo, columns, modulation, weights, inv_freqs, w_pos, rope_pairs, core_len, halo, q_scale):
    batch, t, h, _w, dim = x.shape
    head_dim = weights[6].shape[0]
    k = x.new_empty(batch, t, h, halo + columns.shape[0], dim // head_dim, head_dim)
    return x.new_empty(batch, t, h, core_len, dim // head_dim, head_dim), k, torch.empty_like(k)
