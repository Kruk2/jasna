"""The diffusion blocks' residual SwiGLU MLP in one Triton kernel, same arithmetic as
``transformer.modulated_swiglu`` plus the residual add: ``x += w_down(silu(gate) * up)`` with
``gate, up = w_gate(y), w_up(y)`` and ``y = rms_norm(x) * (1 + scale) + shift``.

Each program owns ``BLOCK_ROWS`` rows: it keeps ``y`` and the output accumulator in registers and
walks the hidden width ``BLOCK_HIDDEN`` columns at a time, so the hidden activation never reaches
memory. Gate, up and the hidden product are rounded to the input dtype as the torch GEMMs store
them. Used on AMD (``transformer.enable_fused_kernels``), where it is 1.5x faster than the compiled
torch chunk; NVIDIA keeps that chunk.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from jasna.models.ltx_vae.swiglu_hip import BLOCK_HIDDEN, BLOCK_ROWS, NUM_WARPS

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
    _residual_swiglu_kernel[(triton.cdiv(rows, BLOCK_ROWS),)](
        x, norm_weight, scale, shift, w_gate, w_up, w_down, rows,
        DIM=dim, HIDDEN=w_gate.shape[0], BLOCK_ROWS=BLOCK_ROWS, BLOCK_HIDDEN=BLOCK_HIDDEN,
        num_warps=NUM_WARPS, num_stages=1,
    )
