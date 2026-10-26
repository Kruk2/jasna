"""The diffusion blocks' fused kernels (``diffusion_triton``) built ahead of time for AMD
(``scripts/build_ltx_vae_hipfbs.py``), for GPUs without Triton: Windows ROCm ships none."""

import torch

from jasna.media.cuda_kernel import aot_kernel
from jasna.models.ltx_vae.hip_launch import KERNEL_LAUNCH
from jasna.models.ltx_vae.transformer import (
    DIFFUSION_DIM,
    DIFFUSION_HEAD_DIM,
    DIFFUSION_HIDDEN,
    DIFFUSION_ROPE_PAIRS,
    FUSED_QKV_DIMS,
)

# Launch configs, fastest on a Radeon AI PRO R9700 (gfx1201). Residual SwiGLU: rows per program,
# hidden columns per step, warps (swept 64/128 rows, 32/64 columns, 4/8 warps, 1/2 stages). Slab
# Q/K/V: rows per program and warps (swept 16-128 rows, 4-16 warps, at widths 256 and 512).
SWIGLU_BLOCK_ROWS = 128
SWIGLU_BLOCK_HIDDEN = 64
SWIGLU_WARPS = 8
QKV_BLOCK_ROWS = 32
QKV_WARPS = 8
SWIGLU_MODULE = f"ltx_swiglu_{DIFFUSION_DIM}x{DIFFUSION_HIDDEN}"


def qkv_module(dim: int) -> str:
    return f"ltx_slab_qkv_{dim}x{DIFFUSION_HEAD_DIM}"


_SWIGLU_POINTERS = ("x_ptr", "norm_ptr", "scale_ptr", "shift_ptr", "gate_ptr", "up_ptr", "down_ptr")
SWIGLU_ARGUMENTS = {**{name: "*bf16" for name in _SWIGLU_POINTERS}, "rows": "i32"}
_QKV_BF16 = ("norm_ptr", "scale_ptr", "shift_ptr", "wq_ptr", "bq_ptr", "wk_ptr", "bk_ptr", "wv_ptr", "bv_ptr", "qn_ptr", "kn_ptr")
_QKV_FP32 = ("inv_ptr", "wpos_ptr")
_QKV_OUT = ("q_ptr", "k_ptr", "v_ptr")
_QKV_INTS = ("rows", "H", "W", "EXT", "halo", "core")
QKV_ARGUMENTS = {
    "x_ptr": "*bf16:wide", "halo_ptr": "*bf16", "col_ptr": "*i64", **{name: "*bf16" for name in _QKV_BF16},
    **{name: "*fp32" for name in _QKV_FP32}, **{name: "*bf16" for name in _QKV_OUT},
    **{name: "i32" for name in _QKV_INTS}, "q_scale": "fp32",
}


def _launch(module: str, arguments: dict[str, str], rows_per_program: int, rows: int, device: torch.device, args) -> None:
    aot_kernel(KERNEL_LAUNCH, module, device, arguments).launch(
        (-(-rows // rows_per_program), 1, 1), torch.cuda.current_stream(device).cuda_stream, *args,
    )


def _require_bf16(tensors, shapes_match: bool) -> None:
    if not shapes_match or {t.dtype for t in tensors} != {torch.bfloat16} or not all(t.is_contiguous() for t in tensors):
        raise ValueError(f"the ahead-of-time diffusion-block kernels take contiguous bf16 tensors of width {DIFFUSION_DIM}"
                         f" (Q/K/V: {' or '.join(map(str, FUSED_QKV_DIMS))})")


def residual_swiglu_(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> None:
    """``diffusion_triton.residual_swiglu_`` at the shipped decoder's shapes."""
    tensors = (x, norm_weight, scale, shift, w_gate, w_up, w_down)
    _require_bf16(tensors, x.shape[1] == DIFFUSION_DIM and w_gate.shape == (DIFFUSION_HIDDEN, DIFFUSION_DIM))
    rows = x.shape[0]
    _launch(SWIGLU_MODULE, SWIGLU_ARGUMENTS, SWIGLU_BLOCK_ROWS, rows, x.device, (*tensors, rows))


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
    """``diffusion_triton.slab_qkv`` at the shipped decoder's shapes."""
    batch, t, h, w, dim = x.shape
    _require_bf16((x, left_halo, *modulation, *weights),
                  dim in FUSED_QKV_DIMS and bool(modulation) == (dim == DIFFUSION_DIM)
                  and weights[7].shape == (DIFFUSION_HEAD_DIM,) and tuple(rope_pairs) == DIFFUSION_ROPE_PAIRS[:2]
                  and columns.dtype == torch.int64 and columns.is_contiguous())
    ext = halo + columns.numel()
    q = x.new_empty(batch, t, h, core_len, dim // DIFFUSION_HEAD_DIM, DIFFUSION_HEAD_DIM)
    k = x.new_empty(batch, t, h, ext, dim // DIFFUSION_HEAD_DIM, DIFFUSION_HEAD_DIM)
    v = torch.empty_like(k)
    rows = batch * t * h * ext
    norm, scale, shift = modulation or (x, x, x)
    pointers = (x, left_halo, columns, norm, scale, shift, *weights, inv_freqs.float().contiguous(),
                w_pos.float().contiguous(), q, k, v)
    _launch(qkv_module(dim), QKV_ARGUMENTS, QKV_BLOCK_ROWS, rows, x.device,
            (*pointers, rows, h, w, ext, halo, core_len, q_scale))
    return q, k, v
