"""The diffusion blocks' fused kernels (``diffusion_triton``) built ahead of time for AMD
(``scripts/build_ltx_vae_hipfbs.py``), for GPUs without Triton: Windows ROCm ships none."""

import ctypes
import functools

import torch

from jasna.accelerator import amd_gpu_architecture
from jasna.media.cuda_kernel import _TRITON_SCRATCH_ARGS, Kernel

# Launch configs, fastest on a Radeon AI PRO R9700 (gfx1201). Residual SwiGLU: rows per program,
# hidden columns per step, warps (swept 64/128 rows, 32/64 columns, 4/8 warps, 1/2 stages). Slab
# Q/K/V: rows per program and warps (swept 16-128 rows, 4-16 warps, at widths 256 and 512).
SWIGLU_BLOCK_ROWS = 128
SWIGLU_BLOCK_HIDDEN = 64
SWIGLU_WARPS = 8
QKV_BLOCK_ROWS = 32
QKV_WARPS = 8
DIM = 256
HIDDEN = 1024
HEAD_DIM = 64
ROPE_PAIRS = (8, 12, 12)
"""The shipped DiffVAE decoder's diffusion blocks: width, MLP width, head dim, RoPE pairs per (T, H, W)."""
QKV_DIMS = (DIM, 512)
"""Widths the Q/K/V kernel is built for: the diffusion blocks' and the 512-wide deterministic stages 3-4
(the 1024/2048-wide stages 1-2 do not fit its one-tile projections). Only the diffusion blocks' kernel
modulates its slab."""
SWIGLU_MODULE = f"ltx_swiglu_{DIM}x{HIDDEN}"


def qkv_module(dim: int) -> str:
    return f"ltx_slab_qkv_{dim}x{HEAD_DIM}"

_SWIGLU_POINTERS = ("x_ptr", "norm_ptr", "scale_ptr", "shift_ptr", "gate_ptr", "up_ptr", "down_ptr")
SWIGLU_ARGUMENTS = {**{name: "*bf16" for name in _SWIGLU_POINTERS}, "rows": "i32"}
_SWIGLU_TYPES = (ctypes.c_void_p,) * len(_SWIGLU_POINTERS) + (ctypes.c_int32,)
_QKV_SLAB = ("x_ptr", "halo_ptr")
_QKV_BF16 = ("norm_ptr", "scale_ptr", "shift_ptr", "wq_ptr", "bq_ptr", "wk_ptr", "bk_ptr", "wv_ptr", "bv_ptr", "qn_ptr", "kn_ptr")
_QKV_FP32 = ("inv_ptr", "wpos_ptr")
_QKV_OUT = ("q_ptr", "k_ptr", "v_ptr")
_QKV_INTS = ("rows", "H", "W", "EXT", "halo", "core")
QKV_ARGUMENTS = {
    "x_ptr": "*bf16:wide", "halo_ptr": "*bf16", "col_ptr": "*i64", **{name: "*bf16" for name in _QKV_BF16},
    **{name: "*fp32" for name in _QKV_FP32}, **{name: "*bf16" for name in _QKV_OUT},
    **{name: "i32" for name in _QKV_INTS}, "q_scale": "fp32",
}
_QKV_TYPES = (ctypes.c_void_p,) * (len(_QKV_SLAB) + 1 + len(_QKV_BF16) + len(_QKV_FP32) + len(_QKV_OUT)) + (
    ctypes.c_int32,) * len(_QKV_INTS) + (ctypes.c_float,)


@functools.cache
def _kernel(architecture: str, module: str, arg_types: tuple) -> tuple[Kernel, int]:
    from jasna.models.ltx_vae.hip_launch import KERNEL_LAUNCH

    kernels = KERNEL_LAUNCH.get(architecture, {})
    if module not in kernels:
        raise RuntimeError(f"no ahead-of-time kernel {module} on {architecture}")
    function, _grid, threads, shared = kernels[module]
    return Kernel(module, function, (*arg_types, *(ctypes.c_void_p,) * _TRITON_SCRATCH_ARGS), shared), threads


def _launch(module: str, arg_types: tuple, rows_per_program: int, rows: int, device: torch.device, args) -> None:
    kernel, threads = _kernel(amd_gpu_architecture(device), module, arg_types)
    kernel.launch(
        (-(-rows // rows_per_program), 1, 1), (threads, 1, 1), (*args, *(0,) * _TRITON_SCRATCH_ARGS),
        torch.cuda.current_stream(device).cuda_stream,
    )


def _require_bf16(tensors, shapes_match: bool) -> None:
    if not shapes_match or {t.dtype for t in tensors} != {torch.bfloat16} or not all(t.is_contiguous() for t in tensors):
        raise ValueError(f"the ahead-of-time diffusion-block kernels take contiguous bf16 tensors of width {DIM}"
                         f" (Q/K/V: {' or '.join(map(str, QKV_DIMS))})")


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
    _require_bf16(tensors, x.shape[1] == DIM and w_gate.shape == (HIDDEN, DIM))
    rows = x.shape[0]
    _launch(SWIGLU_MODULE, _SWIGLU_TYPES, SWIGLU_BLOCK_ROWS, rows, x.device, (*(t.data_ptr() for t in tensors), rows))


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
                  dim in QKV_DIMS and bool(modulation) == (dim == DIM) and weights[7].shape == (HEAD_DIM,)
                  and tuple(rope_pairs) == ROPE_PAIRS[:2] and columns.dtype == torch.int64 and columns.is_contiguous())
    ext = halo + columns.numel()
    q = x.new_empty(batch, t, h, core_len, dim // HEAD_DIM, HEAD_DIM)
    k = x.new_empty(batch, t, h, ext, dim // HEAD_DIM, HEAD_DIM)
    v = torch.empty_like(k)
    rows = batch * t * h * ext
    norm, scale, shift = modulation or (x, x, x)
    pointers = (x, left_halo, columns, norm, scale, shift, *weights, inv_freqs.float().contiguous(),
                w_pos.float().contiguous(), q, k, v)
    _launch(qkv_module(dim), _QKV_TYPES, QKV_BLOCK_ROWS, rows, x.device,
            (*(p.data_ptr() for p in pointers), rows, h, w, ext, halo, core_len, q_scale))
    return q, k, v
