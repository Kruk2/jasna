"""The diffusion blocks' residual SwiGLU MLP through ``swiglu_triton``'s kernel built ahead of time for AMD
(``scripts/build_ltx_vae_hipfbs.py``), for GPUs without Triton: Windows ROCm ships none."""

import ctypes
import functools

import torch

from jasna.accelerator import amd_gpu_architecture
from jasna.media.cuda_kernel import _TRITON_SCRATCH_ARGS, Kernel

# Rows per program, hidden columns per step and warps: fastest on a Radeon AI PRO R9700 (gfx1201)
# in a sweep over 64/128 rows, 32/64 hidden columns, 4/8 warps and 1/2 stages.
BLOCK_ROWS = 128
BLOCK_HIDDEN = 64
NUM_WARPS = 8
SHAPES = ((256, 1024),)
"""(dim, hidden) of the shipped DiffVAE decoder's diffusion-block MLP."""

_POINTERS = ("x_ptr", "norm_ptr", "scale_ptr", "shift_ptr", "gate_ptr", "up_ptr", "down_ptr")
KERNEL_ARGUMENTS = {**{name: "*bf16" for name in _POINTERS}, "rows": "i32"}
_ARG_TYPES = (ctypes.c_void_p,) * len(_POINTERS) + (ctypes.c_int32,) + (ctypes.c_void_p,) * _TRITON_SCRATCH_ARGS


def module_name(shape: tuple[int, int]) -> str:
    return "ltx_swiglu_{}x{}".format(*shape)


@functools.cache
def _kernel(architecture: str, shape: tuple[int, int]) -> tuple[Kernel, int]:
    from jasna.models.ltx_vae.hip_launch import KERNEL_LAUNCH

    kernels = KERNEL_LAUNCH.get(architecture, {})
    if module_name(shape) not in kernels:
        raise RuntimeError(f"no ahead-of-time SwiGLU kernel for {shape} on {architecture}")
    function, _grid, threads, shared = kernels[module_name(shape)]
    return Kernel(module_name(shape), function, _ARG_TYPES, shared), threads


def residual_swiglu_(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> None:
    """``swiglu_triton.residual_swiglu_`` for bf16 at the shipped decoder's shapes."""
    tensors = (x, norm_weight, scale, shift, w_gate, w_up, w_down)
    if {t.dtype for t in tensors} != {torch.bfloat16} or not all(t.is_contiguous() for t in tensors):
        raise ValueError("the ahead-of-time SwiGLU kernel takes contiguous bf16 tensors")
    kernel, threads = _kernel(amd_gpu_architecture(x.device), (x.shape[1], w_gate.shape[0]))
    rows = x.shape[0]
    kernel.launch(
        (-(-rows // BLOCK_ROWS), 1, 1), (threads, 1, 1),
        (*(t.data_ptr() for t in tensors), rows, *(0,) * _TRITON_SCRATCH_ARGS),
        torch.cuda.current_stream(x.device).cuda_stream,
    )
