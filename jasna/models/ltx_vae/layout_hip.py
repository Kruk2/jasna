"""``layout_triton``'s channels-first copy built ahead of time for AMD (``scripts/build_ltx_vae_hipfbs.py``)."""

import ctypes
import functools

import torch

from jasna.accelerator import amd_gpu_architecture
from jasna.media.cuda_kernel import _TRITON_SCRATCH_ARGS, Kernel

MODULE = "ltx_channels_first_fp16_bf16"
BLOCK_S = 64
BLOCK_C = 64
WARPS = 4
ARGUMENTS = {"x_ptr": "*fp16:wide", "out_ptr": "*bf16:wide", "S": "i32", "C": "i32"}
_ARG_TYPES = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, *(ctypes.c_void_p,) * _TRITON_SCRATCH_ARGS)


@functools.cache
def _kernel(architecture: str) -> tuple[Kernel, int]:
    from jasna.models.ltx_vae.hip_launch import KERNEL_LAUNCH

    kernels = KERNEL_LAUNCH.get(architecture, {})
    if MODULE not in kernels:
        raise RuntimeError(f"no ahead-of-time kernel {MODULE} on {architecture}")
    function, _grid, threads, shared = kernels[MODULE]
    return Kernel(MODULE, function, _ARG_TYPES, shared), threads


def channels_first_bf16(x: torch.Tensor) -> torch.Tensor:
    """A channels-last 3D fp16 ``x`` as a contiguous bf16 tensor."""
    if x.dtype != torch.float16 or x.dim() != 5 or not x.is_contiguous(memory_format=torch.channels_last_3d):
        raise ValueError(f"the ahead-of-time channels-first copy takes channels-last 3D fp16, got {x.dtype} {x.stride()}")
    n, c = x.shape[:2]
    s = x[0, 0].numel()
    out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    kernel, threads = _kernel(amd_gpu_architecture(x.device))
    kernel.launch((-(-s // BLOCK_S), -(-c // BLOCK_C), n), (threads, 1, 1),
                  (x.data_ptr(), out.data_ptr(), s, c, *(0,) * _TRITON_SCRATCH_ARGS),
                  torch.cuda.current_stream(x.device).cuda_stream)
    return out
