"""Neighborhood attention through ``na_triton``'s kernel built ahead of time for AMD
(``scripts/build_ltx_vae_hipfbs.py``): every AMD GPU runs it, with or without Triton."""

import torch

from jasna.accelerator import amd_gpu_architecture
from jasna.media.cuda_kernel import aot_kernel
from jasna.models.ltx_vae import na_geometry
from jasna.models.ltx_vae.hip_launch import KERNEL_LAUNCH

HEAD_DIM = 64
WINDOWS = ((9, 9, 9), (3, 7, 7), (3, 5, 5))
"""The shipped DiffVAE decoder's stage-5 and stage 1-4 windows."""

_POINTERS = ("q_ptr", "k_ptr", "v_ptr", "o_ptr")
_SIZES = ("T", "H", "W", "query_w_start", "query_w_len", "num_heads", "tiles_h", "tiles_w")
_STRIDES = ("q_stride_b", "q_stride_t", "q_stride_h", "q_stride_w",
            "kv_stride_b", "kv_stride_t", "kv_stride_h", "kv_stride_w")
KERNEL_ARGUMENTS = {
    **{name: "*bf16" for name in _POINTERS},
    **{name: "i32" for name in _SIZES},
    **{name: "i32:16" for name in _STRIDES},
}
_BUFFER_LOAD_BYTES = 2**31


def module_name(window: tuple[int, int, int]) -> str:
    return "ltx_na3d_{}x{}x{}".format(*window)


@torch.library.custom_op("jasna::na3d_aot", mutates_args=())
def na3d(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kernel_size: list[int], query_w_start: int) -> torch.Tensor:
    """``na_triton.na3d`` for bf16, head dim 64; an opaque op, so a compiled slab calls it in one piece."""
    if {q.dtype, k.dtype, v.dtype} != {torch.bfloat16} or k.shape[5] != HEAD_DIM:
        raise ValueError(f"the ahead-of-time NA kernel takes bf16 with head dim {HEAD_DIM}, got {q.dtype} {tuple(k.shape)}")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    window = tuple(kernel_size)
    config = na_geometry.config_for(window, amd_gpu_architecture(q.device))
    launch = na_geometry.na_launch(q, k, v, window, query_w_start, config.tile, config.block_n, config.box_rows)
    if any(stride % 16 for stride in launch.scalars[len(_SIZES):]):
        raise ValueError(f"the ahead-of-time NA kernel needs strides divisible by 16, got {launch.scalars[len(_SIZES):]}")
    if max(t.numel() * t.element_size() for t in (q, k, v)) >= _BUFFER_LOAD_BYTES:
        raise ValueError(f"K/V {tuple(k.shape)} exceed the 2 GB range of AMD buffer loads")
    kernel = aot_kernel(KERNEL_LAUNCH, module_name(launch.kernel), q.device, KERNEL_ARGUMENTS)
    out = torch.empty_like(q)
    kernel.launch((*launch.grid, 1), torch.cuda.current_stream(q.device).cuda_stream, q, k, v, out, *launch.scalars)
    return out


@na3d.register_fake
def _na3d_fake(q, k, v, kernel_size, query_w_start):
    return torch.empty_like(q)
