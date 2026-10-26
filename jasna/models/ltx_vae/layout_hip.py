"""``layout_triton``'s channels-first copy built ahead of time for AMD (``scripts/build_ltx_vae_hipfbs.py``)."""

import torch

from jasna.media.cuda_kernel import aot_kernel
from jasna.models.ltx_vae.hip_launch import KERNEL_LAUNCH

MODULE = "ltx_channels_first_fp16_bf16"
BLOCK_S = 64
BLOCK_C = 64
WARPS = 4
ARGUMENTS = {"x_ptr": "*fp16:wide", "out_ptr": "*bf16:wide", "S": "i32", "C": "i32"}


def channels_first_bf16(x: torch.Tensor) -> torch.Tensor:
    """A channels-last 3D fp16 ``x`` as a contiguous bf16 tensor."""
    if x.dtype != torch.float16 or x.dim() != 5 or not x.is_contiguous(memory_format=torch.channels_last_3d):
        raise ValueError(f"the ahead-of-time channels-first copy takes channels-last 3D fp16, got {x.dtype} {x.stride()}")
    n, c = x.shape[:2]
    s = x[0, 0].numel()
    out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    kernel = aot_kernel(KERNEL_LAUNCH, MODULE, x.device, ARGUMENTS)
    kernel.launch((-(-s // BLOCK_S), -(-c // BLOCK_C), n), torch.cuda.current_stream(x.device).cuda_stream, x, out, s, c)
    return out
