"""The eager AMD encoder's head-to-tail copy (``video_encoder._layer_input``): the kitchen head's
channels-last fp16 output into the MIOpen tail's channels-first bf16 input, as a tiled transpose.
ROCm's generic strided copy runs it at ~10 GB/s; built ahead of time (``layout_hip``)."""

import triton
import triton.language as tl


@triton.jit
def _channels_first_kernel(x_ptr, out_ptr, S, C, BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr):
    # x: (N, S, C) memory (channels-last, S = D * H * W), out: (N, C, S).
    s = tl.program_id(0) * BLOCK_S + tl.arange(0, BLOCK_S)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    plane = tl.program_id(2).to(tl.int64) * S * C
    mask = (s[:, None] < S) & (c[None, :] < C)
    tile = tl.load(x_ptr + plane + s[:, None].to(tl.int64) * C + c[None, :], mask=mask)
    tl.store(out_ptr + plane + c[None, :].to(tl.int64) * S + s[:, None], tile.to(out_ptr.dtype.element_ty), mask=mask)
