"""Triton causal 3x3x3 conv of the encoder's fp16 layers (``jasna::causal_conv3d``), JIT-compiled for
the convs ``video_encoder.ENCODER_HEADS`` lists for the GPU (NVIDIA ones so far).

An implicit GEMM over channels-last activations that accumulates in fp16, as comfy-kitchen's
``fp16_conv3d`` does (GeForce cards run fp16-accumulate tensor cores ~1.6x faster than fp32
accumulation). The causal repeat of the first frame and the zero border are read in place, where
kitchen needs a padded copy of its input, so it also beats kitchen on shapes both run, and it takes
the 1024-channel layers kitchen declines. Every launch sums each output in the same order (tap by
tap, input channels in order), so its tile sizes change only the speed. A ``promoted`` launch sums
each step's ``block_in`` channels in fp16 and adds those sums up in fp32: ~10 % slower, ~15x less
error per conv, for GPUs whose fp16 sums over all 27 x 1024 products cost accuracy (RTX 40 and 30).
Bias and residual are added in fp32 before the one rounding. ``CausalConv3d``'s torch conv stays
the reference.
"""

import torch
import triton
import triton.language as tl

TAPS = tl.constexpr(27)


@triton.jit
def _causal_conv3d_kernel(
    x_ptr, w_ptr, b_ptr, r_ptr, y_ptr, voxels, T, H, W,
    w_stride_out, w_stride_in, w_stride_t, w_stride_h, w_stride_w,
    C: tl.constexpr, K: tl.constexpr, HAS_RESIDUAL: tl.constexpr, PROMOTED: tl.constexpr,
    BLOCK_VOXELS: tl.constexpr, BLOCK_OUT: tl.constexpr, BLOCK_IN: tl.constexpr, GROUP: tl.constexpr,
):
    # Rows are output voxels (b, t, h, w) in channels-last order, columns output channels; the
    # reduction runs over the 27 taps x C input channels, BLOCK_IN channels of one tap per step.
    voxel_tiles = tl.cdiv(voxels, BLOCK_VOXELS)
    out_tiles: tl.constexpr = K // BLOCK_OUT
    pid = tl.program_id(0)
    first_voxel_tile = pid // (GROUP * out_tiles) * GROUP
    group_rows = min(voxel_tiles - first_voxel_tile, GROUP)
    voxel_tile = first_voxel_tile + pid % (GROUP * out_tiles) % group_rows
    out_tile = pid % (GROUP * out_tiles) // group_rows

    voxel = voxel_tile * BLOCK_VOXELS + tl.arange(0, BLOCK_VOXELS)
    out_channel = out_tile * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    in_channel = tl.arange(0, BLOCK_IN)
    in_range = voxel < voxels
    w = voxel % W
    h = voxel // W % H
    t = voxel // (W * H) % T
    frame0 = voxel // (W * H * T) * T

    acc = tl.zeros((BLOCK_VOXELS, BLOCK_OUT), dtype=tl.float16)
    total = tl.zeros((BLOCK_VOXELS, BLOCK_OUT), dtype=tl.float32)
    chunks: tl.constexpr = C // BLOCK_IN
    for step in tl.range(0, TAPS * chunks):
        tap = step // chunks
        c0 = step % chunks * BLOCK_IN
        dt = tap // 9
        dh = tap // 3 % 3
        dw = tap % 3
        source_t = tl.maximum(t + dt - 2, 0)
        source_h = h + dh - 1
        source_w = w + dw - 1
        inside = in_range & (source_h >= 0) & (source_h < H) & (source_w >= 0) & (source_w < W)
        source = (((frame0 + source_t) * H + source_h) * W + source_w).to(tl.int64) * C + c0
        a = tl.load(x_ptr + source[:, None] + in_channel[None, :], mask=inside[:, None], other=0.0)
        tap_offset = dt * w_stride_t + dh * w_stride_h + dw * w_stride_w
        b = tl.load(w_ptr + out_channel[None, :] * w_stride_out + (c0 + in_channel)[:, None] * w_stride_in + tap_offset)
        if PROMOTED:
            total += tl.dot(a, b, tl.zeros((BLOCK_VOXELS, BLOCK_OUT), dtype=tl.float16), out_dtype=tl.float16).to(tl.float32)
        else:
            acc = tl.dot(a, b, acc, out_dtype=tl.float16)

    if PROMOTED:
        out = total
    else:
        out = acc.to(tl.float32)
    out += tl.load(b_ptr + out_channel).to(tl.float32)[None, :]
    offsets = voxel.to(tl.int64)[:, None] * K + out_channel[None, :]
    if HAS_RESIDUAL:
        out += tl.load(r_ptr + offsets, mask=in_range[:, None], other=0.0).to(tl.float32)
    tl.store(y_ptr + offsets, out.to(tl.float16), mask=in_range[:, None])


@torch.library.custom_op("jasna::causal_conv3d", mutates_args=())
def causal_conv3d(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor | None, launch: list[int]
) -> torch.Tensor:
    """``residual + conv3d(x with its first frame repeated twice in front, weight, bias, padding (0, 1, 1))``
    for channels-last fp16 ``x`` ``(B, C, T, H, W)`` and ``residual`` ``(B, K, T, H, W)``; ``launch`` is
    ``video_encoder.ConvLaunch``'s fields."""
    batch, channels, frames, height, width = x.shape
    out_channels = weight.shape[0]
    block_voxels, block_out, block_in, warps, stages, group, promoted = launch
    y = torch.empty((batch, out_channels, frames, height, width), device=x.device, dtype=x.dtype,
                    memory_format=torch.channels_last_3d)
    voxels = batch * frames * height * width
    _causal_conv3d_kernel[(triton.cdiv(voxels, block_voxels) * (out_channels // block_out),)](
        x, weight, bias, y if residual is None else residual, y, voxels, frames, height, width,
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
        C=channels, K=out_channels, HAS_RESIDUAL=residual is not None, PROMOTED=bool(promoted), BLOCK_VOXELS=block_voxels,
        BLOCK_OUT=block_out, BLOCK_IN=block_in, GROUP=group, num_warps=warps, num_stages=stages,
    )
    return y


@causal_conv3d.register_fake
def _(x, weight, bias, residual, launch):
    batch, _, frames, height, width = x.shape
    return torch.empty((batch, weight.shape[0], frames, height, width), device=x.device, dtype=x.dtype,
                       memory_format=torch.channels_last_3d)
