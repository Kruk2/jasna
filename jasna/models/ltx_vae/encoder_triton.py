"""The LTX VAE encoder's element-wise layers fused into two kernels, built ahead of time for AMD
(``encoder_hip``, ``scripts/build_ltx_encoder_hipfbs.py``). Every tensor is channels-last 3D: a row is
one (batch, frame, y, x) position with its channels contiguous."""

import triton
import triton.language as tl


@triton.jit
def _norm_pad_kernel(
    x_ptr, bias_ptr, residual_ptr, sum_ptr, out_ptr, rows, T, H, W, C, lead, pad,
    stride_n, stride_t, stride_h, stride_w, stride_c,
    NORM: tl.constexpr, ADD: tl.constexpr, BLOCK_ROWS: tl.constexpr, BLOCK_C: tl.constexpr,
):
    # x: (N, T, H, W, C) positions at the given strides, bias: (C,), residual and sum: contiguous (rows, C).
    # v = x + bias (+ residual, rounded to its dtype and stored to sum when NORM), then silu(pixel_norm(v)) when NORM, goes to
    # out (N, lead + T, H + 2 pad, W + 2 pad, C) at frame lead + t, with frame 0 also in the lead frames.
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    c = tl.arange(0, BLOCK_C)
    mask = (row < rows)[:, None] & (c < C)[None, :]
    w = row % W
    h = (row // W) % H
    t = (row // (W * H)) % T
    n = (row // (W * H * T)).to(tl.int64)
    source = n * stride_n + t.to(tl.int64) * stride_t + h.to(tl.int64) * stride_h + w.to(tl.int64) * stride_w
    value = tl.load(x_ptr + source[:, None] + c[None, :].to(tl.int64) * stride_c, mask=mask, other=0.0).to(tl.float32)
    value += tl.load(bias_ptr + c, mask=c < C, other=0.0).to(tl.float32)[None, :]
    if ADD:
        dense = row.to(tl.int64)[:, None] * C + c[None, :]
        value = (value + tl.load(residual_ptr + dense, mask=mask, other=0.0).to(tl.float32)).to(residual_ptr.dtype.element_ty)
        if NORM:
            tl.store(sum_ptr + dense, value, mask=mask)
        value = value.to(tl.float32)
    if NORM:
        mean_square = tl.sum(value * value, axis=1) / C
        value = value * tl.rsqrt(mean_square + 1e-8)[:, None]
        value = value * tl.sigmoid(value)
    value = value.to(out_ptr.dtype.element_ty)
    padded_h = H + 2 * pad
    padded_w = W + 2 * pad
    plane = ((h + pad) * padded_w + w + pad).to(tl.int64)
    frames = lead + T
    target = (n * frames + lead + t) * padded_h * padded_w + plane
    tl.store(out_ptr + target[:, None] * C + c[None, :], value, mask=mask)
    for frame in range(lead):
        lead_target = (n * frames + frame) * padded_h * padded_w + plane
        tl.store(out_ptr + lead_target[:, None] * C + c[None, :], value, mask=mask & (t == 0)[:, None])


@triton.jit
def _downsample_add_kernel(
    y_ptr, bias_ptr, x_ptr, out_ptr, rows, T, H, W, stride_t, stride_h, stride_w, group,
    y_stride_n, y_stride_t, y_stride_h, y_stride_w, x_stride_n, x_stride_t, x_stride_h, x_stride_w,
    C_OUT: tl.constexpr, BLOCK_ROWS: tl.constexpr,
):
    # SpaceToDepthDownsample's tail: space_to_depth(y + bias) + the group mean of space_to_depth(x), into a
    # contiguous (N, T, H, W, C_OUT) out. y is the conv's output, x the block's input (channels at stride 1).
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    c = tl.arange(0, C_OUT)
    mask = (row < rows)[:, None]
    w = row % W
    h = (row // W) % H
    t = (row // (W * H)) % T
    n = (row // (W * H * T)).to(tl.int64)
    positions = stride_t * stride_h * stride_w
    y_row = n * y_stride_n + (t * stride_t).to(tl.int64) * y_stride_t + (h * stride_h).to(tl.int64) * y_stride_h \
        + (w * stride_w).to(tl.int64) * y_stride_w
    x_row = n * x_stride_n + (t * stride_t).to(tl.int64) * x_stride_t + (h * stride_h).to(tl.int64) * x_stride_h \
        + (w * stride_w).to(tl.int64) * x_stride_w
    offset = c % positions
    y_column = (offset // (stride_h * stride_w)) * y_stride_t + (offset // stride_w % stride_h) * y_stride_h \
        + (offset % stride_w) * y_stride_w + c // positions
    value = tl.load(y_ptr + y_row[:, None] + y_column[None, :].to(tl.int64), mask=mask).to(tl.float32)
    value += tl.load(bias_ptr + c // positions).to(tl.float32)[None, :]
    skip = tl.zeros((BLOCK_ROWS, C_OUT), dtype=tl.float32)
    for member in range(group):
        k = c * group + member
        offset = k % positions
        x_column = (offset // (stride_h * stride_w)) * x_stride_t + (offset // stride_w % stride_h) * x_stride_h \
            + (offset % stride_w) * x_stride_w + k // positions
        skip += tl.load(x_ptr + x_row[:, None] + x_column[None, :].to(tl.int64), mask=mask).to(tl.float32)
    value += skip / group
    tl.store(out_ptr + row.to(tl.int64)[:, None] * C_OUT + c[None, :], value.to(out_ptr.dtype.element_ty), mask=mask)
