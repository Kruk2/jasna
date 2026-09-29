"""Triton kernels for the LTX quantized linears (NVIDIA GPUs with Triton).

Same arithmetic as the torch path in ``transformer.py``, which stays the reference:
- ``rotate_quantize``: ConvRot rotation + per-row int8 in one read of the activations.
  The rotation runs as radix-4 butterflies in fp32 and is rounded to bf16 like the
  reference's bf16 matmul, so codes differ from it only where the two fp32 summation
  orders round to different bf16 values (a few per million, by one code).
- ``int8_scaled_mm``: int8 GEMM whose epilogue applies the row and channel scales, the
  bias, the bf16 cast and optionally GELU-tanh; bit-exact against ``torch._int_mm`` plus
  the reference rescale.
- ``quantize_fp4``: NVFP4 block quantization with the scales written straight into the
  cuBLAS 128x4 tiled layout; bit-exact against ``transformer.quantize_fp4``.
"""

import functools

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor

EPS = tl.constexpr(1e-6)


@triton.jit
def _rotate_quantize_kernel(
    x_ptr, shift_ptr, scale_ptr, q_ptr, s_ptr, stride_x, half_rows, K: tl.constexpr, MODULATE: tl.constexpr
):
    # One row per program, held in registers, so the activations are read once.
    # x @ H for the regular Hadamard H = (r4 x r4 x r4 x r4) / 16 per 256 columns: one r4
    # stage along an axis is y_j = s - 2 x_(3-j) with s the sum over that axis; the four
    # index reversals compose into one reversal of the 256 columns, done by loading them
    # reversed.
    row = tl.program_id(0)
    j = tl.arange(0, K)
    cols = (j // 256) * 256 + 255 - j % 256
    x = tl.load(x_ptr + row * stride_x + cols).to(tl.float32)
    if MODULATE:
        # rms_norm, then the AdaLN row of this row's half: x * (1 + scale) + shift
        x = x * libdevice.rsqrt(tl.sum(x * x, axis=0) / K + EPS)
        half = row // half_rows
        x = x * (1 + tl.load(scale_ptr + half * K + cols).to(tl.float32)) + tl.load(shift_ptr + half * K + cols).to(tl.float32)
        x = x.to(tl.bfloat16).to(tl.float32)
    x = tl.reshape(x, (K // 256, 4, 4, 4, 4))
    x = 0.5 * (tl.sum(x, axis=1, keep_dims=True) - 2 * x)
    x = 0.5 * (tl.sum(x, axis=2, keep_dims=True) - 2 * x)
    x = 0.5 * (tl.sum(x, axis=3, keep_dims=True) - 2 * x)
    x = 0.5 * (tl.sum(x, axis=4, keep_dims=True) - 2 * x)
    x = tl.reshape(x, (K,)).to(tl.bfloat16).to(tl.float32)
    scale = tl.math.div_rn(tl.max(tl.abs(x), axis=0), 127.0)
    scale = tl.where(scale > 0, scale, 1.0)
    tl.store(s_ptr + row, scale)
    codes = libdevice.rint(tl.math.div_rn(x, scale))
    tl.store(q_ptr + row * K + j, tl.minimum(tl.maximum(codes, -127.0), 127.0).to(tl.int8))


def _rotate_quantize(x, shift, scale, modulate):
    m, k = x.shape
    if k % 256 or k & (k - 1):
        raise ValueError(f"rotate_quantize needs a power-of-two width of at least 256, got {k}")
    codes = torch.empty(m, k, dtype=torch.int8, device=x.device)
    scales = torch.empty(m, dtype=torch.float32, device=x.device)
    _rotate_quantize_kernel[(m,)](
        x, shift, scale, codes, scales, x.stride(0), max(m // 2, 1), K=k, MODULATE=modulate, num_warps=8
    )
    return codes, scales


def rotate_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(rows, k)`` bf16 -> (int8 codes ``(rows, k)``, fp32 row scales ``(rows,)``)."""
    return _rotate_quantize(x, x, x, False)


def modulate_rotate_quantize(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``rotate_quantize(rms_norm(x) * (1 + scale[half]) + shift[half])`` where the first half
    of the rows uses AdaLN row 0 of ``shift``/``scale`` (``(2, k)``) and the second half row 1;
    the modulated row is rounded to bf16 as the torch path stores it."""
    return _rotate_quantize(x, shift.contiguous(), scale.contiguous(), True)


@triton.jit
def _tile_coords(tile, M, N, BM: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr):
    grid_m = tl.cdiv(M, BM)
    width = GROUP_M * tl.cdiv(N, BN)
    first_m = (tile // width) * GROUP_M
    size_m = min(grid_m - first_m, GROUP_M)
    return first_m + (tile % width) % size_m, (tile % width) // size_m


@triton.jit
def _mm_epilogue(
    acc, pid_m, pid_n, as_ptr, bs_ptr, bias_ptr, res_ptr, gate_ptr, c_ptr, M, N, half_rows, row_offset,
    HAS_BIAS: tl.constexpr, GELU: tl.constexpr, RESIDUAL: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
):
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m_mask = offs_m < M
    n_mask = offs_n < N
    mask = m_mask[:, None] & n_mask[None, :]
    a_scales = tl.load(as_ptr + offs_m, mask=m_mask, other=0.0)
    b_scales = tl.load(bs_ptr + offs_n, mask=n_mask, other=0.0)
    out = acc.to(tl.float32) * (a_scales[:, None] * b_scales[None, :])
    if HAS_BIAS:
        out = libdevice.add_rn(out, tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :])
    out = out.to(tl.bfloat16)
    if GELU:
        v = out.to(tl.float32)
        out = (0.5 * v * (1 + libdevice.tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)))).to(tl.bfloat16)
    if RESIDUAL:
        # x + gate[half] * out, the gate row picked by the row's half of the whole block input
        half = (offs_m + row_offset) // half_rows
        gate = tl.load(gate_ptr + half[:, None] * N + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(res_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
        out = (residual + out.to(tl.float32) * gate).to(tl.bfloat16)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], out, mask=mask)


@triton.jit
def _int8_mm_kernel(
    a_ptr, b_ptr, as_ptr, bs_ptr, bias_ptr, res_ptr, gate_ptr, c_ptr, M, N, K, half_rows, row_offset,
    HAS_BIAS: tl.constexpr, GELU: tl.constexpr, RESIDUAL: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid_m, pid_n = _tile_coords(tl.program_id(0), M, N, BM, BN, GROUP_M)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + tl.where(offs_m < M, offs_m, 0)[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + tl.where(offs_n < N, offs_n, 0)[None, :] * K + offs_k[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for _ in range(0, K, BK):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc, out_dtype=tl.int32)
        a_ptrs += BK
        b_ptrs += BK
    _mm_epilogue(
        acc, pid_m, pid_n, as_ptr, bs_ptr, bias_ptr, res_ptr, gate_ptr, c_ptr, M, N, half_rows, row_offset,
        HAS_BIAS, GELU, RESIDUAL, BM, BN,
    )


@triton.jit
def _tma_tile(a_desc, b_desc, pid_m, pid_n, k_start, k_stop, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k in range(k_start, k_stop, BK):
        acc = tl.dot(a_desc.load([pid_m * BM, k]), b_desc.load([pid_n * BN, k]).T, acc, out_dtype=tl.int32)
    return acc


@triton.jit
def _int8_mm_stream_k_kernel(
    a_desc, b_desc, as_ptr, bs_ptr, bias_ptr, res_ptr, gate_ptr, c_ptr, ws_ptr, count_ptr,
    M, N, K, half_rows, row_offset, FULL_TILES, TAIL_SPLITS,
    HAS_BIAS: tl.constexpr, GELU: tl.constexpr, RESIDUAL: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr, PROGRAMS: tl.constexpr,
):
    # Persistent: whole waves of tiles as usual, then the last, partial wave split along K
    # over the idle programs. Partial int32 sums are added atomically (exact, in any order)
    # and the last split of a tile runs its epilogue.
    pid = tl.program_id(0)
    for tile in range(pid, FULL_TILES, PROGRAMS):
        pid_m, pid_n = _tile_coords(tile, M, N, BM, BN, GROUP_M)
        acc = _tma_tile(a_desc, b_desc, pid_m, pid_n, 0, K, BM, BN, BK)
        _mm_epilogue(
            acc, pid_m, pid_n, as_ptr, bs_ptr, bias_ptr, res_ptr, gate_ptr, c_ptr, M, N, half_rows, row_offset,
            HAS_BIAS, GELU, RESIDUAL, BM, BN,
        )
    tail_units = (tl.cdiv(M, BM) * tl.cdiv(N, BN) - FULL_TILES) * TAIL_SPLITS
    if pid < tail_units:
        tail = pid // TAIL_SPLITS
        split = pid % TAIL_SPLITS
        k_steps = K // BK
        pid_m, pid_n = _tile_coords(FULL_TILES + tail, M, N, BM, BN, GROUP_M)
        acc = _tma_tile(
            a_desc, b_desc, pid_m, pid_n,
            (split * k_steps // TAIL_SPLITS) * BK, ((split + 1) * k_steps // TAIL_SPLITS) * BK, BM, BN, BK,
        )
        partial = ws_ptr + tail * BM * BN + tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
        tl.atomic_add(partial, acc, sem="relaxed")
        if tl.atomic_add(count_ptr + tail, 1, sem="acq_rel") == TAIL_SPLITS - 1:
            _mm_epilogue(
                tl.load(partial, volatile=True), pid_m, pid_n, as_ptr, bs_ptr, bias_ptr, res_ptr, gate_ptr, c_ptr,
                M, N, half_rows, row_offset, HAS_BIAS, GELU, RESIDUAL, BM, BN,
            )


@functools.cache
def _programs(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _mm(codes, act_scales, qdata, scales, bias, out, *, gelu, residual, gate, half_rows, row_offset):
    m, k = codes.shape
    n = qdata.shape[0]
    if n < 128:
        bm, bn, bk, stages, warps = 32, 32, 256, 3, 2
    elif k > 4096:
        bm, bn, bk, stages, warps = 128, 128, 128, 3, 4
    else:
        bm, bn, bk, stages, warps = 128, 256, 128, 3, 8
    if k % bk:
        raise ValueError(f"int8_scaled_mm needs a width divisible by {bk}, got {k}")
    args = (
        act_scales, scales, bias if bias is not None else scales,
        residual if residual is not None else out, gate if gate is not None else scales, out,
    )
    flags = dict(HAS_BIAS=bias is not None, GELU=gelu, RESIDUAL=residual is not None, BM=bm, BN=bn, BK=bk, GROUP_M=8)
    tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    programs = _programs(codes.device)
    tail = tiles % programs
    # A short grid whose last wave is mostly idle loses most to that wave: split it along K.
    if n >= 128 and tiles < 8 * programs and 0 < tail <= programs // 2:
        workspace = torch.zeros(tail * bm * bn + tail, dtype=torch.int32, device=codes.device)
        _int8_mm_stream_k_kernel[(programs,)](
            TensorDescriptor.from_tensor(codes, [bm, bk]), TensorDescriptor.from_tensor(qdata, [bn, bk]), *args,
            workspace, workspace[tail * bm * bn :], m, n, k, half_rows, row_offset, tiles - tail,
            min(programs // tail, k // bk), PROGRAMS=programs, num_stages=stages, num_warps=warps, **flags,
        )
    else:
        _int8_mm_kernel[(tiles,)](
            codes, qdata, *args, m, n, k, half_rows, row_offset, num_stages=stages, num_warps=warps, **flags
        )
    return out


def int8_scaled_mm(
    codes: torch.Tensor,
    act_scales: torch.Tensor,
    qdata: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
    gelu: bool,
) -> torch.Tensor:
    """``(codes @ qdata.T) * act_scales[:, None] * scales + bias`` as bf16, then GELU-tanh
    when ``gelu``."""
    out = torch.empty(codes.shape[0], qdata.shape[0], dtype=torch.bfloat16, device=codes.device)
    return _mm(codes, act_scales, qdata, scales, bias, out, gelu=gelu, residual=None, gate=None, half_rows=1, row_offset=0)


def int8_feed_forward(
    codes: torch.Tensor,
    act_scales: torch.Tensor,
    up_qdata: torch.Tensor,
    up_scales: torch.Tensor,
    up_bias: torch.Tensor | None,
    down_qdata: torch.Tensor,
    down_scales: torch.Tensor,
    down_bias: torch.Tensor | None,
    residual: torch.Tensor,
    gate: torch.Tensor,
    chunk_rows: int,
) -> torch.Tensor:
    """``residual + gate[half] * down(gelu_tanh(up(x)))`` from the quantized ``x``, in row
    chunks so the wide hidden activation stays small."""
    residual, gate = residual.contiguous(), gate.contiguous()
    out = torch.empty_like(residual)
    rows = codes.shape[0]
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        hidden = int8_scaled_mm(codes[start:stop], act_scales[start:stop], up_qdata, up_scales, up_bias, True)
        hidden_codes, hidden_scales = rotate_quantize(hidden)
        _mm(
            hidden_codes, hidden_scales, down_qdata, down_scales, down_bias, out[start:stop],
            gelu=False, residual=residual[start:stop], gate=gate, half_rows=rows // 2, row_offset=start,
        )
    return out


@triton.jit
def _quantize_fp4_kernel(
    x_ptr, ts_ptr, q_ptr, s_ptr, M, K, stride_x, GELU: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BK + tl.arange(0, BK)
    row_mask = rows < M
    x = tl.load(x_ptr + rows[:, None] * stride_x + cols[None, :], mask=row_mask[:, None], other=0.0).to(tl.float32)
    if GELU:
        x = (0.5 * x * (1 + libdevice.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))).to(tl.bfloat16).to(tl.float32)
    tensor_scale = tl.load(ts_ptr)
    blocks = tl.reshape(x, (BM, BK // 16, 16))
    block_amax = tl.max(tl.abs(blocks), axis=2)
    scale8 = tl.minimum(tl.math.div_rn(tl.math.div_rn(block_amax, 6.0), tensor_scale), 448.0).to(tl.float8e4nv)
    divisor = scale8.to(tl.float32) * tensor_scale
    divisor = tl.where(divisor > 0, divisor, 1.0)
    v = tl.math.div_rn(blocks, divisor[:, :, None])
    mag = tl.abs(v)
    # E2M1 round half to even: ties at 0.25, 1.25, 2.5, 5 go down, at 0.75, 1.75, 3.5 up
    code = (
        (mag > 0.25).to(tl.uint8) + (mag >= 0.75).to(tl.uint8) + (mag > 1.25).to(tl.uint8)
        + (mag >= 1.75).to(tl.uint8) + (mag > 2.5).to(tl.uint8) + (mag >= 3.5).to(tl.uint8)
        + (mag > 5.0).to(tl.uint8)
    ) | ((v < 0).to(tl.uint8) << 3)
    hi, lo = tl.split(tl.reshape(code, (BM, BK // 2, 2)))
    packed_cols = tl.program_id(1) * (BK // 2) + tl.arange(0, BK // 2)
    tl.store(q_ptr + rows[:, None] * (K // 2) + packed_cols[None, :], (hi << 4) | lo, mask=row_mask[:, None])
    # cuBLAS 128x4 tiles: (row // 128, col // 4, row % 32, row // 32 % 4, col % 4)
    scale_cols = tl.program_id(1) * (BK // 16) + tl.arange(0, BK // 16)
    col_tiles = tl.cdiv(K // 16, 4)
    offsets = (
        (((rows[:, None] // 128) * col_tiles + scale_cols[None, :] // 4) * 32 + rows[:, None] % 32) * 4
        + (rows[:, None] // 32) % 4
    ) * 4 + scale_cols[None, :] % 4
    tl.store(s_ptr + offsets, scale8.to(tl.uint8, bitcast=True), mask=row_mask[:, None])


def quantize_fp4(x: torch.Tensor, tensor_scale: torch.Tensor, gelu: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """``(rows, k)`` -> (packed uint8 ``(rows, k/2)``, float8 block scales in the zero-padded
    cuBLAS 128x4 tiled layout), like ``transformer.quantize_fp4``; with ``gelu``, of ``gelu_tanh(x)`` rounded
    to bf16."""
    m, k = x.shape
    if k % 256:
        raise ValueError(f"quantize_fp4 needs a width divisible by 256, got {k}")
    packed = torch.empty(m, k // 2, dtype=torch.uint8, device=x.device)
    scales = torch.zeros(-(-m // 128) * 128, -(-(k // 16) // 4) * 4, dtype=torch.uint8, device=x.device)
    block_rows, block_k = 32, 256
    _quantize_fp4_kernel[(triton.cdiv(m, block_rows), k // block_k)](
        x, tensor_scale, packed, scales, m, k, x.stride(0), GELU=gelu, BM=block_rows, BK=block_k, num_warps=4
    )
    return packed, scales.view(torch.float8_e4m3fn)
