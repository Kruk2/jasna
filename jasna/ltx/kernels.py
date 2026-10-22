"""Triton kernels for the LTX quantized linears (NVIDIA and AMD GPUs with Triton).

Same arithmetic as the torch path in ``transformer.py``, which stays the reference:
- ``rotate_quantize``: ConvRot rotation + per-row int8 in one read of the activations
  (``gated_rotate_quantize``: of a gated attention output read straight from its heads layout).
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
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor

from jasna.ltx.kernel_policy import (
    CUBLAS_EPILOGUE_TILE,
    CUBLAS_EPILOGUE_WARPS,
    CUBLAS_INT8_MM,
    FP4_QUANTIZE_COLS,
    FP4_QUANTIZE_ROWS,
    FP4_QUANTIZE_WARPS,
    FP4_SCALE_TILE_COLS,
    FP4_SCALE_TILE_ROWS,
    GROUP_M,
    ROTATE_QUANTIZE_WARPS,
    SMALL_MM_MAX_N,
    STREAM_K_MAX_WAVES,
    gemm_tile,
    gpu_architecture,
    has_tma,
    programs,
)

EPS = tl.constexpr(1e-6)
# HIP has no add_rn: AMD compiles the GEMM kernels without fp fusion (``GEMM_OPTIONS``), so a
# plain add stays a separate rounding there.
HIP = tl.constexpr(torch.version.hip is not None)
GEMM_OPTIONS = {"enable_fp_fusion": False} if torch.version.hip else {}


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
    _rotate_quantize_row(x, row, j, q_ptr, s_ptr, K)


@triton.jit
def _rotate_quantize_row(x, row, j, q_ptr, s_ptr, K: tl.constexpr):
    """Stores the int8 codes and scale of fp32 row ``x``, loaded with its 256-column blocks reversed."""
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


@triton.jit
def _gated_rotate_quantize_kernel(o_ptr, g_ptr, q_ptr, s_ptr, rows, K: tl.constexpr, HEAD_DIM: tl.constexpr):
    # Row t of ``o (heads, rows, HEAD_DIM)`` read as one token row of width K, each head times
    # 2 * sigmoid of its gate logit ``g[t, head]`` and rounded to bf16 as the torch path stores it.
    row = tl.program_id(0)
    j = tl.arange(0, K)
    cols = (j // 256) * 256 + 255 - j % 256
    head = cols // HEAD_DIM
    x = tl.load(o_ptr + head * rows * HEAD_DIM + row * HEAD_DIM + cols % HEAD_DIM).to(tl.float32)
    gate = tl.load(g_ptr + row * (K // HEAD_DIM) + head).to(tl.float32)
    x = (x * (2.0 * tl.sigmoid(gate))).to(tl.bfloat16).to(tl.float32)
    _rotate_quantize_row(x, row, j, q_ptr, s_ptr, K)


def _rotate_quantize(x, shift, scale, modulate):
    m, k = x.shape
    if k % 256 or k & (k - 1):
        raise ValueError(f"rotate_quantize needs a power-of-two width of at least 256, got {k}")
    codes = torch.empty(m, k, dtype=torch.int8, device=x.device)
    scales = torch.empty(m, dtype=torch.float32, device=x.device)
    _rotate_quantize_kernel[(m,)](
        x, shift, scale, codes, scales, x.stride(0), max(m // 2, 1), K=k, MODULATE=modulate,
        num_warps=ROTATE_QUANTIZE_WARPS,
    )
    return codes, scales


def rotate_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(rows, k)`` bf16 -> (int8 codes ``(rows, k)``, fp32 row scales ``(rows,)``)."""
    return _rotate_quantize(x, x, x, False)


def gated_rotate_quantize(heads: torch.Tensor, gate_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``rotate_quantize`` of attention output ``heads`` ``(heads, rows, head_dim)`` bf16 as token rows,
    each head times ``2 * sigmoid(gate_logits[row, head])``."""
    count, rows, head_dim = heads.shape
    k = count * head_dim
    if k % 256 or k & (k - 1):
        raise ValueError(f"gated_rotate_quantize needs a power-of-two width of at least 256, got {k}")
    codes = torch.empty(rows, k, dtype=torch.int8, device=heads.device)
    scales = torch.empty(rows, dtype=torch.float32, device=heads.device)
    _gated_rotate_quantize_kernel[(rows,)](
        heads, gate_logits, codes, scales, rows, K=k, HEAD_DIM=head_dim, num_warps=ROTATE_QUANTIZE_WARPS,
    )
    return codes, scales


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
        bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]
        if HIP:
            out = out + bias
        else:
            out = libdevice.add_rn(out, bias)
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
    M, N, K, half_rows, row_offset, FULL_TILES, TAIL_SPLITS, PROGRAMS,
    HAS_BIAS: tl.constexpr, GELU: tl.constexpr, RESIDUAL: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
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


@triton.jit
def _int8_epilogue_kernel(
    acc_ptr, as_ptr, bs_ptr, bias_ptr, res_ptr, gate_ptr, c_ptr, M, N, half_rows, row_offset,
    HAS_BIAS: tl.constexpr, GELU: tl.constexpr, RESIDUAL: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    acc = tl.load(acc_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask, other=0)
    _mm_epilogue(
        acc, pid_m, pid_n, as_ptr, bs_ptr, bias_ptr, res_ptr, gate_ptr, c_ptr, M, N, half_rows, row_offset,
        HAS_BIAS, GELU, RESIDUAL, BM, BN,
    )


def stream_k_splits(m: int, n: int, k: int, bm: int, bn: int, bk: int, sms: int) -> int:
    """K splits per tile of the last, partial wave when stream-K pays for this GEMM (0: it doesn't).
    ``tiles % sms`` tiles are left for that wave; each takes ``sms // tail`` idle SMs, at most
    one per ``bk`` step."""
    tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    tail = tiles % sms
    if n < SMALL_MM_MAX_N or tiles >= STREAM_K_MAX_WAVES * sms or not 0 < tail <= sms // 2:
        return 0
    return min(sms // tail, k // bk)


_cached_gpu_architecture = functools.cache(gpu_architecture)


def _mm(codes, act_scales, qdata, scales, bias, out, *, gelu, residual, gate, half_rows, row_offset):
    m, k = codes.shape
    n = qdata.shape[0]
    args = (
        act_scales, scales, bias if bias is not None else scales,
        residual if residual is not None else out, gate if gate is not None else scales, out,
    )
    architecture = _cached_gpu_architecture(codes.device)
    if architecture in CUBLAS_INT8_MM:
        bm, bn = CUBLAS_EPILOGUE_TILE
        # torch._int_mm needs more than 16 rows; on SM 7.5 also a multiple of 16 output features
        acc = torch._int_mm(F.pad(codes, (0, 0, 0, 17 - m)) if m <= 16 else codes, qdata.t())
        _int8_epilogue_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
            acc, *args, m, n, half_rows, row_offset,
            HAS_BIAS=bias is not None, GELU=gelu, RESIDUAL=residual is not None, BM=bm, BN=bn,
            num_warps=CUBLAS_EPILOGUE_WARPS, **GEMM_OPTIONS,
        )
        return out
    bm, bn, bk, stages, warps = gemm_tile(n, k, architecture)
    if k % bk:
        raise ValueError(f"int8_scaled_mm needs a width divisible by {bk}, got {k}")
    flags = dict(HAS_BIAS=bias is not None, GELU=gelu, RESIDUAL=residual is not None, BM=bm, BN=bn, BK=bk, GROUP_M=GROUP_M)
    tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    sms = programs(codes.device)
    # A short grid whose last wave is mostly idle loses most to that wave: split it along K.
    splits = stream_k_splits(m, n, k, bm, bn, bk, sms) if has_tma(architecture) else 0
    if splits:
        tail = tiles % sms
        workspace = torch.zeros(tail * bm * bn + tail, dtype=torch.int32, device=codes.device)
        _int8_mm_stream_k_kernel[(sms,)](
            TensorDescriptor.from_tensor(codes, [bm, bk]), TensorDescriptor.from_tensor(qdata, [bn, bk]), *args,
            workspace, workspace[tail * bm * bn :], m, n, k, half_rows, row_offset, tiles - tail, splits, sms,
            num_stages=stages, num_warps=warps, **flags,
        )
    else:
        _int8_mm_kernel[(tiles,)](
            codes, qdata, *args, m, n, k, half_rows, row_offset, num_stages=stages, num_warps=warps, **flags,
            **GEMM_OPTIONS,
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


def int8_residual_mm(
    codes: torch.Tensor,
    act_scales: torch.Tensor,
    qdata: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """``residual + gate[half] * int8_scaled_mm(...)``: the first half of the rows takes gate row 0, the
    second row 1."""
    residual, gate = residual.contiguous(), gate.contiguous()
    out = torch.empty_like(residual)
    rows = codes.shape[0]
    return _mm(codes, act_scales, qdata, scales, bias, out, gelu=False, residual=residual, gate=gate,
               half_rows=rows // 2, row_offset=0)


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


def fp4_scale_shape(rows: int, k: int) -> tuple[int, int]:
    """Shape of the zero-padded tiled block scales of ``(rows, k)`` NVFP4 values."""
    cols = k // 16
    return -(-rows // FP4_SCALE_TILE_ROWS) * FP4_SCALE_TILE_ROWS, -(-cols // FP4_SCALE_TILE_COLS) * FP4_SCALE_TILE_COLS


def quantize_fp4(x: torch.Tensor, tensor_scale: torch.Tensor, gelu: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """``(rows, k)`` -> (packed uint8 ``(rows, k/2)``, float8 block scales in the zero-padded
    cuBLAS 128x4 tiled layout), like ``transformer.quantize_fp4``; with ``gelu``, of ``gelu_tanh(x)`` rounded
    to bf16."""
    m, k = x.shape
    if k % FP4_QUANTIZE_COLS:
        raise ValueError(f"quantize_fp4 needs a width divisible by {FP4_QUANTIZE_COLS}, got {k}")
    packed = torch.empty(m, k // 2, dtype=torch.uint8, device=x.device)
    scales = torch.zeros(fp4_scale_shape(m, k), dtype=torch.uint8, device=x.device)
    _quantize_fp4_kernel[(triton.cdiv(m, FP4_QUANTIZE_ROWS), k // FP4_QUANTIZE_COLS)](
        x, tensor_scale, packed, scales, m, k, x.stride(0), GELU=gelu, BM=FP4_QUANTIZE_ROWS, BK=FP4_QUANTIZE_COLS,
        num_warps=FP4_QUANTIZE_WARPS,
    )
    return packed, scales.view(torch.float8_e4m3fn)
