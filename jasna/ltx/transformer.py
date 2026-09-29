"""Video-only LTX-2.5 transformer as plain functions over per-block weight dicts.

A port of the Lightricks LTX-2.5 ``ltx-core`` 22B forward (``model/transformer``, commit
6ea1527869a5ce57452e215595eae189a7cf65cc; LTX-2.x Community License) restricted to what
restoration runs: video tokens, cross-attention AdaLN, gated attention, split RoPE, and a
per-window token layout of ``n`` target tokens followed by ``n`` reference tokens sharing
the same positions. Target tokens carry timestep ``sigma`` and reference tokens 0, so
every AdaLN modulation is one row per half instead of one per token.

Blocks take their weights as a dict so the executor can hand in resident tensors or views
into a streaming slot. A Linear is stored as ``<name>.weight`` (bf16), as ConvRot W8A8
(``<name>.qdata`` int8 rotated codes, ``<name>.scales`` fp32 per output channel), or as NVFP4
W4A4 (``<name>.weight`` uint8 packed E2M1, high nibble first, ``<name>.weight_scale``
float8_e4m3fn block scales in the cuBLAS 128x4 tiled layout, ``<name>.weight_scale_2`` fp32).
"""

from __future__ import annotations

import functools
import importlib.util
import math
from collections.abc import Mapping
from typing import NamedTuple
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

Weights = Mapping[str, torch.Tensor]

NUM_BLOCKS = 48
HEADS = 32
EPS = 1e-6
TIMESTEP_SCALE = 1000.0
ROPE_THETA = 10000.0
ROPE_MAX_POS = (20, 2048, 2048)
POSITION_FPS = 24.0
LATENT_TEMPORAL_SCALE = 8
LATENT_SPATIAL_SCALE = 32
ROT_SIZE = 256
ACT_QMAX = 127
FP4_BLOCK = 16
FP4_MAX = 6.0
FP8_E4M3_MAX = 448.0
# E2M1 magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6: midpoints rounding to the even code sit below
# the tie, the others above, so a tie lands on the even code (round half to even).
_FP4_TIES_DOWN = (0.25, 1.25, 2.5, 5.0)
_FP4_TIES_UP = (0.75, 1.75, 3.5)
CHUNK_ROWS = 4096
SDPA_PRIORITY = [
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]

LINEAR_NAMES = tuple(
    f"{attn}.{proj}"
    for attn in ("attn1", "attn2")
    for proj in ("to_q", "to_k", "to_v", "to_out.0", "to_gate_logits")
) + ("ff.net.0.proj", "ff.net.2")


def _build_regular_hadamard(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    r4 = torch.tensor([[1.0, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]])
    h = r4.clone()
    while h.shape[0] < size:
        h = torch.kron(h, r4)
    return (h / size**0.5).to(device=device, dtype=dtype)


_cached_regular_hadamard = functools.cache(_build_regular_hadamard)


def _regular_hadamard(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return _build_regular_hadamard(size, device, dtype)
    return _cached_regular_hadamard(size, device, dtype)


def rotate(x: torch.Tensor) -> torch.Tensor:
    """ConvRot block regular-Hadamard rotation along the last dim (self-inverse)."""
    h = _regular_hadamard(ROT_SIZE, x.device, x.dtype)
    shape = x.shape
    return torch.matmul(x.reshape(-1, shape[-1] // ROT_SIZE, ROT_SIZE), h).reshape(shape)


def quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row int8 (round half to even), rows padded to a multiple of 32
    for ``torch._int_mm``. Returns (int8 codes, fp32 scales)."""
    rows = x.shape[0]
    padded = -(-rows // 32) * 32
    xf = x.float()
    scales = xf.abs().amax(dim=1) / ACT_QMAX
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    codes = torch.round(xf / scales.unsqueeze(1)).clamp_(-ACT_QMAX, ACT_QMAX).to(torch.int8)
    if padded != rows:
        codes = F.pad(codes, (0, 0, 0, padded - rows))
        scales = F.pad(scales, (0, padded - rows), value=1.0)
    return codes, scales


def int8_linear(
    x: torch.Tensor, qdata: torch.Tensor, scales: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    """ConvRot W8A8: rotate, quantize per token, INT8 tensor-core GEMM, rescale. With the
    Triton kernels on, one fused quantize pass and one GEMM with the rescale in its epilogue;
    otherwise rows run in chunks of ``CHUNK_ROWS`` (every step is per row, so chunking only
    caps the fp32/int32 temporaries)."""
    out_features, in_features = qdata.shape
    flat = x.reshape(-1, in_features)
    if _quant_kernels is not None and x.is_cuda:
        codes, act_scales = _quant_kernels.int8_rotate_quantize(flat)
        out = _quant_kernels.int8_scaled_mm(codes, act_scales, qdata, scales, bias)
        return out.reshape(*x.shape[:-1], out_features)
    out = torch.empty(flat.shape[0], out_features, dtype=x.dtype, device=x.device)
    for start in range(0, flat.shape[0], CHUNK_ROWS):
        chunk = flat[start : start + CHUNK_ROWS]
        rows = chunk.shape[0]
        codes, act_scales = quantize_rows(rotate(chunk))
        acc = torch._int_mm(codes, qdata.t())[:rows]
        result = acc.float() * (act_scales[:rows].unsqueeze(1) * scales)
        if bias is not None:
            result = result + bias.float()
        out[start : start + rows] = result
    return out.reshape(*x.shape[:-1], out_features)


def fp4_codes(x: torch.Tensor) -> torch.Tensor:
    """4-bit E2M1 codes (sign bit 3), round half to even, saturating at +-6."""
    magnitude = x.abs()
    index = sum((magnitude > tie).to(torch.uint8) for tie in _FP4_TIES_DOWN)
    index = index + sum((magnitude >= tie).to(torch.uint8) for tie in _FP4_TIES_UP)
    return index | ((x < 0).to(torch.uint8) << 3)


def swizzle_block_scales(scales: torch.Tensor) -> torch.Tensor:
    """``(rows, k/16)`` float8 block scales -> zero-padded cuBLAS 128x4 tiled layout."""
    rows, cols = scales.shape
    padded = F.pad(scales.view(torch.uint8), (0, -cols % 4, 0, -rows % 128))
    tiled_rows, tiled_cols = padded.shape
    tiles = padded.view(tiled_rows // 128, 4, 32, tiled_cols // 4, 4).permute(0, 3, 2, 1, 4)
    return tiles.reshape(tiled_rows, tiled_cols).view(torch.float8_e4m3fn)


def fp4_tensor_scale(x: torch.Tensor) -> torch.Tensor:
    """Dynamic per-tensor decode scale ``amax / (6 * 448)`` (1 for an all-zero tensor)."""
    scale = x.abs().amax().float() / (FP4_MAX * FP8_E4M3_MAX)
    return torch.where(scale > 0, scale, torch.ones_like(scale))


def quantize_fp4(x: torch.Tensor, tensor_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(rows, k)`` -> (packed uint8 ``(rows, k/2)``, tiled block scales), one block scale
    per 16 values under the given per-tensor decode scale."""
    xf = x.float()
    blocks = xf.view(x.shape[0], -1, FP4_BLOCK)
    block_scales = (blocks.abs().amax(-1) / FP4_MAX / tensor_scale).clamp(max=FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    divisor = block_scales.float() * tensor_scale
    divisor = torch.where(divisor > 0, divisor, torch.ones_like(divisor))
    codes = fp4_codes(blocks / divisor.unsqueeze(-1)).view(x.shape[0], -1, 2)
    packed = (codes[..., 0] << 4) | codes[..., 1]
    return packed, swizzle_block_scales(block_scales)


def _fp4_mm(
    packed: torch.Tensor,
    act_blocks: torch.Tensor,
    act_scale: torch.Tensor,
    weight: torch.Tensor,
    block_scales: torch.Tensor,
    tensor_scale: torch.Tensor,
    bias: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    return F.scaled_mm(
        packed.view(torch.float4_e2m1fn_x2),
        weight.view(torch.float4_e2m1fn_x2).t(),
        scale_a=[act_blocks, act_scale.reshape(1)],
        scale_recipe_a=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
        scale_b=[block_scales, tensor_scale.reshape(1)],
        scale_recipe_b=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
        swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
        swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
        bias=bias,
        output_dtype=dtype,
    )


def nvfp4_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    block_scales: torch.Tensor,
    tensor_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """NVFP4 W4A4: activations quantized per call (one dynamic tensor scale for the call), then
    the block-scaled FP4 tensor-core GEMM. Both operands pack high nibble first; the block scale
    covers whole pairs, so the order only permutes each dot product's terms."""
    out_features = weight.shape[0]
    flat = x.reshape(-1, weight.shape[1] * 2)
    act_scale = fp4_tensor_scale(flat)
    if _quant_kernels is not None and x.is_cuda:
        packed, act_blocks = _quant_kernels.quantize_fp4(flat, act_scale, False)
        out = _fp4_mm(packed, act_blocks, act_scale, weight, block_scales, tensor_scale, bias, x.dtype)
        return out.reshape(*x.shape[:-1], out_features)
    out = torch.empty(flat.shape[0], out_features, dtype=x.dtype, device=x.device)
    for start in range(0, flat.shape[0], CHUNK_ROWS):
        chunk = flat[start : start + CHUNK_ROWS]
        packed, act_blocks = quantize_fp4(chunk, act_scale)
        out[start : start + chunk.shape[0]] = _fp4_mm(
            packed, act_blocks, act_scale, weight, block_scales, tensor_scale, bias, x.dtype
        )
    return out.reshape(*x.shape[:-1], out_features)


def gelu_linear(w: Weights, name: str, h: torch.Tensor) -> torch.Tensor:
    """``linear(w, name, gelu_tanh(h))``; the NVFP4 kernels quantize straight from ``h``."""
    weight = w.get(f"{name}.weight")
    if _quant_kernels is not None and h.is_cuda and weight is not None and weight.dtype == torch.uint8:
        flat = h.reshape(-1, weight.shape[1] * 2)
        act_scale = fp4_tensor_scale(F.gelu(flat, approximate="tanh"))
        packed, act_blocks = _quant_kernels.quantize_fp4(flat, act_scale, True)
        out = _fp4_mm(
            packed, act_blocks, act_scale, weight, w[f"{name}.weight_scale"], w[f"{name}.weight_scale_2"],
            w.get(f"{name}.bias"), h.dtype,
        )
        return out.reshape(*h.shape[:-1], weight.shape[0])
    return linear(w, name, F.gelu(h, approximate="tanh"))


class QuantizedRows(NamedTuple):
    """ConvRot int8 codes ``(rows, k)`` and row scales of a ``[1, rows, k]`` block activation,
    quantized once for every int8 linear that reads it (Triton kernels only)."""

    codes: torch.Tensor
    scales: torch.Tensor


def linear(w: Weights, name: str, x: torch.Tensor | QuantizedRows) -> torch.Tensor:
    bias = w.get(f"{name}.bias")
    if isinstance(x, QuantizedRows):
        return _quant_kernels.int8_scaled_mm(x.codes, x.scales, w[f"{name}.qdata"], w[f"{name}.scales"], bias)[None]
    weight = w.get(f"{name}.weight")
    if weight is None:
        return int8_linear(x, w[f"{name}.qdata"], w[f"{name}.scales"], bias)
    if weight.dtype == torch.uint8:
        return nvfp4_linear(x, weight, w[f"{name}.weight_scale"], w[f"{name}.weight_scale_2"], bias)
    return F.linear(x, weight, bias)


_quant_kernels = None


def enable_quant_kernels(device: torch.device) -> bool:
    """Route the INT8 and NVFP4 linears through the fused Triton kernels (``kernels.py``,
    tuned on consumer Blackwell, SM 12.x) as opaque ops, so they compile as calls; an INT8
    block also fuses its norms, AdaLN modulation and feed-forward residual into them. Returns
    whether they are in use."""
    global _quant_kernels
    if (
        _quant_kernels is None
        and device.type == "cuda"
        and torch.version.cuda is not None
        and torch.cuda.get_device_capability(device)[0] == 12
        and importlib.util.find_spec("triton") is not None
    ):
        from jasna.ltx import kernels

        @torch.library.custom_op("jasna::int8_rotate_quantize", mutates_args=())
        def int8_rotate_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return kernels.rotate_quantize(x.contiguous())

        @int8_rotate_quantize.register_fake
        def _(x):
            return x.new_empty(x.shape, dtype=torch.int8), x.new_empty(x.shape[0], dtype=torch.float32)

        @torch.library.custom_op("jasna::int8_modulate_rotate_quantize", mutates_args=())
        def int8_modulate_rotate_quantize(
            x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return kernels.modulate_rotate_quantize(x.contiguous(), shift, scale)

        @int8_modulate_rotate_quantize.register_fake
        def _(x, shift, scale):
            return x.new_empty(x.shape, dtype=torch.int8), x.new_empty(x.shape[0], dtype=torch.float32)

        @torch.library.custom_op("jasna::int8_scaled_mm", mutates_args=())
        def int8_scaled_mm(
            codes: torch.Tensor,
            act_scales: torch.Tensor,
            qdata: torch.Tensor,
            scales: torch.Tensor,
            bias: torch.Tensor | None,
        ) -> torch.Tensor:
            return kernels.int8_scaled_mm(codes, act_scales, qdata, scales, bias, False)

        @int8_scaled_mm.register_fake
        def _(codes, act_scales, qdata, scales, bias):
            return codes.new_empty(codes.shape[0], qdata.shape[0], dtype=torch.bfloat16)

        @torch.library.custom_op("jasna::int8_feed_forward", mutates_args=())
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
        ) -> torch.Tensor:
            return kernels.int8_feed_forward(
                codes, act_scales, up_qdata, up_scales, up_bias, down_qdata, down_scales, down_bias,
                residual, gate, CHUNK_ROWS,
            )

        @int8_feed_forward.register_fake
        def _(codes, act_scales, up_qdata, up_scales, up_bias, down_qdata, down_scales, down_bias, residual, gate):
            return torch.empty_like(residual, memory_format=torch.contiguous_format)

        @torch.library.custom_op("jasna::quantize_fp4", mutates_args=())
        def quantize_fp4_op(
            x: torch.Tensor, tensor_scale: torch.Tensor, gelu: bool
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return kernels.quantize_fp4(x.contiguous(), tensor_scale, gelu)

        @quantize_fp4_op.register_fake
        def _(x, tensor_scale, gelu):
            rows, k = x.shape
            return (
                x.new_empty(rows, k // 2, dtype=torch.uint8),
                x.new_empty(-(-rows // 128) * 128, -(-(k // FP4_BLOCK) // 4) * 4, dtype=torch.float8_e4m3fn),
            )

        _quant_kernels = SimpleNamespace(
            int8_rotate_quantize=int8_rotate_quantize,
            int8_modulate_rotate_quantize=int8_modulate_rotate_quantize,
            int8_scaled_mm=int8_scaled_mm,
            int8_feed_forward=int8_feed_forward,
            quantize_fp4=quantize_fp4_op,
        )
    return _quant_kernels is not None


_sage_attention = None


def enable_sage_attention() -> bool:
    """Route video self-attention through SageAttention 2 (an opaque op, so it compiles as
    one call) when it is installed. Returns whether it is in use."""
    global _sage_attention
    if _sage_attention is None and importlib.util.find_spec("sageattention") is not None:
        from sageattention import sageattn

        @torch.library.custom_op("jasna::sage_attention", mutates_args=())
        def sage_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return sageattn(q, k, v, tensor_layout="HND", is_causal=False).contiguous()

        sage_attention.register_fake(lambda q, k, v: torch.empty_like(q, memory_format=torch.contiguous_format))
        _sage_attention = sage_attention
    return _sage_attention is not None


def self_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Video self-attention ``[1, heads, tokens, d]``: SageAttention 2 once enabled."""
    if _sage_attention is not None and q.is_cuda:
        return _sage_attention(q, k, v)
    with sdpa_kernel(SDPA_PRIORITY, set_priority=True):
        return F.scaled_dot_product_attention(q, k, v)


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), weight=weight, eps=EPS)


# --------------------------------------------------------------------------- #
# Positions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rope:
    """cos/sin ``[1, heads, n, head_dim/2]`` for the ``n`` target tokens; the reference
    half reuses them (identical positions)."""

    cos: torch.Tensor
    sin: torch.Tensor


def _freq_grid(dim: int) -> torch.Tensor:
    count = dim // (2 * len(ROPE_MAX_POS))
    grid = np.power(
        ROPE_THETA,
        np.linspace(0.0, np.log(ROPE_THETA) / np.log(ROPE_THETA), count, dtype=np.float64),
    )
    return torch.as_tensor(grid * math.pi / 2, dtype=torch.float32)


def latent_positions(frames: int, height: int, width: int) -> torch.Tensor:
    """``[3, n]`` token-centre positions (seconds, pixel y, pixel x), causal first frame."""
    f, h, w = torch.meshgrid(torch.arange(frames), torch.arange(height), torch.arange(width), indexing="ij")
    starts = torch.stack([f, h, w]).reshape(3, -1)
    bounds = torch.stack([starts, starts + 1], dim=-1)
    scale = torch.tensor([LATENT_TEMPORAL_SCALE, LATENT_SPATIAL_SCALE, LATENT_SPATIAL_SCALE]).view(3, 1, 1)
    pixel = bounds * scale
    pixel[0] = (pixel[0] + 1 - LATENT_TEMPORAL_SCALE).clamp(min=0)
    pixel = pixel.float()
    pixel[0] = pixel[0] / POSITION_FPS
    return (pixel[..., 0] + pixel[..., 1]) / 2.0


def build_rope(frames: int, height: int, width: int, *, dim: int, heads: int, device: torch.device) -> Rope:
    positions = latent_positions(frames, height, width)
    fractional = torch.stack([positions[i] / ROPE_MAX_POS[i] for i in range(3)], dim=-1)[None]
    indices = _freq_grid(dim)
    freqs = (indices * (fractional.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)
    pad = dim // 2 - freqs.shape[-1]
    cos, sin = freqs.cos(), freqs.sin()
    cos = torch.cat([torch.ones_like(cos[:, :, :pad]), cos], dim=-1)
    sin = torch.cat([torch.zeros_like(sin[:, :, :pad]), sin], dim=-1)
    tokens = cos.shape[1]
    cos = cos.reshape(1, tokens, heads, -1).swapaxes(1, 2)
    sin = sin.reshape(1, tokens, heads, -1).swapaxes(1, 2)
    return Rope(cos=cos.to(device=device, dtype=torch.bfloat16), sin=sin.to(device=device, dtype=torch.bfloat16))


def _apply_rope(x: torch.Tensor, rope: Rope, heads: int) -> torch.Tensor:
    """``x`` ``[1, 2n, heads*d]`` -> ``[1, heads, 2n, d]`` rotated (split halves)."""
    n = rope.cos.shape[2]
    halves = x.unflatten(-1, (heads, 2, -1)).transpose(1, 2).unflatten(2, (2, n))  # [1,H,2,n,2,d/2]
    cos = rope.cos[:, :, None, :, None, :]
    sin = rope.sin[:, :, None, :, None, :]
    out = halves * cos
    out[..., :1, :].addcmul_(-sin, halves[..., 1:, :])
    out[..., 1:, :].addcmul_(sin, halves[..., :1, :])
    return out.flatten(4).flatten(2, 3)


# --------------------------------------------------------------------------- #
# Timestep conditioning
# --------------------------------------------------------------------------- #
def _sinusoid(t: torch.Tensor) -> torch.Tensor:
    half = 128
    exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    emb = t[:, None].float() * torch.exp(exponent)[None, :]
    return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)


def adaln(w: Weights, prefix: str, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """AdaLayerNormSingle on timesteps ``t`` (already scaled): (modulation, embedding)."""
    proj = _sinusoid(t).to(torch.bfloat16)
    emb = linear(w, f"{prefix}.emb.timestep_embedder.linear_2", F.silu(linear(w, f"{prefix}.emb.timestep_embedder.linear_1", proj)))
    return linear(w, f"{prefix}.linear", F.silu(emb)), emb


@dataclass(frozen=True)
class StepConditioning:
    """Everything one denoising step needs besides the tokens, for (target, reference)."""

    block_rows: torch.Tensor  # [1, 2, 9, D]: AdaLN rows per half, before each block's table
    output_rows: torch.Tensor  # [1, 2, D]: output-norm embedding per half
    prompt_rows: torch.Tensor  # [1, 1, 2, D]: cross-attention K/V modulation before each table


def step_conditioning(w: Weights, sigma: float, device: torch.device) -> StepConditioning:
    sigma_t = torch.tensor([sigma], dtype=torch.float32, device=device)
    t = torch.cat([sigma_t, torch.zeros_like(sigma_t)]) * TIMESTEP_SCALE
    rows, embedded = adaln(w, "adaln_single", t)
    prompt, _ = adaln(w, "prompt_adaln_single", sigma_t * TIMESTEP_SCALE)
    dim = embedded.shape[-1]
    return StepConditioning(
        block_rows=rows.view(1, 2, -1, dim),
        output_rows=embedded.view(1, 2, dim),
        prompt_rows=prompt.view(1, 1, 2, dim),
    )


# --------------------------------------------------------------------------- #
# Block
# --------------------------------------------------------------------------- #
def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``x * (1 + scale) + shift`` with one row per half: x ``[1, 2n, D]``, rows ``[1, 2, D]``."""
    halves = x.unflatten(1, (2, -1))
    return (halves * (1 + scale[:, :, None]) + shift[:, :, None]).flatten(1, 2)


def _gate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (x.unflatten(1, (2, -1)) * gate[:, :, None]).flatten(1, 2)


def _attention(
    w: Weights,
    prefix: str,
    x: torch.Tensor | QuantizedRows,
    context: torch.Tensor | QuantizedRows,
    rope: Rope | None,
    heads: int,
    *,
    skip: bool,
) -> torch.Tensor:
    """Gated attention output before its ``to_out.0`` projection."""
    v = linear(w, f"{prefix}.to_v", context)
    if skip:
        out = v
    else:
        q = rms_norm(linear(w, f"{prefix}.to_q", x), w[f"{prefix}.q_norm.weight"])
        k = rms_norm(linear(w, f"{prefix}.to_k", context), w[f"{prefix}.k_norm.weight"])
        if rope is not None:
            q, k = _apply_rope(q, rope, heads), _apply_rope(k, rope, heads)
        else:
            q = q.unflatten(-1, (heads, -1)).transpose(1, 2)
            k = k.unflatten(-1, (heads, -1)).transpose(1, 2)
        v_heads = v.unflatten(-1, (heads, -1)).transpose(1, 2)
        if rope is not None:
            out = self_attention(q, k, v_heads)
        else:
            with sdpa_kernel(SDPA_PRIORITY, set_priority=True):
                out = F.scaled_dot_product_attention(q, k, v_heads)
        out = out.transpose(1, 2).flatten(2)
    gates = 2.0 * torch.sigmoid(linear(w, f"{prefix}.to_gate_logits", x))
    return (out.unflatten(-1, (heads, -1)) * gates.unsqueeze(-1)).flatten(2)


def block_forward(
    w: Weights,
    x: torch.Tensor,
    cond: StepConditioning,
    context: torch.Tensor,
    rope: Rope,
    *,
    heads: int = HEADS,
    skip_self_attention: bool = False,
) -> torch.Tensor:
    """One transformer block on ``x`` ``[1, 2n, D]`` (target half, then reference half)."""
    ada = w["scale_shift_table"][None, None] + cond.block_rows  # [1, 2, 9, D]
    fused = _quant_kernels is not None and x.is_cuda and "ff.net.2.qdata" in w
    attn_in = _modulated(x, ada[:, :, 0], ada[:, :, 1], fused)
    attn_out = _attention(w, "attn1", attn_in, attn_in, rope, heads, skip=skip_self_attention)
    x = x + _gate(linear(w, "attn1.to_out.0", attn_out), ada[:, :, 2])
    kv = w["prompt_scale_shift_table"][None, None] + cond.prompt_rows
    query = _modulated(x, ada[:, :, 6], ada[:, :, 7], fused)
    prompt = context * (1 + kv[:, :, 1]) + kv[:, :, 0]
    if fused:
        prompt = QuantizedRows(*_quant_kernels.int8_rotate_quantize(prompt[0]))
    attn_out = _attention(w, "attn2", query, prompt, None, heads, skip=False)
    x = x + _gate(linear(w, "attn2.to_out.0", attn_out), ada[:, :, 8])
    ff_in = _modulated(x, ada[:, :, 3], ada[:, :, 4], fused)
    if fused:
        return _quant_kernels.int8_feed_forward(
            ff_in.codes,
            ff_in.scales,
            w["ff.net.0.proj.qdata"],
            w["ff.net.0.proj.scales"],
            w.get("ff.net.0.proj.bias"),
            w["ff.net.2.qdata"],
            w["ff.net.2.scales"],
            w.get("ff.net.2.bias"),
            x[0],
            ada[0, :, 5],
        )[None]
    ff_out = torch.empty_like(ff_in)
    for start in range(0, ff_in.shape[1], CHUNK_ROWS):
        hidden = linear(w, "ff.net.0.proj", ff_in[:, start : start + CHUNK_ROWS])
        ff_out[:, start : start + CHUNK_ROWS] = gelu_linear(w, "ff.net.2", hidden)
    return x + _gate(ff_out, ada[:, :, 5])


def _modulated(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, fused: bool
) -> torch.Tensor | QuantizedRows:
    """``_modulate(rms_norm(x), shift, scale)``, or its int8 codes computed straight from ``x``."""
    if fused:
        return QuantizedRows(*_quant_kernels.int8_modulate_rotate_quantize(x[0], shift[0], scale[0]))
    return _modulate(rms_norm(x), shift, scale)


# --------------------------------------------------------------------------- #
# Input / output
# --------------------------------------------------------------------------- #
def embed_tokens(w: Weights, tokens: torch.Tensor, tokens_per_frame: int) -> torch.Tensor:
    """Project ``[1, 2n, 128]`` latent tokens; the target's first (single-pixel-frame)
    latent frame gets the keyframe marker embedding."""
    x = linear(w, "patchify_proj", tokens)
    x[:, :tokens_per_frame] = x[:, :tokens_per_frame] + w["keyframes_abs_pos_embedding"]
    return x


def velocity(w: Weights, x: torch.Tensor, cond: StepConditioning) -> torch.Tensor:
    """Predicted velocity of the target half."""
    n = x.shape[1] // 2
    mod = w["scale_shift_table"][None, None] + cond.output_rows[:, :1, None]  # [1, 1, 2, D]
    target = F.layer_norm(x[:, :n], (x.shape[-1],), eps=EPS)
    target = target * (1 + mod[:, :, 1]) + mod[:, :, 0]
    return linear(w, "proj_out", target)
