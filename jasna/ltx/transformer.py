"""Video-only LTX-2.5 transformer as plain functions over per-block weight dicts.

A port of the Lightricks LTX-2.5 ``ltx-core`` 22B forward (``model/transformer``, commit
6ea1527869a5ce57452e215595eae189a7cf65cc; LTX-2.x Community License) restricted to what
restoration runs: video tokens, cross-attention AdaLN, gated attention, split RoPE, and a
per-window token layout of ``n`` target tokens followed by ``n`` reference tokens sharing
the same positions. Target tokens carry timestep ``sigma`` and reference tokens 0, so
every AdaLN modulation is one row per half instead of one per token.

Blocks take their weights as a dict so the executor can hand in resident tensors or views
into a streaming slot. A Linear is stored either as ``<name>.weight`` (bf16) or as ConvRot
W8A8 (``<name>.qdata`` int8 rotated codes, ``<name>.scales`` fp32 per output channel).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from dataclasses import dataclass

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


@functools.cache
def _regular_hadamard(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    r4 = torch.tensor([[1.0, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]])
    h = r4.clone()
    while h.shape[0] < size:
        h = torch.kron(h, r4)
    return (h / size**0.5).to(device=device, dtype=dtype)


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
    """ConvRot W8A8: rotate, quantize per token, INT8 tensor-core GEMM, rescale.
    Rows run in chunks of ``CHUNK_ROWS``; every step is per row, so chunking only caps
    the fp32/int32 temporaries."""
    out_features, in_features = qdata.shape
    flat = x.reshape(-1, in_features)
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


def linear(w: Weights, name: str, x: torch.Tensor) -> torch.Tensor:
    bias = w.get(f"{name}.bias")
    weight = w.get(f"{name}.weight")
    if weight is not None:
        return F.linear(x, weight, bias)
    return int8_linear(x, w[f"{name}.qdata"], w[f"{name}.scales"], bias)


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
    x: torch.Tensor,
    context: torch.Tensor,
    rope: Rope | None,
    heads: int,
    *,
    skip: bool,
) -> torch.Tensor:
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
        with sdpa_kernel(SDPA_PRIORITY, set_priority=True):
            out = F.scaled_dot_product_attention(q, k, v_heads)
        out = out.transpose(1, 2).flatten(2)
    gates = 2.0 * torch.sigmoid(linear(w, f"{prefix}.to_gate_logits", x))
    out = (out.unflatten(-1, (heads, -1)) * gates.unsqueeze(-1)).flatten(2)
    return linear(w, f"{prefix}.to_out.0", out)


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
    attn_in = _modulate(rms_norm(x), ada[:, :, 0], ada[:, :, 1])
    attn_out = _attention(w, "attn1", attn_in, attn_in, rope, heads, skip=skip_self_attention)
    x = x + _gate(attn_out, ada[:, :, 2])
    kv = w["prompt_scale_shift_table"][None, None] + cond.prompt_rows
    query = _modulate(rms_norm(x), ada[:, :, 6], ada[:, :, 7])
    prompt = context * (1 + kv[:, :, 1]) + kv[:, :, 0]
    x = x + _gate(_attention(w, "attn2", query, prompt, None, heads, skip=False), ada[:, :, 8])
    ff_in = _modulate(rms_norm(x), ada[:, :, 3], ada[:, :, 4])
    ff_out = torch.empty_like(ff_in)
    for start in range(0, ff_in.shape[1], CHUNK_ROWS):
        hidden = F.gelu(linear(w, "ff.net.0.proj", ff_in[:, start : start + CHUNK_ROWS]), approximate="tanh")
        ff_out[:, start : start + CHUNK_ROWS] = linear(w, "ff.net.2", hidden)
    return x + _gate(ff_out, ada[:, :, 5])


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
