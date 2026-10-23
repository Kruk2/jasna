"""Neighborhood-attention transformer blocks of the diffusion VAE decoder, ``chunked_eager`` path only.

Ported from ``ltx_core/model/video_vae/transformer/`` (layers, qkv, rope_math, det_attn_rope,
attention, swiglu, blocks, chunked/*): jasna's Triton neighborhood attention on CUDA, eager tiled
SDPA (same window semantics) on the CPU, torch SwiGLU, W-chunked diffusion residual with halos and deferred stage-4 context inject.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils._triton import has_triton_package

from jasna.accelerator import is_amd_device
from jasna.models.ltx_vae.diffusion_hip import QKV_DIMS
from jasna.models.ltx_vae.eager_na import na3d
from jasna.tensor_utils import compile_static

EAGER_ROPE_T_TILES = 4
DIFFUSION_W_CHUNKS = 4
SWIGLU_TILE_TOKENS = 16_384
DECODER_COMPILE_MODE = "default"
_NORM_EPS = 1e-6


def pixel_shuffle_channels_last(x: torch.Tensor, stride: tuple[int, int, int]) -> torch.Tensor:
    """``b t h w (c p1 p2 p3) -> b (t p1) (h p2) (w p3) c``."""
    b, t, h, w, cp = x.shape
    p1, p2, p3 = stride
    c = cp // (p1 * p2 * p3)
    x = x.reshape(b, t, h, w, c, p1, p2, p3).permute(0, 1, 5, 2, 6, 3, 7, 4)
    return x.reshape(b, t * p1, h * p2, w * p3, c)


class LinearPixelShuffleUpsample(nn.Module):
    """Linear channel-expand, then channels-last pixel shuffle.

    With ``stride[0] == 2`` the shuffle yields a duplicate leading frame; ``drop_leading_frame``
    must be True only for the chunk holding the tensor's true temporal origin.
    """

    def __init__(self, in_channels: int, stride: tuple[int, int, int], out_channels_reduction_factor: int) -> None:
        super().__init__()
        self.stride = stride
        self.proj = nn.Linear(in_channels, math.prod(stride) * in_channels // out_channels_reduction_factor, bias=True)

    def forward(self, x: torch.Tensor, drop_leading_frame: bool) -> torch.Tensor:
        x = pixel_shuffle_channels_last(self.proj(x), self.stride)
        if self.stride[0] == 2 and drop_leading_frame:
            x = x[:, 1:, :, :, :]
        return x


class AdaLNZero(nn.Module):
    """``t_emb`` -> 7 modulation chunks (scale/shift/gate slots; gates are unused)."""

    NUM_CHUNKS: int = 7

    def __init__(self, dim: int, t_emb_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(t_emb_dim, self.NUM_CHUNKS * dim, bias=True)

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        chunks = self.proj(F.silu(t_emb)).chunk(self.NUM_CHUNKS, dim=-1)
        return tuple(c[:, None, None, None, :] for c in chunks)


def default_rope_dim_split(head_dim: int) -> tuple[int, int, int]:
    """Split of ``head_dim`` across (T, H, W) RoPE chunks."""
    d_t = (head_dim // 4) // 2 * 2
    d_hw = (head_dim - d_t) // 2
    if d_hw % 2 != 0:
        d_t -= 2
        d_hw = (head_dim - d_t) // 2
    return (d_t, d_hw, d_hw)


def rope_inv_freqs(dim: int) -> torch.Tensor:
    exponents = np.arange(0, dim, 2, dtype=np.float64) / dim
    return torch.from_numpy(1.0 / np.power(10000.0, exponents)).to(torch.float32)


def _broadcast_axis_tables(table_t: torch.Tensor, table_h: torch.Tensor, table_w: torch.Tensor) -> torch.Tensor:
    """Per-axis ``(length, pairs)`` tables -> ``(T, H, W, 1, all pairs)``."""
    t, h, w = table_t.shape[0], table_h.shape[0], table_w.shape[0]
    return torch.cat(
        [
            table_t[:, None, None].expand(t, h, w, -1),
            table_h[None, :, None].expand(t, h, w, -1),
            table_w[None, None].expand(t, h, w, -1),
        ],
        dim=-1,
    )[:, :, :, None]


def _rotate_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    xe = pairs[..., 0].to(torch.float32)
    xo = pairs[..., 1].to(torch.float32)
    return torch.stack([xe * cos - xo * sin, xe * sin + xo * cos], dim=-1).reshape(x.shape)


def apply_abs_rope_slab(x: torch.Tensor, attn: NeighborhoodAttention3D, w_pos: torch.Tensor) -> torch.Tensor:
    """Rotate one W-extent ``(B, T, H, W, NH, HD)`` with absolute RoPE (fp32 math); T/H positions are
    local, W positions are ``w_pos``. cos/sin come from small per-axis tables: a compiled caller
    rotates every channel pair in one elementwise pass, eager runs ``EAGER_ROPE_T_TILES`` T tiles to
    bound the fp32 temporaries."""
    t, h = x.shape[1], x.shape[2]
    angles = (
        torch.arange(t, dtype=torch.float32, device=x.device)[:, None] * attn.rope_inv_t,
        torch.arange(h, dtype=torch.float32, device=x.device)[:, None] * attn.rope_inv_h,
        w_pos[:, None] * attn.rope_inv_w,
    )
    cos = _broadcast_axis_tables(*(angle.cos() for angle in angles))
    sin = _broadcast_axis_tables(*(angle.sin() for angle in angles))
    if torch.compiler.is_compiling():
        return _rotate_pairs(x, cos, sin).to(x.dtype)
    out = torch.empty_like(x)
    step = -(-t // EAGER_ROPE_T_TILES)
    for start in range(0, t, step):
        tile = slice(start, start + step)
        out[:, tile] = _rotate_pairs(x[:, tile], cos[tile], sin[tile])
    return out


class QKVProjections(nn.Module):
    """Separate Q/K/V linears (checkpoints ship a fused ``qkv``; the loader splits it)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.to_q(x), self.to_k(x), self.to_v(x)


_fused_na3d = None
_fused_residual_swiglu = None
_fused_slab_qkv = None


@torch.compiler.disable
def _fp16_eager_na3d(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kernel_size: list[int], query_w_start: int
) -> torch.Tensor:
    """The pure-torch ``na3d`` run in fp16, which has a fused SDPA kernel before SM 8.0 (bf16 has
    none). Not traced, so the compiled slab breaks once around it instead of inside its tile loops."""
    return na3d(q.half(), k.half(), v.half(), tuple(kernel_size), query_w_start).to(q.dtype)


def enable_fused_kernels(device: torch.device) -> None:
    """Run CUDA neighborhood attention through jasna's Triton kernel (``na_triton``): JIT, or on
    AMD without Triton (Windows ROCm) built ahead of time (``na_hip``); before SM 8.0, where Triton
    has no tensor-core dot, through the fp16 pure-torch ``na3d``. On AMD the diffusion blocks'
    residual MLP and the slab Q/K/V projections run as fused kernels (``diffusion_triton``, ahead of
    time ``diffusion_hip``); the Q/K/V kernel also serves the 512-wide deterministic stages."""
    global _fused_na3d, _fused_residual_swiglu, _fused_slab_qkv
    if device.type == "cuda" and torch.cuda.get_device_capability(device) < (8, 0):
        _fused_na3d = _fp16_eager_na3d
        return
    if not has_triton_package():
        from jasna.models.ltx_vae import diffusion_hip, na_hip

        _fused_na3d = na_hip.na3d
        _fused_residual_swiglu = diffusion_hip.residual_swiglu_
        _fused_slab_qkv = diffusion_hip.slab_qkv
        return
    from jasna.models.ltx_vae import na_triton

    _fused_na3d = na_triton.na3d
    if is_amd_device(device):
        from jasna.models.ltx_vae import diffusion_triton

        _fused_residual_swiglu = diffusion_triton.residual_swiglu_
        _fused_slab_qkv = diffusion_triton.slab_qkv


def neighborhood_attention(
    attn: NeighborhoodAttention3D, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, query_w_start: int
) -> torch.Tensor:
    """NA of the queries ``q`` of W columns ``[query_w_start, query_w_start + q.shape[3])`` over the
    whole K/V grid: the Triton kernel on CUDA once enabled, else the pure-torch ``na3d`` (same
    window semantics). Q is already scaled."""
    if q.dtype != v.dtype or k.dtype != v.dtype:
        q = q.to(dtype=v.dtype)
        k = k.to(dtype=v.dtype)
    if _fused_na3d is not None and q.is_cuda:
        return _fused_na3d(q, k, v, list(attn.kernel_size), query_w_start)
    return na3d(q, k, v, attn.kernel_size, query_w_start)


class NeighborhoodAttention3D(nn.Module):
    """3D neighborhood attention with absolute RoPE on channels-last ``(B, T, H, W, C)``."""

    def __init__(self, dim: int, kernel_size: tuple[int, int, int], head_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.kernel_size = tuple(kernel_size)
        self.scale = head_dim**-0.5
        self.rope_dim_split = default_rope_dim_split(head_dim)

        self.register_buffer("rope_inv_t", rope_inv_freqs(self.rope_dim_split[0]), persistent=False)
        self.register_buffer("rope_inv_h", rope_inv_freqs(self.rope_dim_split[1]), persistent=False)
        self.register_buffer("rope_inv_w", rope_inv_freqs(self.rope_dim_split[2]), persistent=False)

        self.qkv = QKVProjections(dim)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.q_norm = nn.RMSNorm(head_dim, eps=_NORM_EPS)
        self.k_norm = nn.RMSNorm(head_dim, eps=_NORM_EPS)

    def fused_qkv(
        self, x: torch.Tensor, left_halo: torch.Tensor, columns: torch.Tensor, modulation: list[torch.Tensor],
        w_pos: torch.Tensor, core_len: int, halo: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Q (scaled) of the core columns and K/V of every column of the slab ``left_halo`` then ``x``'s W
        ``columns`` (modulated by the norm weight, scale and shift in ``modulation`` unless empty) through
        the fused Q/K/V kernel."""
        qkv = self.qkv
        return _fused_slab_qkv(
            x,
            left_halo,
            columns,
            modulation,
            [qkv.to_q.weight, qkv.to_q.bias, qkv.to_k.weight, qkv.to_k.bias, qkv.to_v.weight, qkv.to_v.bias,
             self.q_norm.weight, self.k_norm.weight],
            torch.cat([self.rope_inv_t, self.rope_inv_h, self.rope_inv_w]),
            w_pos,
            [self.rope_inv_t.numel(), self.rope_inv_h.numel()],
            core_len,
            halo,
            float(self.scale),
        )

    def project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, t, h, w, _ = x.shape
        q, k, v = self.qkv(x)
        shape = (batch, t, h, w, self.num_heads, self.head_dim)
        return q.view(shape), k.view(shape), v.view(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic-stage NA over the whole volume."""
        batch, t, h, w, _ = x.shape
        kt, kh, kw = self.kernel_size
        if t < kt or h < kh or w < kw:
            raise ValueError(
                f"3D neighborhood attention requires spatial dims >= kernel_size; "
                f"got (T,H,W)=({t},{h},{w}) vs kernel={self.kernel_size}"
            )
        w_pos = torch.arange(w, dtype=torch.float32, device=x.device)
        if _fused_slab_qkv is not None and x.is_cuda and self.dim in QKV_DIMS:
            x = x.contiguous()
            columns = torch.arange(w, device=x.device)
            q, k, v = self.fused_qkv(x, x, columns, [], w_pos, w, 0)
        else:
            q, k, v = self.project_qkv(x)
            q = self.q_norm(q)
            k = self.k_norm(k)
            q = q * self.scale
            q = apply_abs_rope_slab(q, self, w_pos)
            k = apply_abs_rope_slab(k, self, w_pos)
        out = neighborhood_attention(self, q.contiguous(), k.contiguous(), v.contiguous(), 0)
        out = out.reshape(batch, t, h, w, self.dim)
        return self.proj(out)


class SwiGLU(nn.Module):
    """Gated MLP weights: ``w_down(silu(w_gate(x)) * w_up(x))``."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)


def _swiglu_hidden_dim(dim: int) -> int:
    return (int(dim * 4.0) + 15) // 16 * 16


def _swiglu_into(xc: torch.Tensor, mlp: SwiGLU, workspace: torch.Tensor, out: torch.Tensor) -> None:
    """``out = w_down(silu(xc @ W_gateᵀ) * (xc @ W_upᵀ))`` through a reusable ``workspace``."""
    torch.mm(xc, mlp.w_gate.weight.t(), out=workspace)
    F.silu(workspace, inplace=True)
    workspace.mul_(F.linear(xc, mlp.w_up.weight))
    torch.mm(workspace, mlp.w_down.weight.t(), out=out)


def swiglu_token_chunked(x: torch.Tensor, mlp: SwiGLU) -> torch.Tensor:
    """SwiGLU over ``SWIGLU_TILE_TOKENS``-token chunks with one ``(chunk, hidden)`` workspace."""
    leading = x.shape[:-1]
    dim = x.shape[-1]
    x_flat = x.reshape(-1, dim).contiguous()
    n_tok = x_flat.shape[0]
    out_flat = torch.empty((n_tok, dim), device=x.device, dtype=x.dtype)
    workspace = torch.empty(
        (min(n_tok, SWIGLU_TILE_TOKENS), mlp.w_gate.weight.shape[0]), device=x.device, dtype=x.dtype
    )
    for start in range(0, n_tok, SWIGLU_TILE_TOKENS):
        end = min(n_tok, start + SWIGLU_TILE_TOKENS)
        _swiglu_into(x_flat[start:end], mlp, workspace[: end - start], out_flat[start:end])
    return out_flat.view(*leading, dim)


def modulated_swiglu(
    xc: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """``swiglu(modulate(rms_norm(xc)))`` of one token chunk; ``w_gate_up`` stacks the gate and up
    weights so both come from one GEMM."""
    y = F.rms_norm(xc, (xc.shape[-1],), norm_weight, _NORM_EPS) * (1.0 + scale) + shift
    gate, up = F.linear(y, w_gate_up).chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, w_down)


def residual_modulating_mlp(
    x: torch.Tensor, mlp: SwiGLU, norm: nn.RMSNorm, scale: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    """In-place token-chunked ``x += swiglu(modulate(rms_norm(x)))``."""
    if not x.is_contiguous():
        x = x.contiguous()
    dim = x.shape[-1]
    x_flat = x.reshape(-1, dim)
    s = scale.reshape(-1, dim)
    sh = shift.reshape(-1, dim)
    if _fused_residual_swiglu is not None and x.is_cuda:
        for start in range(0, x_flat.shape[0], SWIGLU_TILE_TOKENS):
            _fused_residual_swiglu(
                x_flat[start : start + SWIGLU_TILE_TOKENS], norm.weight, s[0], sh[0],
                mlp.w_gate.weight, mlp.w_up.weight, mlp.w_down.weight,
            )
        return x
    chunk_mlp = _mlp_chunk if x.is_cuda else modulated_swiglu
    w_gate_up = torch.cat([mlp.w_gate.weight, mlp.w_up.weight])
    for start in range(0, x_flat.shape[0], SWIGLU_TILE_TOKENS):
        xc = x_flat[start : start + SWIGLU_TILE_TOKENS]
        xc.add_(chunk_mlp(xc, norm.weight, s, sh, w_gate_up, mlp.w_down.weight))
    return x


class NABlock(nn.Module):
    """Pre-norm transformer block: NA -> SwiGLU MLP with residual adds (channels-last)."""

    def __init__(self, dim: int, kernel_size: tuple[int, int, int], head_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=_NORM_EPS)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim=head_dim)
        self.norm2 = nn.RMSNorm(dim, eps=_NORM_EPS)
        self.mlp = SwiGLU(dim, _swiglu_hidden_dim(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + swiglu_token_chunked(self.norm2(x), self.mlp)


def _attn_on_w_slab(
    attn: NeighborhoodAttention3D,
    norm: nn.RMSNorm,
    left_halo: torch.Tensor,
    x: torch.Tensor,
    columns: torch.Tensor,
    w_pos: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    core_len: int,
) -> torch.Tensor:
    """One W slab, ``left_halo`` then ``x``'s W ``columns`` (the fused Q/K/V kernel, or compiled, reads
    that gather in the slab's first read): rms_norm + modulate + QKV / in-slab RoPE / NA / proj, output
    for the ``core_len`` columns after the ``kernel_w // 2`` halo only."""
    batch, t, h = x.shape[:3]
    halo = attn.kernel_size[2] // 2
    if _fused_slab_qkv is not None and x.is_cuda and scale.numel() == attn.dim:
        q, k, v = attn.fused_qkv(x, left_halo, columns, [norm.weight, scale, shift], w_pos, core_len, halo)
    else:
        x_chunk = torch.cat([left_halo, x.index_select(3, columns)], dim=3)
        ext_w = x_chunk.shape[3]
        y = F.rms_norm(x_chunk, (attn.dim,), norm.weight, _NORM_EPS)
        y = y * (1.0 + scale) + shift
        head_shape = (batch, t, h, ext_w, attn.num_heads, attn.head_dim)
        core_shape = (batch, t, h, core_len, attn.num_heads, attn.head_dim)
        y_core = y[:, :, :, halo : halo + core_len].reshape(-1, attn.dim)
        q = F.linear(y_core, attn.qkv.to_q.weight, attn.qkv.to_q.bias).view(core_shape)
        q = F.rms_norm(q, (attn.head_dim,), attn.q_norm.weight, _NORM_EPS) * float(attn.scale)
        q = apply_abs_rope_slab(q, attn, w_pos[halo : halo + core_len])
        k = F.linear(y, attn.qkv.to_k.weight, attn.qkv.to_k.bias).view(head_shape)
        k = F.rms_norm(k, (attn.head_dim,), attn.k_norm.weight, _NORM_EPS)
        k = apply_abs_rope_slab(k, attn, w_pos)
        v = F.linear(y, attn.qkv.to_v.weight, attn.qkv.to_v.bias).view(head_shape)
        del y

    out = neighborhood_attention(attn, q.contiguous(), k.contiguous(), v.contiguous(), halo)
    del q, k, v
    return F.linear(out.reshape(batch, t, h, core_len, attn.dim), attn.proj.weight, attn.proj.bias)


def w_chunked_attention_residual(
    x: torch.Tensor,
    attn: NeighborhoodAttention3D,
    norm: nn.RMSNorm,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    """In-place ``x += NA(modulate(norm(x)))`` over ``DIFFUSION_W_CHUNKS`` fixed-extent W slabs.

    Each slab carries ``kernel_w // 2`` halo columns from its neighbors, the left ones as they were
    before the previous slab's update; missing halo slots at the true W boundary are edge-replicated
    so NA never sees zero padding.
    """
    w_chunks = DIFFUSION_W_CHUNKS
    halo = attn.kernel_size[2] // 2
    w = x.shape[3]
    chunk_w = (w + w_chunks - 1) // w_chunks
    extent = chunk_w + 2 * halo
    left_halo = x[:, :, :, :1].expand(-1, -1, -1, halo, -1).contiguous()
    slab_attention = _slab_attention if x.is_cuda else _attn_on_w_slab

    for i in range(w_chunks):
        core_start = i * chunk_w
        core_end = min(w, (i + 1) * chunk_w)
        core_len = core_end - core_start
        if core_len <= 0:
            break

        right_end = min(w, core_end + halo) if i + 1 < w_chunks else core_end
        columns = torch.arange(core_start, core_start + extent - halo, device=x.device)
        columns = torch.where(columns < right_end, columns, core_end - 1)
        w_pos = torch.arange(extent, device=x.device, dtype=torch.float32) + (core_start - halo)
        slab_left_halo = left_halo
        if i + 1 < w_chunks:
            take = min(halo, core_len)
            left_halo = x.new_zeros(*x.shape[:3], halo, x.shape[4])
            left_halo[:, :, :, halo - take :] = x[:, :, :, core_end - take : core_end]
        out = slab_attention(attn, norm, slab_left_halo, x, columns, w_pos, scale, shift, core_len)
        x[:, :, :, core_start:core_end, :].add_(out)

    return x


_slab_attention = _attn_on_w_slab
_mlp_chunk = modulated_swiglu


def compile_diffusion_blocks(device: torch.device, dtype: torch.dtype) -> None:
    """Compile the stage-5 attention slab, MLP chunk and context inject (computing in ``dtype``)
    with static shapes (one graph per shape, shared by all blocks); CUDA tensors use them from then on."""
    global _slab_attention, _mlp_chunk, _add_context
    if _slab_attention is _attn_on_w_slab:
        _slab_attention = compile_static(_attn_on_w_slab, mode=DECODER_COMPILE_MODE, device=device, dtype=dtype)
        _mlp_chunk = compile_static(modulated_swiglu, mode=DECODER_COMPILE_MODE, device=device, dtype=dtype)
        _add_context = compile_static(add_folded_context, mode=DECODER_COMPILE_MODE, device=device, dtype=dtype)


def fold_context_projection(
    upsample: LinearPixelShuffleUpsample, context_proj: nn.Linear
) -> tuple[torch.Tensor, torch.Tensor]:
    """``context_proj ∘ upsample.proj`` as one linear (fp32 fold) whose output channels keep the
    pixel-shuffle order: one GEMM at stage-4 resolution instead of a second one after the shuffle."""
    positions = math.prod(upsample.stride)
    in_channels = upsample.proj.in_features
    up_weight = upsample.proj.weight.float().view(-1, positions, in_channels)
    ctx_weight = context_proj.weight.float()
    weight = torch.einsum("oc,cpi->opi", ctx_weight, up_weight).reshape(-1, in_channels)
    bias = ctx_weight @ upsample.proj.bias.float().view(-1, positions) + context_proj.bias.float()[:, None]
    dtype = upsample.proj.weight.dtype
    return weight.to(dtype), bias.reshape(-1).to(dtype)


def add_folded_context(
    x_slab: torch.Tensor,
    feat_chunk: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple[int, int, int],
    drop_leading_frame: bool,
) -> None:
    """``x_slab += pixel_shuffle(feat_chunk @ weightᵀ + bias)``, cropped to ``x_slab``'s width."""
    batch, t, h, w, channels = feat_chunk.shape
    folded = F.linear(feat_chunk.reshape(-1, channels), weight, bias).view(batch, t, h, w, -1)
    ctx = pixel_shuffle_channels_last(folded, stride)
    if stride[0] == 2 and drop_leading_frame:
        ctx = ctx[:, 1:]
    x_slab.add_(ctx[:, :, :, : x_slab.shape[3]])


_add_context = add_folded_context


def inject_deferred_context(
    x: torch.Tensor,
    stage4_feat: torch.Tensor,
    upsample: LinearPixelShuffleUpsample,
    context_proj: nn.Linear,
    *,
    drop_leading_frame: bool,
) -> torch.Tensor:
    """``x += context_proj(upsample(stage4_feat))`` over ``DIFFUSION_W_CHUNKS`` stage-4 W slabs."""
    weight, bias = fold_context_projection(upsample, context_proj)
    add_context = _add_context if x.is_cuda else add_folded_context
    w_hi = x.shape[3]
    lo = 0
    for feat_chunk in torch.chunk(stage4_feat, DIFFUSION_W_CHUNKS, dim=3):
        hi = min(w_hi, lo + feat_chunk.shape[3] * upsample.stride[2])
        add_context(x[:, :, :, lo:hi, :], feat_chunk, weight, bias, tuple(upsample.stride), drop_leading_frame)
        lo = hi
    return x


class ChunkedDiffusionNABlock(nn.Module):
    """Diffusion NA + SwiGLU block with shared AdaLN modulation and deferred stage-4 context."""

    def __init__(self, dim: int, kernel_size: tuple[int, int, int], context_channels: int, head_dim: int) -> None:
        super().__init__()
        self.context_proj = nn.Linear(context_channels, dim, bias=True)
        self.scale_shift_table = nn.Parameter(torch.zeros(AdaLNZero.NUM_CHUNKS, dim))
        self.norm1 = nn.RMSNorm(dim, eps=_NORM_EPS)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim=head_dim)
        self.norm2 = nn.RMSNorm(dim, eps=_NORM_EPS)
        self.mlp = SwiGLU(dim, _swiglu_hidden_dim(dim))

    def _modulation(
        self, modulation: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = [
            modulation[i] + self.scale_shift_table[i].view(1, 1, 1, 1, -1) for i in range(AdaLNZero.NUM_CHUNKS)
        ]
        return scale_msa, shift_msa, scale_mlp, shift_mlp

    def forward(
        self,
        x: torch.Tensor,
        stage4_feat: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        stage4_upsample: LinearPixelShuffleUpsample,
        *,
        drop_leading_frame: bool,
    ) -> torch.Tensor:
        scale_msa, shift_msa, scale_mlp, shift_mlp = self._modulation(modulation)
        x = inject_deferred_context(
            x, stage4_feat, stage4_upsample, self.context_proj, drop_leading_frame=drop_leading_frame
        )
        x = w_chunked_attention_residual(x, self.attn, self.norm1, scale_msa, shift_msa)
        return residual_modulating_mlp(x, self.mlp, self.norm2, scale_mlp, shift_mlp)
