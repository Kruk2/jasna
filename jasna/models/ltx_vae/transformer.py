"""Neighborhood-attention transformer blocks of the diffusion VAE decoder, ``chunked_eager`` path only.

Ported from ``ltx_core/model/video_vae/transformer/`` (layers, qkv, rope_math, det_attn_rope,
attention, swiglu, blocks, chunked/*): eager tiled-SDPA neighborhood attention, torch SwiGLU,
W-chunked diffusion residual with halos and deferred stage-4 context inject.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from jasna.models.ltx_vae.eager_na import na3d

DET_ROPE_NUM_TILES = 4
DIFFUSION_W_CHUNKS = 4
SWIGLU_TILE_TOKENS = 16_384
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


def rot_abs_axis_impl(xc: torch.Tensor, pos: torch.Tensor, inv: torch.Tensor, axis: int) -> torch.Tensor:
    """Absolute RoPE (fp32 math) on one axis chunk ``xc[..., D]``."""
    out_dtype = xc.dtype
    pairs = xc.reshape(*xc.shape[:-1], xc.shape[-1] // 2, 2)
    xe = pairs[..., 0].to(torch.float32)
    xo = pairs[..., 1].to(torch.float32)
    shape = [1, 1, 1, 1, 1, inv.shape[0]]
    shape[axis] = pos.shape[0]
    ang = (pos[:, None] * inv[None, :]).reshape(shape)
    c = ang.cos()
    s = ang.sin()
    re = xe * c - xo * s
    ro = xe * s + xo * c
    out = torch.stack([re, ro], dim=-1).reshape(xc.shape)
    return out.to(out_dtype) if out.dtype != out_dtype else out


def apply_abs_rope_slab(x: torch.Tensor, attn: NeighborhoodAttention3D, w_pos: torch.Tensor) -> torch.Tensor:
    """Rotate one W-extent ``(B, T, H, W, NH, HD)``; T/H positions are local, W positions are ``w_pos``."""
    d_t, d_h, _ = attn.rope_dim_split
    t_pos = torch.arange(x.shape[1], dtype=torch.float32, device=x.device)
    h_pos = torch.arange(x.shape[2], dtype=torch.float32, device=x.device)
    xt = rot_abs_axis_impl(x[..., :d_t], t_pos, attn.rope_inv_t, axis=1)
    xh = rot_abs_axis_impl(x[..., d_t : d_t + d_h], h_pos, attn.rope_inv_h, axis=2)
    xw = rot_abs_axis_impl(x[..., d_t + d_h :], w_pos, attn.rope_inv_w, axis=3)
    return torch.cat([xt, xh, xw], dim=-1)


def apply_tiled_abs_rope(x: torch.Tensor, attn: NeighborhoodAttention3D) -> torch.Tensor:
    """Full-volume abs-RoPE computed over ``DET_ROPE_NUM_TILES`` W slabs."""
    w_off = 0
    parts: list[torch.Tensor] = []
    for slab in torch.chunk(x, DET_ROPE_NUM_TILES, dim=3):
        w_slab = slab.shape[3]
        w_pos = torch.arange(w_slab, dtype=torch.float32, device=x.device) + w_off
        parts.append(apply_abs_rope_slab(slab, attn, w_pos))
        w_off = w_off + w_slab
    return torch.cat(parts, dim=3)


class QKVProjections(nn.Module):
    """Separate Q/K/V linears (checkpoints ship a fused ``qkv``; the loader splits it)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.to_q(x), self.to_k(x), self.to_v(x)


def eager_sdpa_attention(
    attn: NeighborhoodAttention3D, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    if q.dtype != v.dtype or k.dtype != v.dtype:
        q = q.to(dtype=v.dtype)
        k = k.to(dtype=v.dtype)
    return na3d(q, k, v, kernel_size=attn.kernel_size)


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
        q, k, v = self.project_qkv(x)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q * self.scale
        q = apply_tiled_abs_rope(q, self)
        k = apply_tiled_abs_rope(k, self)
        out = eager_sdpa_attention(self, q.contiguous(), k.contiguous(), v.contiguous())
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


def residual_modulating_mlp(
    x: torch.Tensor, mlp: SwiGLU, norm: nn.RMSNorm, scale: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    """In-place token-chunked ``x += swiglu(modulate(rms_norm(x)))``."""
    if not x.is_contiguous():
        x = x.contiguous()
    dim = x.shape[-1]
    x_flat = x.reshape(-1, dim)
    n_tok = x_flat.shape[0]
    s = scale.reshape(-1, dim)
    sh = shift.reshape(-1, dim)
    max_chunk = min(n_tok, SWIGLU_TILE_TOKENS)
    workspace = torch.empty((max_chunk, mlp.w_gate.weight.shape[0]), device=x.device, dtype=x.dtype)
    y_buf = torch.empty((max_chunk, dim), device=x.device, dtype=x.dtype)
    out_buf = torch.empty((max_chunk, dim), device=x.device, dtype=x.dtype)
    for start in range(0, n_tok, SWIGLU_TILE_TOKENS):
        end = min(n_tok, start + SWIGLU_TILE_TOKENS)
        n = end - start
        xc = x_flat[start:end]
        y = y_buf[:n]
        y.copy_(F.rms_norm(xc, (dim,), norm.weight, _NORM_EPS))
        y.mul_(1.0 + s).add_(sh)
        _swiglu_into(y, mlp, workspace[:n], out_buf[:n])
        xc.add_(out_buf[:n])
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
    x_chunk: torch.Tensor,
    w_pos: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    """One W slab: rms_norm + modulate + QKV / in-slab RoPE / NA / proj."""
    batch, t, h, ext_w, _ = x_chunk.shape
    y = F.rms_norm(x_chunk, (attn.dim,), norm.weight, _NORM_EPS)
    y = y * (1.0 + scale) + shift

    head_shape = (batch, t, h, ext_w, attn.num_heads, attn.head_dim)
    q = F.linear(y, attn.qkv.to_q.weight, attn.qkv.to_q.bias).view(head_shape)
    q = F.rms_norm(q, (attn.head_dim,), attn.q_norm.weight, _NORM_EPS) * float(attn.scale)
    q = apply_abs_rope_slab(q, attn, w_pos)
    k = F.linear(y, attn.qkv.to_k.weight, attn.qkv.to_k.bias).view(head_shape)
    k = F.rms_norm(k, (attn.head_dim,), attn.k_norm.weight, _NORM_EPS)
    k = apply_abs_rope_slab(k, attn, w_pos)
    v = F.linear(y, attn.qkv.to_v.weight, attn.qkv.to_v.bias).view(head_shape)
    del y

    out = eager_sdpa_attention(attn, q.contiguous(), k.contiguous(), v.contiguous())
    del q, k, v
    return F.linear(out.reshape(batch, t, h, ext_w, attn.dim), attn.proj.weight, attn.proj.bias)


def w_chunked_attention_residual(
    x: torch.Tensor,
    attn: NeighborhoodAttention3D,
    norm: nn.RMSNorm,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    """In-place ``x += NA(modulate(norm(x)))`` over ``DIFFUSION_W_CHUNKS`` fixed-extent W slabs.

    Each slab carries ``kernel_w // 2`` halo columns copied from its neighbors; missing
    halo slots at the true W boundary are edge-replicated so NA never sees zero padding.
    """
    w_chunks = DIFFUSION_W_CHUNKS
    halo = attn.kernel_size[2] // 2
    _, _, _, w, c = x.shape
    chunk_w = (w + w_chunks - 1) // w_chunks
    extent = chunk_w + 2 * halo
    left_halo: torch.Tensor | None = None

    for i in range(w_chunks):
        core_start = i * chunk_w
        core_end = min(w, (i + 1) * chunk_w)
        core_len = core_end - core_start

        buf = x.new_zeros(*x.shape[:3], extent, c)
        if i > 0:
            lh = left_halo.shape[3]
            buf[:, :, :, halo - lh : halo, :] = left_halo
        buf[:, :, :, halo : halo + core_len, :] = x[:, :, :, core_start:core_end, :]
        right_filled = 0
        if i + 1 < w_chunks:
            right_end = min(w, core_end + halo)
            right = x[:, :, :, core_end:right_end, :]
            right_filled = right.shape[3]
            buf[:, :, :, halo + core_len : halo + core_len + right_filled, :] = right

        if i == 0 and halo > 0 and core_len > 0:
            edge_l = buf[:, :, :, halo : halo + 1, :]
            buf[:, :, :, :halo, :] = edge_l.expand(*x.shape[:3], halo, c)
        missing_right = extent - (halo + core_len + right_filled)
        if missing_right > 0 and core_len > 0:
            edge_r = buf[:, :, :, halo + core_len - 1 : halo + core_len, :]
            lo_r = halo + core_len + right_filled
            buf[:, :, :, lo_r:extent, :] = edge_r.expand(*x.shape[:3], missing_right, c)

        if i + 1 < w_chunks:
            take = min(halo, core_len)
            left_halo = x[:, :, :, core_end - take : core_end, :].clone()

        w_pos = torch.arange(extent, device=x.device, dtype=torch.float32) + (core_start - halo)
        out = _attn_on_w_slab(attn, norm, buf, w_pos, scale, shift)
        x[:, :, :, core_start:core_end, :].add_(out[:, :, :, halo : halo + core_len, :])

    return x


def _upsample_then_ctx(
    feat: torch.Tensor, upsample: LinearPixelShuffleUpsample, context_proj: nn.Linear, *, drop_leading_frame: bool
) -> torch.Tensor:
    """``context_proj(pixel_shuffle(upsample.proj(feat)))``."""
    up = pixel_shuffle_channels_last(F.linear(feat, upsample.proj.weight, upsample.proj.bias), upsample.stride)
    if upsample.stride[0] == 2 and drop_leading_frame:
        up = up[:, 1:, :, :, :]
    return F.linear(up, context_proj.weight, context_proj.bias)


def inject_deferred_context(
    x: torch.Tensor,
    stage4_feat: torch.Tensor,
    upsample: LinearPixelShuffleUpsample,
    context_proj: nn.Linear,
    *,
    drop_leading_frame: bool,
) -> torch.Tensor:
    """Chunk stage-4 feat along W, upsample + ``context_proj`` each slab, ``add_`` into ``x``."""
    w_hi = x.shape[3]
    lo = 0
    for feat_chunk in torch.chunk(stage4_feat, DIFFUSION_W_CHUNKS, dim=3):
        hi = min(w_hi, lo + feat_chunk.shape[3] * upsample.stride[2])
        ctx = _upsample_then_ctx(feat_chunk, upsample, context_proj, drop_leading_frame=drop_leading_frame)
        x[:, :, :, lo:hi, :].add_(ctx[:, :, :, : hi - lo, :])
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
