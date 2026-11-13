"""The decoder's fused kernels (``diffusion_triton``) built ahead of time for AMD
(``scripts/build_ltx_vae_hipfbs.py``): every AMD GPU runs them, with or without Triton.

- ``residual_swiglu_``: ``transformer.modulated_swiglu`` plus the residual add, ``x += w_down(silu(gate) * up)``
  with ``gate, up = w_gate(y), w_up(y)`` and ``y = rms_norm(x) * (1 + scale) + shift``, at every
  ``FUSED_MLP_DIMS`` width, as the gate/up and down kernels.
- ``slab_qkv``: a W slab's Q/K/V projections with Q/K RMSNorm and absolute RoPE (``_attn_on_w_slab``),
  an opaque op, so a compiled slab calls it in one piece.
- ``proj_residual_`` / ``context_add_``: a diffusion block's attention output projection and stage-4 context
  inject with their adds into the block input.
"""

import torch

from jasna.accelerator import amd_gpu_architecture
from jasna.media.cuda_kernel import aot_kernel
from jasna.models.ltx_vae import transformer
from jasna.models.ltx_vae.hip_launch import KERNEL_LAUNCH
from jasna.models.ltx_vae.transformer import (
    DIFFUSION_CONTEXT_DIM,
    DIFFUSION_CONTEXT_STRIDE,
    DIFFUSION_DIM,
    DIFFUSION_HEAD_DIM,
    DIFFUSION_ROPE_PAIRS,
    FUSED_MLP_DIMS,
    FUSED_QKV_DIMS,
    _swiglu_hidden_dim,
)

# Per-program (rows, columns, K step, warps, stages) of each kernel: per kernel the config with the least
# worst-case slowdown on an RX 9060 XT (gfx1200) and an RX 7900 XTX (gfx1100) of 32-128 rows x 64-256 columns x
# 32-64 K x 4-8 warps x 1-4 stages at the decode's shapes. The MLP's are (gate/up, down) per width. Slab Q/K/V:
# rows per program and warps, fastest on a Radeon AI PRO R9700 (gfx1201) of 16-128 rows x 4-16 warps.
SWIGLU_LAUNCH = {
    256: ((32, 256, 64, 8, 1), (32, 256, 64, 4, 1)),
    512: ((32, 256, 64, 8, 1), (32, 256, 64, 4, 1)),
    1024: ((64, 128, 64, 8, 1), (64, 256, 64, 8, 1)),
}
PROJ_RESIDUAL_LAUNCH = (32, 256, 64, 4, 1)
CONTEXT_ADD_LAUNCH = (32, 256, 64, 8, 1)
QKV_BLOCK_ROWS = 32
QKV_WARPS = 8
FP16_ACCUMULATE_ARCHITECTURES: frozenset[str] = frozenset()
"""AMD architectures whose MLP, context and Q/K/V GEMMs accumulate in fp16 (as NVIDIA's
``NvidiaLaunch.fp16_accumulate``, with bf16 weights and Q/K/V), built into their bundles by
``scripts/build_ltx_vae_hipfbs.py``. None: no AMD GPU has been measured with it."""
PROJ_RESIDUAL_MODULE = f"ltx_proj_residual_{DIFFUSION_DIM}"
CONTEXT_ADD_MODULE = f"ltx_context_add_{DIFFUSION_CONTEXT_DIM}x{DIFFUSION_DIM}"


def swiglu_modules(dim: int) -> tuple[str, str]:
    hidden = _swiglu_hidden_dim(dim)
    return f"ltx_swiglu_gate_up_{dim}x{hidden}", f"ltx_swiglu_down_{dim}x{hidden}"


def qkv_module(dim: int) -> str:
    return f"ltx_slab_qkv_{dim}x{DIFFUSION_HEAD_DIM}"


def accumulates_in_fp16(architecture: str) -> bool:
    return architecture in FP16_ACCUMULATE_ARCHITECTURES


GATE_UP_ARGUMENTS = {**{name: "*bf16" for name in ("x_ptr", "norm_ptr", "scale_ptr", "shift_ptr", "gate_ptr",
                                                   "up_ptr", "h_ptr")}, "rows": "i32"}
DOWN_ARGUMENTS = {"h_ptr": "*bf16", "down_ptr": "*bf16", "x_ptr": "*bf16", "rows": "i32"}


def swiglu_arguments(architecture: str) -> tuple[dict[str, str], dict[str, str]]:
    """The gate/up and down kernels' arguments on ``architecture``: the hidden activation between them is fp16
    where the GEMMs accumulate in fp16."""
    hidden = "*fp16" if accumulates_in_fp16(architecture) else "*bf16"
    return {**GATE_UP_ARGUMENTS, "h_ptr": hidden}, {**DOWN_ARGUMENTS, "h_ptr": hidden}


PROJ_RESIDUAL_ARGUMENTS = {"a_ptr": "*bf16", "w_ptr": "*bf16", "b_ptr": "*bf16", "x_ptr": "*bf16:wide",
                           **{name: "i32" for name in ("rows", "core_len", "x_line_stride", "core_start")}}
CONTEXT_ADD_ARGUMENTS = {"f_ptr": "*bf16", "w_ptr": "*bf16", "b_ptr": "*bf16", "x_ptr": "*bf16:wide",
                         **{name: "i32" for name in ("rows", "f_t", "f_h", "f_w", "f_stride_b", "f_stride_t",
                                                     "f_stride_h", "f_stride_w", "x_t", "x_h", "x_w", "x_w_start",
                                                     "x_w_len", "drop_frames")}}
_QKV_BF16 = ("norm_ptr", "scale_ptr", "shift_ptr", "wq_ptr", "bq_ptr", "wk_ptr", "bk_ptr", "wv_ptr", "bv_ptr", "qn_ptr", "kn_ptr")
_QKV_FP32 = ("inv_ptr", "wpos_ptr")
_QKV_OUT = ("q_ptr", "k_ptr", "v_ptr")
_QKV_INTS = ("rows", "H", "W", "EXT", "halo", "core")
QKV_ARGUMENTS = {
    "x_ptr": "*bf16:wide", "halo_ptr": "*bf16", "col_ptr": "*i64", **{name: "*bf16" for name in _QKV_BF16},
    **{name: "*fp32" for name in _QKV_FP32}, **{name: "*bf16" for name in _QKV_OUT},
    **{name: "i32" for name in _QKV_INTS}, "q_scale": "fp32",
}


def _launch(module: str, arguments: dict[str, str], grid: tuple[int, int], device: torch.device, args) -> None:
    aot_kernel(KERNEL_LAUNCH, module, device, arguments).launch(
        (*grid, 1), torch.cuda.current_stream(device).cuda_stream, *args,
    )


def _require_bf16(tensors, shapes_match: bool) -> None:
    if not shapes_match or {t.dtype for t in tensors} != {torch.bfloat16} or not all(t.is_contiguous() for t in tensors):
        raise ValueError("the ahead-of-time decoder kernels take contiguous bf16 tensors of the widths they are built "
                         f"for (MLP: {', '.join(map(str, FUSED_MLP_DIMS))}; Q/K/V: "
                         f"{' or '.join(map(str, FUSED_QKV_DIMS))}; diffusion blocks: {DIFFUSION_DIM})")


def residual_swiglu_(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> None:
    """In place on contiguous ``x`` ``(rows, dim)``: one row each of ``scale``/``shift`` ``(dim,)``,
    ``w_gate``/``w_up`` ``(hidden, dim)``, ``w_down`` ``(dim, hidden)``, at a ``FUSED_MLP_DIMS`` width, over
    ``transformer.SWIGLU_TILE_TOKENS``-row chunks with one hidden-activation workspace."""
    rows, dim = x.shape
    hidden = _swiglu_hidden_dim(dim)
    tensors = (x, norm_weight, scale, shift, w_gate, w_up, w_down)
    _require_bf16(tensors, dim in FUSED_MLP_DIMS and w_gate.shape == (hidden, dim))
    architecture = amd_gpu_architecture(x.device)
    chunk = transformer.SWIGLU_TILE_TOKENS
    h = x.new_empty(min(rows, chunk), hidden, dtype=torch.float16 if accumulates_in_fp16(architecture) else x.dtype)
    (gate_up_rows, gate_up_cols, *_), (down_rows, down_cols, *_) = SWIGLU_LAUNCH[dim]
    gate_up, down = swiglu_modules(dim)
    gate_up_arguments, down_arguments = swiglu_arguments(architecture)
    for start in range(0, rows, chunk):
        x_chunk = x[start : start + chunk]
        chunk_rows = x_chunk.shape[0]
        _launch(gate_up, gate_up_arguments, (-(-chunk_rows // gate_up_rows), hidden // gate_up_cols), x.device,
                (x_chunk, norm_weight, scale, shift, w_gate, w_up, h, chunk_rows))
        _launch(down, down_arguments, (-(-chunk_rows // down_rows), dim // down_cols), x.device,
                (h, w_down, x_chunk, chunk_rows))


def proj_residual_(
    x: torch.Tensor, out: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, core_start: int
) -> None:
    """``diffusion_triton.proj_residual_`` at the diffusion blocks' width."""
    batch, t, h, w, dim = x.shape
    _require_bf16((x, out, weight, bias), dim == DIFFUSION_DIM and weight.shape == (dim, dim))
    core_len = out.shape[3]
    rows = batch * t * h * core_len
    block_rows, block_cols, *_ = PROJ_RESIDUAL_LAUNCH
    _launch(PROJ_RESIDUAL_MODULE, PROJ_RESIDUAL_ARGUMENTS, (-(-rows // block_rows), dim // block_cols), x.device,
            (out, weight, bias, x, rows, core_len, w * dim, core_start))


def context_add_(
    x: torch.Tensor, w_start: int, w_len: int, feat: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
    stride: tuple[int, int, int], drop_leading_frame: bool,
) -> None:
    """``diffusion_triton.context_add_`` at the diffusion blocks' widths and stride."""
    batch, t, h, w, in_dim = feat.shape
    dim = x.shape[4]
    positions = stride[0] * stride[1] * stride[2]
    _require_bf16((x, weight, bias), in_dim == DIFFUSION_CONTEXT_DIM and dim == DIFFUSION_DIM
                  and tuple(stride) == DIFFUSION_CONTEXT_STRIDE and feat.dtype == torch.bfloat16
                  and feat.stride(4) == 1)
    rows = batch * t * h * w
    block_rows, block_cols, *_ = CONTEXT_ADD_LAUNCH
    _launch(CONTEXT_ADD_MODULE, CONTEXT_ADD_ARGUMENTS, (-(-rows // block_rows), positions * dim // block_cols),
            x.device, (feat, weight, bias, x, rows, t, h, w, *feat.stride()[:4], x.shape[1], x.shape[2], x.shape[3],
                       w_start, w_len, int(stride[0] == 2 and drop_leading_frame)))


@torch.library.custom_op("jasna::slab_qkv_aot", mutates_args=())
def slab_qkv(
    x: torch.Tensor,
    left_halo: torch.Tensor,
    columns: torch.Tensor,
    modulation: list[torch.Tensor],
    weights: list[torch.Tensor],
    inv_freqs: torch.Tensor,
    w_pos: torch.Tensor,
    rope_pairs: list[int],
    core_len: int,
    halo: int,
    q_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q of the core columns ``[halo, halo + core_len)`` (scaled by ``q_scale``) and K/V of every column of
    the W slab ``(B, T, H, halo + len(columns), DIM)`` made of ``left_halo`` ``(B, T, H, halo, DIM)`` then
    ``x``'s ``(B, T, H, W, DIM)`` W ``columns``, as ``(B, T, H, W, heads, head_dim)``. ``modulation``: empty,
    or the norm weight, scale and shift ``(DIM,)`` the slab is modulated with (``rms_norm(slab) * (1 + scale)
    + shift``). ``weights``: to_q weight, bias, to_k weight, bias, to_v weight, bias, q_norm weight, k_norm
    weight; ``inv_freqs`` the (T, H, W) RoPE frequencies, ``rope_pairs`` the T and H pair counts, ``w_pos``
    each slab column's W."""
    x, left_halo, columns = x.contiguous(), left_halo.contiguous(), columns.contiguous()
    batch, t, h, w, dim = x.shape
    _require_bf16((x, left_halo, *modulation, *weights),
                  dim in FUSED_QKV_DIMS and bool(modulation) == (dim == DIFFUSION_DIM)
                  and weights[7].shape == (DIFFUSION_HEAD_DIM,) and tuple(rope_pairs) == DIFFUSION_ROPE_PAIRS[:2]
                  and columns.dtype == torch.int64 and columns.is_contiguous())
    ext = halo + columns.numel()
    q = x.new_empty(batch, t, h, core_len, dim // DIFFUSION_HEAD_DIM, DIFFUSION_HEAD_DIM)
    k = x.new_empty(batch, t, h, ext, dim // DIFFUSION_HEAD_DIM, DIFFUSION_HEAD_DIM)
    v = torch.empty_like(k)
    rows = batch * t * h * ext
    norm, scale, shift = modulation or (x, x, x)
    pointers = (x, left_halo, columns, norm, scale, shift, *weights, inv_freqs.float().contiguous(),
                w_pos.float().contiguous(), q, k, v)
    _launch(qkv_module(dim), QKV_ARGUMENTS, (-(-rows // QKV_BLOCK_ROWS), 1), x.device,
            (*pointers, rows, h, w, ext, halo, core_len, q_scale))
    return q, k, v


@slab_qkv.register_fake
def _slab_qkv_fake(x, left_halo, columns, modulation, weights, inv_freqs, w_pos, rope_pairs, core_len, halo, q_scale):
    batch, t, h, _w, dim = x.shape
    k = x.new_empty(batch, t, h, halo + columns.shape[0], dim // DIFFUSION_HEAD_DIM, DIFFUSION_HEAD_DIM)
    return x.new_empty(batch, t, h, core_len, dim // DIFFUSION_HEAD_DIM, DIFFUSION_HEAD_DIM), k, torch.empty_like(k)
