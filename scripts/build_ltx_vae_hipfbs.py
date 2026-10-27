"""Build the LTX VAE decoder's Triton kernels into committed AMD .hipfb bundles.

Compiled ahead of time, bf16: ``na_triton._na3d_fwd_kernel`` (head dim 64) once per window size of the
shipped DiffVAE decoder with that size's AMD ``na_geometry`` config, ``diffusion_triton``'s residual MLP
(gate/up and down, every ``FUSED_MLP_DIMS`` width), attention projection, context inject and slab Q/K/V kernels
at the shipped shapes (Q/K/V also at the 512-wide deterministic stages), and the eager encoder's channels-first
copy (``layout_triton``). Every AMD GPU launches these through ``na_hip``, ``diffusion_hip`` and ``layout_hip``.

Usage (Linux, ROCm Torch + Triton, ROCm's clang-offload-bundler):
    python scripts/build_ltx_vae_hipfbs.py

Writes ``jasna/media/ltx_na3d_*.hipfb``, ``ltx_swiglu_*.hipfb``, ``ltx_proj_residual_*.hipfb``,
``ltx_context_add_*.hipfb``, ``ltx_slab_qkv_*.hipfb``, ``ltx_channels_first_*.hipfb`` and the launch table
``jasna/models/ltx_vae/hip_launch.py`` (the grids come from the launchers).
"""
from __future__ import annotations

from triton_hipfb import REPO, build

from jasna.models.ltx_vae import diffusion_hip as aot
from jasna.models.ltx_vae import transformer as T
from jasna.models.ltx_vae import layout_hip, na_hip
from jasna.models.ltx_vae.diffusion_triton import (
    _context_add_kernel,
    _proj_residual_kernel,
    _slab_qkv_kernel,
    _swiglu_down_kernel,
    _swiglu_gate_up_kernel,
)
from jasna.models.ltx_vae.layout_triton import _channels_first_kernel
from jasna.models.ltx_vae.na_geometry import config_for
from jasna.models.ltx_vae.na_triton import _na3d_fwd_kernel

LAUNCH_TABLE = REPO / "jasna" / "models" / "ltx_vae" / "hip_launch.py"
_GRID = (0, 0, 0)


def _kernel_specs():
    """stem -> (kernel, arguments, constants, num_warps, num_stages, grid); the launchers compute the grids."""
    specs = {}
    for window in na_hip.WINDOWS:
        config = config_for(window, amd=True)
        tile_t, tile_h, tile_w = config.tile
        constants = {
            "KERNEL_T": window[0], "KERNEL_H": window[1], "KERNEL_W": window[2], "HEAD_DIM": na_hip.HEAD_DIM,
            "TILE_T": tile_t, "TILE_H": tile_h, "TILE_W": tile_w, "BLOCK_N": config.block_n,
            "BOX_ROWS": config.box_rows,
        }
        specs[na_hip.module_name(window)] = (
            _na3d_fwd_kernel, na_hip.KERNEL_ARGUMENTS, constants, config.num_warps, config.num_stages, _GRID,
        )
    for dim in T.FUSED_MLP_DIMS:
        hidden = T._swiglu_hidden_dim(dim)
        (rows, cols, k, warps, stages), (down_rows, down_cols, down_k, down_warps, down_stages) = aot.SWIGLU_LAUNCH[dim]
        gate_up, down = aot.swiglu_modules(dim)
        specs[gate_up] = (
            _swiglu_gate_up_kernel, aot.GATE_UP_ARGUMENTS,
            {"DIM": dim, "HIDDEN": hidden, "BLOCK_ROWS": rows, "BLOCK_HIDDEN": cols, "BLOCK_K": k}, warps, stages, _GRID,
        )
        specs[down] = (
            _swiglu_down_kernel, aot.DOWN_ARGUMENTS,
            {"DIM": dim, "HIDDEN": hidden, "BLOCK_ROWS": down_rows, "BLOCK_COLS": down_cols, "BLOCK_K": down_k},
            down_warps, down_stages, _GRID,
        )
    rows, cols, k, warps, stages = aot.PROJ_RESIDUAL_LAUNCH
    specs[aot.PROJ_RESIDUAL_MODULE] = (
        _proj_residual_kernel, aot.PROJ_RESIDUAL_ARGUMENTS,
        {"DIM": T.DIFFUSION_DIM, "BLOCK_ROWS": rows, "BLOCK_COLS": cols, "BLOCK_K": k}, warps, stages, _GRID,
    )
    rows, cols, k, warps, stages = aot.CONTEXT_ADD_LAUNCH
    stride_t, stride_h, stride_w = T.DIFFUSION_CONTEXT_STRIDE
    specs[aot.CONTEXT_ADD_MODULE] = (
        _context_add_kernel, aot.CONTEXT_ADD_ARGUMENTS,
        {"IN_DIM": T.DIFFUSION_CONTEXT_DIM, "DIM": T.DIFFUSION_DIM, "STRIDE_T": stride_t, "STRIDE_H": stride_h,
         "STRIDE_W": stride_w, "BLOCK_ROWS": rows, "BLOCK_COLS": cols, "BLOCK_K": k}, warps, stages, _GRID,
    )
    for dim in T.FUSED_QKV_DIMS:
        specs[aot.qkv_module(dim)] = (
            _slab_qkv_kernel, aot.QKV_ARGUMENTS,
            {"DIM": dim, "HEAD_DIM": T.DIFFUSION_HEAD_DIM, "PAIRS_T": T.DIFFUSION_ROPE_PAIRS[0],
             "PAIRS_H": T.DIFFUSION_ROPE_PAIRS[1], "BLOCK_ROWS": aot.QKV_BLOCK_ROWS, "MODULATE": dim == T.DIFFUSION_DIM},
            aot.QKV_WARPS, 1, _GRID,
        )
    specs[layout_hip.MODULE] = (
        _channels_first_kernel, layout_hip.ARGUMENTS, {"BLOCK_S": layout_hip.BLOCK_S, "BLOCK_C": layout_hip.BLOCK_C},
        layout_hip.WARPS, 1, _GRID,
    )
    return specs


def main() -> None:
    build(_kernel_specs(), LAUNCH_TABLE, "LTX VAE decoder kernels", "scripts/build_ltx_vae_hipfbs.py")


if __name__ == "__main__":
    main()
