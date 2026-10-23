"""Build the LTX VAE decoder's Triton kernels into committed AMD .hipfb bundles.

Compiled ahead of time, bf16: ``na_triton._na3d_fwd_kernel`` (head dim 64) once per window size of the
shipped DiffVAE decoder with that size's AMD ``na_geometry`` config, and ``diffusion_triton``'s residual
SwiGLU and slab Q/K/V kernels at the shipped diffusion-block shapes (Q/K/V also at the 512-wide
deterministic stages), and the eager encoder's channels-first copy (``layout_triton``). Windows ROCm ships no Triton, so ``na_hip`` and ``diffusion_hip`` launch these there.

Usage (Linux, ROCm Torch + Triton, ROCm's clang-offload-bundler):
    python scripts/build_ltx_vae_hipfbs.py

Writes ``jasna/media/ltx_na3d_*.hipfb``, ``ltx_swiglu_*.hipfb``, ``ltx_slab_qkv_*.hipfb``,
``ltx_channels_first_*.hipfb`` and the launch table
``jasna/models/ltx_vae/hip_launch.py`` (the grids come from the launchers).
"""
from __future__ import annotations

from triton_hipfb import ARCHITECTURES, MEDIA, REPO, WARP_SIZE, _write_launch_table, bundle, compile_code_object

from jasna.models.ltx_vae import diffusion_hip as aot
from jasna.models.ltx_vae import layout_hip, na_hip
from jasna.models.ltx_vae.diffusion_triton import _residual_swiglu_kernel, _slab_qkv_kernel
from jasna.models.ltx_vae.layout_triton import _channels_first_kernel
from jasna.models.ltx_vae.na_geometry import config_for
from jasna.models.ltx_vae.na_triton import _na3d_fwd_kernel

LAUNCH_TABLE = REPO / "jasna" / "models" / "ltx_vae" / "hip_launch.py"


def _kernel_specs():
    """stem -> (kernel, arguments, constants, num_warps, num_stages)"""
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
            _na3d_fwd_kernel, na_hip.KERNEL_ARGUMENTS, constants, config.num_warps, config.num_stages,
        )
    specs[aot.SWIGLU_MODULE] = (
        _residual_swiglu_kernel, aot.SWIGLU_ARGUMENTS,
        {"DIM": aot.DIM, "HIDDEN": aot.HIDDEN, "BLOCK_ROWS": aot.SWIGLU_BLOCK_ROWS, "BLOCK_HIDDEN": aot.SWIGLU_BLOCK_HIDDEN},
        aot.SWIGLU_WARPS, 1,
    )
    for dim in aot.QKV_DIMS:
        specs[aot.qkv_module(dim)] = (
            _slab_qkv_kernel, aot.QKV_ARGUMENTS,
            {"DIM": dim, "HEAD_DIM": aot.HEAD_DIM, "PAIRS_T": aot.ROPE_PAIRS[0], "PAIRS_H": aot.ROPE_PAIRS[1],
             "BLOCK_ROWS": aot.QKV_BLOCK_ROWS, "MODULATE": dim == aot.DIM},
            aot.QKV_WARPS, 1,
        )
    specs[layout_hip.MODULE] = (
        _channels_first_kernel, layout_hip.ARGUMENTS, {"BLOCK_S": layout_hip.BLOCK_S, "BLOCK_C": layout_hip.BLOCK_C},
        layout_hip.WARPS, 1,
    )
    return specs


def main() -> None:
    launch: dict[str, dict[str, tuple]] = {architecture: {} for architecture in ARCHITECTURES}
    for stem, (kernel, arguments, constants, num_warps, num_stages) in _kernel_specs().items():
        options = {"num_warps": num_warps, "num_stages": num_stages}
        code_objects = {}
        for architecture in ARCHITECTURES:
            code, name, shared = compile_code_object(kernel, arguments, constants, options, architecture)
            code_objects[architecture] = code
            launch[architecture][stem] = (name, (0, 0, 0), num_warps * WARP_SIZE, shared)
        output = MEDIA / f"{stem}.hipfb"
        bundle(code_objects, output)
        print(f"{output.name:34s} {output.stat().st_size:8d} bytes  {' '.join(ARCHITECTURES)}")
    _write_launch_table(launch, LAUNCH_TABLE, "LTX VAE decoder kernels", "scripts/build_ltx_vae_hipfbs.py")


if __name__ == "__main__":
    main()
