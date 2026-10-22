"""Build the LTX VAE decoder's Triton kernels into committed AMD .hipfb bundles.

Compiled ahead of time, bf16: ``na_triton._na3d_fwd_kernel`` (head dim 64) once per window size of the
shipped DiffVAE decoder with that size's AMD ``na_geometry`` config, and
``swiglu_triton._residual_swiglu_kernel`` once per diffusion-block MLP shape. Windows ROCm ships no
Triton, so ``na_hip`` and ``swiglu_hip`` launch these there.

Usage (Linux, ROCm Torch + Triton, ROCm's clang-offload-bundler):
    python scripts/build_ltx_vae_hipfbs.py

Writes ``jasna/media/ltx_na3d_*.hipfb``, ``jasna/media/ltx_swiglu_*.hipfb`` and the launch table
``jasna/models/ltx_vae/hip_launch.py`` (the grids come from the launchers).
"""
from __future__ import annotations

from triton_hipfb import ARCHITECTURES, MEDIA, REPO, WARP_SIZE, _write_launch_table, bundle, compile_code_object

from jasna.models.ltx_vae import na_hip, swiglu_hip
from jasna.models.ltx_vae.na_geometry import config_for
from jasna.models.ltx_vae.na_triton import _na3d_fwd_kernel
from jasna.models.ltx_vae.swiglu_triton import _residual_swiglu_kernel

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
    for dim, hidden in swiglu_hip.SHAPES:
        constants = {
            "DIM": dim, "HIDDEN": hidden, "BLOCK_ROWS": swiglu_hip.BLOCK_ROWS, "BLOCK_HIDDEN": swiglu_hip.BLOCK_HIDDEN,
        }
        specs[swiglu_hip.module_name((dim, hidden))] = (
            _residual_swiglu_kernel, swiglu_hip.KERNEL_ARGUMENTS, constants, swiglu_hip.NUM_WARPS, 1,
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
