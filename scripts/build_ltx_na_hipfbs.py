"""Build the LTX VAE decoder's neighborhood-attention kernel into committed AMD .hipfb bundles.

``na_triton._na3d_fwd_kernel`` compiled ahead of time, bf16, head dim 64, once per window size of
the shipped DiffVAE decoder with that size's AMD ``na_geometry`` config. Windows ROCm ships no
Triton, so ``na_hip`` launches these there.

Usage (Linux, ROCm Torch + Triton, ROCm's clang-offload-bundler):
    python scripts/build_ltx_na_hipfbs.py

Writes ``jasna/media/ltx_na3d_*.hipfb`` and the launch table ``jasna/models/ltx_vae/na_hip_launch.py``
(the grid comes from ``na_geometry.na_launch`` at launch).
"""
from __future__ import annotations

from triton_hipfb import ARCHITECTURES, MEDIA, REPO, WARP_SIZE, _write_launch_table, bundle, compile_code_object

from jasna.models.ltx_vae.na_geometry import config_for
from jasna.models.ltx_vae.na_hip import HEAD_DIM, KERNEL_ARGUMENTS, WINDOWS, module_name
from jasna.models.ltx_vae.na_triton import _na3d_fwd_kernel

LAUNCH_TABLE = REPO / "jasna" / "models" / "ltx_vae" / "na_hip_launch.py"


def main() -> None:
    launch: dict[str, dict[str, tuple]] = {architecture: {} for architecture in ARCHITECTURES}
    for window in WINDOWS:
        config = config_for(window, amd=True)
        tile_t, tile_h, tile_w = config.tile
        constants = {
            "KERNEL_T": window[0], "KERNEL_H": window[1], "KERNEL_W": window[2], "HEAD_DIM": HEAD_DIM,
            "TILE_T": tile_t, "TILE_H": tile_h, "TILE_W": tile_w, "BLOCK_N": config.block_n,
            "BOX_ROWS": config.box_rows,
        }
        options = {"num_warps": config.num_warps, "num_stages": config.num_stages}
        stem = module_name(window)
        code_objects = {}
        for architecture in ARCHITECTURES:
            code, name, shared = compile_code_object(_na3d_fwd_kernel, KERNEL_ARGUMENTS, constants, options, architecture)
            code_objects[architecture] = code
            launch[architecture][stem] = (name, (0, 0, 0), config.num_warps * WARP_SIZE, shared)
        output = MEDIA / f"{stem}.hipfb"
        bundle(code_objects, output)
        print(f"{output.name:34s} {output.stat().st_size:8d} bytes  {' '.join(ARCHITECTURES)}")
    _write_launch_table(launch, LAUNCH_TABLE, "LTX VAE neighborhood-attention kernel", "scripts/build_ltx_na_hipfbs.py")


if __name__ == "__main__":
    main()
