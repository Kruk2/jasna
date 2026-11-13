"""Build the LTX VAE encoder's fused kernels (``encoder_triton``) into committed AMD .hipfb bundles.

Compiled ahead of time for every ``encoder_hip.NORM_PAD_BUILDS`` entry and ``DOWNSAMPLE_WIDTHS`` width;
every AMD GPU without Triton runs the encoder through ``encoder_hip``.

Usage (Linux, ROCm Torch + Triton, ROCm's clang-offload-bundler):
    python scripts/build_ltx_encoder_hipfbs.py

Writes ``jasna/media/ltx_encoder_*.hipfb`` and the launch table ``jasna/models/ltx_vae/encoder_hip_launch.py``.
"""
from __future__ import annotations

from triton_hipfb import REPO, build

from jasna.models.ltx_vae import encoder_hip as aot
from jasna.models.ltx_vae.encoder_triton import _downsample_add_kernel, _norm_pad_kernel

LAUNCH_TABLE = REPO / "jasna" / "models" / "ltx_vae" / "encoder_hip_launch.py"
_GRID = (0, 0, 0)


def _kernel_specs(architecture: str):
    """stem -> (kernel, arguments, constants, num_warps, num_stages, grid); the launchers compute the grids."""
    specs = {}
    for source, target, block_c, norm, add in aot.NORM_PAD_BUILDS:
        specs[aot.norm_pad_module(source, target, block_c, norm, add)] = (
            _norm_pad_kernel, aot.norm_pad_arguments(source, target),
            {"NORM": norm, "ADD": add, "BLOCK_ROWS": aot.block_rows(block_c), "BLOCK_C": block_c}, aot.WARPS, 1, _GRID,
        )
    for channels in aot.DOWNSAMPLE_WIDTHS:
        specs[aot.downsample_module(channels)] = (
            _downsample_add_kernel, aot.DOWNSAMPLE_ARGUMENTS,
            {"C_OUT": channels, "BLOCK_ROWS": aot.block_rows(channels)}, aot.WARPS, 1, _GRID,
        )
    return specs


def main() -> None:
    build(_kernel_specs, LAUNCH_TABLE, "LTX VAE encoder kernels", "scripts/build_ltx_encoder_hipfbs.py")


if __name__ == "__main__":
    main()
