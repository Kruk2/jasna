"""Build the fused RF-DETR segmentation-head kernel into committed AMD .hipfb bundles.

One kernel (Triton source: ``jasna/mosaic/rfdetr_triton.py``) runs a whole ``DepthwiseConvBlock``: 3x3 depthwise conv + bias,
LayerNorm, Linear, exact GELU and the residual add, on channels-last fp16
features with fp32 math, rounded to fp16 where torch rounds. Built per feature
size: 144x144 (576 px models) and 192x192 (768 px models), 256 channels.

Usage (Linux, ROCm Torch + Triton, ROCm's clang-offload-bundler):
    python scripts/build_rfdetr_hipfbs.py          # build every architecture
    python scripts/build_rfdetr_hipfbs.py --tune   # time tile candidates on the local GPU

Writes ``jasna/media/rfdetr_dwblock_*.hipfb`` and the launch table
``jasna/mosaic/rfdetr_hip_launch.py`` (grid y is the batch, set at launch).
Tiles come from ``TILE``, picked with ``--tune`` on a gfx1201 (RDNA4).
"""
from __future__ import annotations

import argparse

import triton
from triton_hipfb import REPO, build

from jasna.mosaic.rfdetr_fused_head import ARGUMENTS
from jasna.mosaic.rfdetr_triton import dwblock_kernel

LAUNCH_TABLE = REPO / "jasna" / "mosaic" / "rfdetr_hip_launch.py"

CHANNELS = 256
FEATURE_SIZES = (144, 192)
LAYER_NORM_EPS = 1e-6
TILE = {"BLOCK_P": 32, "BLOCK_N": 64, "num_warps": 8}


def _constants(size: int, block_p: int, block_n: int) -> dict:
    return {"C": CHANNELS, "H": size, "W": size, "EPS": LAYER_NORM_EPS, "BLOCK_P": block_p, "BLOCK_N": block_n}


# module stem -> (kernel, arguments, constexprs, num_warps, num_stages, grid)
def _kernel_specs() -> dict[str, tuple]:
    return {
        f"rfdetr_dwblock_{size}": (
            dwblock_kernel, ARGUMENTS, _constants(size, TILE["BLOCK_P"], TILE["BLOCK_N"]), TILE["num_warps"], 1,
            (triton.cdiv(size * size, TILE["BLOCK_P"]), 1, 1),
        )
        for size in FEATURE_SIZES
    }


def tune() -> None:
    import torch
    from triton.testing import do_bench

    device = torch.device("cuda")
    weight = lambda *shape: torch.randn(*shape, device=device, dtype=torch.float16) * 0.05
    dw_w, dw_b, ln_w, ln_b, pw_w, pw_b = weight(CHANNELS, 9), weight(CHANNELS), weight(CHANNELS) + 1, weight(CHANNELS), weight(CHANNELS, CHANNELS), weight(CHANNELS)
    for size in FEATURE_SIZES:
        x = torch.randn(4, CHANNELS, size, size, device=device, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
        out = torch.empty_like(x)
        print(f"{size}x{size}, batch 4 (BLOCK_P, BLOCK_N, num_warps): us")
        for bp, bn, nw in [(16, 64, 4), (32, 64, 4), (16, 64, 8), (32, 64, 8), (32, 128, 8), (64, 64, 8), (32, 64, 16), (64, 64, 16)]:
            run = lambda: dwblock_kernel[(triton.cdiv(size * size, bp), x.shape[0])](
                x, dw_w, dw_b, ln_w, ln_b, pw_w, pw_b, out, **_constants(size, bp, bn), num_warps=nw, num_stages=1,
            )
            print(f"  {(bp, bn, nw)}: {do_bench(run, warmup=50, rep=300) * 1000:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tune", action="store_true", help="time tile candidates on the local GPU instead of building")
    if parser.parse_args().tune:
        tune()
    else:
        specs = _kernel_specs()
        build(lambda architecture: specs, LAUNCH_TABLE, "fused RF-DETR segmentation-head kernel",
              "scripts/build_rfdetr_hipfbs.py")


if __name__ == "__main__":
    main()
