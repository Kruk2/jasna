"""Build the fused BasicVSR++ propagation kernels into committed AMD .hipfb bundles.

The kernels (Triton source: ``jasna/restorer/basicvsrpp_triton.py``) need Triton,
which AMD ships for Linux only, so they are compiled ahead of time here and loaded at run time on Linux and Windows through
``jasna.media.cuda_kernel`` like the other ``.hipfb`` bundles. Shapes are fixed
to the propagation step (batch 1, 64 channels, 64x64 features, fp16).

Usage (Linux, ROCm Torch + Triton, ROCm's clang-offload-bundler):
    python scripts/build_basicvsrpp_hipfbs.py          # build every architecture
    python scripts/build_basicvsrpp_hipfbs.py --tune   # time tile candidates on the local GPU

Writes ``jasna/media/basicvsrpp_*.hipfb`` and the launch table
``jasna/restorer/basicvsrpp_hip_launch.py``. Tiles come from ``TILES``, picked
with ``--tune`` on a gfx1201 (RDNA4); RDNA3 uses the same tiles.
"""
from __future__ import annotations

import argparse

from triton_hipfb import REPO, build

from jasna.restorer.basicvsrpp_fused_body import CONV_ARGUMENTS, DCN_ARGUMENTS
from jasna.restorer.basicvsrpp_triton import conv3x3_kernel, dcn_kernel

LAUNCH_TABLE = REPO / "jasna" / "restorer" / "basicvsrpp_hip_launch.py"

CHANNELS = 64
SIZE = 64
DEFORM_GROUPS = 16
MAX_RESIDUE_MAGNITUDE = 10.0

TILES = {
    "conv3x3": {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64, "num_warps": 4},
    "dcn": {"BLOCK_M": 32, "BLOCK_K": 16, "num_warps": 4},
}


def _conv_constants(relu: bool, has_res: bool) -> dict:
    tile = TILES["conv3x3"]
    return {
        "C": CHANNELS, "H": SIZE, "W": SIZE, "RELU": relu, "HAS_RES": has_res,
        "BLOCK_M": tile["BLOCK_M"], "BLOCK_N": tile["BLOCK_N"], "BLOCK_K": tile["BLOCK_K"],
    }


def _dcn_constants() -> dict:
    tile = TILES["dcn"]
    return {
        "C_HALF": CHANNELS, "COUT": CHANNELS, "G": DEFORM_GROUPS, "H": SIZE, "W": SIZE,
        "MAX_RES": MAX_RESIDUE_MAGNITUDE, "BLOCK_M": tile["BLOCK_M"], "BLOCK_K": tile["BLOCK_K"],
    }


# module stem -> (kernel, arguments, constexprs, num_warps, num_stages, grid)
def _kernel_specs() -> dict[str, tuple]:
    conv_grid = (SIZE * SIZE // TILES["conv3x3"]["BLOCK_M"], CHANNELS // TILES["conv3x3"]["BLOCK_N"], 1)
    dcn_grid = (SIZE * SIZE // TILES["dcn"]["BLOCK_M"], 1, 1)
    return {
        "basicvsrpp_conv3x3_relu": (
            conv3x3_kernel, CONV_ARGUMENTS, _conv_constants(True, False), TILES["conv3x3"]["num_warps"], 1, conv_grid,
        ),
        "basicvsrpp_conv3x3_residual": (
            conv3x3_kernel, CONV_ARGUMENTS, _conv_constants(False, True), TILES["conv3x3"]["num_warps"], 1, conv_grid,
        ),
        "basicvsrpp_dcn": (
            dcn_kernel, DCN_ARGUMENTS, _dcn_constants(), TILES["dcn"]["num_warps"], 1, dcn_grid,
        ),
    }


def tune() -> None:
    import torch
    from triton.testing import do_bench

    device = torch.device("cuda")
    feature = lambda channels: torch.randn(1, channels, SIZE, SIZE, device=device, dtype=torch.float16)
    weight = torch.randn(9, CHANNELS, CHANNELS, device=device, dtype=torch.float16) * 0.05
    bias = torch.randn(CHANNELS, device=device, dtype=torch.float16)
    x, res, out = feature(CHANNELS), feature(CHANNELS), feature(CHANNELS)
    print("conv3x3 (BLOCK_M, BLOCK_N, BLOCK_K, num_warps): us")
    for bm, bn, bk, nw in [(16, 64, 64, 2), (32, 64, 32, 4), (32, 64, 64, 4), (64, 64, 32, 4), (64, 64, 64, 4), (32, 32, 64, 2), (64, 32, 64, 4)]:
        grid = (SIZE * SIZE // bm, CHANNELS // bn)
        run = lambda: conv3x3_kernel[grid](
            x, weight, bias, res, out, C=CHANNELS, H=SIZE, W=SIZE, RELU=False, HAS_RES=True,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=nw, num_stages=1,
        )
        print(f"  {(bm, bn, bk, nw)}: {do_bench(run, warmup=50, rep=300) * 1000:.1f}")

    om = feature(27 * DEFORM_GROUPS)
    flow_1, flow_2 = feature(2) * 3, feature(2) * 3
    dcn_weight = torch.randn(9, 2 * CHANNELS, CHANNELS, device=device, dtype=torch.float16) * 0.05
    print("dcn (BLOCK_M, BLOCK_K, num_warps): us")
    for bm, bk, nw in [(32, 16, 4), (64, 16, 4), (32, 32, 4), (64, 32, 4), (64, 32, 8), (128, 16, 8)]:
        run = lambda: dcn_kernel[(SIZE * SIZE // bm,)](
            x, res, om, flow_1, flow_2, dcn_weight, bias, out, C_HALF=CHANNELS, COUT=CHANNELS, G=DEFORM_GROUPS,
            H=SIZE, W=SIZE, MAX_RES=MAX_RESIDUE_MAGNITUDE, BLOCK_M=bm, BLOCK_K=bk, num_warps=nw, num_stages=1,
        )
        print(f"  {(bm, bk, nw)}: {do_bench(run, warmup=50, rep=300) * 1000:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tune", action="store_true", help="time tile candidates on the local GPU instead of building")
    if parser.parse_args().tune:
        tune()
    else:
        build(_kernel_specs(), LAUNCH_TABLE, "fused BasicVSR++ kernels", "scripts/build_basicvsrpp_hipfbs.py")


if __name__ == "__main__":
    main()
