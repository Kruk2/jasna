"""Build the fused BasicVSR++ propagation kernels into committed AMD .hipfb bundles.

The kernels are written in Triton, which AMD ships for Linux only, so they are
compiled ahead of time here and loaded at run time on Linux and Windows through
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

import triton
import triton.language as tl
from triton_hipfb import REPO, build

LAUNCH_TABLE = REPO / "jasna" / "restorer" / "basicvsrpp_hip_launch.py"

CHANNELS = 64
SIZE = 64
DEFORM_GROUPS = 16
MAX_RESIDUE_MAGNITUDE = 10.0

TILES = {
    "conv3x3": {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64, "num_warps": 4},
    "dcn": {"BLOCK_M": 32, "BLOCK_K": 16, "num_warps": 4},
}


@triton.jit
def conv3x3_kernel(
    x, w, b, res, out,
    C: tl.constexpr, H: tl.constexpr, W: tl.constexpr, RELU: tl.constexpr, HAS_RES: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """3x3 conv, stride 1, pad 1, NCHW, implicit GEMM; bias, ReLU and residual fused.

    ``w`` is packed ``[9, C, C]`` (tap, in, out).
    """
    HW: tl.constexpr = H * W
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    oy = offs_m // W
    ox = offs_m % W
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for tap in tl.static_range(9):
        iy = oy + (tap // 3 - 1)
        ix = ox + (tap % 3 - 1)
        inside = (iy >= 0) & (iy < H) & (ix >= 0) & (ix < W)
        pix = iy * W + ix
        for c0 in tl.static_range(0, C, BLOCK_K):
            ck = c0 + tl.arange(0, BLOCK_K)
            a = tl.load(x + ck[None, :] * HW + pix[:, None], mask=inside[:, None], other=0.0)
            bw = tl.load(w + (tap * C + ck[:, None]) * C + offs_n[None, :])
            acc = tl.dot(a, bw, acc)
    acc += tl.load(b + offs_n).to(tl.float32)[None, :]
    if RELU:
        acc = tl.maximum(acc, 0.0)
    o_idx = offs_n[None, :] * HW + offs_m[:, None]
    if HAS_RES:
        acc += tl.load(res + o_idx).to(tl.float32)
    tl.store(out + o_idx, acc.to(tl.float16))


@triton.jit
def dcn_kernel(
    xa, xb, om, f1, f2, w, b, out,
    C_HALF: tl.constexpr, COUT: tl.constexpr, G: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    MAX_RES: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """BasicVSR++ second-order modulated deformable conv, offsets and mask built inline.

    ``om`` is the raw ``conv_offset`` output (offset 1, offset 2, mask logits);
    groups ``0..G/2-1`` sample ``xa`` displaced by ``f1``, the rest ``xb`` by
    ``f2``. Bilinear sampling follows torchvision's ``deform_conv2d`` boundary
    rules in fp32. ``w`` is packed ``[9, 2 * C_HALF, COUT]``.
    """
    HW: tl.constexpr = H * W
    CPG: tl.constexpr = 2 * C_HALF // G
    HALF_G: tl.constexpr = G // 2
    GK: tl.constexpr = BLOCK_K // CPG
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    oy = (offs_m // W).to(tl.float32)
    ox = (offs_m % W).to(tl.float32)
    offs_n = tl.arange(0, COUT)
    acc = tl.zeros((BLOCK_M, COUT), dtype=tl.float32)
    for src in tl.static_range(2):
        xs = xa if src == 0 else xb
        fl = f1 if src == 0 else f2
        fx = tl.load(fl + offs_m).to(tl.float32)
        fy = tl.load(fl + HW + offs_m).to(tl.float32)
        o_base = om + (src * 18 * HALF_G) * HW
        for tap in tl.static_range(9):
            for c0 in range(0, C_HALF, BLOCK_K):
                ck = c0 + tl.arange(0, BLOCK_K)
                gl = c0 // CPG + tl.arange(0, GK)
                ch_off = 2 * (gl * 9 + tap)
                raw_dy = tl.load(o_base + ch_off[None, :] * HW + offs_m[:, None]).to(tl.float32)
                raw_dx = tl.load(o_base + (ch_off + 1)[None, :] * HW + offs_m[:, None]).to(tl.float32)
                raw_m = tl.load(om + (36 * HALF_G + (src * HALF_G + gl) * 9 + tap)[None, :] * HW + offs_m[:, None]).to(tl.float32)
                dy = MAX_RES * (2.0 / (1.0 + tl.exp(-2.0 * raw_dy)) - 1.0) + fy[:, None]
                dx = MAX_RES * (2.0 / (1.0 + tl.exp(-2.0 * raw_dx)) - 1.0) + fx[:, None]
                mk = 1.0 / (1.0 + tl.exp(-raw_m))
                dy = tl.reshape(tl.broadcast_to(tl.expand_dims(dy, 2), (BLOCK_M, GK, CPG)), (BLOCK_M, BLOCK_K))
                dx = tl.reshape(tl.broadcast_to(tl.expand_dims(dx, 2), (BLOCK_M, GK, CPG)), (BLOCK_M, BLOCK_K))
                mk = tl.reshape(tl.broadcast_to(tl.expand_dims(mk, 2), (BLOCK_M, GK, CPG)), (BLOCK_M, BLOCK_K))
                py = oy[:, None] + (tap // 3 - 1) + dy
                px = ox[:, None] + (tap % 3 - 1) + dx
                inside = (py > -1.0) & (py < H) & (px > -1.0) & (px < W)
                y0f = tl.floor(py)
                x0f = tl.floor(px)
                ly = py - y0f
                lx = px - x0f
                y0 = y0f.to(tl.int32)
                x0 = x0f.to(tl.int32)
                y1 = y0 + 1
                x1 = x0 + 1
                cbase = xs + ck[None, :] * HW
                v00 = tl.load(cbase + y0 * W + x0, mask=inside & (y0 >= 0) & (x0 >= 0), other=0.0).to(tl.float32)
                v01 = tl.load(cbase + y0 * W + x1, mask=inside & (y0 >= 0) & (x1 < W), other=0.0).to(tl.float32)
                v10 = tl.load(cbase + y1 * W + x0, mask=inside & (y1 < H) & (x0 >= 0), other=0.0).to(tl.float32)
                v11 = tl.load(cbase + y1 * W + x1, mask=inside & (y1 < H) & (x1 < W), other=0.0).to(tl.float32)
                val = ((1 - ly) * ((1 - lx) * v00 + lx * v01) + ly * ((1 - lx) * v10 + lx * v11)) * mk
                wt = tl.load(w + (tap * 2 * C_HALF + src * C_HALF + ck)[:, None] * COUT + offs_n[None, :])
                acc = tl.dot(val.to(tl.float16), wt, acc)
    acc += tl.load(b + offs_n).to(tl.float32)[None, :]
    tl.store(out + offs_n[None, :] * HW + offs_m[:, None], acc.to(tl.float16))


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


# module stem -> (kernel, pointer arguments, constexprs, num_warps, grid)
def _kernel_specs() -> dict[str, tuple]:
    conv_grid = (SIZE * SIZE // TILES["conv3x3"]["BLOCK_M"], CHANNELS // TILES["conv3x3"]["BLOCK_N"], 1)
    dcn_grid = (SIZE * SIZE // TILES["dcn"]["BLOCK_M"], 1, 1)
    conv_args = ("x", "w", "b", "res", "out")
    return {
        "basicvsrpp_conv3x3_relu": (
            conv3x3_kernel, conv_args, _conv_constants(True, False), TILES["conv3x3"]["num_warps"], conv_grid,
        ),
        "basicvsrpp_conv3x3_residual": (
            conv3x3_kernel, conv_args, _conv_constants(False, True), TILES["conv3x3"]["num_warps"], conv_grid,
        ),
        "basicvsrpp_dcn": (
            dcn_kernel, ("xa", "xb", "om", "f1", "f2", "w", "b", "out"), _dcn_constants(),
            TILES["dcn"]["num_warps"], dcn_grid,
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
