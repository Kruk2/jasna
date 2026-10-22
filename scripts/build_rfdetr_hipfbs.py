"""Build the fused RF-DETR segmentation-head kernel into committed AMD .hipfb bundles.

One kernel runs a whole ``DepthwiseConvBlock``: 3x3 depthwise conv + bias,
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
import triton.language as tl
from triton_hipfb import REPO, build

LAUNCH_TABLE = REPO / "jasna" / "mosaic" / "rfdetr_hip_launch.py"

CHANNELS = 256
FEATURE_SIZES = (144, 192)
LAYER_NORM_EPS = 1e-6
TILE = {"BLOCK_P": 32, "BLOCK_N": 64, "num_warps": 8}


@triton.jit
def dwblock_kernel(
    x, dw_w, dw_b, ln_w, ln_b, pw_w, pw_b, out,
    C: tl.constexpr, H: tl.constexpr, W: tl.constexpr, EPS: tl.constexpr,
    BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """``x``/``out`` are NHWC; ``dw_w`` is ``[C, 9]``, ``pw_w`` the Linear's ``[out, in]``. Grid: (pixel tiles, batch)."""
    HW: tl.constexpr = H * W
    n = tl.program_id(1)
    offs_p = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    pmask = offs_p < HW
    oy = offs_p // W
    ox = offs_p % W
    offs_c = tl.arange(0, C)
    xb = x + n * C * HW
    acc = tl.zeros((BLOCK_P, C), dtype=tl.float32)
    for tap in tl.static_range(9):
        iy = oy + (tap // 3 - 1)
        ix = ox + (tap % 3 - 1)
        inside = pmask & (iy >= 0) & (iy < H) & (ix >= 0) & (ix < W)
        v = tl.load(xb + offs_c[None, :] + (iy * W + ix)[:, None] * C, mask=inside[:, None], other=0.0)
        wt = tl.load(dw_w + offs_c * 9 + tap).to(tl.float32)
        acc += v.to(tl.float32) * wt[None, :]
    acc += tl.load(dw_b + offs_c).to(tl.float32)[None, :]
    acc = acc.to(tl.float16).to(tl.float32)
    mean = tl.sum(acc, axis=1) / C
    d = acc - mean[:, None]
    var = tl.sum(d * d, axis=1) / C
    rstd = 1.0 / tl.sqrt(var + EPS)
    y = d * rstd[:, None] * tl.load(ln_w + offs_c).to(tl.float32)[None, :] + tl.load(ln_b + offs_c).to(tl.float32)[None, :]
    y16 = y.to(tl.float16)
    ob = out + n * C * HW
    for n0 in tl.static_range(0, C, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        wt = tl.load(pw_w + offs_n[None, :] * C + offs_c[:, None])
        z = tl.dot(y16, wt)
        z += tl.load(pw_b + offs_n).to(tl.float32)[None, :]
        z = z.to(tl.float16).to(tl.float32)
        z = 0.5 * z * (1.0 + tl.erf(z * 0.7071067811865476))
        z = z.to(tl.float16).to(tl.float32)
        o_idx = offs_n[None, :] + offs_p[:, None] * C
        r = tl.load(xb + o_idx, mask=pmask[:, None], other=0.0).to(tl.float32)
        tl.store(ob + o_idx, (z + r).to(tl.float16), mask=pmask[:, None])


POINTER_ARGS = ("x", "dw_w", "dw_b", "ln_w", "ln_b", "pw_w", "pw_b", "out")


def _constants(size: int, block_p: int, block_n: int) -> dict:
    return {"C": CHANNELS, "H": size, "W": size, "EPS": LAYER_NORM_EPS, "BLOCK_P": block_p, "BLOCK_N": block_n}


# module stem -> (kernel, pointer arguments, constexprs, num_warps, grid)
def _kernel_specs() -> dict[str, tuple]:
    return {
        f"rfdetr_dwblock_{size}": (
            dwblock_kernel, POINTER_ARGS, _constants(size, TILE["BLOCK_P"], TILE["BLOCK_N"]), TILE["num_warps"],
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
        build(_kernel_specs(), LAUNCH_TABLE, "fused RF-DETR segmentation-head kernel", "scripts/build_rfdetr_hipfbs.py")


if __name__ == "__main__":
    main()
