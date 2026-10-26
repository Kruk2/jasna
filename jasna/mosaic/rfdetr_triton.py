"""Triton source of the fused RF-DETR segmentation-head kernel, which ``scripts/build_rfdetr_hipfbs.py``
builds ahead of time for AMD (``rfdetr_fused_head`` launches it)."""

import triton
import triton.language as tl


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
