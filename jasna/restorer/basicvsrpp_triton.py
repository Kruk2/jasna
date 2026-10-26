"""Triton source of the fused BasicVSR++ propagation kernels, which ``scripts/build_basicvsrpp_hipfbs.py``
builds ahead of time for AMD (``basicvsrpp_fused_body`` launches them)."""

import triton
import triton.language as tl


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
