# LTX INT8 kernels on RTX 3090 / 4090 (2026-09-30)

`scripts/benchmark_ltx_denoise.py`: trial weights, one track of 3 windows, median of 3, compile cache empty per checkout.
Rented vast.ai boxes: RTX 3090 (driver 590.48, 350 W), RTX 4090 (driver 595.91, 430 W); torch 2.12.1+cu130, Triton 3.7.1, no SageAttention.
`main-torch` = a0daf57 (`torch._int_mm` path; the Triton kernels were SM 12 only); `arch-tiles` = f889b39 (Triton INT8 kernels with per-architecture tiles).

| GPU | Model, canvas | main-torch s/window | arch-tiles s/window | Change |
|---|---|---|---|---|
| RTX 3090 | distilled, 512 | 18.54 | 16.21 | −13 % |
| RTX 3090 | distilled, 768 | 60.18 | 55.28 | −8 % |
| RTX 3090 | undistilled, 512 | 48.99 | 42.35 | −14 % |
| RTX 4090 | distilled, 512 | 8.17 | 6.88 | −16 % |
| RTX 4090 | distilled, 768 | 26.12 | 23.05 | −12 % |
| RTX 4090 | undistilled, 512 | 21.68 | 18.22 | −16 % |

First compile: 38–46 s (3090) and 15–21 s (4090) with the kernels vs 54–135 s and 28–70 s without.

GPU health check (8192³ matmul, 20 s sustained): 3090 bf16 72 TFLOPS, int8 241 TOPS; 4090 bf16 167 TFLOPS, int8 613 TOPS.

INT8 GEMM tile sweep (`M` 8192/16384 × attention, FF-up, FF-down shapes; bit-exact vs the reference):

| GPU | Best tile (BM, BN, BK, stages, warps) | Kernel TOPS | vs `torch._int_mm` + rescale |
|---|---|---|---|
| RTX 3090 | 128, 128, 64, 3, 4 | 212–228 | 1.2–2.2× faster |
| RTX 4090 | 256, 128, 128, 3, 8 | 551–577 | 1.6–3.9× faster |

GPU time of one window-step with the kernels (`--profile`, distilled 512):

| GPU | Total ms | Attention (cuDNN SDPA) | INT8 GEMM | rotate+quantize | Other fused Triton |
|---|---|---|---|---|---|
| RTX 3090 | 1986 | 50 % | 40 % | 4 % | 6 % |
| RTX 4090 | 863 | 43 % | 39 % | 7 % | 11 % |

Attention already runs at the bf16 peak (3090 68 TFLOPS, 4090 162–165) and cuDNN and Flash attention tie at LTX shapes (1 × 32 heads × 4k–18k tokens × 128),
so only lower-precision attention (SageAttention 2) can cut it further. `rotate_quantize` runs at 800–917 GB/s with 8 warps; no other warp count is faster.
