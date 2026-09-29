# LTX denoise per GPU architecture (2026-09-30)

`scripts/benchmark_ltx_denoise.py`: trial weights, INT8, one track of 3 windows, median of 3, no SageAttention.
All rows are 4dc6a30 (kernel phase 2 + per-architecture INT8 tiles). Baselines: `main-torch` a0daf57 on RTX 3090/4090
(`benchmarks/2026-09-30_ltx_arch_tiles.csv`, `torch._int_mm` path) and `autotune` 46e4384/ef67e1f on RTX 5090
(`benchmarks/2026-09-29_ltx_denoise.csv`, before the Triton kernels). RTX 3090/4090 on rented vast.ai boxes, RTX 5090 local.

| GPU | Model, canvas | Baseline s/window | 4dc6a30 s/window | Change |
|---|---|---|---|---|
| RTX 3090 | distilled, 512 | 18.54 | 15.73 | −15 % |
| RTX 3090 | distilled, 768 | 60.18 | 54.96 | −9 % |
| RTX 3090 | undistilled, 512 | 48.99 | 41.71 | −15 % |
| RTX 4090 | distilled, 512 | 8.17 | 6.60 | −19 % |
| RTX 4090 | distilled, 768 | 26.12 | 22.49 | −14 % |
| RTX 4090 | undistilled, 512 | 21.68 | 17.56 | −19 % |
| RTX 5090 | distilled, 512 | 5.13 | 3.92 | −24 % |
| RTX 5090 | distilled, 768 | 13.66 | 10.76 | −21 % |
| RTX 5090 | undistilled, 512 | 13.37 | 10.41 | −22 % |

GPU time of one window-step on 4dc6a30 (`--profile`, distilled 512): RTX 3090 1941 ms (attention 50 %, INT8 GEMM 40 %),
RTX 4090 831 ms (attention 45 %, INT8 GEMM 41 %). Attention (cuDNN SDPA) is at the bf16 peak there, so SageAttention 2
is the remaining lever on RTX 30/40.
