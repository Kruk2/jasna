# LTX denoise, RTX 5090 (2026-09-29)

`scripts/benchmark_ltx_denoise.py --fresh-cache`: trial weights, one track of 3 windows, median of 3.
`deterministic` = 81dc8b0 (Inductor deterministic mode); `autotune` = 5aff5aa (max-autotune, cached picks).

| Model, canvas | Quant | deterministic s/window | autotune s/window | Change |
|---|---|---|---|---|
| distilled, 512 | INT8 | 5.11 | 5.13 | 0 % |
| distilled, 512 | NVFP4 | 3.79 | 3.60 | −5 % |
| undistilled, 512 | INT8 | 13.98 | 13.37 | −4 % |
| distilled, 768 | INT8 | 13.90 | 13.66 | −2 % |
| distilled, 768 | NVFP4 | – | 10.05 | |
| undistilled, 512 | NVFP4 | – | 9.48 | |

First compile on an empty cache: 7–15 s deterministic, 18–55 s autotune (once per canvas and model; cached after).
`force_fuse_int_mm_with_mul=True`: 5.32 s/window, 4 % slower than without.

GPU time of one window-step (`--profile`, autotune):

| Run | Total ms | GEMM | Attention | Fused Triton (quant, norms) | Other |
|---|---|---|---|---|---|
| INT8, 512 | 639 | 57 % | 16 % | 18 % | 8 % |
| NVFP4, 512 | 445 | 34 % | 29 % | 24 % | 13 % |
| INT8, 768 | 1688 | 49 % | 31 % | 19 % | 1 % |
| NVFP4, 768 | 1277 | 27 % | 40 % | 22 % | 11 % |

The undistilled step costs 1.39× a distilled one, almost all of it the STG pass (blocks 28–47 run twice).
