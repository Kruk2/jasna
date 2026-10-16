# LTX INT8 kernels per GPU (2026-09-30)

`scripts/benchmark_ltx_denoise.py`: trial weights, INT8, 3 windows, median of 3. cuDNN SDPA attention on the 3090/4090, SageAttention 2 on the 5090.
`a0daf57` torch `_int_mm` path (Triton kernels were 5090-only), `f889b39` Triton kernels with per-GPU tiles, `4dc6a30` plus the block fusions.
5090 baseline is `46e4384`/`ef67e1f` from `2026-09-29_ltx_denoise.csv`.

| GPU | Model, canvas | a0daf57 s/window | f889b39 | 4dc6a30 | Change |
|---|---|---|---|---|---|
| RTX 3090 | distilled, 512 | 18.54 | 16.21 | 15.73 | −15 % |
| RTX 3090 | distilled, 768 | 60.18 | 55.28 | 54.96 | −9 % |
| RTX 3090 | undistilled, 512 | 48.99 | 42.35 | 41.71 | −15 % |
| RTX 4090 | distilled, 512 | 8.17 | 6.88 | 6.60 | −19 % |
| RTX 4090 | distilled, 768 | 26.12 | 23.05 | 22.49 | −14 % |
| RTX 4090 | undistilled, 512 | 21.68 | 18.22 | 17.56 | −19 % |
| RTX 5090 | distilled, 512 | 5.13 | – | 3.92 | −24 % |
| RTX 5090 | distilled, 768 | 13.66 | – | 10.76 | −21 % |
| RTX 5090 | undistilled, 512 | 13.37 | – | 10.41 | −22 % |

INT8 GEMM tiles (BM, BN, BK, stages, warps), one pick won every LTX shape per GPU:
3090 `128,128,64,3,4` (~215 TOPS, 1.2–2.2× `_int_mm` + rescale), 4090 `256,128,128,3,8` (~565 TOPS, 1.6–3.9×).

One window-step on `4dc6a30` (`--profile`, distilled 512): 3090 1941 ms, attention 50 %, GEMM 40 %; 4090 831 ms, attention 45 %, GEMM 41 %.
SDPA runs at the bf16 peak there and cuDNN ties Flash, so the rest of the gain has to come from INT8 attention.
