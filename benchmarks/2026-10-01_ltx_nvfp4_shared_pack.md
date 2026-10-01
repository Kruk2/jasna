# LTX NVFP4: one activation quantization for Q/K/V (2026-10-01)

`scripts/benchmark_ltx_denoise.py --fast`, RTX 5090, trial weights; main `559968c` and change `61a8505`
alternated in one session.

| Model, canvas | 559968c ms/step | 61a8505 ms/step | change |
|---|---|---|---|
| distilled, 512 | 401.8 | 388.6 | −3.3 % |
| distilled, 768 | 1157.3 | 1137.0 | −1.8 % |
| undistilled, 512 | 558.9 | 546.2 | −2.3 % |

Self-attention Q/K/V and cross-attention K/V now share one packed input. Bit-identical per block, VRAM unchanged.
