# LTX NVFP4: one shared activation quantization for Q/K/V (2026-10-01)

`scripts/benchmark_ltx_denoise.py --fast` (NVFP4, trial weights), RTX 5090, 3 windows x all steps,
`ms_per_window_step` per run = median of 3 repeats (undistilled: 1 repeat). `559968c` = main,
`61a8505` = the change. Runs alternate main/variant in one session (batch `c-reversed` and the last
pair of `d`/`e` run the variant first); compare rows only within this file.

| Model, canvas | main 559968c | shared pack 61a8505 | change |
|---|---|---|---|
| distilled, 512 (7 pairs) | 401.8 ms | 388.6 ms | −3.3 % (paired median −2.7 %) |
| distilled, 768 (3 pairs) | 1157.3 ms | 1137.0 ms | −1.8 % |
| undistilled, 512 (2 pairs) | 558.9 ms | 546.2 ms | −2.3 % |

The first batch (`a`) was noisy (−0.4 % to −2.6 %); every later pair gained 2–3 % at 512.
Peak VRAM unchanged.

What changed: self-attention Q/K/V each computed the dynamic tensor scale (an amax pass) and packed
the same input again; cross-attention K/V did the same on the prompt. They now share one packed input.
Profile of one 512 step (`--profile`): `_quantize_fp4_kernel` calls 12 → 9 per block and its share
of GPU time 8.1 % → 6.7 %, the amax passes 10 → 7 per block (2.6 % → 1.4 %).
In the NVFP4 model the gate projections are bf16, so cross-attention Q has nothing to share with.
Output is bit-identical per block (`tests/test_ltx_kernels_gpu.py`).

Not done: FF2's GELU is already fused into Inductor's amax reduction (no GELU temporary is written),
so the remaining cost there is the second read of the FF1 output, which needs the amax inside the
FF1 GEMM epilogue (cuBLAS `scaled_mm` has no such epilogue).
