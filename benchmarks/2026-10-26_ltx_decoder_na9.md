# LTX decode: 9x9x9 stage-5 window + fused slab Q/K/V on NVIDIA (2026-10-26)

Decode of one 121-frame window, each GPU's segment budget
(8 GiB at 512, 16 GiB at 768), trial weights (timings do not depend on them), torch 2.12.1+cu130,
Triton 3.7.1, median of 5, old and new alternated twice. Old = 11x11x11 window, torch slab Q/K/V.

| GPU | 512 old -> new | 768 old -> new |
|---|---|---|
| RTX 5090 | 0.766 -> 0.597 s (-22 %) | 1.669 -> 1.318 s (-21 %) |
| RTX 4090 (Vast, 400 W cap) | 1.088 -> 0.854 s (-21 %) | 2.516 -> 1.980 s (-21 %) |
| RTX 3090 (Vast) | 2.363 -> 1.943 s (-18 %) | 5.392 -> 4.405 s (-18 %) |

On the 5090 that is ~8 % of a Standard (2-step, 512) LTX window.

- 9x9x9 window: 5090 0.78 -> 0.67 s alone; human A/B on four real clips (260-1072 px mosaics, Standard,
  lossless): "basically the same". The tuned decoder weights are unchanged.
- Fused slab Q/K/V (already on AMD): 65.7 dB vs the torch slab on real weights. Launch per width from a
  sweep of 16-128 rows x 2-8 warps on every real decode call on all three GPUs: 32 rows/4 warps at width
  256, 64 rows/4 warps at width 512 (2x faster than 32 rows there on the 3090).
- Not taken: the fused residual SwiGLU on NVIDIA (slower than the compiled torch MLP at every config that
  fits 99 KB of shared memory, best 0.91 vs 0.77 s/window on the 5090); other NA tile configs (best or
  within 2 % of the shipped ones on every GPU, none wins on all three); NATTEN 0.21.7 (3.5-4.5x slower
  than the Triton NA at every window, best cutlass tile config).
- Roofline (5090, 11x11x11): the NA kernel issues ~165 TFLOPS vs 175-200 TFLOPS for cuDNN dense attention
  at head dim 64; half of it is masked work of the 4x4x4 query tile's key box (2.06x at 11^3, 2.37x at 9^3).
- Decode GPU time at 512, 11^3, 5090: GEMM 41 %, NA 38 %, elementwise 12 %, inductor 9 %.

Not measured: AMD. Every AMD GPU runs the prebuilt NA kernels, so AMD needs `ltx_na3d_9x9x9.hipfb` (`scripts/build_ltx_vae_hipfbs.py`) before it can decode at all.
