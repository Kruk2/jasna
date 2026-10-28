# LTX denoise kernel tuning on an RTX 5090

RTX 5090, PyTorch 2.12.1+cu130, Triton 3.7.1, distilled model with placeholder weights, native blocks,
all 48 blocks resident. `scripts/benchmark_ltx_denoise.py --windows 3 --repeats 3`, two alternating runs
of `main` (afe14be4) and the INT8 cross-attention change (c9630a2e). File: `_kernel_tuning.csv`.

## Result, ms per window-step (mean of the two runs)

| canvas | quant | main | c9630a2e |
|---|---|---|---|
| 512 | INT8 | 491.6 | 479.6 (-2.4 %) |
| 768 | INT8 | 1370.4 | 1346.6 (-1.7 %) |
| 512 | NVFP4 | 382.2 | 372.3 (-2.6 %) |
| 768 | NVFP4 | 1128.6 | 1114.3 (-1.3 %) |

Runs of the same code differ by up to 2.5 % (clocks under sustained load), so these gains are rough. The
profile of one 512 INT8 step shows the cross-attention going from 34 ms (cuDNN bf16, ~196 TFLOPS) to
18 ms (INT8 attention plus its quantization), 494 -> 467 ms GPU time.

Output on two real 1080p clips (distill8 INT8 weights, torch path, SDPA vs INT8 cross-attention):
6.5-7.0e-3 relative latent difference and 58.6-59.2 dB decode PSNR. This is less than the existing
difference between the native block and the torch path (8.0-8.5e-3, 57.4-58.2 dB).

## Where the 512 INT8 step goes and what was tried

Measured peaks: cuBLAS INT8 647 TOPS (8192^3), bf16 222 TFLOPS, 1541 GB/s copy.

| part | ms/step | rate | tried |
|---|---|---|---|
| INT8 GEMMs | 296-303 | ~610 TOPS in the pipeline, 670-750 alone | sweep of 138 tiles x stream-K x GROUP_M over all block shapes: the best rule is -1.4 %, inside the noise; shipped tiles kept |
| self-attention | 93 | ~555 TOPS | 64-row query tiles 2-5 % slower, 64-key tiles 14 % slower; shipped tiles kept |
| cross-attention | 34 -> 18 | 196 TFLOPS bf16 -> INT8 | shipped (c9630a2e) |
| rotate/quantize, norm/RoPE, gates, Q/K/V quantize | ~59 | memory-bound | not tried; fusing norm/RoPE into the Q/K quantization (as AMD does) is worth ~1.5 % at most |

NVFP4 512: cuBLAS FP4 GEMMs 147 ms (~1260 TOPS), attention 125 ms before the change, other 93 ms (FP4
quantize 26, gated residual 14, RMS modulate 11, GELU amax 8).

Not measured: RTX 3090 / 4090. They keep SDPA cross-attention until it is measured there.
