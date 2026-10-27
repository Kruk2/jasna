# LTX decoder: fused MLP, projection and context kernels; distilled decoder

Decode seconds per 121-frame window, trial placeholder weights (`LtxModelFiles.placeholder`), each GPU's own
segment decode budget, median of 7 decodes after a warm-up, two alternating runs of `main` and the branch.
"tuned" is the shipped 8-block decoder; "distilled" the 4-block distilled decoder (stage-5 blocks 1, 3, 5, 7
kept and distilled from tuned, same 9x9x9 window).

## What changed

- Residual MLP of the diffusion blocks and deterministic stages 2-4 as two fused GEMMs (norm and modulation in
  the first's prologue, SwiGLU in its epilogue, the residual add in the second's). Stage 1 (width 2048) stays
  torch: the fused MLP is 1-15 % slower there.
- Diffusion-block attention output projection with its bias and residual add written straight into the block
  input (0.6x the GEMM plus strided add), and the folded stage-4 context projection with its pixel shuffle and
  add (0.6x the compiled GEMM, shuffle and add). Bit-exact on NVIDIA.
- Per-GPU launch configs for these kernels and the slab Q/K/V (RTX 3090, 4090, 5090; other GPUs take the config
  with the least worst-case slowdown on those three, at most 1.15x).
- AMD runs the same kernels built ahead of time; its one-kernel MLP is gone.

## NVIDIA, decode s/window

| GPU | decoder | 512 main | 512 branch | 768 main | 768 branch |
|---|---|---|---|---|---|
| RTX 5090 | tuned | 0.576 | 0.511 (-11 %) | 1.286 | 1.150 (-11 %) |
| RTX 5090 | distilled | 0.344 | 0.306 (-11 %) | 0.762 | 0.689 (-10 %) |
| RTX 4090 | tuned | 0.853 | 0.712 (-17 %) | 1.944 | 1.647 (-15 %) |
| RTX 4090 | distilled | 0.506 | 0.425 (-16 %) | 1.164 | 0.987 (-15 %) |
| RTX 3090 | tuned | 1.956 | 1.815 (-7 %) | 4.418 | 4.076 (-8 %) |
| RTX 3090 | distilled | 1.158 | 1.083 (-6 %) | 2.610 | 2.427 (-7 %) |

Against the decoder before the 9x9x9 window (benchmarks/2026-10-26_ltx_decoder_na9.md, 11x11x11, torch Q/K/V):
distilled + branch is 2.5x faster at 512 on an RTX 5090 (0.766 -> 0.306 s), 2.6x on a 4090 (1.088 -> 0.425 s)
and 2.2x on a 3090 (2.363 -> 1.083 s).

Output against `main`: 93.6-94.7 dB PSNR on the same latent (the fused deterministic-stage MLP rounds the SwiGLU
product once where torch rounds the SiLU first).

## AMD, decode s/window (ROCm 10.1, torch 2.14)

| GPU | decoder | 512 main | 512 branch | 768 main | 768 branch |
|---|---|---|---|---|---|
| RX 7900 XTX | tuned | 3.928 | 0.997 (-75 %) | 8.934 | 2.247 (-75 %) |
| RX 7900 XTX | distilled | 2.137 | 0.669 (-69 %) | 4.891 | 1.516 (-69 %) |
| RX 9060 XT | tuned | 2.806 | 1.593 (-43 %) | 8.422 | 3.827 (-55 %) |
| RX 9060 XT | distilled | 1.707 | 1.078 (-37 %) | 4.118 | 2.506 (-39 %) |

The one-kernel MLP that `main` ships for AMD runs 5.8x slower than compiled torch on the RX 7900 XTX (RDNA3);
the two-kernel MLP replaces it. AMD launch configs are one per kernel: the least worst-case slowdown on these two
GPUs (at most 1.07x). Output against `main`: 52-60 dB PSNR.

## Headroom, RTX 5090, distilled decoder, 512 (Nsight Systems, one decode)

Measured peaks: 219 TFLOPS bf16 (8192^3 GEMM), 1581 GB/s device copy.

| kernel | ms | achieved | of peak |
|---|---|---|---|
| neighborhood attention 9x9x9 | 82.3 | 74 TFLOPS useful | 34 % (about 80 % counting the 2.4x masked box) |
| MLP gate/up | 75.8 | 173 TFLOPS | 79 % |
| MLP down | 36.4 | 181 TFLOPS | 82 % |
| slab Q/K/V | 38.9 | 129 TFLOPS | 59 % |
| projection + residual | 7.7 | 1587 GB/s | 100 % |
| context inject | 12.2 | 171 TFLOPS | 78 % |

Everything at 100 % of peak would save about 57 ms of the 288 ms; the remaining fusable work is a few percent.
FP8 GEMMs in the MLP (per-row scales) were tried and dropped: 38.8 dB against bf16 even in stage 5 alone.
The MLP GEMMs beat cuBLAS on the same shapes (160 / 149 TFLOPS); two slab Q/K/V redesigns (one wide dot per
projection, a K-tiled GEMM with the norm in its prologue) were 0-5 % slower than the shipped kernel.
Neighborhood attention issues 187 TFLOPS over its query tiles' KV boxes, 89 % of the 211 TFLOPS of the best
dense attention here (torch SDPA flash / cuDNN, head dim 64); its shipped config is the fastest of about 110
(tiles up to 128 queries, 2-8 warps, 32-128-key blocks).
