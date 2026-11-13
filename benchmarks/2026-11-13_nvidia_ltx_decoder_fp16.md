# LTX VAE decoder on an RTX 5090: fp16 sums in the fused kernels (2026-11-13)

The LTX VAE decoder (full decoder, `vae-decoder-full.safetensors`) before and after running its MLP, context inject
and slab Q/K/V GEMMs on fp16 operands with fp16 sums, the neighborhood attention on fp16 Q/K/V, and retuned launch
configs (`diffusion_triton.NVIDIA_LAUNCH`, SM 12.0 entry with `fp16_accumulate`). Code: main `ec8c6c69` against
`ec8c6c69` plus this change, unless a section says otherwise. Per-run values are in the `.csv`.

## Decode, one window

Real 121-frame windows of Sample 1 (S610, 1080p, 512 canvas) for 512; random latents for 768 (no real 768 windows at
hand). Each run is its own process: load, two warm-up decodes, then 9 (512) or 7 (768) timed decodes; runs alternate
before/after.

| canvas | before (s/window) | after (s/window) | change | peak activations |
|---|---|---|---|---|
| 512 | 0.574, 0.575, 0.575 | 0.475, 0.475, 0.478 | -17 % | 3.20 GiB both |
| 768 | 1.293, 1.283 | 1.078, 1.077 | -16 % | 7.08 GiB both |

## Distance from an fp32 decoder

Same 8 real windows (every 20th of the clip's 143), an fp32 decoder with the torch reference code and the same noise
(drawn in bf16 on both sides), same tile layout. PSNR on [-1, 1] and the mean absolute difference of the uint8 frames
the compose pass makes, averaged over the windows.

| | PSNR vs fp32 | mean abs vs fp32 (of 255) |
|---|---|---|
| before | 63.06 dB | 0.0954 |
| after | 63.08 dB | 0.0950 |

After against before: 66.7 dB, mean 0.044, at most 2.4 of 255 in a window. fp16 keeps three more mantissa bits than
bf16 in every operand and stored activation, which outweighs its fp16 sums. Variants that were further from fp32 were
not used: keeping the attention's running P.V sum in fp16 (63.27 dB on 3 windows where before was 63.37), and fp16
sums in the attention projection (2 % further in its kernel test, no speed gain).

## fp16 range

fp16 overflows above 65504. Over all 143 Sample 1 windows the largest value any fp16 operand, stored activation or
sum reached was 35.5 (Q/K/V) and the largest bound on a partial sum 267 (an attention chunk's P.V sum, at most 32 keys
times the largest value), so the closest quantity stays 245x below the limit and all others 1800x or more. The same
held within 230-290x on a 768 window, on a window's latents scaled 2x and 4x and on random-normal latents at 512 and
768; every decode was finite. Of the 115 million weights cast to fp16, 0.4 % fall below fp16's normal range and 219
round to zero; the largest change is 3e-8, 5e-10 of the weights' total magnitude.

## Compose pass

Sample 1, 1080p, 6056 frames, 143 windows at 512, Standard LTX, compose only from the same saved latents (the
pipeline's `[timing] LTX compose` line), runs alternating.

| run | before | after |
|---|---|---|
| 1 | 84.7 s | 84.2 s |
| 2 | 92.2 s | 78.4 s |
| 3 | 92.4 s | 78.7 s |

Runs 2-3 of each side and the first run after ran alternately while other programs kept about 3 CPU cores busy; the
first run before ran later with an idle desktop, which alone made it 8 s faster. Restored frames after against before
(the first runs): 42.9 dB PSNR in the restored areas, mean difference 1.2 of 255, after the output's video encoding.

## Where the time goes (512, before)

`torch.profiler` over 3 real windows: 584 ms of kernels per window, the GPU busy 96 % of the wall time.

| kernel | ms/window | share | TFLOP/window | TFLOPS |
|---|---|---|---|---|
| neighborhood attention | 193 | 33 % | 12.0 (useful) | 62 useful, ~150 executed |
| MLP gate/up | 142 | 24 % | 21.5 | 151 |
| slab Q/K/V | 78 | 13 % | 10.0 | 128 |
| MLP down | 67 | 12 % | 10.7 | 160 |
| context inject | 35 | 6 % | 4.2 | 121 |
| cuBLAS (stages 1-3) | 24 | 4 % | 4.1 | 171 |
| attention projection | 18 | 3 % | 2.1 | 115 |
| other | ~27 | 5 % | | |

GEMM rates of this card, 8192 cubed: bf16 222, fp16 with fp32 sums 212, fp16 with fp16 sums 323 TFLOPS.

## Why the decoder gains less than the encoder

The encoder's convs already ran at the bf16 rate, so faster products sped up nearly all of it. The decoder's kernels
ran at 50-70 % of it: the products are skinny (K = 256-1024 over millions of rows, so few K steps per tile and a large
share of prologue and epilogue), the attention projection is bound by memory, and the attention, the largest part,
computes 2.4x the useful products (its 4x4x4 query tiles walk a 12-cubed key box) and spends time on its window masks
and softmax. cuBLAS at the same shapes, plain products:

| product | bf16 | fp16, fp16 sums | gain |
|---|---|---|---|
| stage-5 gate/up 16384x256x2048 | 0.105 ms | 0.074 ms | 1.41x |
| stage-5 down 16384x1024x256 | 0.056 ms | 0.033 ms | 1.66x |
| slab Q/K/V 619520x256x768 | 1.26 ms | 1.07 ms | 1.18x |
| projection 495616x256x256 | 0.372 ms | 0.331 ms | 1.12x |
| context 62464x512x2048 | 0.660 ms | 0.440 ms | 1.50x |

The fused kernels per call, before and after (launch configs retuned for fp16 sums):

| kernel | before | after |
|---|---|---|
| stage-5 gate/up (16384 rows) | 0.100 ms | 0.078 ms |
| stage-5 down (16384 rows) | 0.050 ms | 0.035 ms |
| slab Q/K/V (stage 5) | 1.93 ms | 1.23 ms |
| context inject | 0.86 ms | 0.61 ms |
| attention projection | 0.489 ms | 0.489 ms |
| attention (stage 5) | 4.76 ms | 3.88 ms |

- Weights are cast to fp16 once per call. Cast inside the K loop instead (bf16 weights), the Q/K/V kernel was 15 %
  slower than before.
- The attention runs at 184 TFLOPS (bf16) and 226 (fp16) on the products it executes. Without window masks it takes
  3.36 ms, without masks and softmax 3.10 ms (timing only). Other query tiles (2x2x4, 2x4x4, 4x2x4, 2x4x8, 2x2x8, both
  key walks) and a walk where each query row skips the key planes outside its window (25 % fewer products) were not
  faster.
- A single MLP kernel that keeps the hidden activation on the chip reached 211 TFLOPS but lost to the two retuned
  kernels (14.8 against 13.7 ms per stage-5 block).

## Setup

- RTX 5090 (32 GB), Windows 11, Core i9-13900K. Python 3.13, torch 2.12.1+cu130, Triton 3.7.1. Source tree.
- Other programs kept 2-5 CPU cores busy during every run (a backup tool, an antivirus scan, a browser); no other GPU
  job ran.
- `scripts/benchmark_ltx_decoder_kernels.py --verify 7` sweeps and checks an entry on another NVIDIA GPU.

## Limits

One GPU. GPUs without fp16 sums in their entry decode bit-identically to before: the 5090 with its previous entry gave
the same pixels as main on three real 512 windows and a 768 window. One clip; 768 timed on random latents only.
