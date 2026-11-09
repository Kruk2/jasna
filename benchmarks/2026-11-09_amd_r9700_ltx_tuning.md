# LTX on a Radeon AI PRO R9700: compose feather, decoder NA tiles, INT8 GEMM tile (2026-11-09)

Standard LTX (`ltx`, 2-step distilled INT8) with the full VAE encoder and decoder, 512 canvas, on a
Radeon AI PRO R9700 (gfx1201, 32 GB). Main `81111262` against `f685e952`, which adds three changes:

- `60d22fea` the compose feather blur runs as two matrix products, not `conv2d`;
- `9d4a06ce` the decoder's 9x9x9 neighborhood attention uses 2x2x8 query tiles on AMD;
- `f685e952` one 64x128x128 INT8 GEMM tile on AMD.

## End to end

S610, 1080p, 6056 frames, 144 windows. Each tree kept its own MIOpen cache: the cold run starts empty, the
warm runs reuse it. LTX stage seconds:

| run | `81111262` | `f685e952` |
|---|---|---|
| cold | 1120.4 | 820.0 (-27 %) |
| warm 1 | 921.9 | 823.8 (-11 %) |
| warm 2 | 915.7 | 825.6 (-10 %) |

| pass (warm 2) | `81111262` | `f685e952` |
|---|---|---|
| scan | 38.9 s | 37.9 s |
| encode | 151.7 s (1.02 s/window) | 151.8 s (1.02 s/window) |
| denoise | 463.3 s (1.59 s/window-step) | 429.7 s (1.48 s/window-step) |
| compose | 261.8 s (decode 1.51 s/window) | 206.1 s (decode 1.38 s/window) |

- Compose before: MIOpen benchmarked every new feather box size (about 47 ms each). On the cold run that
  was most of the 466 s; the warm runs reuse the shapes of this one clip, which a new video would not.
  After: no MIOpen in the feather, cold and warm compose are the same, and compose is now as long as the
  window decodes it waits for (144 x 1.38 s).
- The warm base runs are its best case: a different video brings new box sizes and pays the search again.

With `930b0da8` (tuned MIOpen solvers for the encoder, installed into the ROCm system databases as a
release build does) and an empty user cache: **794.0 s** cold (encode 128.1 s, 0.86 s/window; scan 38.6,
denoise 426.6, compose 200.7). That is -29 % against the old cold run and faster than the old code's warm
runs (-14 %). 143 windows in this run against 144 above.

## MIOpen tuning of the encoder

Encoder alone, full VAE, production MIOpen settings, s/window and the first window of a fresh cache:

| | 512 first | 512 | 768 first | 768 |
|---|---|---|---|---|
| fresh cache | 62.7 s | 1.053 | 225.9 s | 2.226 |
| warm cache | 5.6 s | 1.046 | 6.5 s | 2.221 |
| tuned entries in the system database | 5.6 s | 0.892 | 6.2 s | 1.863 |

MIOpen reads only the system file named after the exact GPU (architecture + compute units, `gfx120140` for
64 CUs); entries under another CU count or the bare architecture are ignored (first window 63 s again).

## Kernels

Feather blur, 300 real box sizes of this clip, per call:

| | first call of a size | repeat |
|---|---|---|
| MIOpen conv, HYBRID | 47 ms | 0.27 ms |
| MIOpen conv, FAST | 17.6 ms | 16.8 ms |
| band matrix products | 0.5 ms | 0.5 ms |

On an RTX 5090 the matrix products take 0.12 ms against 0.13 ms for cuDNN.

Decoder stage-5 neighborhood attention (9x9x9, the 32 real calls of one 512 decode): 481 ms with 4x2x4
tiles, 342 ms with 2x2x8 (16-key box rows filled exactly, 1600 keys scanned per query instead of 1920).
Full decode 1.390 -> 1.248 s per window in alternating runs.

INT8 GEMM at the transformer block's 8192-row shapes (Triton, the native block's kernel):

| shape | 128x256 / 128x128 tiles | 64x128x128, 1 stage, 4 warps |
|---|---|---|
| 4096x4096 projections | 199 TOPS | 227 TOPS |
| FF up 4096 -> 16384 with GELU | 130 TOPS | 202 TOPS |
| FF down 16384 -> 4096 | 193 TOPS | 204 TOPS |

Native-block denoise 1580 / 1587 -> 1434 / 1448 ms per window-step. For reference on this GPU:
hipBLASLt INT8 reaches about 215 TOPS (118 at K = 16384), bf16 about 140 TFLOPS.

## Setup

- Rented pod: privileged Docker, Ubuntu 26.04, one R9700, Core i5-10400 (6 cores), 15 GB RAM, ROCm 10.1
  userspace, AMF 26.20. Python 3.12, torch 2.14.0+rocm10.1.0, Triton 3.8.0, `rocm-sdk-devel` 10.1.0 for
  the native block build.
- `python -m jasna --ltx-trial --restoration-model-name ltx --ltx-vae-encoder full --ltx-vae-decoder full
  --secondary-restoration none` from a source tree (trial weights run the same kernels and shapes).
- One run per row; the two warm runs agree within 1 %.

## Limits

One GPU (gfx1201). The other AMD targets take the same tiles without a measurement; RDNA3 needs its own
check. Trial weights: speed only, no output quality.
