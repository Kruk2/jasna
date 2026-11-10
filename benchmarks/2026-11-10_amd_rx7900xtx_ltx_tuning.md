# LTX on a Radeon RX 7900 XTX: the AMD tuning of 2026-11-09/10 on gfx1100

The changes measured on an R9700 in `2026-11-09_amd_r9700_ltx_tuning.md` (compose feather without MIOpen,
2x2x8 decoder NA tiles, the 64x128x128 INT8 tile, shipped MIOpen tuning), plus the gfx1100 MIOpen tuning
(`f177eee9`) and the unpipelined 9x9x9 decoder NA on RDNA3 (`407f2589`), on an RX 7900 XTX (gfx1100, 96 CUs,
24 GB). Standard LTX (`ltx`, 2-step distilled INT8), full VAE encoder and decoder, 512.

## End to end

S610, 1080p, 6056 frames. Main `81111262` against `407f2589` (with its MIOpen tuning installed into the
ROCm system databases, as a release build does). Each tree kept its own MIOpen user cache: cold starts
empty, warm reuses it.

| run | `81111262` | `407f2589` |
|---|---|---|
| cold | 1495.6 s | 1242.6 s (-17 %) |
| warm | 1355.9 s | 1253.6 s (-8 %) |

| pass (warm) | `81111262` | `407f2589` |
|---|---|---|
| scan | 42.5 s | 40.1 s |
| encode | 196.7 s (1.33 s/window) | 157.1 s (1.06 s/window) |
| denoise | 838.7 s (2.91 s/window-step) | 818.5 s (2.84 s/window-step) |
| compose | 278.0 s (decode 1.71 s/window) | 237.9 s (decode 1.63 s/window) |

The new tree's cold and warm runs match: nothing is left for MIOpen to search. The old tree's cold compose
(427 s) is the feather's per-shape MIOpen search; its warm run reuses this clip's shapes, which a new video
would not. 143 windows in the cold runs against 144 in the warm ones.

## Tiles on gfx1100

- Decoder 9x9x9 NA, the 32 real calls of a 512 decode: the shipped 2x2x8 tiles with one pipeline stage
  are the fastest of the sweep, 419.2 ms per decode against 468.0 ms for the same tiles with two stages
  (the shipped kernel until `407f2589`). The AOT build of the one-stage kernel matches the JIT kernel
  bit for bit.
- INT8 GEMM at the block's 8192-row shapes: the shared 64x128x128 tile is the fastest for FF up (95 TOPS)
  and FF down (102 TOPS); for the projections 128x128x64 with 2 stages is within 1.5 % (96 against 95
  TOPS), so gfx1100 keeps the shared tile.

## MIOpen tuning of the encoder

Encoder alone, full VAE, production MIOpen settings, Triton (source) path, first window of a fresh cache and
the median of the next four:

| | 512 first | 512 | 768 first | 768 |
|---|---|---|---|---|
| no Jasna tuning | 79.3 s | 1.328 | 175.9 s | 3.060 |
| gfx1101 tuning (V710, release path only) | 78.9 s | 1.326 | 175.6 s | 3.057 |
| gfx1100 tuning | 3.6 s | 1.051 | 4.7 s | 2.431 |

The gfx1101 entries were tuned under Windows, which runs the prebuilt path only, so they hold none of the
source path's channels-last convs: borrowed by gfx1100 they changed nothing. Since `f177eee9` gfx1101 takes
the gfx1100 entries for those convs.

## Setup

Rented pod: privileged Docker, Ubuntu 26.04, one RX 7900 XTX, Ryzen 9 5950X, 30 GB RAM, ROCm 10.1 userspace,
AMF 26.20. Python 3.12, torch 2.14.0+rocm10.1.0, Triton 3.8.0, `rocm-sdk-devel` 10.1.0. `--ltx-trial`
(trial weights run the same kernels and shapes) from source trees; one run per row. All 48 transformer
blocks stay resident.
