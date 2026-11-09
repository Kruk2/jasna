# LTX on a Radeon RX 9060 XT: the AMD tuning of 2026-11-09 on gfx1200

The changes measured on an R9700 in `2026-11-09_amd_r9700_ltx_tuning.md` (compose feather without MIOpen,
2x2x8 decoder NA tiles, the 64x128x128 INT8 tile, shipped MIOpen tuning), checked on an RX 9060 XT
(gfx1200, 32 CUs, 16 GB). Standard LTX (`ltx`, 2-step distilled INT8), full VAE encoder and decoder, 512.

## End to end

S610, 1080p, 6056 frames. Main `81111262` against `1140c2f1` (with its gfx1200 MIOpen tuning installed into
the ROCm system databases, as a release build does). Each tree kept its own MIOpen user cache: cold starts
empty, warm reuses it.

| run | `81111262` | `1140c2f1` |
|---|---|---|
| cold | 1950.8 s | 1481.4 s (-24 %) |
| warm | 1711.3 s | 1489.4 s (-13 %) |

| pass (warm) | `81111262` | `1140c2f1` |
|---|---|---|
| scan | 67.1 s | 63.9 s |
| encode | 273.9 s (1.86 s/window) | 226.8 s (1.54 s/window) |
| denoise | 918.4 s (3.16 s/window-step) | 823.3 s (2.85 s/window-step) |
| compose | 452.0 s (decode 2.75 s/window) | 375.3 s (decode 2.54 s/window) |

The new tree's cold and warm runs match: the tuned encoder and the feather leave nothing for MIOpen to
search.

## Tiles on gfx1200

- Decoder 9x9x9 NA, the 32 real calls of a 512 decode: the shipped 2x2x8 tiles are the fastest of the
  sweep (607.8 ms per decode); the old 4x2x4 tiles are not in the top eight.
- INT8 GEMM at the block's 8192-row shapes: the shared 64x128x128 tile is the fastest for the projections
  (123 TOPS) and FF up (106 TOPS). For FF down (K = 16384) 128x128x128 with 2 stages and 8 warps was 4 %
  faster in isolation (13.1 vs 13.6 ms), but the native-block denoise was slower with it (2910 / 2911 vs
  2892 / 2884 ms per window-step), so gfx1200 keeps the shared tile.

## MIOpen tuning of the encoder

Encoder alone, full VAE, production MIOpen settings, Triton (source) path, first window of a fresh cache and
the median of the next four:

| | 512 first | 512 | 768 first | 768 |
|---|---|---|---|---|
| no Jasna tuning | 306.3 s | 1.879 | 693.4 s | 4.374 |
| tuned entries in the system database | 4.5 s | 1.560 | 6.1 s | 3.532 |

## Setup

Rented pod: privileged Docker, Ubuntu 26.04, one RX 9060 XT, Ryzen 9 3950X, 15 GB RAM, ROCm 10.1 userspace,
AMF 26.20. Python 3.12, torch 2.14.0+rocm10.1.0, Triton 3.8.0, `rocm-sdk-devel` 10.1.0. `--ltx-trial`
(trial weights run the same kernels and shapes) from source trees; one run per row. 37-39 of the 48
transformer blocks stay resident in 16 GB, the rest stream.
