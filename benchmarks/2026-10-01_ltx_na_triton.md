# LTX decode: Triton neighborhood attention (2026-10-01)

RTX 5090, real weights. Kernel: `scripts/benchmark_na3d.py`; decode: `scripts/benchmark_ltx_decode.py`
(median of 5). Measured at `85d690b`, W-chunk rows at `329c6c9`.

## Kernel, ms per call

| Shape | kitchen | NATTEN | Triton |
|---|---|---|---|
| stage 5, 512, W 42 | 35.7 | 42.6 | 7.23 |
| stage 5, 512, W 138 | 110.6 | 144.2 | 28.8 |
| stage 5, 768, W 58 | 72.2 | 89.8 | 16.3 |
| stage 5, 768, W 202 | 248.4 | 319.8 | 66.9 |
| stage 4, 512 | 3.50 | 5.66 | 1.14 |
| stage 4, 768 | 7.96 | 12.82 | 2.69 |

Configs: `4x4x4 n32 w2 s3` (11³ window), `2x4x4 n32 w2 s2` (smaller windows), from a ~280-config sweep.

## Decode, s per window

| Canvas, tile budget | kitchen | Triton |
|---|---|---|
| 512, 24 GiB | 1.805 | 0.862 (−52 %) |
| 768, 24 GiB | 3.800 | 1.915 (−50 %) |
| 512, 6 GiB segment | 2.382 | 1.121 (−53 %) |
| 768, 8 GiB segment | 6.297 | 2.831 (−55 %) |

63 dB against kitchen, VRAM +0.1–0.4 GiB. An INT8/FP8 variant gained only 1–2 % and was dropped.
Fewer W chunks (4 → 2) would save ~5 % but needs a tile-budget model update; unchanged.
