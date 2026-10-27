# LTX: Basic Quality (1 step) vs Standard Quality (2 steps) (2026-10-27)

RTX 5090, CLI from source with `--ltx-trial` (placeholder weights, same speed as a real run except
model loading), one 40 s 1080p video, 24 windows at 512, Sharper large mosaics off. Code: main
`3da76b61` plus the uncommitted Basic Quality wiring (model list, file names and trial step count; no
pipeline change). Runs alternated Basic / Standard: three pairs with INT8, two with Fast mode (NVFP4).
Per-run values in the `.csv`.

| Medians | Standard INT8 | Basic INT8 | Standard NVFP4 | Basic NVFP4 |
|---|---|---|---|---|
| LTX time | 48.3 s | 37.2 s (1.30x faster) | 43.8 s | 33.3 s (1.32x faster) |
| Encode | 10.9 s | 11.1 s | 10.7 s | 10.3 s |
| Denoise | 24.8 s | 13.3 s (1.86x faster) | 19.7 s | 10.2 s (1.93x faster) |
| Compose | 10.5 s | 10.7 s | 11.2 s | 10.7 s |

Basic runs one denoise step instead of two, so only denoise gets faster; VAE encode and
decode/compose stay the same and cap the whole-job gain near 1.3x at this canvas. Relative
speed, with Standard = 1x: Basic is about 0.77x the time (1.3x faster).

Not measured: the 768 canvas (Sharper large mosaics), where denoise is a larger share and the gain
should be somewhat higher, other GPUs, and release builds.
