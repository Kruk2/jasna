# LTX model speed: Basic, Standard, Good, High, Top (2026-11-06)

RTX 5090 (driver 595.104.02), Linux, source checkout at main `e015764b` (0.12.2) with the raw INT8 weights,
CLI with default settings (HEVC, NVIDIA defaults, 512 canvas, default VAEs: Basic half + max, others full + full).
Input: `SONE-610_bench_1080p_h264_8bit.mp4` (3:22, 6056 frames, 143 windows). Each model first ran once on a 10 s
cut to take the compiles and the first-use speed test; those rows are discarded. LTX time is the `[timing] LTX`
total; job wall time adds about 3-7 s. Per-run values in the `.csv`.

| Model | Runs | LTX time | vs Standard |
|---|---|---|---|
| Basic (1 step) | 2 | 163.8 s | 1.82x faster |
| Standard (2 steps) | 2 | 298.7 s | 1.00 |
| Good (4 steps) | 1 | 434.0 s | 1.45x slower |
| High (8 steps) | estimate | about 712 s | about 2.4x slower |
| Top (undistilled) | estimate | about 1615 s | about 5.4x slower |

High and Top were not run on the full clip. Their estimates are the passes every model shares (scan + encode +
compose, 156.7 s, the mean of the Standard and Good runs) plus window-steps times the denoise rate:

- High: 1144 window-steps x 0.485 s (Good's full run; the 1-8 step models share one transformer, and the 10 s
  warm-ups gave the same 0.48 s median for Standard, Good and High).
- Top: 2145 window-steps (15 per window) x 0.68 s (its warm-up median; it runs a different transformer).

The VAE passes are a large share of Basic and Standard and a small one of the others, so the ratios depend on the
GPU: on an RTX 5080 (16 GB) a 0.12.2 beta tester measured Basic 1.44x faster than Standard on a 140 s clip.

Not measured: NVFP4 (`--ltx-fast`), the 768 canvas, other GPUs, AMD, a release build.
