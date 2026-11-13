# LTX whole jobs on an RTX 5090: 0.12.2 vs 0.12.3 (2026-11-13)

RTX 5090, Windows 11, local desktop. Standard LTX (`ltx`, 2-step distilled INT8), full VAE encoder and decoder,
512 canvas, all 48 transformer blocks resident. 0.12.2 = tag `v0.12.2` (`12510264`), 0.12.3 = main `07f9bc38`.
Both ran from source with the same Python environment. Whole jobs (scan, encode, denoise, compose, video
output). Each table shows the mean over the runs, with every run in brackets; per-run values are in the `.csv`.

What changed: the VAE encoder head runs in fp16 with Triton convolutions, and encode windows stay on the GPU
and encode while the next one fills; compose blends frames on fused GPU kernels and keeps the window decoder
busy; the VAE decoder kernels use fp16 sums; scan traces its regions in batches. Denoise did not change.

## S610 (1080p, 6056 frames, 143 windows)

Two rounds of A, B, B, A.

| | 0.12.2 | 0.12.3 | change |
|---|---|---|---|
| Whole job | 365.2 s (365.2, 366.9, 364.2, 364.3) | 291.1 s (294.6, 287.5, 294.9, 287.3) | -20 % |
| LTX total | 356.6 s | 283.1 s | -21 % |
| Scan | 27.6 s | 17.8 s | -36 % |
| Encode | 68.3 s (0.39 s/window) | 43.3 s (0.25 s/window) | -37 % |
| Denoise | 149.4 s (0.49 s/window-step) | 149.0 s (0.48 s/window-step) | 0 % |
| Compose | 111.3 s (decode 0.59 s/window) | 73.1 s (decode 0.47 s/window) | -34 % |
| Peak VRAM | 15978 MiB | 16587 MiB | +4 % |

That is 16.6 frames/s before and 20.8 frames/s after. Every run of a version was within 1 % of the others.

## M284 (1080p, 449 frames, 5 windows)

One round of A, B, B, A.

| | 0.12.2 | 0.12.3 | change |
|---|---|---|---|
| Whole job | 34.7 s (42.0, 27.4) | 28.6 s (32.2, 25.0) | -18 % |
| LTX total | 26.7 s | 20.4 s | -24 % |
| Scan | 1.65 s | 1.25 s | -24 % |
| Encode | 8.6 s (0.39 s/window) | 6.1 s (0.24 s/window) | -30 % |
| Denoise | 10.3 s (0.48 s/window-step) | 8.8 s (0.48 s/window-step) | -15 % |
| Compose | 6.2 s (decode 0.60 s/window) | 4.3 s (decode 0.53 s/window) | -31 % |
| Peak VRAM | 16566 MiB | 15954 MiB | -4 % |

## p840 (1080p, 300 frames, 3 windows)

One round of A, B, B, A.

| | 0.12.2 | 0.12.3 | change |
|---|---|---|---|
| Whole job | 29.9 s (36.1, 23.6) | 23.9 s (25.8, 22.0) | -20 % |
| LTX total | 21.9 s | 16.4 s | -25 % |
| Scan | 1.3 s | 1.0 s | -23 % |
| Encode | 7.1 s (0.40 s/window) | 5.3 s (0.27 s/window) | -25 % |
| Denoise | 8.9 s (0.48 s/window-step) | 6.9 s (0.47 s/window-step) | -23 % |
| Compose | 4.6 s (decode 0.63 s/window) | 3.3 s (decode 0.53 s/window) | -28 % |
| Peak VRAM | 15978 MiB | 15978 MiB | 0 % |

On the short clips most of the LTX time is model loading, which the pass times include. The first run of each
version (A-1, B-1) took 4-15 s longer than its second run, on both versions (cause not measured); the second
runs alone give -7 % (p840) and -9 % (M284) for the whole job. The per-window rates do not depend
on loading and match S610: encode about -35 %, decode -12 to -20 %, denoise the same. The denoise change on
the short clips is loading, not the per-step rate.

## Output

0.12.3 against 0.12.2 on the same clip: 52.5 dB PSNR over all S610 frames (lowest frame 48.0 dB), 53.6 dB on
p840, after the output's video encoding.

## Setup

- RTX 5090 (32 GB, driver 610.74, 575 W), Core i9-13900K, Windows 11. Python 3.13, torch 2.12.1+cu130,
  Triton 3.7.1, VALI decoder. Source checkout of each version; VS 2022 build tools on `PATH`, so the native
  LTX block was built and used.
- Command, the same for both versions (each run is a fresh process):
  `python -m jasna --input <clip> --output <out>.mkv --restoration-model-name ltx --ltx-vae-encoder full --ltx-vae-decoder full --log-level info --no-progress`
  Output HEVC (`hevc_nvenc`, defaults).
- Before the timed runs, one untimed job per version on p840 built the native block, compiled the GPU kernels
  and ran the LTX speed test.
- Whole job = wall time from start to exit, measured outside the program. Pass times are the job's
  `[timing] LTX` lines; peak VRAM is the peak that line reports.
- Local, desktop running: during every run other programs used 3-8 CPU cores (a virus scanner, a backup,
  other scripts). No other GPU job ran during a run.

## Limits

One GPU, 1080p clips only, Standard mode only (not Fast/NVFP4, not other step counts or the half/max VAEs).
One A B B A round for the short clips. A dev install, not the release build, though both versions ran the
same way.
