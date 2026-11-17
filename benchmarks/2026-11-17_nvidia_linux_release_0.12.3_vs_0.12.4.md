# Linux NVIDIA release: 0.12.3 vs 0.12.4 (2026-11-17)

RTX 5090 on the maintainer's desktop, Linux, same day, same config folder for both versions. The v0.12.3 Linux
NVIDIA archive against the 0.12.4 release build (main `59b444e0`), default settings. Standard LTX (`ltx`,
2-step distilled INT8), 512 canvas; all 48 transformer blocks stayed resident in every run. BasicVSR++ with
compiled engines.

Each table shows the mean, with every run in brackets. Per-run values are in the `.csv`.

## LTX whole job

| Clip | Release 0.12.3 | Release 0.12.4 | change |
|---|---|---|---|
| M284 | 61.3 s (61.2, 61.3) | 60.2 s (60.4, 59.9) | -1.1 s (-2 %) |
| S610 | 400.6 s (400.2, 401.0) | 371.6 s (384.0, 359.1) | -29.0 s (-7 %) |

0.12.4 is faster on both clips: on M284 by more than the spread of either version (0.1 and 0.5 s), on S610
its slower run is still 16 s faster than the faster 0.12.3 run.

## LTX per-step rates

Lower is better for s/window, higher is better for frames/s. A step counts as changed only if the difference
is larger than the run-to-run spread.

| Step | Clip | Release 0.12.3 | Release 0.12.4 | |
|---|---|---|---|---|
| Scan, frames/s | M284 | 625 (631, 620) | 625 (651, 599) | same |
| Scan, frames/s | S610 | 340 (337, 343) | 361 (363, 358) | faster |
| Encode, s/window | M284 | 0.46 (0.47, 0.44) | 0.34 (0.34, 0.34) | faster |
| Encode, s/window | S610 | 0.49 (0.48, 0.50) | 0.34 (0.34, 0.34) | faster |
| Denoise, s/window-step | M284 | 0.53 (0.54, 0.51) | 0.55 (0.54, 0.55) | same |
| Denoise, s/window-step | S610 | 0.64 (0.64, 0.64) | 0.67 (0.70, 0.64) | same (see below) |
| Decode, s/window | M284 | 0.70 (0.71, 0.68) | 0.59 (0.59, 0.58) | faster |
| Decode, s/window | S610 | 0.68 (0.67, 0.69) | 0.63 (0.67, 0.58) | faster in one of two runs |
| Compose, frames/s | M284 | 111 (109, 112) | 128 (127, 129) | faster |
| Compose, frames/s | S610 | 61.1 (61.6, 60.6) | 67.5 (62.9, 72.0) | faster in one of two runs |

Encode is about 30 % faster per window in 0.12.4, and M284 decode and compose are faster too. The first 0.12.4
S610 run denoised at 0.70 s/window-step against 0.64 for 0.12.3; a second S610 run per version (the planned
order had one) gave 0.64 for both, so denoise did not change. The same first 0.12.4 run also had the slower
decode and compose of its two runs.

Both versions report 15946 MiB in the LTX log. 0.12.3 prints no app-startup or LTX-worker-start `[timing]`
lines; for 0.12.4 they were 5.5–12.5 s to the first job and 12.3–13.3 s for the worker start.

## BasicVSR++ whole job

`scripts/benchmark_releases.py` with its settings (clip size 180, overlap 15, no secondary restoration),
compiled engines, one discarded warm-up per version, then two runs per version in the order A B B A.

| Clip | Release 0.12.3 | Release 0.12.4 | |
|---|---|---|---|
| S610 1080p H.264 | 57.3 s (62.6, 52.0) | 53.0 s (53.8, 52.2) | same |
| S610 2160p H.264 | 80.8 s (80.7, 80.9) | 80.5 s (80.5, 80.4) | same |

The first 0.12.3 1080p run took 10 s longer than its second; without it the two versions are within 0.2 s.
On 2160p the difference (0.3 s) is about the spread. Peak VRAM of the process: 5.7 / 5.9 GB on 1080p and
11.0 / 11.1 GB on 2160p (0.12.3 / 0.12.4).

## Setup

- Maintainer's desktop with a running desktop session (browsers and an editor open): RTX 5090 32 GB (driver
  620.70.06), Core i9-13900K (32 threads), 64 GB RAM, Fedora-based Linux, kernel 7.3. 2.2–3.1 GB of VRAM was in
  use by other apps before each run (5.0 GB before the second 0.12.3 S610 run). No other GPU job ran during a
  timed run.
- Both archives unpacked as a normal user and run with a test licence (checked online). LTX:
  `jasna --online --input <clip> --output <out>.mp4 --restoration-model-name ltx --log-level info --no-progress`.
  The LTX model files were downloaded once and copied into the 0.12.3 folder.
- LTX order: one M284 warm-up per version, then A B B A on M284, one S610 run per version, then a second S610
  run per version (B A) to recheck the denoise rate.
- BasicVSR++ engines were compiled by an earlier job of each version; no timed run compiled.
- Whole job is the wall time from start to exit, measured outside the program. LTX pass times and rates come
  from the job's `[timing] LTX` lines.

## Limits

- A desktop machine with other apps open, not an idle instance: the desktop's own GPU use changed during the
  runs.
- Two runs per version and clip. The 0.12.4 S610 runs differ by 25 s, so the S610 gain is roughly 16–42 s.
- BasicVSR++ per-run times are taken from the benchmark log file times (about ±1 s); the means are the script's
  own. Only the 1080p and 2160p H.264 clips were run.
- The BasicVSR++ engines of both versions were compiled at the same time, while other jobs used the GPU.
- One GPU, Standard LTX mode only.
