# Windows NVIDIA release: 0.12.2 vs 0.12.3 (2026-11-14)

RTX 5090, Windows 11, local desktop. Standard LTX (`ltx`, 2-step distilled INT8), 512 canvas, default VAE
settings. All 48 transformer blocks stayed resident in every run. Both versions are Windows NVIDIA release
archives, unpacked as shipped:

- **0.12.2:** the public v0.12.2 archive.
- **0.12.3, S610 runs:** an earlier 0.12.3 release build of main `a1ab68b7`. The final release (main `3a492b4c`)
  adds two changes on top: an AMD-only VAE encoder change, and freeing GPU memory before Jasna decides how many
  transformer blocks to keep resident. Neither should change a step on this card, where all 48 blocks fit.
- **0.12.3, M284 check:** the final release build of main `3a492b4c`, one run (see below).

S610 (1080p, 6056 frames, 143 windows): one warm-up per version, then A B B A. Each table shows the mean, with
every run in brackets. Per-run values are in the `.csv`.

## Per-step rates (S610)

Lower is better for s/window, higher is better for frames/s.

| Step | 0.12.2 | 0.12.3 (`a1ab68b7`) | change |
|---|---|---|---|
| Scan, frames/s | 217 (220, 214) | 339 (338, 340) | +56 % |
| Encode, s/window | 0.46 (0.47, 0.45) | 0.355 (0.35, 0.36) | -23 % |
| Denoise, s/window-step | 0.52 (0.52, 0.52) | 0.52 (0.52, 0.52) | 0 % |
| Decode, s/window | 0.76 (0.77, 0.75) | 0.60 (0.60, 0.60) | -21 % |
| Compose, frames/s | 56.5 (55.5, 57.4) | 68.3 (68.2, 68.4) | +21 % |

Denoise is the same. Scan, encode, decode and compose are faster, as in the Linux release
(`2026-11-14_nvidia_linux_release_0.12.2_vs_0.12.3.md`).

## Whole job and fixed parts (S610)

| | 0.12.2 | 0.12.3 (`a1ab68b7`) | change |
|---|---|---|---|
| Whole job | 429.6 s (431.9, 427.4) | 376.3 s (377.5, 375.1) | -12 % |
| LTX total | 404.8 s (407.9, 401.6) | 354.0 s (352.9, 355.0) | -13 % |
| Scan | 28.7 s | 18.2 s | -37 % |
| Encode | 93.2 s | 69.5 s | -25 % |
| Denoise | 170.5 s | 171.0 s | 0 % |
| Compose | 112.4 s | 95.4 s | -15 % |
| LTX memory (LTX log) | 28778 MiB | 15946 MiB | -45 % |

That is 14.1 frames/s before and 16.1 frames/s after.

Stage split from log timestamps:

| Stage | 0.12.2 | 0.12.3 (`a1ab68b7`) |
|---|---|---|
| Start to installation verified | 7.2 s | 7.3 s |
| To plan ready (app start, detector load, scan) | 42.5 s (40.3, 44.6) | 30.8 s (32.8, 28.9) |
| Plan ready to the first encode window done | 20.8 s (20.6, 21.0) | 18.4 s (17.9, 18.9) |
| Tail (last LTX line to exit) | 1.7 s (2.2, 1.3) | 2.2 s (2.4, 2.0) |
| Outside LTX (whole job minus LTX total) | 24.9 s | 22.4 s |

Almost all of the gain is in the per-window steps. The faster scan accounts for most of the shorter time to
plan ready. The wait before the first encode window (model load and licence check) is about 2 s shorter. The tail is within the run-to-run
spread.

## Final release check (M284)

One licensed Standard LTX job on M284 (1080p, 449 frames, 5 windows) with the final 0.12.3 release build (main
`3a492b4c`). It was the first job after unpacking, so it ran the one-time LTX speed test first; only its rates
are shown, not job times.

| Step | 0.12.3 release (`3a492b4c`) |
|---|---|
| Scan, frames/s | 470 |
| Encode, s/window | 0.40 |
| Denoise, s/window-step | 0.48 |
| Decode, s/window | 0.65 |
| Compose, frames/s | 131 |
| Resident blocks | 48/48 |

There is no 0.12.2 M284 run with logs to set beside it, so this is a check of the final build, not a
comparison. An M284 A B B A comparison was run on the earlier build, but its logs were not kept, so it is not in
this file.

## Setup

- RTX 5090 (32 GB, driver 610.74, 575 W), Core i9-13900K, Windows 11. Each run is a fresh process.
- Command, the same for both versions:
  `jasna.exe --online --input <clip> --output <out>.mp4 --restoration-model-name ltx --log-level info`
  (progress bars on; the M284 check also had `--no-progress`). Both used the same settings folder and a test
  licence. 0.12.2 was given the licence code on every run and activated online each time; 0.12.3 used its saved
  activation.
- The LTX model files were real copies in each install, so no run downloaded anything.
- Warm-ups (not in the means) ran the one-time LTX speed test and built the detector engine: 955.4 s (0.12.2)
  and 667.4 s (0.12.3).
- Whole job is the wall time from start to exit, measured outside the program. Pass times and rates come from
  the job's `[timing] LTX` lines.
- Local desktop: during the timed runs other programs (a backup and system services) used 1.7-2.3 CPU cores on
  average. No other GPU job ran during a run.

## Limits

- Two timed S610 runs per version, one clip, Standard mode only.
- The S610 0.12.3 runs used the earlier build `a1ab68b7`, not the final release commit.
- The M284 check is one run of the final build, with no 0.12.2 run to compare.
- Some runs logged late in the job that the licence service could not be reached; no run failed because of it.
