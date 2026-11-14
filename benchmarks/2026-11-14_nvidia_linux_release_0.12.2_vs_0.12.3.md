# Linux NVIDIA release: 0.12.2 vs 0.12.3 (2026-11-14)

RTX 5090 on a rented instance, Linux. Standard LTX (`ltx`, 2-step distilled INT8), 512 canvas. All 48
transformer blocks stayed resident in every run. Two comparisons:

- **Release:** the public v0.12.2 Linux NVIDIA archive vs the 0.12.3 release build (main `3a492b4c`), default
  settings.
- **Dev install:** source checkouts of tag `v0.12.2` (`12510264`) and of main `a1ab68b7`, an earlier 0.12.3
  commit than the release, in their own venvs, run with `--ltx-trial` (placeholder weights with the real shapes)
  and the full VAE encoder and decoder. This shows which changes are rate changes in the code and which are
  fixed costs of the release build.

Each table shows the mean, with every run in brackets. Per-run values are in the `.csv`.

## Per-step rates

Lower is better for s/window, higher is better for frames/s. A step counts as a regression only if 0.12.3 is
slower than 0.12.2 by more than the run-to-run spread. No step regressed.

| Step | Clip | Release 0.12.2 | Release 0.12.3 | Dev 0.12.2 | Dev `a1ab68b7` |
|---|---|---|---|---|---|
| Scan, frames/s | M284 | 381 (383, 380) | 482 (482, 482) | 388 (388, 389) | 497 (496, 498) |
| Scan, frames/s | S610 | 341 | 450 | 370 | 498 |
| Encode, s/window | M284 | 0.43 (0.43, 0.43) | 0.33 (0.33, 0.33) | 0.36 (0.36, 0.36) | 0.23 (0.23, 0.23) |
| Encode, s/window | S610 | 0.43 | 0.34 | 0.36 | 0.25 |
| Denoise, s/window-step | M284 | 0.47 (0.47, 0.47) | 0.47 (0.47, 0.47) | 0.46 (0.46, 0.46) | 0.46 (0.46, 0.46) |
| Denoise, s/window-step | S610 | 0.51 | 0.51 | 0.47 | 0.47 |
| Decode, s/window | M284 | 0.68 (0.68, 0.68) | 0.56 (0.57, 0.55) | 0.55 (0.55, 0.55) | 0.48 (0.48, 0.47) |
| Decode, s/window | S610 | 0.67 | 0.54 | 0.55 | 0.45 |
| Compose, frames/s | M284 | 115 (115, 115) | 139 (137, 140) | 134 (133, 134) | 166 (166, 167) |
| Compose, frames/s | S610 | 62.3 | 76.3 | 75.5 | 92.9 |

Denoise is the same in both versions. Scan, encode, decode and compose are faster in 0.12.3 in both the
release and the dev install, by about the same amount. The release encodes with the default (half) VAE encoder
and the dev runs with the full one, so the absolute encode times of the two columns do not compare.

## Whole job and fixed parts

| | Release 0.12.2 | Release 0.12.3 | change | Dev 0.12.2 | Dev `a1ab68b7` | change |
|---|---|---|---|---|---|---|
| S610 whole job | 352.8 s | 303.1 s | -14 % | 297.6 s | 255.8 s | -14 % |
| M284 whole job | 50.0 s (47.1, 53.0) | 38.9 s (38.8, 38.9) | -22 % | 17.7 s (17.6, 17.7) | 16.1 s (16.0, 16.1) | -9 % |

Stage split from log timestamps (dev runs print no install check, so their first stage is folded into the
next one):

| Stage | M284 0.12.2 | M284 0.12.3 | S610 0.12.2 | S610 0.12.3 | S610 dev 0.12.2 | S610 dev `a1ab68b7` |
|---|---|---|---|---|---|---|
| Start to installation verified | 2.0 s | 2.0 s | 2.0 s | 2.0 s | — | — |
| To plan ready (scan done) | 8.1 s | 4.8 s | 23.5 s | 17.3 s | 18.1 s | 14.0 s |
| To the first encode window | 21.4 s (17.4, 25.4) | 11.8 s | 18.1 s | 3.1 s | 9.2 s | 3.1 s |
| Encode | 2.1 s | 1.7 s | 61.5 s | 48.6 s | 51.5 s | 35.8 s |
| Denoise | 8.9 s | 9.6 s | 146.9 s | 146.8 s | 137.0 s | 136.3 s |
| Compose | 5.9 s | 6.2 s | 99.1 s | 82.4 s | 80.7 s | 65.7 s |
| Tail | 1.6 s | 2.8 s | 1.7 s | 2.8 s | 1.0 s | 0.9 s |

"To the first encode window" is the encode pass time minus windows × s/window: program start, licence check,
model load and kernel set-up. Model loading got faster. The 0.12.3 release starts loading LTX while the scan
runs, so on S610 the load overlaps the 13 s scan. On the 1 s M284 scan it cannot, and 11.8 s of load
remains. The release spends about 10 s more on load than the dev install, in both versions: that is the
release's fixed cost (loading the licensed model files and the online licence check), not a rate. The 0.12.3
release adds about 1 s of fixed tail time per job. LTX also uses less memory in 0.12.3 (15946 MiB vs 28778 MiB
in the LTX log).

## 16 GB card (RTX 5070 Ti)

The same two release archives on a rented RTX 5070 Ti 16 GB, Standard LTX on M284. After one warm-up per
version, the runs went A B B A, plus one more run per version with whole-card VRAM sampled every 0.5 s (run 3).

| | Release 0.12.2 | Release 0.12.3 | change |
|---|---|---|---|
| Resident blocks | 42/48 (42, 42, 42) | 42/48 (42, 42, 42) | same |
| Scan, frames/s | 243 (243, 243) | 282 (282, 283) | faster |
| Encode, s/window | 0.97 (0.97, 0.97) | 0.72 (0.71, 0.72) | faster |
| Denoise, s/window-step | 1.04 (1.04, 1.04, 1.05) | 1.03 (1.03, 1.04, 1.03) | same |
| Decode, s/window | 1.44 (1.44, 1.43) | 1.12 (1.11, 1.13) | faster |
| Compose, frames/s | 56.6 (56.3, 56.8) | 72.9 (73.2, 72.5) | faster |
| Whole job | 72.4 s (72.7, 72.1, 72.4) | 62.2 s (62.2, 62.1, 62.3) | -14 % |
| Peak VRAM, whole card (run 3) | 13433 MiB | 13379 MiB | -54 MiB |

0.12.3 keeps as many blocks resident as 0.12.2 (42 of 48 in every timed run) and no step is slower. The
warm-up runs kept 40–42 blocks on both versions. Setup: RTX 5070 Ti 16 GB (driver 580.196.07, open kernel
module), Ryzen 9 5950X (30 vCPU), 98 GB RAM, Ubuntu 22.04 VM. The model files were copied in, so no run
downloaded anything. Both versions used the same config folder.

## Setup

- Rented instance: RTX 5090 32 GB (driver 580.196.07, open kernel module), Ryzen 9 9950X3D2 (30 vCPU), 73 GB
  RAM, Ubuntu 22.04 VM. Each run is a fresh process, and no other GPU job ran during a run.
- Release: the archives unpacked as a normal user and run as
  `jasna --online --input <clip> --output <out>.mp4 --restoration-model-name ltx --log-level info --no-progress`
  with a test licence (checked online). Both versions used the same config folder.
- Dev: Python 3.13, torch 2.12.1+cu130, Triton 3.7.1, the pinned VALI decoder wheel, and the CUDA 13.0 toolkit on
  `PATH`. Each install ran
  `python -m jasna --ltx-trial --input <clip> --output <out>.mp4 --restoration-model-name ltx --ltx-vae-encoder full --ltx-vae-decoder full --log-level info --no-progress`.
- Order: one M284 warm-up per version, then A B B A on M284 and one S610 run per version (release). For dev,
  one warm-up per version and clip, then A B B A on M284 and one S610 run per version.
- Whole job is the wall time from start to exit, measured outside the program. Pass times and rates come from
  the job's `[timing] LTX` lines.

## Limits

- S610 has one run per version.
- The 0.12.2 release's M284 load time varied between its two runs (17.4 and 25.4 s).
- The dev runs used main `a1ab68b7`, which is earlier than the release commit `3a492b4c`.
- A CPU-only test container used 8 cores during the dev M284 runs.
- One GPU, 1080p clips, Standard mode only.
