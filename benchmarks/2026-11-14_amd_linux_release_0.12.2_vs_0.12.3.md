# AMD Radeon AI PRO R9700: Linux release builds, v0.12.2 vs v0.12.3 (2026-11-14)

Radeon AI PRO R9700 (gfx1201, 32 GB) on a rented instance: Core i5-10400 (6 cores), 15 GB RAM, Ubuntu 26.04
(kernel 7.0), AMF 26.20 for decoding and HEVC encoding. Both versions are the Linux AMD release archives, unpacked
as shipped: v0.12.2 is the published `v0.12.2` release, v0.12.3 is the release build of main `3a492b4c`. Both
bundle PyTorch 2.14.0+rocm10.1.0 (HIP 7.16). Standard LTX (2 steps) with the full VAE encoder and decoder, on the
real model weights. Each version kept its own cache folder; both used the same settings folder, which also holds
the MIOpen kernel database. Job time is the whole job, measured around the process. M284 (15 s 1080p30, 449
frames, 5 windows): each version first ran one discarded job, then the timed jobs ran in the order A B B A. S610
(1080p, 6056 frames, 143-144 windows) ran later in the order A B A B A B; the table uses the two completed runs of
each version (see the caveat below). The tables show the mean of the two runs; per-run values are in the `.csv`.

| Job | v0.12.2 | v0.12.3 |
|---|---|---|
| M284 | 114.1 s | 116.8 s (+2.7 s, +2 %) |
| S610 | 1144.1 s | 1036.8 s (-107.3 s, -9 %) |

## Rates per step (M284)

| LTX step | v0.12.2 | v0.12.3 |
|---|---|---|
| scan | 164.9 frames/s | 184.6 frames/s (+12 %) |
| encode | 1.55 s/window | 1.43 s/window (-8 %) |
| denoise | 1.73 s/window-step | 1.57 s/window-step (-9 %) |
| decode (compose) | 2.00 s/window | 1.81 s/window (-10 %) |
| transformer blocks resident | 48/48 | 48/48 |

Every step is faster in v0.12.3. Each run of a version gave the same rates within 0.02 s.

## Where the time goes (M284)

| LTX pass | v0.12.2 | v0.12.3 |
|---|---|---|
| scan | 4.8 s | 4.3 s |
| encode | 35.1 s | 37.7 s |
| denoise | 33.8 s | 32.7 s |
| compose | 17.7 s | 19.1 s |
| LTX total | 91.3 s | 93.7 s (+2.4 s) |
| outside LTX (start, detector, setup, finishing the file) | 22.8 s | 23.1 s |

The faster steps save about 3.4 s on M284's 5 windows, but v0.12.3 spends more time per pass outside the steps
themselves: about +3 s in encode, +0.5 s in denoise and +2.5 s in compose (pass time minus rate times windows).
That time does not grow with the video length, so the gain shows on longer videos, as on S610.

## S610 (1080p, 6056 frames)

| LTX pass | v0.12.2 | v0.12.3 |
|---|---|---|
| scan | 50.0 s (125.8 frames/s) | 41.7 s (151.0 frames/s, +20 %) |
| encode | 241.0 s (1.55 s/window) | 213.3 s (1.48 s/window, -5 %) |
| denoise | 545.7 s (1.91 s/window-step) | 500.9 s (1.76 s/window-step, -8 %) |
| compose | 286.0 s (decode 1.93 s/window) | 257.2 s (decode 1.70 s/window, -12 %) |
| LTX total | 1122.6 s | 1013.0 s (-10 %) |
| outside LTX | 21.6 s | 23.8 s |

Both versions planned 143 or 144 windows from run to run (the same scene cuts and tracks). Run-to-run spread:
6.2 s (v0.12.2) and 8.9 s (v0.12.3).

Caveat: the rented R9700 lost the GPU in some S610 runs on both versions (2 of 6 in the interleaved set: one
v0.12.2 run and one v0.12.3 run, always when the encode pass started), and the job then stopped making progress
until the time limit ended it. The other video jobs on the same instance were not affected. Those runs are listed
in the `.csv` without times and are not in the means.

## Other notes

- LTX memory in the LTX log dropped from 28.2 GB (28826 MiB) in v0.12.2 to 15.6 GB (15994 MiB) in v0.12.3.
  Whole-card peak with v0.12.3 was 15.2 GB on M284 and 15.8 GB on S610.
- Spread between the two runs: 0.5 s (v0.12.2) and 0.3 s (v0.12.3) on M284.
- Discarded first runs on M284: 117.2 s (v0.12.2) and 118.8 s (v0.12.3).
- An earlier v0.12.3 test build (main `1a3aa7f3`, with a fused VAE encoder) measured on the same instance: encode
  1.24 s/window, the other rates as above; its rows are in the `.csv`. The release went back to the 0.12.2 VAE
  encoder path.
- Command, the same for both versions:
  `jasna --input <clip> --output <out>.mkv --restoration-model-name ltx --ltx-vae-encoder full
  --ltx-vae-decoder full --log-level debug --no-progress`

Not measured: other AMD GPUs, Windows, Basic quality.
