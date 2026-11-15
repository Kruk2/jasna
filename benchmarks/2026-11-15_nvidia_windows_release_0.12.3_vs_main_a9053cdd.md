# Windows NVIDIA release: 0.12.3 vs a release build of main `a9053cdd` (2026-11-15)

RTX 5090, Windows 11, local desktop. Standard LTX (`ltx`, 2-step distilled INT8), 512 canvas, default VAE
settings. All 48 transformer blocks stayed resident in every run. Both are Windows NVIDIA release archives,
unpacked as shipped:

- **0.12.3:** the released archive (main `3a492b4c`).
- **main `a9053cdd`:** a release build made for testing. It is not a released version and still reports 0.12.3.

S610 (1080p, 6056 frames, 144 windows): one warm-up per build on M284, then the timed runs in the order
A B B A B B (A = 0.12.3, B = main). Each table shows the mean, with every run in brackets. Per-run values are in
the `.csv`.

## Whole job and passes (S610)

| | 0.12.3 | main `a9053cdd` | change |
|---|---|---|---|
| Whole job | 386.6 s (373.2, 400.1) | 333.2 s (331.1, 332.1, 337.0, 332.7) | -14 % |
| LTX total | 353.1 s (353.4, 352.7) | 312.9 s (313.7, 312.8, 313.2, 311.7) | -11 % |
| Scan | 18.5 s | 18.0 s | -3 % |
| Encode | 69.7 s (69.5, 69.9) | 49.8 s (50.8, 49.8, 49.5, 49.1) | -29 % |
| Denoise | 170.7 s (171.0, 170.4) | 168.4 s (168.4, 168.5, 168.7, 167.9) | -1 % |
| Compose | 94.2 s (94.1, 94.2) | 76.7 s (76.8, 76.5, 77.1, 76.5) | -19 % |
| LTX memory (LTX log) | 15946 MiB | 15946 MiB | 0 % |

The second 0.12.3 run took 27 s longer than the first before its job began (45 s instead of about 19 s until
the detector was loaded); its LTX passes match the first run. Against the first run alone the whole job is
-11 %, the same as the LTX total. That is 16.2 frames/s before and 18.2 frames/s after.

## Per-step rates (S610)

| Step | 0.12.3 | main `a9053cdd` | change |
|---|---|---|---|
| Scan, frames/s | 333 | 343 | +3 % |
| Encode, s/window | 0.35 | 0.28 | -20 % |
| Denoise, s/window-step | 0.51 | 0.52 | same |
| Decode, s/window | 0.59 | 0.49-0.50 | -16 % |
| Compose, frames/s | 69.3 | 83.5 | +20 % |

This build removes time the release spent between windows with the GPU idle, so the gain grows with the length
of the video. The short warm-up clip shows the same rates once its one-time work is done (M284: encode 0.37 to
0.27 s/window, decode 0.63 to 0.52).

## Fixed time (main `a9053cdd`)

From the `[timing]` log lines, mean of the four runs. 0.12.3 does not print these lines.

| Log line | Time |
|---|---|
| App startup to the first job | 15.5 s (process start 4.9 s, startup check 6.6 s, until the job 4.0 s) |
| LTX worker start | 19.4 s (process start 7.4 s, startup check and setup 12.1 s) |
| LTX license check | 0.6 s |
| Model load: transformer file | 20.3 s |
| Model load: encoder | 3.8 s |
| Model load: transformer | 16.3 s |
| Model load: decoder | 4.1 s |

The worker start and the transformer file load run while the scan is running, so they do not add up to the
whole-job time.

## Against earlier files

- **main `9b038644`** (`2026-11-14_nvidia_windows_release_0.12.3_vs_main_9b038644.md`, the day before, 143
  windows): whole job 342.3 s, LTX total 325.6 s, encode 57.7 s (0.35 s/window), compose 86.4 s (decode
  0.57 s/window). That build had shortened the fixed time; this one adds the per-window gain.
- **0.12.2** was not run again. Against its filed whole-job time (429.6 s,
  `2026-11-14_nvidia_windows_release_0.12.2_vs_0.12.3.md`), main `a9053cdd` is at -22 % and 0.12.3 in its first
  run here at -13 %.

## Setup

- RTX 5090 (32 GB, driver 610.74, 575 W), Core i9-13900K, Windows 11. Each run is a fresh process.
- Command, the same for both builds:
  `jasna.exe --online --input <clip> --output <out>.mp4 --restoration-model-name ltx --log-level info`
  (progress bars on). Each build had its own empty settings folder. Both used the same test license, given on
  every run.
- The LTX model files were real copies in each install, so no run downloaded anything.
- Warm-ups (not in the means) ran the one-time LTX speed test and built the detector engine on M284 (1080p,
  449 frames): 389.8 s (0.12.3) and 324.9 s (main).
- Whole job is the wall time from start to exit, measured outside the program. Pass times and rates come from
  the job's `[timing] LTX` lines.
- Local desktop: during the timed runs other programs (a backup client, a virus scanner, a download client)
  used 0.7-2.6 CPU cores on average. No other GPU job ran.

## Limits

- Two timed runs of 0.12.3 and four of main, one clip, Standard mode only.
- One more run of main ended early and is not in the tables: it stopped with an access violation at the end
  of the scan, as one run of each build did the day before. It was repeated. Its cause is not known yet.
- Two runs of main on M284 gave the same frames; their files differ only in the stored creation time. The
  frames differ slightly from 0.12.3's (mean absolute difference 0.65 of 255 per channel), and swapping only the
  detector engine between installs moves them by as much (0.64), so this is the size of install-to-install
  variation, not a measured quality change. The S610 outputs were not compared frame by frame.
