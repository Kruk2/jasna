# Windows NVIDIA release: 0.12.3 vs a release build of main `9b038644` (2026-11-14)

RTX 5090, Windows 11, local desktop. Standard LTX (`ltx`, 2-step distilled INT8), 512 canvas, default VAE
settings. All 48 transformer blocks stayed resident in every run. Both are Windows NVIDIA release archives,
unpacked as shipped:

- **0.12.3:** the released archive (main `3a492b4c`).
- **main `9b038644`:** a release build made only for this measurement. It is not a released version and still
  reports 0.12.3.

S610 (1080p, 6056 frames, 143 windows): one warm-up per build on M284, then the timed runs in the order
A B B A B B (A = 0.12.3, B = main). Each table shows the mean, with every run in brackets. Per-run values are in
the `.csv`.

## Whole job and passes (S610)

| | 0.12.3 | main `9b038644` | change |
|---|---|---|---|
| Whole job | 370.5 s (368.7, 372.2) | 342.3 s (342.7, 337.0, 344.7, 344.9) | -7.6 % |
| LTX total | 349.3 s (349.1, 349.5) | 325.6 s (326.2, 320.9, 327.0, 328.4) | -6.8 % |
| Scan | 17.5 s | 17.4 s | 0 % |
| Encode | 69.7 s (69.7, 69.7) | 57.7 s (57.8, 55.8, 58.3, 58.9) | -17 % |
| Denoise | 169.1 s (168.7, 169.4) | 164.2 s (164.2, 163.3, 164.4, 164.7) | -2.9 % |
| Compose | 93.1 s (93.3, 92.9) | 86.4 s (86.9, 84.5, 86.8, 87.4) | -7.2 % |
| Outside LTX (whole job minus LTX total) | 21.2 s | 16.7 s | -21 % |
| LTX memory (LTX log) | 15946 MiB | 15946 MiB | 0 % |

That is 16.3 frames/s before and 17.7 frames/s after.

## Per-step rates (S610)

| Step | 0.12.3 | main `9b038644` | change |
|---|---|---|---|
| Scan, frames/s | 353 | 354 | 0 % |
| Encode, s/window | 0.35 | 0.35 | 0 % |
| Denoise, s/window-step | 0.51 | 0.51 | 0 % |
| Decode, s/window | 0.59 | 0.57 | -3 % |
| Compose, frames/s | 69.8 | 73.3 | +5 % |

Most of the gain is time that does not depend on the length of the video: the wait before the first encode
window is about 12 s shorter, and startup, the wait before the first denoise step and the wait before the first
decode are a few seconds shorter each. Per window, only decode got faster (about 3 s over this clip).

## Where the remaining fixed time goes (main `9b038644`)

From the new `[timing]` log lines, mean of the four runs. 0.12.3 does not print these lines.

| Log line | Time |
|---|---|
| App startup to the first job | 13.9 s (process start 4.6 s, startup check 5.2 s, until the job 4.1 s) |
| LTX worker start | 18.5 s (process start 7.0 s, startup check and setup 11.5 s) |
| LTX license check | 0.6 s |
| Model load: transformer file | 19.1 s |
| Model load: encoder | 3.8 s |
| Model load: transformer | 14.9 s |
| Model load: decoder | 3.7 s |

The worker start and the transformer file load run while the scan is running, so they do not add up to the
whole-job time.

## Against 0.12.2

0.12.2 was not run again. Against its filed whole-job time from the same day (429.6 s,
`2026-11-14_nvidia_windows_release_0.12.2_vs_0.12.3.md`), 0.12.3 is at -14 % in these runs and main
`9b038644` at -20 %. The filed 0.12.3 runs of that file (376.3 s) used an earlier build and a slightly busier
machine.

## Setup

- RTX 5090 (32 GB, driver 610.74, 575 W), Core i9-13900K, Windows 11. Each run is a fresh process.
- Command, the same for both builds:
  `jasna.exe --online --input <clip> --output <out>.mp4 --restoration-model-name ltx --log-level info`
  (progress bars on). Each build had its own empty settings folder. Both used the same test license, given on
  every run.
- The LTX model files were real copies in each install, so no run downloaded anything.
- Warm-ups (not in the means) ran the one-time LTX speed test and built the detector engine on M284 (1080p,
  449 frames): 354.8 s (0.12.3) and 293.5 s (main).
- Whole job is the wall time from start to exit, measured outside the program. Pass times and rates come from
  the job's `[timing] LTX` lines.
- Local desktop: during the timed runs other programs used 0.5-0.8 CPU cores on average. No other GPU job ran.

## Limits

- Two timed runs of 0.12.3 and four of main, one clip, Standard mode only.
- Two more runs ended early and are not in the tables: one of each build stopped with an access violation at
  the end of the scan, about 35 s into the job. The same crash was seen two days earlier, so it is not new in
  main; its cause is not known yet.
- The warm-up outputs of the two builds are not byte-identical. The S610 outputs were not compared.
