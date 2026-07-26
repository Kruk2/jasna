# Linux release and performance benchmark

- Date: 2026-07-26
- OS: Linux
- Hardware: NVIDIA RTX 5090 32 GB, driver 595.84; Intel i9-13900K
- Targets: official Linux NVIDIA releases
  [v0.4.1](https://github.com/Kruk2/jasna/releases/tag/v0.4.1),
  [v0.5.0](https://github.com/Kruk2/jasna/releases/tag/v0.5.0),
  [v0.6.2](https://github.com/Kruk2/jasna/releases/tag/v0.6.2),
  [v0.7.2](https://github.com/Kruk2/jasna/releases/tag/v0.7.2),
  [v0.8.1](https://github.com/Kruk2/jasna/releases/tag/v0.8.1), and v0.9.0
  source commit `4a171c9`
- Harness: `scripts/benchmark_releases.py`

## Method

- Settings: `--max-clip-size 180 --temporal-overlap 15
  --secondary-restoration none`; defaults otherwise, including each version's
  default models and HEVC output.
- Each target first restored `assets/test_clip1_1080p.mp4`; that output and all
  measured outputs were discarded. Frozen-release and Lada rows are one
  measured run after warmup. Refreshed v0.9.0 rows are medians of three paired
  runs, interleaved per input with alternating target order. The recent-commit
  and feature-pair results are medians of three fresh subprocess runs.
- Source targets forced the VALI decoder. Frozen releases used their own
  release defaults.
- The v0.9.0 source target used a locally built PyAV 18.0.0 with explicit CUDA
  stream support. Its paired runs used tmpfs scratch space. Released targets
  used their bundled dependencies.
- RAM RSS and per-process VRAM were sampled every 0.5 seconds. Values below are
  median/peak MiB for the benchmark process; compositor VRAM is excluded.
- SONE files are six encodings of the same 6,056-frame, 202.07-second source.
  The VR input is 8K, 900 frames, and 15 seconds.
- The frozen binaries exported `LD_LIBRARY_PATH` to child tools. Scratch-only
  `ffmpeg`, `ffprobe`, and `mkvmerge` launchers unset it before calling the
  existing `/usr/bin` tools. No Jasna release files, drivers, or system
  settings were changed.
- This was a live desktop run, not a locked-down lab environment. Treat small
  single-digit differences as noise unless a paired result repeats the effect.

## Release results

| Version | Input | Wall | fps | RAM med/peak MiB | VRAM med/peak MiB | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| v0.4.1 | 720p H.264 8-bit | 01:35.2 | 63.6 | 4065/5215 | 17982/20414 | OK |
| v0.4.1 | 1080p H.264 8-bit | 01:46.3 | 57.0 | 5586/5755 | 19393/21941 | OK |
| v0.4.1 | 1080p HEVC 10-bit | 01:48.3 | 55.9 | 4421/4810 | 19581/22033 | OK |
| v0.4.1 | 2160p H.264 8-bit | 03:23.0 | 29.8 | 6578/6750 | 26596/31226 | OK |
| v0.4.1 | 2160p HEVC 10-bit | 03:24.1 | 29.7 | 6642/6784 | 27270/31242 | OK |
| v0.4.1 | 2160p AV1 10-bit | 03:22.1 | 30.0 | 6640/6774 | 27211/30865 | OK |
| v0.5.0 | 720p H.264 8-bit | 00:45.3 | 133.7 | 2498/2605 | 8846/9206 | OK |
| v0.5.0 | 1080p H.264 8-bit | 00:47.1 | 128.7 | 2658/2805 | 9775/10407 | OK |
| v0.5.0 | 1080p HEVC 10-bit | 00:48.2 | 125.8 | 2691/2826 | 10115/10905 | OK |
| v0.5.0 | 2160p H.264 8-bit | 01:22.0 | 73.9 | 3532/3657 | 13190/15380 | OK |
| v0.5.0 | 2160p HEVC 10-bit | 01:30.6 | 66.8 | 3535/3644 | 14316/16268 | OK |
| v0.5.0 | 2160p AV1 10-bit | 01:30.1 | 67.2 | 3520/3635 | 14089/16634 | OK |
| v0.6.2 | 720p H.264 8-bit | 00:44.7 | 135.5 | 2403/2509 | 8994/9306 | OK |
| v0.6.2 | 1080p H.264 8-bit | 00:45.5 | 133.0 | 2579/2669 | 9983/10569 | OK |
| v0.6.2 | 1080p HEVC 10-bit | 00:47.3 | 128.0 | 2593/2691 | 10223/10953 | OK |
| v0.6.2 | 2160p H.264 8-bit | 01:16.2 | 79.4 | 3400/3483 | 12722/15634 | OK |
| v0.6.2 | 2160p HEVC 10-bit | 01:24.4 | 71.8 | 3425/3503 | 14098/16528 | OK |
| v0.6.2 | 2160p AV1 10-bit | 01:20.4 | 75.4 | 3415/3495 | 13784/16606 | OK |
| v0.7.2 | 720p H.264 8-bit | 00:47.5 | 127.5 | 2471/2487 | 9012/9464 | OK |
| v0.7.2 | 1080p H.264 8-bit | 00:44.4 | 136.3 | 2645/2683 | 9889/10789 | OK |
| v0.7.2 | 1080p HEVC 10-bit | 00:46.3 | 130.7 | 2667/2705 | 10235/10953 | OK |
| v0.7.2 | 2160p H.264 8-bit | 01:16.0 | 79.7 | 3474/3522 | 12789/15604 | OK |
| v0.7.2 | 2160p HEVC 10-bit | 01:19.6 | 76.1 | 3496/3543 | 13806/16568 | OK |
| v0.7.2 | 2160p AV1 10-bit | 01:20.4 | 75.3 | 3492/3538 | 13792/16572 | OK |
| v0.8.1 | 720p H.264 8-bit | 00:48.7 | 124.3 | 2227/2407 | 8832/9438 | OK |
| v0.8.1 | 1080p H.264 8-bit | 00:49.5 | 122.3 | 2596/2638 | 9807/10325 | OK |
| v0.8.1 | 1080p HEVC 10-bit | 00:49.6 | 122.0 | 2635/2679 | 9855/10573 | OK |
| v0.8.1 | 2160p H.264 8-bit | 01:30.0 | 67.3 | 3537/3577 | 12042/15362 | OK |
| v0.8.1 | 2160p HEVC 10-bit | 01:33.8 | 64.6 | 3616/3662 | 12434/15792 | OK |
| v0.8.1 | 2160p AV1 10-bit | — | — | — | — | Excluded: v0.8.1 release bug |
| v0.8.1 | 8K VR HEVC 8-bit 60 fps | 00:45.5 | 19.8 | 7410/7413 | 18928/19312 | OK |
| v0.9.0 `4a171c9` | 720p H.264 8-bit | 00:34.4 | 175.9 | 2319/2342 | 8820/9330 | OK |
| v0.9.0 `4a171c9` | 1080p H.264 8-bit | 00:38.6 | 156.8 | 2480/2508 | 9527/10361 | OK |
| v0.9.0 `4a171c9` | 1080p HEVC 10-bit | 00:36.1 | 167.8 | 2481/2508 | 9637/10537 | OK |
| v0.9.0 `4a171c9` | 2160p H.264 8-bit | 01:15.4 | 80.3 | 3356/3412 | 11708/15724 | OK |
| v0.9.0 `4a171c9` | 2160p HEVC 10-bit | 01:19.3 | 76.4 | 3095/3398 | 12248/16322 | OK |
| v0.9.0 `4a171c9` | 2160p AV1 10-bit | 01:04.7 | 93.6 | 3360/3394 | 12106/16142 | OK |
| v0.9.0 `4a171c9` | 8K VR HEVC 8-bit 60 fps | 00:34.5 | 26.1 | 6648/6909 | 17548/18696 | OK |

v0.5.0 is used because
[v0.5.1](https://github.com/Kruk2/jasna/releases/tag/v0.5.1) has no Linux
release artifact. v0.7.2 is the latest v0.7.x release. The three v0.8.1 Linux
NVIDIA archive parts matched their published SHA-256 digests.
The v0.8.1 2160p AV1 result was excluded because that release has a bug on this
input.

## Recent performance commits

Broad end-to-end runs used 1080p HEVC, 2160p HEVC, and 8K VR with the normal
feature defaults. Each cell is the median wall time of three runs.

| Commit | Change | 1080p HEVC | 2160p HEVC | 8K VR |
| --- | --- | ---: | ---: | ---: |
| `9b6a089` | Before the performance series | 43.7 s | 87.6 s | 42.7 s |
| `5456a7c` | RGB to NV12/P010 encode kernel | 46.3 s | 87.1 s | 42.2 s |
| `eca9035` | Cache detector normalization constants | 47.0 s | 88.6 s | 42.5 s |
| `d348739` | Skip unused blend frame copy | 49.2 s | 88.9 s | 45.4 s |
| `f77b5f4` | `.cube` LUT kernel | 44.1 s | 88.4 s | 48.2 s |
| `c64724f` | Detector preprocess kernel | 43.6 s | 78.1 s | 37.1 s |
| `07fabc2` | Denoise kernel | 38.4 s | 77.4 s | 36.9 s |
| `7d9cc8c` | Cache VR axes and rotations | 38.6 s | 77.8 s | 37.8 s |

The full series reduced median wall time from baseline to the final pre-stream
performance commit by 11.7% at 1080p, 11.2% at 2160p, and 11.5% on the default
8K path. The detector preprocess commit produced the clearest always-on step.
LUT, denoise, and explicit VR projection are disabled in this broad run, so
their neighboring rows must not be used to estimate those commits.

### Feature-specific paired runs

| Feature and input | Before | After | Wall-time change | RAM med/peak MiB | VRAM med/peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| No-detection blend, synthetic 2160p black | 20.65 s | 17.65 s | -14.5% | 2840/2842 → 2842/2843 | 8648/8648 → 8648/8648 |
| Identity 2x2 `.cube` LUT, 2160p HEVC | 86.64 s | 78.63 s | -9.2% | 3360/3401 → 3357/3397 | 12604/15916 → 12314/15536 |
| Medium denoise, 1080p HEVC | 44.19 s | 44.30 s | +0.3% (noise) | 2593/2624 → 2596/2627 | 9633/10239 → 9629/10307 |
| Explicit `sbs-fisheye`, 8K VR | 40.97 s | 39.89 s | -2.6% | 6967/6975 → 6969/6975 | 17186/18100 → 17392/17782 |

The synthetic blend clip has 1,800 frames and no expected detections. The LUT,
denoise, and VR options were explicitly enabled for both sides of each pair.

## Lada Flatpak results

Lada 0.11.0 was measured from its user-scoped Linux Flatpak with the accurate
`v2` detector, CUDA FP16, `--max-clip-length 180`, and the
`hevc-nvidia-gpu-hq` encoding preset. One warmup was discarded. RAM includes
all processes in the Flatpak cgroup, and no 8K input was run.

| Input | Wall | fps | RAM med/peak MiB | VRAM med/peak MiB | Jasna v0.9.0 speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 720p H.264 8-bit | 01:47.3 | 56.4 | 5134/5361 | 1893/3142 | 3.1x |
| 1080p H.264 8-bit | 02:02.4 | 49.5 | 6738/7101 | 2020/3351 | 3.2x |
| 1080p HEVC 10-bit | 02:06.5 | 47.9 | 7558/8103 | 2045/3613 | 3.5x |
| 2160p H.264 8-bit | 04:55.6 | 20.5 | 13748/16576 | 2653/4225 | 3.9x |
| 2160p HEVC 10-bit | 05:10.7 | 19.5 | 12700/17454 | 2613/4079 | 3.9x |
| 2160p AV1 10-bit | 05:20.9 | 18.9 | 15016/18570 | 2649/3871 | 5.0x |

## Legacy-video interest check

The files from the legacy README table were run last on `9b6a089` (the parent
of all seven performance commits) and v0.9.0. Official v0.7.2 was then run on
HUBLK and the three short test inputs; ABF was excluded from that follow-up as
requested. Their original maximum clip sizes were retained; temporal overlap
was 8 and secondary restoration was disabled. These are single full-file runs
after target warmup.

| Input | Version | Wall | fps | RAM med/peak MiB | VRAM med/peak MiB | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| ABF-017 4K (260,913 frames) | `9b6a089` | 43:20.0 | 100.3 | 1652/3530 | 9034/9564 | OK |
| ABF-017 4K (260,913 frames) | v0.9.0 | 42:16.6 | 102.9 | 2009/3378 | 9296/9446 | OK |
| HUBLK-063 1080p (342,237 frames) | v0.7.2 | 23:38.1 | 241.3 | 1302/2790 | 10345/10345 | OK |
| HUBLK-063 1080p (342,237 frames) | `9b6a089` | 17:42.6 | 322.1 | 841/2698 | 10019/10093 | OK |
| HUBLK-063 1080p (342,237 frames) | v0.9.0 | 18:00.9 | 316.6 | 1781/2661 | 9673/9673 | OK |
| DASS-570_2m (3,636 frames) | v0.7.2 | 01:05.4 | 55.6 | 1006/2563 | 3285/3339 | OK |
| DASS-570_2m (3,636 frames) | `9b6a089` | 00:22.7 | 159.9 | 2582/2588 | 2981/3025 | OK |
| DASS-570_2m (3,636 frames) | v0.9.0 | 00:22.0 | 165.6 | 2457/2463 | 2883/2973 | OK |
| NASK-223_Test (6,413 frames) | v0.7.2 | 01:06.6 | 96.2 | 2657/2665 | 3289/3289 | OK |
| NASK-223_Test (6,413 frames) | `9b6a089` | 01:03.9 | 100.4 | 2603/2612 | 3075/3075 | OK |
| NASK-223_Test (6,413 frames) | v0.9.0 | 01:01.1 | 105.0 | 2469/2477 | 2953/3029 | OK |
| test-007 (3,813 frames) | v0.7.2 | 00:27.4 | 139.3 | 2601/2643 | 3469/3481 | OK |
| test-007 (3,813 frames) | `9b6a089` | 00:30.7 | 124.0 | 2587/2615 | 3205/3255 | OK |
| test-007 (3,813 frames) | v0.9.0 | 00:26.8 | 142.5 | 2462/2484 | 3105/3105 | OK |
v0.9.0 was 2.4% faster on ABF, 3.5% faster on DASS, 4.4% faster on NASK, and
12.9% faster on test-007. HUBLK was 1.7% slower, within the noise expected from
a single live-desktop run.

v0.7.2 completed HUBLK in 23:38.1, 23.6% faster than the historical v0.6.2
result of 30:58 but 33.5% slower than `9b6a089`. Its DASS result was unusually
slow despite no warning or fallback in the log.

## Raw data

- `2026-07-26_release_matrix.csv`
- `2026-07-26_perf_commit_series.csv`
- `2026-07-26_perf_feature_pairs.csv`
- `2026-07-26_legacy_videos.csv`
- `2026-07-26_lada_flatpak.csv`
