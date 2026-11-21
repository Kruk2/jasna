# BasicVSR++ on a Radeon AI PRO R9700: AMF encoder pipelining (2026-11-21)

`5229c642` overlaps the copy of each restored frame to the host with the AMF encode (two pinned host buffers,
an encode worker thread). Its parent `d5ba960f` copies and encodes one frame after the other. Raw rows:
`2026-11-21_amd_r9700_basicvsrpp_encoder_pipelining.csv`.

S610, 1080p H.264, 6056 frames, HEVC output, `--max-clip-size 180 --temporal-overlap 15
--secondary-restoration none --no-compile-basicvsrpp`, HIP graphs on (default). Runs alternate old/new after a
warm-up run; the columns are the mean of 2 runs.

| | `d5ba960f` | `5229c642` |
|---|---|---|
| wall | 119.5 s | 113.9 s (-4.7 %) |
| pipeline | 106.7 s (56.8 fps) | 100.6 s (60.2 fps, -5.7 %) |
| blend-encode: blend | 53.5 s | 39.3 s |
| blend-encode: write | 30.9 s | 24.7 s |
| blend-encode: waiting for restored clips | 19.9 s | 34.5 s |
| detect-track | 102.5 s | 96.9 s |
| restore | 75.2 s | 64.3 s |
| peak VRAM | 5703 MiB | 5330 MiB |

- The blend/encode thread got 20 s faster and now waits on the restorer instead. The pipeline is now limited by
  detection (the decode-detect thread is busy for almost the whole run), which on this pod is held back by its
  4-core CPU. A faster host should gain more.
- Output is bit-exact: an encode-only run (decode, then AMF HEVC encode of every frame) gives the same md5 for
  all 6056 decoded frames on both commits.
- No GPU fault event (`amd-smi event`) during the runs.

## Setup

Rented pod: privileged Docker, Ubuntu 26.04, one Radeon AI PRO R9700 (gfx1201, 32 GB, 256 MB CPU-visible VRAM),
Intel Core i3-8100 (4 cores), 15 GB RAM, amdgpu 7.1.9, ROCm 10.1 userspace, AMF 26.20. Python 3.12,
torch 2.14.0+rocm10.1.0, source trees.
