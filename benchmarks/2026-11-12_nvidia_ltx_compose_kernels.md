# LTX compose on fused NVIDIA kernels (2026-11-12)

RTX 5090 (driver 610.74), Windows 11, torch 2.12.1+cu130 (CUDA 13.0), Triton 3.7.1. Standard LTX (`ltx`,
2-step distilled INT8), full VAE decoder, 512. Code: main `a453bc84` plus this change; A runs force every
frame onto the Torch compositing path, B runs use the new `ltx_compose` kernels.

## Compose pass

S610, 1080p, 6056 frames, 143 windows. Every run resumes the same saved plans and latents, so only the
compose pass runs; the time is its `[timing] LTX compose` line. Runs alternated A, B, B, A, A, B. The
desktop was in use and other processes used about 5 CPU cores during the runs; no other GPU job ran.

| run | A (Torch) | B (kernels) |
|---|---|---|
| 1 | 112.9 s | 94.3 s |
| 2 | 106.9 s | 93.6 s |
| 3 | 106.6 s | 95.1 s |
| mean | 108.8 s | 94.3 s (-13 %) |

The decode rate was the same on both sides (0.57-0.59 s per window). Per-run values in the `.csv`.

## Why

Each composited frame queued about 150 short Torch kernels on its stream, and each waited behind the
window decoder's kernels; the GPU time of the compositing itself was about 0.5 ms per frame. The kernels
do the mask dilation, the colour match and the whole layer blend in four launches; the feather blur and
the canvas resampling stay Torch matrix products. Kernel launches on the frame stream, profiled over 200
frames of S610 1080p:

| | A (Torch) | B (kernels) |
|---|---|---|
| GPU operations per frame | 147 | 18 |
| GPU time per frame | 0.50 ms | 0.17 ms |

## Output

Bit-identical to the Torch path on the RTX 5090:
- S610 1080p and 2160p (144 windows): the MD5 of every one of the 6056 composited frames matches
  between A and B, and so does the output file.
- All six timed 1080p runs wrote the same output file.
- 300 random frames (360p, 1080p, 2160p; one to three tracks, one or two windows each, crops at the
  frame edge, crossfades) match exactly.

Frames with a track handover, VR projection and the CPU keep the Torch path.

## Other GPUs

On rented cloud instances (Linux):
- RTX 4090 and RTX 3090: 0 of 300 random frames differ. With the driver's PTX compilation turned off,
  so the kernels load from each generation's own compiled code, 0 of 30 (4090) and 0 of 300 (3090).
- Radeon AI PRO R9700, RX 7900 XTX and RX 9060 XT (torch 2.14.0+rocm10.1.0): 0 of 300 random frames
  differ each. On the RX 7900 XTX all 6056 composited frames of S610 1080p match as well, with the same
  output file (compose-only runs with placeholder decoder weights of the real shapes: same work,
  made-up pixels).

## AMD timing

RX 7900 XTX, the same S610 compose-only runs, one run per side:

| run | A (Torch) | B (kernels) |
|---|---|---|
| compose | 236.2 s | 232.8 s (-1 %) |

The window decoder bounds the pass there (1.60 s per window, about 229 s for the 143 windows), so fewer
launches change little. AMD ships the kernels but keeps the Torch path.

Not measured: a quiet machine, compose timing on other NVIDIA GPUs, 2160p timing (only the exactness
runs, which wait for every frame), AMD on Windows.
