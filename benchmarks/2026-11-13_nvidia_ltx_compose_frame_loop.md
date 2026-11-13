# LTX compose: keeping the window decoder busy (2026-11-13)

RTX 5090, Windows 11, torch 2.12.1+cu130 (CUDA 13.0). Standard LTX (`ltx`, 2-step distilled INT8), full
VAE decoder, 512. Code: main `268011ea` (A) against main `268011ea` plus this change (B).

## Compose pass

S610, 1080p, 6056 frames, 143 windows. Every run resumes the same saved plans and latents, so only the
compose pass runs; the time is its `[timing] LTX compose` line. Runs alternated A, B, B, A, A, B in one
session. Other processes used about 2 CPU cores during the runs (a backup and the virus scanner); no other
GPU job ran.

| run | A (main) | B (this change) |
|---|---|---|
| 1 | 97.4 s | 87.1 s |
| 2 | 92.7 s | 85.8 s |
| 3 | 92.6 s | 85.6 s |
| mean | 94.2 s | 86.2 s (-8.5 %) |

The decode time per window was 0.57 s on A and 0.55 s on B (it no longer includes reading the latent from
disk). Whole S610 jobs, from scan to compose, in one session without another GPU job: compose 90.6 s on
A, 83.1 s and 83.2 s on B. Per-run values in the `.csv`.

## Why

The window decoder (about 0.55 s of GPU work per window, about 79 s for the clip) bounds the pass. On
main it was busy for about 85 of 92 s of the frame loop: between windows it waited about 35 ms for a
cache flush and for the next latent to load, and where windows are sparse it ran out of work, since only
two windows were decoded ahead while the frame loop waited on the video reader. Now:
- while decoded windows fit on the GPU, up to 16 windows decode ahead (2 when they wait in host memory, as before);
- the next latents load on a thread of their own, and the GPU memory cache is only handed back before a
  decode when that could change how the window is decoded;
- a decoded window reaches the frames through a GPU event instead of a wait on the CPU;
- the video is read on a thread of its own, so its per-batch wait no longer stalls the frame loop.

With this the decoder is busy for about 80 of 83 s; the rest is in stretches with few windows, where
the video encoder sets the pace.

## Output

Bit-identical: every compose-only run on both sides wrote the same file (MD5
`5c803fa25ed98a8dcd746502ff2e2d35`), and so did ten more consecutive B runs. Five whole B jobs and one
whole A job (scan to compose) wrote one identical file as well. On S610 2160p the MD5 of all 6056
composited frames matches between A and B, and so does the output file.

## AMD

AMD GPUs with the precompiled kernels now composite frames on the fused compose kernels too, like NVIDIA.
They were bit-identical to the Torch path on the RX 7900 XTX, R9700 and RX 9060 XT, and on the RX 7900 XTX
not slower (232.8 s against 236.2 s, see `2026-11-12_nvidia_ltx_compose_kernels.md`), so both vendors share
one path. The frame-loop changes above are the same code on both vendors; they were not timed on AMD.

Not measured: a quiet machine, other GPUs, AMD with this change, 2160p timing (only the exactness runs,
which wait for every frame).
