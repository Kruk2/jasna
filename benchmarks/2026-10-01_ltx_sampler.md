# LTX sampler: no host syncs, shared prompt K/V (RTX 5090, 2026-10-01)

`scripts/benchmark_ltx_denoise.py` (trial weights, INT8, 3 windows, median of 3), run back to back in one
session in the order 559968c, 5534c5e, 6d2d2b7, 559968c. Commits:

- 559968c: baseline (main).
- 5534c5e: sigma schedule kept on the CPU and the fusion ramp copied without blocking, so a step never
  syncs the device.
- 6d2d2b7: 5534c5e plus each block's prompt cross-attention K/V computed once per block and step and
  shared by every window of the group and the STG pass.

| Model, canvas | 559968c s (two runs) | 5534c5e s | 6d2d2b7 s | Change vs baseline |
|---|---|---|---|---|
| distilled, 512 | 11.83 / 11.81 | 11.83 | 11.69 | −1.1 % |
| distilled, 768 | 32.59 / 32.62 | 32.58 | 32.49 | −0.3 % |
| undistilled (STG), 512 | 31.33 / 31.31 | 31.22 | 31.02 | −1.0 % |

GPU time of one distilled 512 window-step (`--profile`): 488.4 ms → 481.9 ms (−1.3 %); the int8 GEMM calls drop
from 864 to 672 (two of three windows no longer project the 1024 prompt tokens). The saving per window-step
is (windows − 1) / windows of about 10 ms at 512, so it grows with the track's window count and shrinks
at 768, where the video tokens dominate. Peak VRAM +0.02 GiB.

The sync removal alone is within noise: the GPU was never starved, since the CPU queues blocks far ahead.

Output: eager (`TORCH_COMPILE_DISABLE=1`) final latents of all three commits are bit-identical. The compiled
path differs between two runs of the *same* baseline commit by 0.4 % relative L2 after 2 steps on trial
weights, and the new commits sit at that floor. Likely cause, not proven: every process re-ran 87
coordinate-descent tuning benchmarks despite an FX graph cache hit, so picks can change between runs.
