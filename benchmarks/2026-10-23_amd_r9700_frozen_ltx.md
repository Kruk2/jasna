# Linux AMD frozen release, LTX trial on an R9700, 2026-10-23

The frozen Linux AMD 0.11.3 release now runs the LTX trial almost as fast as the source
tree on a Radeon AI PRO R9700 (gfx1201): 103.7 s for the LTX stage of the 300-frame test
clip, down from 138.9 s, against 99.5 s for the source tree with Triton. The release now
ships the native LTX block (prebuilt kernels plus its extension), and the eager encoder and
decoder the release runs without Triton got faster.

## Build

- Built from a clean `git archive` of main `49d6f59` with
  `build_linux_amd.sh`, 24 CPUs, in 85 min. All build, release and archive gates passed.
- The archive is 7.9 GB in four parts, the same size as before.
- New in the dist: `ltx/native/prebuilt/` with the gfx1100/1101/1102/1200/1201 INT8 kernels
  and `jasna_ltx_native.so` for those targets.

## Environment

- One R9700 with 32 GB, a 4-core i3-12100F and 30 GB RAM. ROCm 10.1 userspace, AMF 26.20.
- Every GPU run took the pod's single GPU lock. Frozen runs used a fresh `HOME`, so the
  first LTX run started with empty caches.
- The source tree used Python 3.12, torch 2.12.0+rocm10.0.0 and Triton 3.8.0.

## Results

300-frame test clip, LTX trial, `--secondary-restoration none`, warm runs.

| | Frozen before (2c3c773) | Frozen now (49d6f59) | Source tree, Triton |
|---|---|---|---|
| LTX stage | 138.9 s | 103.7, 103.6 s | 99.5 s (100.9 s on 2c3c773) |
| Encode | 2.29 s/window | 1.32 s/window | 1.01 |
| Denoise | 2.11 s/step | 1.52 s/step | 1.50 |
| Decode | 1.96 s/window | 1.75 s/window | 1.70 |
| Wall time | 146 s | 112 s | 105 s (2c3c773) |
| Startup to first frame | 7.6 s | 7.6 s | 4.6 to 5.3 s |

- The log no longer says "native blocks off". A file watch saw the run open
  `ltx/native/prebuilt/jasna_ltx_native.so` and the gfx1201 kernel images.
- First run with empty caches: 161.5 s wall (before: 200 s). It includes the trial's
  one-time speed profile measurement (47 s).
- Owner clip, 5310 frames 1080p, RF-DETR v6 + BasicVSR++: 94.45 fps (median before: 94.7 fps).
- Licensed LTX without `--ltx-trial`: refused before any download, rc 1, with the
  "LTX quality models need an NVIDIA GPU for now" message.

## What changed

Each change has its own commit and test.

| Change | Effect (R9700, 512 window) |
|---|---|
| Ship the native LTX block in AMD releases | denoise 2.11 -> 1.51 s/step |
| Encoder causal pad keeps the input memory format | eager encode 2.26 -> 1.39 s |
| Eager encoder tail runs in NCDHW | eager encode 1.38 -> 1.27 s |
| Decoder slab gathered and modulated inside the fused Q/K/V kernel | eager decode 1.85 -> 1.63 s, compiled 1.60 -> 1.58 s |
| Tiled transpose for the encoder head-to-tail copy | eager encode 1.27 -> 1.16 s |

- The tiled transpose landed after this build (main `aeca5b8`). The source tree run without
  Triton, which matches the release, measured 101.6 and 102.1 s for the LTX stage with it,
  encode 1.20 s/window.
- NVIDIA is unchanged. On an RTX 5090 the compiled encoder and decoder give bit-identical
  output at the same speed.

## Left

- Startup is still about 3 s slower than the source tree: 1.6 s is the installation check,
  and the rest is frozen imports.
- Encode is still 0.2 s/window slower than the compiled encoder. MIOpen transposes inside
  every NCDHW conv (86 ms per window), and the eager norms and SiLUs are separate passes.
- Decode is 0.05 s/window slower. The eager context inject still does a pixel-shuffle copy
  (53 ms per window).
