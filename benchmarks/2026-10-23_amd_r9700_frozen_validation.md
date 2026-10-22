# Linux AMD frozen release on an R9700, 2026-10-23

The frozen Linux AMD 0.11.3 release now runs end to end on a Radeon AI PRO R9700
(gfx1201). BasicVSR++ runs at the same speed as the source tree, and the output
matches it within run-to-run noise. The LTX trial completes, but it is 1.38x
slower than the source tree, because the release ships neither Triton nor native
LTX block kernels.

## Build

- Built from a clean `git archive` of main `2c3c773` with
  `build_linux_amd.sh`, 24 CPUs, in 81 min.
- The archive is 7.9 GB in four parts and unpacks to 17 GB. All build, release and
  archive gates passed.

## Environment

- One R9700 with 32 GB, a 4-core i3-12100F and 30 GB RAM. ROCm 10.1 userspace, AMF 26.20.
- Every GPU run took the pod's single GPU lock. The shell had no locale set (`LANG=`).
- The source tree for comparison used Python 3.12, torch 2.12.0+rocm10.0.0 and Triton 3.8.0.

## Results

CLI defaults unless noted (`--secondary-restoration none`). fps = frames / pipeline time.

| Check | Frozen | Source tree |
|---|---|---|
| `--version`, `--help` | pass (0.11.3) | |
| AMF check | passes; with AMF hidden it refuses with the install hint, rc 1 | |
| Licensed LTX without `--ltx-trial` | refused before any download, rc 1 | |
| 5310-frame 1080p clip, RF-DETR v6 + BasicVSR++ | 102.1, 84.1, 101.3, 88.2 fps (warm; first run 82.6) | 96.7, 94.5 fps |
| Same clip, PSNR vs source output | 48.12 dB (source run-to-run: 48.12 dB) | |
| 300-frame test clip, RF-DETR v6 | 59.1 fps warm, 25.1 fps first run, 300 frames | |
| Test clip, PSNR vs source output | 48.7 dB (source run-to-run: 47.8 dB) | |
| Test clip, `rfdetr-vr-v1` | 58.8 fps, 300 frames | |
| Test clip, `lada-yolo-v4` | 35.8 fps, 300 frames; PSNR vs source 49.5 dB | |
| Test clip, LTX trial, warm | 138.9 s LTX stage, 300 frames | 100.9 s |
| LTX encode / denoise / decode | 2.29 s/window, 2.11 s/step, 1.96 s/window | 1.08, 1.51, 1.72 |
| Startup to first frame | 7.6 s | 4.6 to 5.3 s |

- All 17 `.hipfb` files ship. A file watch saw every kernel the runs use load:
  the HIP media kernels, the fused BasicVSR++ propagation (`FusedPropagateBody`),
  the RF-DETR fused head (144 and 192), and the LTX VAE kernels plus the comfy-kitchen
  HIP extension. The BasicVSR++ and RF-DETR logs never mention Triton.
- In the LTX trial, the native block reports "Triton is not installed to compile
  its kernels and none are shipped", so it uses the eager torch path. The trial
  measures a speed profile on its first run, which makes that run slower (200 s wall).
- First run with empty caches: the run succeeds, and MIOpen logs
  `MIOPEN_FIND_MODE = HYBRID(3)`. MIOpen builds its kernels on first use, which
  makes that run about 8 s slower.
- Frozen BasicVSR++ fps varies more between runs (84 to 102) than the source tree.
  The median (94.7) matches the source tree.

## Fixed during validation

The first frozen builds failed. These fixes landed, each with a test:

- The Nuitka freeze stopped on PyYAML metadata without its `_yaml` shim.
- The new 512-wide LTX slab kernel was not bundled.
- `--help` failed with `UnicodeEncodeError` under the C locale. Linux builds now
  freeze Python's UTF-8 mode.
- RF-DETR failed to load. Nuitka finds bundled metadata only by its exact name,
  so the `huggingface-hub` version check failed. The frozen app now matches
  normalized names.
- Nuitka blanks bytecode docstrings, and Transformers' docstring decorators rejected
  them. AMD bytecode modules now keep their docstrings, and rfdetr ships as bytecode.
- Release builds now handle AMD architecture names correctly.
