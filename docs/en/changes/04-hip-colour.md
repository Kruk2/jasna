# HIP color conversion kernels

English (default) | [中文](../../zh/changes/04-hip-colour.md)

Feature: `04-hip-colour`. Base source: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## Purpose

Add AMD HIP RGB/YUV conversion, architecture-pinned code objects, identity checks, bounded scratch reuse, and explicit stream ownership. Preserve the NVIDIA implementation and existing color interfaces.

## Usage and default behavior

Linux uses the validated HIP product route where admitted. Windows HIP color remains opt-in and requires the matching manifest, code object, architecture, and loaded runtime; use the build/probe scripts shipped with this feature.

## Direct prerequisites

- [Windows ROCm import and vendor compatibility](03-windows-rocm-compat.md)

Dependencies describe the combined implementation. PRs must remain feature-scoped; an unmerged prerequisite is not native acceptance.

## Safety and support limits

NV12 and P010 have separate layout/bit-depth contracts. A kernel identity or synchronization failure is an error, not permission to silently stage frames through the CPU.

## Validation and reproduction

```bash
python -m pytest -q tests/test_hip_colour_kernel_product.py tests/test_hip_kernel.py tests/test_lut_kernel.py tests/test_rgb_to_yuv_kernel.py tests/test_yuv_scratch_reuse.py tests/test_yuv_to_rgb.py
```

The preceding exact-source integrated stack passed **3141 tests, with 225 skips and 178 passing subtests** on Linux in an isolated CPU environment. This documentation update does not change runtime behavior. These counts describe the combined stack, not 26 independently passing dependency-free branches.

Historical Linux native acceptance covered 13 jobs (9427.04 seconds) with strict decode/timeline/seam checks on its recorded processing baseline. It does not automatically certify the newer Windows changes. This documentation refresh runs no real-video/GPU workload. Windows SDK/native asset rebuilding, whole-card telemetry, and real-hardware A/B are **WAIVED_BY_USER_NOT_RUN**, not PASS. Other untested Windows/NVIDIA/LTX and paid-model AMD paths remain uncertified.

## Implementation and detailed records

- [docs/WINDOWS_AMD_HIP_COLOR_KERNELS_CN.md](../../../docs/WINDOWS_AMD_HIP_COLOR_KERNELS_CN.md)
- [jasna/media/hip_color_kernels.gfx1100.windows.json](../../../jasna/media/hip_color_kernels.gfx1100.windows.json)
- [jasna/media/hip_kernel.py](../../../jasna/media/hip_kernel.py)
- [jasna/media/rgb_to_yuv.gfx1100.hsaco](../../../jasna/media/rgb_to_yuv.gfx1100.hsaco)
- [jasna/media/rgb_to_yuv.gfx1100.windows.co](../../../jasna/media/rgb_to_yuv.gfx1100.windows.co)
- [jasna/media/rgb_to_yuv.py](../../../jasna/media/rgb_to_yuv.py)
- [jasna/media/yuv_to_rgb.gfx1100.hsaco](../../../jasna/media/yuv_to_rgb.gfx1100.hsaco)
- [jasna/media/yuv_to_rgb.gfx1100.windows.co](../../../jasna/media/yuv_to_rgb.gfx1100.windows.co)
- [jasna/media/yuv_to_rgb.py](../../../jasna/media/yuv_to_rgb.py)
- [scripts/build_hip_code_objects.sh](../../../scripts/build_hip_code_objects.sh)
- [scripts/build_hip_code_objects_windows.ps1](../../../scripts/build_hip_code_objects_windows.ps1)
- [scripts/probe_amd_hip_color_kernels.py](../../../scripts/probe_amd_hip_color_kernels.py)
- [tests/test_hip_colour_kernel_product.py](../../../tests/test_hip_colour_kernel_product.py)
- [tests/test_hip_kernel.py](../../../tests/test_hip_kernel.py)
- [tests/test_lut_kernel.py](../../../tests/test_lut_kernel.py)
- [tests/test_rgb_to_yuv_kernel.py](../../../tests/test_rgb_to_yuv_kernel.py)
- [tests/test_yuv_scratch_reuse.py](../../../tests/test_yuv_scratch_reuse.py)
- [tests/test_yuv_to_rgb.py](../../../tests/test_yuv_to_rgb.py)
