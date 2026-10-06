# Identity-gated Windows AMD Math SDPA

English (default) | [中文](../../zh/changes/23-windows-sdpa-compat.md)

Feature: `23-windows-sdpa-compat`. Base source: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## Purpose

Apply the recorded Windows gfx1100 Math SDPA policy before constructing RF-DETR. Validate Torch 2.12.0+rocm10.0.0, HIP 7.15.26333, architecture, loaded runtime API, and DLL SHA-256 rather than trusting a package label.

## Usage and default behavior

JASNA_WINDOWS_AMD_SDPA_POLICY accepts auto, math, or default. Auto changes only the exact verified profile; math rejects an unverified profile; default observes existing flags. FP16 selection remains independent, and Math SDPA still runs on the GPU.

## Direct prerequisites

- [Windows ROCm import and vendor compatibility](03-windows-rocm-compat.md)
- [Validated RF-DETR MIGraphX selection](10-rfdetr-migraphx.md)

Dependencies describe the combined implementation. PRs must remain feature-scoped; an unmerged prerequisite is not native acceptance.

## Safety and support limits

Linux, CPU, and NVIDIA are neither probed nor changed. Process defaults do not override explicit sdpa_kernel contexts; this is not certification of LTX attention or a universal Windows ROCm workaround.

## Validation and reproduction

```bash
python -m pytest -q tests/test_windows_sdpa_policy.py
```

The preceding exact-source integrated stack passed **3141 tests, with 225 skips and 178 passing subtests** on Linux in an isolated CPU environment. This documentation update does not change runtime behavior. These counts describe the combined stack, not 26 independently passing dependency-free branches.

Historical Linux native acceptance covered 13 jobs (9427.04 seconds) with strict decode/timeline/seam checks on its recorded processing baseline. It does not automatically certify the newer Windows changes. This documentation refresh runs no real-video/GPU workload. Windows SDK/native asset rebuilding, whole-card telemetry, and real-hardware A/B are **WAIVED_BY_USER_NOT_RUN**, not PASS. Other untested Windows/NVIDIA/LTX and paid-model AMD paths remain uncertified.

## Implementation and detailed records

- [jasna/mosaic/rfdetr_torch_runner.py](../../../jasna/mosaic/rfdetr_torch_runner.py)
- [jasna/mosaic/windows_sdpa_policy.py](../../../jasna/mosaic/windows_sdpa_policy.py)
- [tests/test_windows_sdpa_policy.py](../../../tests/test_windows_sdpa_policy.py)
