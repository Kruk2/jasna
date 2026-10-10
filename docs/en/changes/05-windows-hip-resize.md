# Opt-in Windows HIP resize normalization

English (default) | [中文](../../zh/changes/05-windows-hip-resize.md)

Feature: `05-windows-hip-resize`. Base source: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## Purpose

Implement a precompiled Windows AMD ResizeNormalizer backend while preserving the shared preprocessing/postprocessing interface, kernel ABI, stream selection, and NVIDIA/Linux defaults.

## Usage and default behavior

JASNA_WINDOWS_HIP_RESIZE=1 requests the backend. Admission is limited to the recorded gfx1100 runtime/code-object identity and supported B=1..4, C=3 geometry. The existing contract document records exact version/hash and stride restrictions.

## Direct prerequisites

- [HIP color conversion kernels](04-hip-colour.md)

Dependencies describe the combined implementation. PRs must remain feature-scoped; an unmerged prerequisite is not native acceptance.

## Safety and support limits

The option defaults off. Out-of-scope geometry retains the original Torch expression; a mismatched explicitly selected bundle is an error. Historical component measurements do not certify a newly upgraded SDK or all Windows GUI routes.

## Validation and reproduction

```bash
python -m pytest -q tests/test_windows_hip_resize_contract.py
```

The preceding exact-source integrated stack passed **3141 tests, with 225 skips and 178 passing subtests** on Linux in an isolated CPU environment. This documentation update does not change runtime behavior. These counts describe the combined stack, not 26 independently passing dependency-free branches.

Historical Linux native acceptance covered 13 jobs (9427.04 seconds) with strict decode/timeline/seam checks on its recorded processing baseline. It does not automatically certify the newer Windows changes. This documentation refresh runs no real-video/GPU workload. Windows SDK/native asset rebuilding, whole-card telemetry, and real-hardware A/B are **WAIVED_BY_USER_NOT_RUN**, not PASS. Other untested Windows/NVIDIA/LTX and paid-model AMD paths remain uncertified.

## Implementation and detailed records

- [docs/WINDOWS_HIP_RESIZE_ACCEPTANCE_CN.md](../../../docs/WINDOWS_HIP_RESIZE_ACCEPTANCE_CN.md)
- [jasna/media/hip_resize_normalize.gfx1100.windows.json](../../../jasna/media/hip_resize_normalize.gfx1100.windows.json)
- [jasna/media/resize_normalize.gfx1100.windows.co](../../../jasna/media/resize_normalize.gfx1100.windows.co)
- [jasna/media/resize_normalize.py](../../../jasna/media/resize_normalize.py)
- [jasna/media/windows_hip_resize_contract.py](../../../jasna/media/windows_hip_resize_contract.py)
- [scripts/build_windows_hip_resize.py](../../../scripts/build_windows_hip_resize.py)
- [tests/test_windows_hip_resize_contract.py](../../../tests/test_windows_hip_resize_contract.py)
