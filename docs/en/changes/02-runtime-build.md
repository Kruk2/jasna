# Reproducible FFmpeg and PyAV builds

English (default) | [中文](../../zh/changes/02-runtime-build.md)

Feature: `02-runtime-build`. Base source: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## Purpose

Provide Linux and Windows build scripts with pinned FFmpeg/PyAV/AMF sources and auditable FFmpeg patches for transfer formats, stale frame contexts, resolution changes, keyframe reset, projection tags, and contiguous host input.

## Usage and default behavior

Use scripts/build_unified_ffmpeg_pyav_ubuntu.sh or scripts/build_unified_ffmpeg_pyav_windows.ps1. Install their verified output through the runtime installer; building alone must not switch the desktop launcher.

## Direct prerequisites

- [Pinned unified media runtime and installer](01-runtime-contract.md)

Dependencies describe the combined implementation. PRs must remain feature-scoped; an unmerged prerequisite is not native acceptance.

## Safety and support limits

Reproducible source pins are not a promise that an unbuilt SDK combination works. Windows SDK/native asset rebuilding is explicitly waived and NOT_RUN in this delivery.

## Validation and reproduction

```bash
python -m pytest -q tests/test_unified_build_scripts.py
```

The preceding exact-source integrated stack passed **3141 tests, with 225 skips and 178 passing subtests** on Linux in an isolated CPU environment. This documentation update does not change runtime behavior. These counts describe the combined stack, not 26 independently passing dependency-free branches.

Historical Linux native acceptance covered 13 jobs (9427.04 seconds) with strict decode/timeline/seam checks on its recorded processing baseline. It does not automatically certify the newer Windows changes. This documentation refresh runs no real-video/GPU workload. Windows SDK/native asset rebuilding, whole-card telemetry, and real-hardware A/B are **WAIVED_BY_USER_NOT_RUN**, not PASS. Other untested Windows/NVIDIA/LTX and paid-model AMD paths remain uncertified.

## Implementation and detailed records

- [docs/UNIFIED_BUILD_PIPELINE_CN.md](../../../docs/UNIFIED_BUILD_PIPELINE_CN.md)
- [patches/ffmpeg/0001-amf-transfer-use-context-sw-format.patch](../../../patches/ffmpeg/0001-amf-transfer-use-context-sw-format.patch)
- [patches/ffmpeg/0002-amfdec-replace-stale-frames-context.patch](../../../patches/ffmpeg/0002-amfdec-replace-stale-frames-context.patch)
- [patches/ffmpeg/0003-amfdec-fix-dynamic-resolution-reinit.patch](../../../patches/ffmpeg/0003-amfdec-fix-dynamic-resolution-reinit.patch)
- [patches/ffmpeg/0004-matroska-projection-tag-spherical.patch](../../../patches/ffmpeg/0004-matroska-projection-tag-spherical.patch)
- [patches/ffmpeg/0005-amfdec-reset-state-at-keyframes.patch](../../../patches/ffmpeg/0005-amfdec-reset-state-at-keyframes.patch)
- [patches/ffmpeg/0006-amfenc-wrap-contiguous-host-input.patch](../../../patches/ffmpeg/0006-amfenc-wrap-contiguous-host-input.patch)
- [scripts/build_unified_ffmpeg_pyav_ubuntu.sh](../../../scripts/build_unified_ffmpeg_pyav_ubuntu.sh)
- [scripts/build_unified_ffmpeg_pyav_windows.ps1](../../../scripts/build_unified_ffmpeg_pyav_windows.ps1)
- [tests/test_unified_build_scripts.py](../../../tests/test_unified_build_scripts.py)
