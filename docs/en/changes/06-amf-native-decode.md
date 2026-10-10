# AMF native decoding and frame ownership

English (default) | [中文](../../zh/changes/06-amf-native-decode.md)

Feature: `06-amf-native-decode`. Base source: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## Purpose

Use the Linux AMF Vulkan-to-HIP device-to-device route with explicit surface/cache/file-descriptor lifetimes and synchronization. Carry decoder seek/source lifetime correctness and guarded Windows handling at the backend boundary.

## Usage and default behavior

Use the shared reader/backend selection rather than a scan-specific decoder. Check D2D audit counters, frame ordering, native close errors, and format admission in logs. An explicitly selected native backend must fail visibly when its contract is violated.

## Direct prerequisites

- [Shared native job and diagnostic contracts](00-shared-native-job-contracts.md)
- [Pinned unified media runtime and installer](01-runtime-contract.md)
- [HIP color conversion kernels](04-hip-colour.md)
- [Bounded Windows D3D11–HIP resident media](07-windows-resident-media.md)

Dependencies describe the combined implementation. PRs must remain feature-scoped; an unmerged prerequisite is not native acceptance.

## Safety and support limits

rocDecode is removed and must not be reintroduced. Linux Vulkan/HIP and Windows D3D11/HIP are separate adapters; Linux acceptance does not certify the Windows combination.

## Validation and reproduction

```bash
python -m pytest -q tests/test_amf_interop_core.py tests/test_decoder_source_lifetime.py tests/test_video_decoder_backends.py tests/test_video_decoder_seek.py tests/test_video_decoder_software.py
```

The preceding exact-source integrated stack passed **3141 tests, with 225 skips and 178 passing subtests** on Linux in an isolated CPU environment. This documentation update does not change runtime behavior. These counts describe the combined stack, not 26 independently passing dependency-free branches.

Historical Linux native acceptance covered 13 jobs (9427.04 seconds) with strict decode/timeline/seam checks on its recorded processing baseline. It does not automatically certify the newer Windows changes. This documentation refresh runs no real-video/GPU workload. Windows SDK/native asset rebuilding, whole-card telemetry, and real-hardware A/B are **WAIVED_BY_USER_NOT_RUN**, not PASS. Other untested Windows/NVIDIA/LTX and paid-model AMD paths remain uncertified.

## Implementation and detailed records

- [docs/AMF_AV1_NATIVE_CN.md](../../../docs/AMF_AV1_NATIVE_CN.md)
- [docs/AMF_INTEROP_CORE_CN.md](../../../docs/AMF_INTEROP_CORE_CN.md)
- [docs/AMF_INTEROP_EVENT_POOL_CN.md](../../../docs/AMF_INTEROP_EVENT_POOL_CN.md)
- [docs/LINUX_AMD_AMF_CACHE_FD_LIFECYCLE_CN.md](../../../docs/LINUX_AMD_AMF_CACHE_FD_LIFECYCLE_CN.md)
- [docs/LINUX_AMD_AUTO_DECODE_CN.md](../../../docs/LINUX_AMD_AUTO_DECODE_CN.md)
- [docs/ROCDECODE_REMOVAL_CN.md](../../../docs/ROCDECODE_REMOVAL_CN.md)
- [docs/SHARED_DECODER_SOURCE_LIFETIME_ACCEPTANCE_CN.md](../../../docs/SHARED_DECODER_SOURCE_LIFETIME_ACCEPTANCE_CN.md)
- [docs/WINDOWS_AMD_CORRECTNESS_CN.md](../../../docs/WINDOWS_AMD_CORRECTNESS_CN.md)
- [jasna/media/video_decoder.py](../../../jasna/media/video_decoder.py)
- [scripts/amf_surface_probe.pyx](../../../scripts/amf_surface_probe.pyx)
- [scripts/build_amf_surface_probe.py](../../../scripts/build_amf_surface_probe.py)
- [tests/test_amf_interop_core.py](../../../tests/test_amf_interop_core.py)
- [tests/test_decoder_source_lifetime.py](../../../tests/test_decoder_source_lifetime.py)
- [tests/test_video_decoder_backends.py](../../../tests/test_video_decoder_backends.py)
- [tests/test_video_decoder_seek.py](../../../tests/test_video_decoder_seek.py)
- [tests/test_video_decoder_software.py](../../../tests/test_video_decoder_software.py)
