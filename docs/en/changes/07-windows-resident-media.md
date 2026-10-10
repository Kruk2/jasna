# Bounded Windows D3D11–HIP resident media

English (default) | [中文](../../zh/changes/07-windows-resident-media.md)

Feature: `07-windows-resident-media`. Base source: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## Purpose

Add the Python coordinator, Cython wrapper, native implementation, builder, and product probe for explicit D3D11–HIP resident decode/encode surfaces. Keep a bounded ownership/fence contract.

## Usage and default behavior

JASNA_WINDOWS_D3D11_HIP_RESIDENT=1 requests the component; the default is off. The recorded admission covers 1920x1080 and 3840x2160. Encoder pools contain four surfaces and must match their AMF settings.

## Direct prerequisites

- [Pinned unified media runtime and installer](01-runtime-contract.md)
- [HIP color conversion kernels](04-hip-colour.md)

Dependencies describe the combined implementation. PRs must remain feature-scoped; an unmerged prerequisite is not native acceptance.

## Safety and support limits

8192x4096 is rejected by the dual-reader product memory guard even if a single-reader probe succeeded. This is not a default Windows 8K optimization; freezing/distribution and broader hardware acceptance remain outside the certified scope.

## Validation and reproduction

```bash
python -m pytest -q tests/test_windows_d3d11_hip_resident.py
```

The preceding exact-source integrated stack passed **3141 tests, with 225 skips and 178 passing subtests** on Linux in an isolated CPU environment. This documentation update does not change runtime behavior. These counts describe the combined stack, not 26 independently passing dependency-free branches.

Historical Linux native acceptance covered 13 jobs (9427.04 seconds) with strict decode/timeline/seam checks on its recorded processing baseline. It does not automatically certify the newer Windows changes. This documentation refresh runs no real-video/GPU workload. Windows SDK/native asset rebuilding, whole-card telemetry, and real-hardware A/B are **WAIVED_BY_USER_NOT_RUN**, not PASS. Other untested Windows/NVIDIA/LTX and paid-model AMD paths remain uncertified.

## Implementation and detailed records

- [docs/WINDOWS_RESIDENT_MEDIA_CN.md](../../../docs/WINDOWS_RESIDENT_MEDIA_CN.md)
- [jasna/media/windows_d3d11_hip_resident.py](../../../jasna/media/windows_d3d11_hip_resident.py)
- [scripts/amf_d3d11_hip_resident.pyx](../../../scripts/amf_d3d11_hip_resident.pyx)
- [scripts/build_amf_d3d11_hip_resident.py](../../../scripts/build_amf_d3d11_hip_resident.py)
- [scripts/native/amf_d3d11_hip_resident_native.hpp](../../../scripts/native/amf_d3d11_hip_resident_native.hpp)
- [scripts/probe_windows_d3d11_hip_resident_product.py](../../../scripts/probe_windows_d3d11_hip_resident_product.py)
- [tests/test_windows_d3d11_hip_resident.py](../../../tests/test_windows_d3d11_hip_resident.py)
