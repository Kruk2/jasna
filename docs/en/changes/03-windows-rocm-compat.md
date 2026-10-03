# Windows ROCm import and vendor compatibility

English (default) | [中文](../../zh/changes/03-windows-rocm-compat.md)

Feature: `03-windows-rocm-compat`. Base source: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## Purpose

Make public-source imports and system checks distinguish AMD/ROCm from NVIDIA/CUDA. Provide a narrow MMEngine compatibility layer and keep TensorRT optional instead of making AMD source runs import NVIDIA-only components.

## Usage and default behavior

Use the existing source launcher and model interfaces. Detection/media tests declare their vendor assumptions; importing a module must not require another vendor's unavailable libraries.

## Direct prerequisites

None.

Dependencies describe the combined implementation. PRs must remain feature-scoped; an unmerged prerequisite is not native acceptance.

## Safety and support limits

This is compatibility infrastructure, not a new inference backend or evidence of Windows end-to-end performance. NVIDIA TensorRT behavior remains vendor-specific.

## Validation and reproduction

```bash
python -m pytest -q tests/test_mmengine_windows_rocm_compat.py tests/test_os_utils.py tests/test_trt_utils.py
```

The preceding exact-source integrated stack passed **3141 tests, with 225 skips and 178 passing subtests** on Linux in an isolated CPU environment. This documentation update does not change runtime behavior. These counts describe the combined stack, not 26 independently passing dependency-free branches.

Historical Linux native acceptance covered 13 jobs (9427.04 seconds) with strict decode/timeline/seam checks on its recorded processing baseline. It does not automatically certify the newer Windows changes. This documentation refresh runs no real-video/GPU workload. Windows SDK/native asset rebuilding, whole-card telemetry, and real-hardware A/B are **WAIVED_BY_USER_NOT_RUN**, not PASS. Other untested Windows/NVIDIA/LTX and paid-model AMD paths remain uncertified.

## Implementation and detailed records

- [docs/DETECTION_TEST_VENDOR_ISOLATION_CN.md](../../../docs/DETECTION_TEST_VENDOR_ISOLATION_CN.md)
- [docs/MEDIA_TEST_ENVIRONMENT_ISOLATION_CN.md](../../../docs/MEDIA_TEST_ENVIRONMENT_ISOLATION_CN.md)
- [docs/WINDOWS_MATCHED_GUI_ACCEPTANCE_CN.md](../../../docs/WINDOWS_MATCHED_GUI_ACCEPTANCE_CN.md)
- [jasna/models/basicvsrpp/__init__.py](../../../jasna/models/basicvsrpp/__init__.py)
- [jasna/models/basicvsrpp/mmengine_compat.py](../../../jasna/models/basicvsrpp/mmengine_compat.py)
- [jasna/os_utils.py](../../../jasna/os_utils.py)
- [jasna/trt/__init__.py](../../../jasna/trt/__init__.py)
- [tests/conftest.py](../../../tests/conftest.py)
- [tests/test_mmengine_windows_rocm_compat.py](../../../tests/test_mmengine_windows_rocm_compat.py)
- [tests/test_os_utils.py](../../../tests/test_os_utils.py)
- [tests/test_trt_utils.py](../../../tests/test_trt_utils.py)
