# Explicit AMD performance and capacity probes

English (default) | [中文](../../zh/changes/21-performance-probes.md)

Feature: `21-performance-probes`. Base source: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## Purpose

Add opt-in probes for async D2H dual GOP, native YUV roundtrip, RF-DETR B2, and single-decode capacity. Record comparable timings, resource/lifetime evidence, and source/output identities.

## Usage and default behavior

Use the corresponding scripts/probe_* entry point explicitly with --help. Compare the same source, range, ROI, format, hardware, and warmup policy; use bounded samples before production promotion.

## Direct prerequisites

- [Shared pipeline resource and failure safety](12-pipeline-resource-safety.md)

Dependencies describe the combined implementation. PRs must remain feature-scoped; an unmerged prerequisite is not native acceptance.

## Safety and support limits

These scripts do not automatically enable B2, single-decode, native roundtrip, or new queue depths in the GUI. Capacity and synthetic microbenchmarks are not whole-video speed or quality certification.

## Validation and reproduction

```bash
python -m pytest -q tests/test_probe_amd_dual_gop_async_d2h.py tests/test_probe_amd_native_yuv_roundtrip.py tests/test_probe_rfdetr_migraphx_b2.py tests/test_probe_single_decode_capacity.py
```

The preceding exact-source integrated stack passed **3141 tests, with 225 skips and 178 passing subtests** on Linux in an isolated CPU environment. This documentation update does not change runtime behavior. These counts describe the combined stack, not 26 independently passing dependency-free branches.

Historical Linux native acceptance covered 13 jobs (9427.04 seconds) with strict decode/timeline/seam checks on its recorded processing baseline. It does not automatically certify the newer Windows changes. This documentation refresh runs no real-video/GPU workload. Windows SDK/native asset rebuilding, whole-card telemetry, and real-hardware A/B are **WAIVED_BY_USER_NOT_RUN**, not PASS. Other untested Windows/NVIDIA/LTX and paid-model AMD paths remain uncertified.

## Implementation and detailed records

- [docs/AMD_8K_PIPELINE_OPTIMIZATION_AUDIT_20260905_CN.md](../../../docs/AMD_8K_PIPELINE_OPTIMIZATION_AUDIT_20260905_CN.md)
- [scripts/probe_amd_dual_gop_async_d2h.py](../../../scripts/probe_amd_dual_gop_async_d2h.py)
- [scripts/probe_amd_native_yuv_roundtrip.py](../../../scripts/probe_amd_native_yuv_roundtrip.py)
- [scripts/probe_rfdetr_migraphx_b2.py](../../../scripts/probe_rfdetr_migraphx_b2.py)
- [scripts/probe_rfdetr_migraphx_b2_product.py](../../../scripts/probe_rfdetr_migraphx_b2_product.py)
- [scripts/probe_single_decode_capacity.py](../../../scripts/probe_single_decode_capacity.py)
- [tests/test_probe_amd_dual_gop_async_d2h.py](../../../tests/test_probe_amd_dual_gop_async_d2h.py)
- [tests/test_probe_amd_native_yuv_roundtrip.py](../../../tests/test_probe_amd_native_yuv_roundtrip.py)
- [tests/test_probe_rfdetr_migraphx_b2.py](../../../tests/test_probe_rfdetr_migraphx_b2.py)
- [tests/test_probe_single_decode_capacity.py](../../../tests/test_probe_single_decode_capacity.py)
