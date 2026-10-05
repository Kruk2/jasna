# Validated BasicVSR++ MIGraphX B1 restoration

English (default) | [中文](../../zh/changes/11-basicvsrpp-migraphx.md)

Feature: `11-basicvsrpp-migraphx`. Base source: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## Purpose

Add a bounded B1 restoration implementation, artifact builder/probe, and strict validation of model/source/runtime/extension identity. Distinguish the selected Torch-MIGraphX binary from another already loaded extension.

## Usage and default behavior

JASNA_BASICVSRPP_MIGRAPHX_B1 requests the route; JASNA_BASICVSRPP_MIGRAPHX_B1_DIR selects its artifact directory. Use the included builder/probe only with matching model and runtime identities.

## Direct prerequisites

- [Windows ROCm import and vendor compatibility](03-windows-rocm-compat.md)
- [Validated RF-DETR MIGraphX selection](10-rfdetr-migraphx.md)

Dependencies describe the combined implementation. PRs must remain feature-scoped; an unmerged prerequisite is not native acceptance.

## Safety and support limits

A native event/illegal-memory-access failure must propagate with its original cause. Do not silently switch to CPU, skip restoration, or reuse an artifact merely because its filename matches.

## Validation and reproduction

```bash
python -m pytest -q tests/test_basicvsrpp_migraphx_b1_product.py tests/test_basicvsrpp_mosaic_restorer.py tests/test_build_basicvsrpp_migraphx_b1.py
```

The preceding exact-source integrated stack passed **3141 tests, with 225 skips and 178 passing subtests** on Linux in an isolated CPU environment. This documentation update does not change runtime behavior. These counts describe the combined stack, not 26 independently passing dependency-free branches.

Historical Linux native acceptance covered 13 jobs (9427.04 seconds) with strict decode/timeline/seam checks on its recorded processing baseline. It does not automatically certify the newer Windows changes. This documentation refresh runs no real-video/GPU workload. Windows SDK/native asset rebuilding, whole-card telemetry, and real-hardware A/B are **WAIVED_BY_USER_NOT_RUN**, not PASS. Other untested Windows/NVIDIA/LTX and paid-model AMD paths remain uncertified.

## Implementation and detailed records

- [docs/LINUX_AMD_HIP_MIGRAPHX_OPTIMIZATION_CN.md](../../../docs/LINUX_AMD_HIP_MIGRAPHX_OPTIMIZATION_CN.md)
- [jasna/restorer/basicvsrpp_migraphx_b1.py](../../../jasna/restorer/basicvsrpp_migraphx_b1.py)
- [jasna/restorer/basicvsrpp_mosaic_restorer.py](../../../jasna/restorer/basicvsrpp_mosaic_restorer.py)
- [scripts/build_basicvsrpp_migraphx_b1.py](../../../scripts/build_basicvsrpp_migraphx_b1.py)
- [scripts/probe_basicvsrpp_migraphx_b1.py](../../../scripts/probe_basicvsrpp_migraphx_b1.py)
- [tests/test_basicvsrpp_migraphx_b1_product.py](../../../tests/test_basicvsrpp_migraphx_b1_product.py)
- [tests/test_basicvsrpp_mosaic_restorer.py](../../../tests/test_basicvsrpp_mosaic_restorer.py)
- [tests/test_build_basicvsrpp_migraphx_b1.py](../../../tests/test_build_basicvsrpp_migraphx_b1.py)
