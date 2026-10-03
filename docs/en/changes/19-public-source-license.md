# Public-source optional license boundary

English (default) | [中文](../../zh/changes/19-public-source-license.md)

Feature: `19-public-source-license`. Base source: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## Purpose

Keep free model workflows importable when the private jasna.protection package is absent. Route compiler/image entry points through one boundary; preserve official store behavior when the private package is installed.

## Usage and default behavior

The absent-package shim returns no license, reports unlicensed, and rejects activation explicitly. Missing dependencies inside an installed private package are real errors and must not be hidden.

Path-only compiler tests import the existing shared engine-path helper directly, so their CPU checks do not require TensorRT. Their assertions and the product implementation are unchanged.

## Direct prerequisites

None.

Dependencies describe the combined implementation. PRs must remain feature-scoped; an unmerged prerequisite is not native acceptance.

## Safety and support limits

This does not implement paid modules, bypass activation, decrypt supporter weights, or certify paid-model AMD compatibility. Official importable private components are still required for those features.

## Validation and reproduction

```bash
python -m pytest -q tests/test_engine_compiler.py tests/test_license_api.py
```

The preceding exact-source integrated stack passed **3141 tests, with 225 skips and 178 passing subtests** on Linux in an isolated CPU environment. This documentation update does not change runtime behavior. These counts describe the combined stack, not 26 independently passing dependency-free branches.

Historical Linux native acceptance covered 13 jobs (9427.04 seconds) with strict decode/timeline/seam checks on its recorded processing baseline. It does not automatically certify the newer Windows changes. This documentation refresh runs no real-video/GPU workload. Windows SDK/native asset rebuilding, whole-card telemetry, and real-hardware A/B are **WAIVED_BY_USER_NOT_RUN**, not PASS. Other untested Windows/NVIDIA/LTX and paid-model AMD paths remain uncertified.

## Implementation and detailed records

- [docs/PUBLIC_SOURCE_LICENSE_BOUNDARY_CN.md](../../../docs/PUBLIC_SOURCE_LICENSE_BOUNDARY_CN.md)
- [jasna/engine_compiler.py](../../../jasna/engine_compiler.py)
- [jasna/image_restore.py](../../../jasna/image_restore.py)
- [jasna/license_api.py](../../../jasna/license_api.py)
- [tests/test_engine_compiler.py](../../../tests/test_engine_compiler.py)
- [tests/test_license_api.py](../../../tests/test_license_api.py)
