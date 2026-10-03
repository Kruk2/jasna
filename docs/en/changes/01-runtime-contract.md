# Pinned unified media runtime and installer

English (default) | [中文](../../zh/changes/01-runtime-contract.md)

Feature: `01-runtime-contract`. Base source: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## Purpose

Treat PyAV and its FFmpeg shared libraries as one pinned ABI unit. Add runtime/source/file identity validation, atomic installation, and child-process launchers without implementing a second processing pipeline.

## Usage and default behavior

Run scripts/run_jasna_unified.sh --preflight-only on Linux, or scripts/run_jasna_unified_windows.ps1 -PreflightOnly on Windows. JASNA_UNIFIED_RUNTIME_ROOT selects an installed runtime. Use scripts/install_unified_runtime.py --help for installation options.

## Direct prerequisites

None.

Dependencies describe the combined implementation. PRs must remain feature-scoped; an unmerged prerequisite is not native acceptance.

## Safety and support limits

Only the launched child environment is changed. Hash/ABI failures stop before media imports; system PyAV/FFmpeg or CPU fallback is not silently selected. Runtime assets are built/installed separately and are not bundled in this PR.

## Validation and reproduction

```bash
python -m pytest -q tests/test_runtime_contract.py tests/test_unified_runtime_installer.py
```

The preceding exact-source integrated stack passed **3141 tests, with 225 skips and 178 passing subtests** on Linux in an isolated CPU environment. This documentation update does not change runtime behavior. These counts describe the combined stack, not 26 independently passing dependency-free branches.

Historical Linux native acceptance covered 13 jobs (9427.04 seconds) with strict decode/timeline/seam checks on its recorded processing baseline. It does not automatically certify the newer Windows changes. This documentation refresh runs no real-video/GPU workload. Windows SDK/native asset rebuilding, whole-card telemetry, and real-hardware A/B are **WAIVED_BY_USER_NOT_RUN**, not PASS. Other untested Windows/NVIDIA/LTX and paid-model AMD paths remain uncertified.

## Implementation and detailed records

- [docs/UNIFIED_RUNTIME_CN.md](../../../docs/UNIFIED_RUNTIME_CN.md)
- [docs/UNIFIED_RUNTIME_INSTALL_CN.md](../../../docs/UNIFIED_RUNTIME_INSTALL_CN.md)
- [jasna/runtime_contract.py](../../../jasna/runtime_contract.py)
- [pyproject.toml](../../../pyproject.toml)
- [scripts/install_unified_runtime.py](../../../scripts/install_unified_runtime.py)
- [scripts/run_jasna_unified.py](../../../scripts/run_jasna_unified.py)
- [scripts/run_jasna_unified.sh](../../../scripts/run_jasna_unified.sh)
- [scripts/run_jasna_unified_windows.ps1](../../../scripts/run_jasna_unified_windows.ps1)
- [tests/test_runtime_contract.py](../../../tests/test_runtime_contract.py)
- [tests/test_unified_runtime_installer.py](../../../tests/test_unified_runtime_installer.py)
