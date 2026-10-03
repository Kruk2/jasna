# CPU-only SD 1.5 regression isolation

English (default) | [中文](../../zh/changes/20-cpu-regression-isolation.md)

Feature: `20-cpu-regression-isolation`. Base source: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## Purpose

Make the SD 1.5 unit fixture declare its mocked hardware/import state so it is reproducible on CPU test hosts independently of the physical GPU vendor.

## Usage and default behavior

Run tests/test_sd15_inpaint_restorer.py in the isolated CPU test environment. Synthetic/mocked modules are test fixtures, not product inference backends.

## Direct prerequisites

- [GUI controls, diagnostics, and reliable progress](17-gui-settings-diagnostics.md)

Dependencies describe the combined implementation. PRs must remain feature-scoped; an unmerged prerequisite is not native acceptance.

## Safety and support limits

Passing this fixture does not install private models, activate paid functionality, or prove SD 1.5 AMD GPU compatibility. This feature changes tests only.

## Validation and reproduction

```bash
python -m pytest -q tests/test_sd15_inpaint_restorer.py
```

The preceding exact-source integrated stack passed **3141 tests, with 225 skips and 178 passing subtests** on Linux in an isolated CPU environment. This documentation update does not change runtime behavior. These counts describe the combined stack, not 26 independently passing dependency-free branches.

Historical Linux native acceptance covered 13 jobs (9427.04 seconds) with strict decode/timeline/seam checks on its recorded processing baseline. It does not automatically certify the newer Windows changes. This documentation refresh runs no real-video/GPU workload. Windows SDK/native asset rebuilding, whole-card telemetry, and real-hardware A/B are **WAIVED_BY_USER_NOT_RUN**, not PASS. Other untested Windows/NVIDIA/LTX and paid-model AMD paths remain uncertified.

## Implementation and detailed records

- [tests/test_sd15_inpaint_restorer.py](../../../tests/test_sd15_inpaint_restorer.py)
