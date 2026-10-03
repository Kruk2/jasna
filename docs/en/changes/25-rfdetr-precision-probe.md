# Permutation-aware RF-DETR precision diagnostics

English (default) | [中文](../../zh/changes/25-rfdetr-precision-probe.md)

Feature: `25-rfdetr-precision-probe`. Base source: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## Purpose

Compare synthetic Windows B1/B4 FP32/FP16 RF-DETR outputs with Hungarian proposal matching so proposal reordering is not mistaken for precision failure. Record identities and explicit evidence limits.

## Usage and default behavior

Run scripts/probe_windows_rfdetr_precision.py --weights <weights> --output <report>. Optional --batches accepts 1 and 4. CPU execution is a declared numerical reference, not a silent fallback for failed GPU inference.

## Direct prerequisites

- [Identity-gated Windows AMD Math SDPA](23-windows-sdpa-compat.md)

Dependencies describe the combined implementation. PRs must remain feature-scoped; an unmerged prerequisite is not native acceptance.

## Safety and support limits

Synthetic comparison does not certify real detection/restore quality. Quality remains NOT_CERTIFIED and throughput/performance remains NOT_RUN unless separately measured on real hardware.

## Validation and reproduction

```bash
python -m pytest -q tests/test_rfdetr_precision_matching.py
```

The preceding exact-source integrated stack passed **3141 tests, with 225 skips and 178 passing subtests** on Linux in an isolated CPU environment. This documentation update does not change runtime behavior. These counts describe the combined stack, not 26 independently passing dependency-free branches.

Historical Linux native acceptance covered 13 jobs (9427.04 seconds) with strict decode/timeline/seam checks on its recorded processing baseline. It does not automatically certify the newer Windows changes. This documentation refresh runs no real-video/GPU workload. Windows SDK/native asset rebuilding, whole-card telemetry, and real-hardware A/B are **WAIVED_BY_USER_NOT_RUN**, not PASS. Other untested Windows/NVIDIA/LTX and paid-model AMD paths remain uncertified.

## Implementation and detailed records

- [scripts/probe_windows_rfdetr_precision.py](../../../scripts/probe_windows_rfdetr_precision.py)
- [tests/test_rfdetr_precision_matching.py](../../../tests/test_rfdetr_precision_matching.py)
