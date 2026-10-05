# Adaptive automatic pre-scan and missed-range fixes

English (default) | [中文](../../zh/changes/14-automatic-prescan.md)

Feature: `14-automatic-prescan`. Base source: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## Purpose

Use shared readers/detectors for adaptive coarse and fine scans, signature-validated checkpoints, coverage-based routing, and timestamp-jitter-aware hit merging. Avoid decoder-epoch churn on long high-resolution inputs.

## Usage and default behavior

GUI automatic mode normally coarse-scans near 4 seconds and fine-scans candidates near 0.5 seconds. No credible hits permits validated copy; default coverage of 85% selects full; otherwise use Smart Render. Explicit manual spans/full processing take priority.

## Direct prerequisites

- [AMF native decoding and frame ownership](06-amf-native-decode.md)
- [Validated RF-DETR MIGraphX selection](10-rfdetr-migraphx.md)

Dependencies describe the combined implementation. PRs must remain feature-scoped; an unmerged prerequisite is not native acceptance.

## Safety and support limits

Jitter tolerance must not bridge a genuinely missing sample. Invalid old signatures trigger rescan. Auto full fallback for an unrepresentable source GOP still runs restoration only on confirmed PTS ranges; an explicit Smart Render request is not silently changed.

## Validation and reproduction

```bash
python -m pytest -q tests/test_mosaic_scan.py tests/test_pre_scan_routing.py
```

The preceding exact-source integrated stack passed **3141 tests, with 225 skips and 178 passing subtests** on Linux in an isolated CPU environment. This documentation update does not change runtime behavior. These counts describe the combined stack, not 26 independently passing dependency-free branches.

Historical Linux native acceptance covered 13 jobs (9427.04 seconds) with strict decode/timeline/seam checks on its recorded processing baseline. It does not automatically certify the newer Windows changes. This documentation refresh runs no real-video/GPU workload. Windows SDK/native asset rebuilding, whole-card telemetry, and real-hardware A/B are **WAIVED_BY_USER_NOT_RUN**, not PASS. Other untested Windows/NVIDIA/LTX and paid-model AMD paths remain uncertified.

## Implementation and detailed records

- [docs/AUTOMATIC_PRE_SCAN_ROUTING_CN.md](../../../docs/AUTOMATIC_PRE_SCAN_ROUTING_CN.md)
- [docs/MOSAIC_SCAN_UNIFIED_AMD_CN.md](../../../docs/MOSAIC_SCAN_UNIFIED_AMD_CN.md)
- [jasna/gui/mosaic_scan.py](../../../jasna/gui/mosaic_scan.py)
- [jasna/gui/pre_scan_routing.py](../../../jasna/gui/pre_scan_routing.py)
- [tests/test_mosaic_scan.py](../../../tests/test_mosaic_scan.py)
- [tests/test_pre_scan_routing.py](../../../tests/test_pre_scan_routing.py)
