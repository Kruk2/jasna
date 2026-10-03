# Exact VR studio and projection routing

English (default) | [中文](../../zh/changes/18-vr-projection-studios.md)

Feature: `18-vr-projection-studios`. Base source: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## Purpose

Extend shared VR studio projection rules using exact token recognition, retaining explicit user projection choices and avoiding collisions between similar studio prefixes.

## Usage and default behavior

Auto routing includes the added CCVR, KBVR, KMVR, DSVR, MAXVR, and JPSVR fisheye mappings. Select raw/fisheye explicitly when file naming does not reliably describe the projection; consult the VR guide.

## Direct prerequisites

None.

Dependencies describe the combined implementation. PRs must remain feature-scoped; an unmerged prerequisite is not native acceptance.

## Safety and support limits

Studio naming is a routing heuristic, not proof of a projection or guaranteed detection quality. Keep the detector threshold and backend separate from projection decisions.

## Validation and reproduction

```bash
python -m pytest -q tests/test_vr180.py
```

The preceding exact-source integrated stack passed **3141 tests, with 225 skips and 178 passing subtests** on Linux in an isolated CPU environment. This documentation update does not change runtime behavior. These counts describe the combined stack, not 26 independently passing dependency-free branches.

Historical Linux native acceptance covered 13 jobs (9427.04 seconds) with strict decode/timeline/seam checks on its recorded processing baseline. It does not automatically certify the newer Windows changes. This documentation refresh runs no real-video/GPU workload. Windows SDK/native asset rebuilding, whole-card telemetry, and real-hardware A/B are **WAIVED_BY_USER_NOT_RUN**, not PASS. Other untested Windows/NVIDIA/LTX and paid-model AMD paths remain uncertified.

## Implementation and detailed records

- [docs/zh/vr180.md](../../../docs/zh/vr180.md)
- [jasna/vr180.py](../../../jasna/vr180.py)
- [tests/test_vr180.py](../../../tests/test_vr180.py)
