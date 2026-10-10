# Bounded Linux HEVC dual-GOP encoding

English (default) | [中文](../../zh/changes/09-dual-gop.md)

Feature: `09-dual-gop`. Base source: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## Purpose

Alternate independent closed GOPs between two persistent AMF encoding sessions and assemble their results in timeline order. Bound staging, surfaces, pending GOPs, and shutdown ownership.

## Usage and default behavior

GUI presets request dual GOP; CLI use is explicit through --amd-dual-gop-encode. Final HEVC output admission requires Main10/P010 at least 3840x2160 pixels or Main/NV12 at least 5760x2880 pixels, positive even dimensions, and usable source-rate information.

## Direct prerequisites

- [AMD encoder contracts and source-rate Peak VBR](08-encoder-source-rate.md)
- [Smart Render seams and durable resume](13-smart-render-resume.md)

Dependencies describe the combined implementation. PRs must remain feature-scoped; an unmerged prerequisite is not native acceptance.

## Safety and support limits

Smart Render also requires compatible HEVC source packets. Non-admitted GUI jobs retain single-session encoding. This is Linux GOP-level parallelism, not Windows split-frame, and speedup must be measured against a matched baseline.

## Validation and reproduction

```bash
python -m pytest -q tests/test_dual_gop_encoder.py
```

The preceding exact-source integrated stack passed **3141 tests, with 225 skips and 178 passing subtests** on Linux in an isolated CPU environment. This documentation update does not change runtime behavior. These counts describe the combined stack, not 26 independently passing dependency-free branches.

Historical Linux native acceptance covered 13 jobs (9427.04 seconds) with strict decode/timeline/seam checks on its recorded processing baseline. It does not automatically certify the newer Windows changes. This documentation refresh runs no real-video/GPU workload. Windows SDK/native asset rebuilding, whole-card telemetry, and real-hardware A/B are **WAIVED_BY_USER_NOT_RUN**, not PASS. Other untested Windows/NVIDIA/LTX and paid-model AMD paths remain uncertified.

## Implementation and detailed records

- [docs/AMD_DUAL_GOP_ENCODER_CN.md](../../../docs/AMD_DUAL_GOP_ENCODER_CN.md)
- [jasna/media/dual_gop_encoder.py](../../../jasna/media/dual_gop_encoder.py)
- [tests/test_dual_gop_encoder.py](../../../tests/test_dual_gop_encoder.py)
