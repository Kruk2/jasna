# Smart Render seams and durable resume

English (default) | [中文](../../zh/changes/13-smart-render-resume.md)

Feature: `13-smart-render-resume`. Base source: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## Purpose

Combine copied and rendered spans while validating parameter sets, timestamp continuity, frame/duration contracts, and span boundaries. Keep durable fragment manifests and invalidate incompatible cached work.

## Usage and default behavior

Use the existing segment/Smart Render route. Completed compatible fragments may be reused; incomplete or mismatched fragments must not be accepted solely because a file exists. Full-span empty effect ranges mean all frames, not no restoration.

## Direct prerequisites

- [AMF native decoding and frame ownership](06-amf-native-decode.md)
- [AMD encoder contracts and source-rate Peak VBR](08-encoder-source-rate.md)

Dependencies describe the combined implementation. PRs must remain feature-scoped; an unmerged prerequisite is not native acceptance.

## Safety and support limits

Smart Render preserves source packet contracts and therefore cannot freely change the output codec/profile. Structural checks, strict decode, seam checks, and actual restoration evidence serve different purposes.

## Validation and reproduction

```bash
python -m pytest -q tests/test_smart_render_workspace.py tests/test_splice.py tests/test_splice_media.py
```

The preceding exact-source integrated stack passed **3141 tests, with 225 skips and 178 passing subtests** on Linux in an isolated CPU environment. This documentation update does not change runtime behavior. These counts describe the combined stack, not 26 independently passing dependency-free branches.

Historical Linux native acceptance covered 13 jobs (9427.04 seconds) with strict decode/timeline/seam checks on its recorded processing baseline. It does not automatically certify the newer Windows changes. This documentation refresh runs no real-video/GPU workload. Windows SDK/native asset rebuilding, whole-card telemetry, and real-hardware A/B are **WAIVED_BY_USER_NOT_RUN**, not PASS. Other untested Windows/NVIDIA/LTX and paid-model AMD paths remain uncertified.

## Implementation and detailed records

- [docs/SMART_RENDER_RESUME_WORKSPACE_CN.md](../../../docs/SMART_RENDER_RESUME_WORKSPACE_CN.md)
- [docs/SMART_RENDER_TIMESTAMP_SEAM_CN.md](../../../docs/SMART_RENDER_TIMESTAMP_SEAM_CN.md)
- [docs/en/segments.md](../../../docs/en/segments.md)
- [docs/ja/segments.md](../../../docs/ja/segments.md)
- [docs/zh/segments.md](../../../docs/zh/segments.md)
- [jasna/media/splice.py](../../../jasna/media/splice.py)
- [jasna/smart_render_workspace.py](../../../jasna/smart_render_workspace.py)
- [tests/test_smart_render_workspace.py](../../../tests/test_smart_render_workspace.py)
- [tests/test_splice.py](../../../tests/test_splice.py)
- [tests/test_splice_media.py](../../../tests/test_splice_media.py)
