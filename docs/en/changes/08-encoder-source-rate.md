# AMD encoder contracts and source-rate Peak VBR

English (default) | [中文](../../zh/changes/08-encoder-source-rate.md)

Feature: `08-encoder-source-rate`. Base source: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## Purpose

Preserve codec/profile/bit-depth, buffered frame/PTS/LUT ownership, and AMF option bounds. Add the GUI choice between automatic source-rate HEVC Peak VBR and manual constant-QP/CQ.

## Usage and default behavior

In supported Linux AMD HEVC GUI routes, automatic mode uses source-derived vbr_peak for full processing and Smart Render; manual mode uses cqp and the user's CQ. Full may transcode H.264 input to HEVC; Smart Render packet copying additionally requires a compatible source stream.

## Direct prerequisites

- [HIP color conversion kernels](04-hip-colour.md)
- [Bounded Windows D3D11–HIP resident media](07-windows-resident-media.md)

Dependencies describe the combined implementation. PRs must remain feature-scoped; an unmerged prerequisite is not native acceptance.

## Safety and support limits

Rate control does not guarantee identical file size: repaired content and copied spans differ. AV1 Main10/P010 has its own preanalysis/rate-control rules; neither Linux HEVC performance nor CPU tests certify untested Windows formats.

## Validation and reproduction

```bash
python -m pytest -q tests/test_amd_support.py tests/test_hevc_smart_render_encoder.py tests/test_media_init.py tests/test_video_encoder_mux.py tests/test_video_encoder_unit.py
```

The preceding exact-source integrated stack passed **3141 tests, with 225 skips and 178 passing subtests** on Linux in an isolated CPU environment. This documentation update does not change runtime behavior. These counts describe the combined stack, not 26 independently passing dependency-free branches.

Historical Linux native acceptance covered 13 jobs (9427.04 seconds) with strict decode/timeline/seam checks on its recorded processing baseline. It does not automatically certify the newer Windows changes. This documentation refresh runs no real-video/GPU workload. Windows SDK/native asset rebuilding, whole-card telemetry, and real-hardware A/B are **WAIVED_BY_USER_NOT_RUN**, not PASS. Other untested Windows/NVIDIA/LTX and paid-model AMD paths remain uncertified.

## Implementation and detailed records

- [docs/AMD_ENCODER_CORRECTNESS_CN.md](../../../docs/AMD_ENCODER_CORRECTNESS_CN.md)
- [docs/HEVC_SMART_RENDER_ENCODER_CN.md](../../../docs/HEVC_SMART_RENDER_ENCODER_CN.md)
- [docs/WINDOWS_AMD_HEVC_VBR_PEAK_TODO_CN.md](../../../docs/WINDOWS_AMD_HEVC_VBR_PEAK_TODO_CN.md)
- [jasna/media/media_files.py](../../../jasna/media/media_files.py)
- [jasna/media/probe.py](../../../jasna/media/probe.py)
- [jasna/media/video_encoder.py](../../../jasna/media/video_encoder.py)
- [tests/test_amd_support.py](../../../tests/test_amd_support.py)
- [tests/test_hevc_smart_render_encoder.py](../../../tests/test_hevc_smart_render_encoder.py)
- [tests/test_media_init.py](../../../tests/test_media_init.py)
- [tests/test_video_encoder_mux.py](../../../tests/test_video_encoder_mux.py)
- [tests/test_video_encoder_unit.py](../../../tests/test_video_encoder_unit.py)
