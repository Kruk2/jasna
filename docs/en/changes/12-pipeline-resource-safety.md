# Shared pipeline resource and failure safety

English (default) | [中文](../../zh/changes/12-pipeline-resource-safety.md)

Feature: `12-pipeline-resource-safety`. Base source: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## Purpose

Share restoration session construction and the decode/detect, primary, secondary, blend/encode pipeline. Add capacity-scaled GPU/host budgets, bounded queue ownership, native stall detection, and first-error propagation.

## Usage and default behavior

Memory reclamation uses capacity-scaled watermarks and pressure/debounce policies rather than a universal remaining-4-GiB trigger. On worker failure/cancel, release blocked producers and drain only the cancellation path; healthy queues are untouched.

## Direct prerequisites

- [Shared native job and diagnostic contracts](00-shared-native-job-contracts.md)
- [AMF native decoding and frame ownership](06-amf-native-decode.md)
- [Bounded Windows D3D11–HIP resident media](07-windows-resident-media.md)
- [AMD encoder contracts and source-rate Peak VBR](08-encoder-source-rate.md)
- [Bounded Linux HEVC dual-GOP encoding](09-dual-gop.md)
- [Validated BasicVSR++ MIGraphX B1 restoration](11-basicvsrpp-migraphx.md)
- [Smart Render seams and durable resume](13-smart-render-resume.md)
- [Public-source optional license boundary](19-public-source-license.md)

Dependencies describe the combined implementation. PRs must remain feature-scoped; an unmerged prerequisite is not native acceptance.

## Safety and support limits

Preserve the first real exception through cleanup. User Stop is not a fabricated worker failure. Resource limits are safety controls, not automatic performance certification for every card or format.

## Validation and reproduction

```bash
python -m pytest -q tests/test_dual_gop_encoder.py tests/test_frame_queue.py tests/test_main.py tests/test_main_validation.py tests/test_owned_vram_reader.py tests/test_pipeline_run.py tests/test_pipeline_run_sync.py tests/test_pipeline_segments.py tests/test_pipeline_threads.py tests/test_progressbar.py tests/test_session_config.py tests/test_session_factory.py tests/test_shared_pipeline_cleanup.py tests/test_streaming.py tests/test_tvai_secondary_restorer.py tests/test_video_session.py tests/test_vram_offloader.py tests/test_windows_vram_reader_factory.py
```

The preceding exact-source integrated stack passed **3141 tests, with 225 skips and 178 passing subtests** on Linux in an isolated CPU environment. This documentation update does not change runtime behavior. These counts describe the combined stack, not 26 independently passing dependency-free branches.

Historical Linux native acceptance covered 13 jobs (9427.04 seconds) with strict decode/timeline/seam checks on its recorded processing baseline. It does not automatically certify the newer Windows changes. This documentation refresh runs no real-video/GPU workload. Windows SDK/native asset rebuilding, whole-card telemetry, and real-hardware A/B are **WAIVED_BY_USER_NOT_RUN**, not PASS. Other untested Windows/NVIDIA/LTX and paid-model AMD paths remain uncertified.

## Implementation and detailed records

- [docs/PIPELINE_WORKER_FAILURE_PROPAGATION_CN.md](../../../docs/PIPELINE_WORKER_FAILURE_PROPAGATION_CN.md)
- [docs/en/cli.md](../../../docs/en/cli.md)
- [docs/ja/cli.md](../../../docs/ja/cli.md)
- [docs/zh/cli.md](../../../docs/zh/cli.md)
- [jasna/cli_help.py](../../../jasna/cli_help.py)
- [jasna/frame_queue.py](../../../jasna/frame_queue.py)
- [jasna/gui/video_session.py](../../../jasna/gui/video_session.py)
- [jasna/main.py](../../../jasna/main.py)
- [jasna/pipeline.py](../../../jasna/pipeline.py)
- [jasna/pipeline_threads.py](../../../jasna/pipeline_threads.py)
- [jasna/progressbar.py](../../../jasna/progressbar.py)
- [jasna/restorer/tvai_secondary_restorer.py](../../../jasna/restorer/tvai_secondary_restorer.py)
- [jasna/session_config.py](../../../jasna/session_config.py)
- [jasna/session_factory.py](../../../jasna/session_factory.py)
- [jasna/streaming_pipeline.py](../../../jasna/streaming_pipeline.py)
- [jasna/vram_offloader.py](../../../jasna/vram_offloader.py)
- [tests/test_dual_gop_encoder.py](../../../tests/test_dual_gop_encoder.py)
- [tests/test_frame_queue.py](../../../tests/test_frame_queue.py)
- [tests/test_main.py](../../../tests/test_main.py)
- [tests/test_main_validation.py](../../../tests/test_main_validation.py)
- [tests/test_owned_vram_reader.py](../../../tests/test_owned_vram_reader.py)
- [tests/test_pipeline_run.py](../../../tests/test_pipeline_run.py)
- [tests/test_pipeline_run_sync.py](../../../tests/test_pipeline_run_sync.py)
- [tests/test_pipeline_segments.py](../../../tests/test_pipeline_segments.py)
- [tests/test_pipeline_threads.py](../../../tests/test_pipeline_threads.py)
- [tests/test_progressbar.py](../../../tests/test_progressbar.py)
- [tests/test_session_config.py](../../../tests/test_session_config.py)
- [tests/test_session_factory.py](../../../tests/test_session_factory.py)
- [tests/test_shared_pipeline_cleanup.py](../../../tests/test_shared_pipeline_cleanup.py)
- [tests/test_streaming.py](../../../tests/test_streaming.py)
- [tests/test_tvai_secondary_restorer.py](../../../tests/test_tvai_secondary_restorer.py)
- [tests/test_video_session.py](../../../tests/test_video_session.py)
- [tests/test_vram_offloader.py](../../../tests/test_vram_offloader.py)
- [tests/test_windows_vram_reader_factory.py](../../../tests/test_windows_vram_reader_factory.py)
