# Isolated native video jobs and durable outputs

English (default) | [中文](../../zh/changes/16-isolated-video-jobs.md)

Feature: `16-isolated-video-jobs`. Base source: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## Purpose

Run native video attempts in isolated processes, exchange bounded worker events, classify process exit/cancel correctly, and publish validated outputs atomically with durable fragment recovery.

## Usage and default behavior

Use the normal GUI queue. Session/worker refresh after a bounded range is internal: cumulative progress, speed, remaining-time estimates, and original failure context must survive refresh. Windows guarded attempts remain explicitly scoped.

## Direct prerequisites

- [Shared native job and diagnostic contracts](00-shared-native-job-contracts.md)
- [Pinned unified media runtime and installer](01-runtime-contract.md)
- [HIP color conversion kernels](04-hip-colour.md)
- [AMF native decoding and frame ownership](06-amf-native-decode.md)
- [Bounded Linux HEVC dual-GOP encoding](09-dual-gop.md)
- [Shared pipeline resource and failure safety](12-pipeline-resource-safety.md)
- [Smart Render seams and durable resume](13-smart-render-resume.md)
- [Adaptive automatic pre-scan and missed-range fixes](14-automatic-prescan.md)
- [Preserved input folders and output resume validation](15-preserved-folder-outputs.md)

Dependencies describe the combined implementation. PRs must remain feature-scoped; an unmerged prerequisite is not native acceptance.

## Safety and support limits

A child process producing a file is not sufficient for success. Check terminal events, exit status, media contracts, and final publication. Stop must not create work for later batch items; native crash isolation does not establish a driver root-cause fix.

## Validation and reproduction

```bash
python -m pytest -q tests/test_batch_resume_output.py tests/test_frozen_patch_entrypoints.py tests/test_gui_job_ordering.py tests/test_gui_preserve_input_structure.py tests/test_gui_processor_stop.py tests/test_gui_video_job_isolation.py tests/test_isolated_failure_diagnostics.py tests/test_main_entry.py tests/test_pre_scan_processor.py tests/test_video_job_process.py tests/test_windows_gpu_recovery.py tests/test_windows_guarded_attempt.py tests/test_windows_native_logs.py tests/test_windows_video_worker.py
```

The preceding exact-source integrated stack passed **3141 tests, with 225 skips and 178 passing subtests** on Linux in an isolated CPU environment. This documentation update does not change runtime behavior. These counts describe the combined stack, not 26 independently passing dependency-free branches.

Historical Linux native acceptance covered 13 jobs (9427.04 seconds) with strict decode/timeline/seam checks on its recorded processing baseline. It does not automatically certify the newer Windows changes. This documentation refresh runs no real-video/GPU workload. Windows SDK/native asset rebuilding, whole-card telemetry, and real-hardware A/B are **WAIVED_BY_USER_NOT_RUN**, not PASS. Other untested Windows/NVIDIA/LTX and paid-model AMD paths remain uncertified.

## Implementation and detailed records

- [docs/FINAL_OUTPUT_DURABILITY_CN.md](../../../docs/FINAL_OUTPUT_DURABILITY_CN.md)
- [docs/GUI_ISOLATED_FAILURE_DIAGNOSTICS_ACCEPTANCE_CN.md](../../../docs/GUI_ISOLATED_FAILURE_DIAGNOSTICS_ACCEPTANCE_CN.md)
- [docs/WINDOWS_GUARDED_PARENT_ACCEPTANCE_CN.md](../../../docs/WINDOWS_GUARDED_PARENT_ACCEPTANCE_CN.md)
- [jasna/__main__.py](../../../jasna/__main__.py)
- [jasna/gui/isolated_worker_streams.py](../../../jasna/gui/isolated_worker_streams.py)
- [jasna/gui/processor.py](../../../jasna/gui/processor.py)
- [jasna/gui/video_job_process.py](../../../jasna/gui/video_job_process.py)
- [jasna/gui/windows_guard_exit_result.py](../../../jasna/gui/windows_guard_exit_result.py)
- [jasna/gui/windows_guarded_attempt.py](../../../jasna/gui/windows_guarded_attempt.py)
- [jasna/gui/windows_video_worker.py](../../../jasna/gui/windows_video_worker.py)
- [tests/test_batch_resume_output.py](../../../tests/test_batch_resume_output.py)
- [tests/test_frozen_patch_entrypoints.py](../../../tests/test_frozen_patch_entrypoints.py)
- [tests/test_gui_job_ordering.py](../../../tests/test_gui_job_ordering.py)
- [tests/test_gui_preserve_input_structure.py](../../../tests/test_gui_preserve_input_structure.py)
- [tests/test_gui_processor_stop.py](../../../tests/test_gui_processor_stop.py)
- [tests/test_gui_video_job_isolation.py](../../../tests/test_gui_video_job_isolation.py)
- [tests/test_isolated_failure_diagnostics.py](../../../tests/test_isolated_failure_diagnostics.py)
- [tests/test_main_entry.py](../../../tests/test_main_entry.py)
- [tests/test_pre_scan_processor.py](../../../tests/test_pre_scan_processor.py)
- [tests/test_video_job_process.py](../../../tests/test_video_job_process.py)
- [tests/test_windows_gpu_recovery.py](../../../tests/test_windows_gpu_recovery.py)
- [tests/test_windows_guarded_attempt.py](../../../tests/test_windows_guarded_attempt.py)
- [tests/test_windows_native_logs.py](../../../tests/test_windows_native_logs.py)
- [tests/test_windows_video_worker.py](../../../tests/test_windows_video_worker.py)
