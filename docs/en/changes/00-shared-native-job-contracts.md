# Shared native job and diagnostic contracts

English (default) | [中文](../../zh/changes/00-shared-native-job-contracts.md)

Feature: `00-shared-native-job-contracts`. Base source: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## Purpose

Share job/settings data, hardware policy, recoverability records, durable run logging, and native diagnostic event contracts across platforms. Keep vendor differences at adapter boundaries.

## Usage and default behavior

GUI processing batch remains B4 unless the user explicitly requests --batch-size 8 in custom parameters. That GUI-only flag is removed before encoder options are constructed. Run logs exclude secret fields.

## Direct prerequisites

- [Windows ROCm import and vendor compatibility](03-windows-rocm-compat.md)
- [HIP color conversion kernels](04-hip-colour.md)

Dependencies describe the combined implementation. PRs must remain feature-scoped; an unmerged prerequisite is not native acceptance.

## Safety and support limits

Declaring the Windows whole-card identity/logging contract does not certify every adapter. Whole-card telemetry validation is waived for this delivery, not reported as PASS.

## Validation and reproduction

```bash
python -m pytest -q tests/test_gui_settings_persistence_paths.py tests/test_hardware_policy.py tests/test_native_worker.py tests/test_preset_migration.py tests/test_run_log.py tests/test_run_log_windows_adapter.py tests/test_system_stats.py tests/test_windows_global_vram.py tests/test_windows_native_logs.py
```

The preceding exact-source integrated stack passed **3141 tests, with 225 skips and 178 passing subtests** on Linux in an isolated CPU environment. This documentation update does not change runtime behavior. These counts describe the combined stack, not 26 independently passing dependency-free branches.

Historical Linux native acceptance covered 13 jobs (9427.04 seconds) with strict decode/timeline/seam checks on its recorded processing baseline. It does not automatically certify the newer Windows changes. This documentation refresh runs no real-video/GPU workload. Windows SDK/native asset rebuilding, whole-card telemetry, and real-hardware A/B are **WAIVED_BY_USER_NOT_RUN**, not PASS. Other untested Windows/NVIDIA/LTX and paid-model AMD paths remain uncertified.

## Implementation and detailed records

- [docs/CRASH_RESILIENT_RUN_LOG_CN.md](../../../docs/CRASH_RESILIENT_RUN_LOG_CN.md)
- [docs/GUI_BATCH_SIZE_CN.md](../../../docs/GUI_BATCH_SIZE_CN.md)
- [docs/WINDOWS_GLOBAL_VRAM_IDENTITY_ACCEPTANCE_CN.md](../../../docs/WINDOWS_GLOBAL_VRAM_IDENTITY_ACCEPTANCE_CN.md)
- [docs/WINDOWS_NATIVE_FFMPEG_LOGS_ACCEPTANCE_CN.md](../../../docs/WINDOWS_NATIVE_FFMPEG_LOGS_ACCEPTANCE_CN.md)
- [jasna/gui/gpu_recovery.py](../../../jasna/gui/gpu_recovery.py)
- [jasna/gui/hardware_policy.py](../../../jasna/gui/hardware_policy.py)
- [jasna/gui/models.py](../../../jasna/gui/models.py)
- [jasna/gui/run_log.py](../../../jasna/gui/run_log.py)
- [jasna/gui/system_stats.py](../../../jasna/gui/system_stats.py)
- [jasna/native_worker.py](../../../jasna/native_worker.py)
- [jasna/windows_global_vram.py](../../../jasna/windows_global_vram.py)
- [jasna/windows_native_logs.py](../../../jasna/windows_native_logs.py)
- [tests/test_gui_settings_persistence_paths.py](../../../tests/test_gui_settings_persistence_paths.py)
- [tests/test_hardware_policy.py](../../../tests/test_hardware_policy.py)
- [tests/test_native_worker.py](../../../tests/test_native_worker.py)
- [tests/test_preset_migration.py](../../../tests/test_preset_migration.py)
- [tests/test_run_log.py](../../../tests/test_run_log.py)
- [tests/test_run_log_windows_adapter.py](../../../tests/test_run_log_windows_adapter.py)
- [tests/test_system_stats.py](../../../tests/test_system_stats.py)
- [tests/test_windows_global_vram.py](../../../tests/test_windows_global_vram.py)
- [tests/test_windows_native_logs.py](../../../tests/test_windows_native_logs.py)
