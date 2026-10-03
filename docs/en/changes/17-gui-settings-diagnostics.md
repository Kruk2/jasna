# GUI controls, diagnostics, and reliable progress

English (default) | [中文](../../zh/changes/17-gui-settings-diagnostics.md)

Feature: `17-gui-settings-diagnostics`. Base source: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## Purpose

Expose shared scan/routing/rate-control/batch settings, preserve vendor-effective defaults, and add durable diagnostic logs, queue terminal-state handling, close/stop behavior, and HiDPI corrections.

## Usage and default behavior

Automatic source-rate versus manual CQ controls follow the selected route's capability. Settings serialize independently of locale; warnings and worker lifecycle events must not reset useful cumulative speed/ETA displays.

## Direct prerequisites

- [Shared native job and diagnostic contracts](00-shared-native-job-contracts.md)
- [Isolated native video jobs and durable outputs](16-isolated-video-jobs.md)
- [Public-source optional license boundary](19-public-source-license.md)

Dependencies describe the combined implementation. PRs must remain feature-scoped; an unmerged prerequisite is not native acceptance.

## Safety and support limits

Vendor-specific controls do not imply the same native implementation on AMD and NVIDIA. Existing upstream LTX UI integration is retained, but LTX/paid-model AMD native compatibility is not certified by these GUI changes.

## Validation and reproduction

```bash
python -m pytest -q tests/test_gui_about_dialog.py tests/test_gui_close_shutdown.py tests/test_gui_components.py tests/test_gui_file_actions.py tests/test_gui_hidpi_scaling.py tests/test_gui_icons.py tests/test_gui_ltx_models.py tests/test_gui_official_sellers.py tests/test_gui_queue_layout.py tests/test_gui_segments.py tests/test_gui_settings_sections.py tests/test_gui_staged_fixes.py tests/test_gui_tvai_validation.py tests/test_gui_video_job_isolation.py tests/test_gui_video_player.py tests/test_gui_windows_hip_warmup.py tests/test_gui_wizard_gpu.py tests/test_hardware_policy.py tests/test_post_export_action.py tests/test_raw_player.py tests/test_restoration_preview.py tests/test_run_log.py tests/test_segment_editor.py
```

The preceding exact-source integrated stack passed **3141 tests, with 225 skips and 178 passing subtests** on Linux in an isolated CPU environment. This documentation update does not change runtime behavior. These counts describe the combined stack, not 26 independently passing dependency-free branches.

Historical Linux native acceptance covered 13 jobs (9427.04 seconds) with strict decode/timeline/seam checks on its recorded processing baseline. It does not automatically certify the newer Windows changes. This documentation refresh runs no real-video/GPU workload. Windows SDK/native asset rebuilding, whole-card telemetry, and real-hardware A/B are **WAIVED_BY_USER_NOT_RUN**, not PASS. Other untested Windows/NVIDIA/LTX and paid-model AMD paths remain uncertified.

## Implementation and detailed records

- [docs/WINDOWS_GUI_PROGRESS_CALLBACK_ACCEPTANCE_CN.md](../../../docs/WINDOWS_GUI_PROGRESS_CALLBACK_ACCEPTANCE_CN.md)
- [docs/WINDOWS_GUI_TERMINAL_STATUS_ACCEPTANCE_CN.md](../../../docs/WINDOWS_GUI_TERMINAL_STATUS_ACCEPTANCE_CN.md)
- [docs/en/gui.md](../../../docs/en/gui.md)
- [docs/ja/gui.md](../../../docs/ja/gui.md)
- [docs/zh/gui.md](../../../docs/zh/gui.md)
- [jasna/gui/app.py](../../../jasna/gui/app.py)
- [jasna/gui/components.py](../../../jasna/gui/components.py)
- [jasna/gui/job_list_item.py](../../../jasna/gui/job_list_item.py)
- [jasna/gui/locales/en.py](../../../jasna/gui/locales/en.py)
- [jasna/gui/locales/ja.py](../../../jasna/gui/locales/ja.py)
- [jasna/gui/locales/ko.py](../../../jasna/gui/locales/ko.py)
- [jasna/gui/locales/th.py](../../../jasna/gui/locales/th.py)
- [jasna/gui/locales/zh.py](../../../jasna/gui/locales/zh.py)
- [jasna/gui/ltx_models.py](../../../jasna/gui/ltx_models.py)
- [jasna/gui/mask_feedback.py](../../../jasna/gui/mask_feedback.py)
- [jasna/gui/scaling.py](../../../jasna/gui/scaling.py)
- [jasna/gui/settings_panel.py](../../../jasna/gui/settings_panel.py)
- [jasna/gui/settings_sections/advanced.py](../../../jasna/gui/settings_sections/advanced.py)
- [jasna/gui/settings_sections/basic.py](../../../jasna/gui/settings_sections/basic.py)
- [jasna/gui/settings_sections/encoding.py](../../../jasna/gui/settings_sections/encoding.py)
- [jasna/gui/validation.py](../../../jasna/gui/validation.py)
- [jasna/gui/video_player.py](../../../jasna/gui/video_player.py)
- [tests/test_gui_about_dialog.py](../../../tests/test_gui_about_dialog.py)
- [tests/test_gui_close_shutdown.py](../../../tests/test_gui_close_shutdown.py)
- [tests/test_gui_components.py](../../../tests/test_gui_components.py)
- [tests/test_gui_file_actions.py](../../../tests/test_gui_file_actions.py)
- [tests/test_gui_hidpi_scaling.py](../../../tests/test_gui_hidpi_scaling.py)
- [tests/test_gui_icons.py](../../../tests/test_gui_icons.py)
- [tests/test_gui_ltx_models.py](../../../tests/test_gui_ltx_models.py)
- [tests/test_gui_official_sellers.py](../../../tests/test_gui_official_sellers.py)
- [tests/test_gui_queue_layout.py](../../../tests/test_gui_queue_layout.py)
- [tests/test_gui_segments.py](../../../tests/test_gui_segments.py)
- [tests/test_gui_settings_sections.py](../../../tests/test_gui_settings_sections.py)
- [tests/test_gui_staged_fixes.py](../../../tests/test_gui_staged_fixes.py)
- [tests/test_gui_tvai_validation.py](../../../tests/test_gui_tvai_validation.py)
- [tests/test_gui_video_job_isolation.py](../../../tests/test_gui_video_job_isolation.py)
- [tests/test_gui_video_player.py](../../../tests/test_gui_video_player.py)
- [tests/test_gui_windows_hip_warmup.py](../../../tests/test_gui_windows_hip_warmup.py)
- [tests/test_gui_wizard_gpu.py](../../../tests/test_gui_wizard_gpu.py)
- [tests/test_hardware_policy.py](../../../tests/test_hardware_policy.py)
- [tests/test_post_export_action.py](../../../tests/test_post_export_action.py)
- [tests/test_raw_player.py](../../../tests/test_raw_player.py)
- [tests/test_restoration_preview.py](../../../tests/test_restoration_preview.py)
- [tests/test_run_log.py](../../../tests/test_run_log.py)
- [tests/test_segment_editor.py](../../../tests/test_segment_editor.py)
