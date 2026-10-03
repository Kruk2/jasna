# GUI 控制、診斷與可靠進度

[English (default)](../../en/changes/17-gui-settings-diagnostics.md)

功能: `17-gui-settings-diagnostics`. 源碼基準: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## 目的

呈現共用掃描／路由／碼率／batch 設定、保留各廠商有效預設，加入持久診斷日誌、佇列終態處理、關閉／停止行為與 HiDPI 修正。

## 使用方式與預設行為

自動源碼率與手動 CQ 控制依所選路線能力顯示。設定序列化不依賴語言；警告及 worker 生命週期事件不應重置有效的累計速度／ETA 顯示。

## 直接前置

- [共用原生任務與診斷契約](00-shared-native-job-contracts.md)
- [原生視訊任務隔離與持久輸出](16-isolated-video-jobs.md)
- [公開源碼可選授權邊界](19-public-source-license.md)

依賴描述整合實作；PR 必須保持單功能範圍。未合併的前置不等於原生驗收。

## 安全與支援限制

廠商專用控制不代表 AMD／NVIDIA 使用相同原生實作。保留上游 LTX UI 整合，但這些 GUI 變更不認證 LTX／付費模型 AMD 原生相容性。

## 驗證與重現

```bash
python -m pytest -q tests/test_gui_about_dialog.py tests/test_gui_close_shutdown.py tests/test_gui_components.py tests/test_gui_file_actions.py tests/test_gui_hidpi_scaling.py tests/test_gui_icons.py tests/test_gui_ltx_models.py tests/test_gui_official_sellers.py tests/test_gui_queue_layout.py tests/test_gui_segments.py tests/test_gui_settings_sections.py tests/test_gui_staged_fixes.py tests/test_gui_tvai_validation.py tests/test_gui_video_job_isolation.py tests/test_gui_video_player.py tests/test_gui_windows_hip_warmup.py tests/test_gui_wizard_gpu.py tests/test_hardware_policy.py tests/test_post_export_action.py tests/test_raw_player.py tests/test_restoration_preview.py tests/test_run_log.py tests/test_segment_editor.py
```

先前精確源碼整合集合在 Linux 隔離 CPU 環境通過 **3141 項測試、225 項跳過、178 項子測試**。此次文件更新不改執行行為。這是整合集合結果，不代表 26 個無前置獨立分支各自全部通過。

歷史 Linux 原生驗收涵蓋 13 個任務（9427.04 秒），在記錄的處理基準上完成嚴格解碼／時間軸／接縫檢查，不能自動認證後來新增的 Windows 變更。此次文件更新不執行真實影片／GPU 負載。Windows SDK／原生資產重建、全卡遙測、實機 A/B 為 **WAIVED_BY_USER_NOT_RUN**，不是 PASS；其他未測 Windows／NVIDIA／LTX 與付費模型 AMD 路線仍未認證。

## 實作與詳細記錄

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
