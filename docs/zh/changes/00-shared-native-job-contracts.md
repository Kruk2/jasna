# 共用原生任務與診斷契約

[English (default)](../../en/changes/00-shared-native-job-contracts.md)

功能: `00-shared-native-job-contracts`. 源碼基準: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## 目的

跨平台共用任務／設定資料、硬體策略、可恢復性記錄、持久執行日誌與原生診斷事件契約。顯卡廠商差異留在介接邊界。

## 使用方式與預設行為

GUI 處理批次預設仍為 B4，除非使用者在自訂參數明確指定 --batch-size 8。這個 GUI 標記會在建立編碼器選項之前移除。執行日誌排除機密欄位。

## 直接前置

- [Windows ROCm 匯入與顯卡廠商相容](03-windows-rocm-compat.md)
- [HIP 色彩轉換核心](04-hip-colour.md)

依賴描述整合實作；PR 必須保持單功能範圍。未合併的前置不等於原生驗收。

## 安全與支援限制

定義 Windows 全卡身分／日誌契約不代表所有介接器均已認證。本次交付豁免全卡遙測驗證，不會標記為 PASS。

## 驗證與重現

```bash
python -m pytest -q tests/test_gui_settings_persistence_paths.py tests/test_hardware_policy.py tests/test_native_worker.py tests/test_preset_migration.py tests/test_run_log.py tests/test_run_log_windows_adapter.py tests/test_system_stats.py tests/test_windows_global_vram.py tests/test_windows_native_logs.py
```

先前精確源碼整合集合在 Linux 隔離 CPU 環境通過 **3141 項測試、225 項跳過、178 項子測試**。此次文件更新不改執行行為。這是整合集合結果，不代表 26 個無前置獨立分支各自全部通過。

歷史 Linux 原生驗收涵蓋 13 個任務（9427.04 秒），在記錄的處理基準上完成嚴格解碼／時間軸／接縫檢查，不能自動認證後來新增的 Windows 變更。此次文件更新不執行真實影片／GPU 負載。Windows SDK／原生資產重建、全卡遙測、實機 A/B 為 **WAIVED_BY_USER_NOT_RUN**，不是 PASS；其他未測 Windows／NVIDIA／LTX 與付費模型 AMD 路線仍未認證。

## 實作與詳細記錄

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
