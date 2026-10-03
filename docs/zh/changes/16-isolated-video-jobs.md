# 原生視訊任務隔離與持久輸出

[English (default)](../../en/changes/16-isolated-video-jobs.md)

功能: `16-isolated-video-jobs`. 源碼基準: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## 目的

在隔離進程執行原生視訊任務、交換有界 worker 事件、正確分類進程退出／取消，並以持久分段恢復原子發布經驗證輸出。

## 使用方式與預設行為

使用正常 GUI 佇列。有界區間後的 session／worker 重建屬內部動作；累計進度、速度、剩餘時間估計及原始失敗上下文需延續。Windows 受保護 attempt 維持明確限定範圍。

## 直接前置

- [共用原生任務與診斷契約](00-shared-native-job-contracts.md)
- [固定版本統一媒體 runtime 與安裝器](01-runtime-contract.md)
- [HIP 色彩轉換核心](04-hip-colour.md)
- [AMF 原生解碼與幀所有權](06-amf-native-decode.md)
- [有界 Linux HEVC 雙 GOP 編碼](09-dual-gop.md)
- [共用流水線資源與失敗安全](12-pipeline-resource-safety.md)
- [Smart Render 接縫與持久續跑](13-smart-render-resume.md)
- [自適應自動預掃描與漏區間修復](14-automatic-prescan.md)
- [保持輸入資料夾與輸出續跑驗證](15-preserved-folder-outputs.md)

依賴描述整合實作；PR 必須保持單功能範圍。未合併的前置不等於原生驗收。

## 安全與支援限制

子進程產生檔案不足以判定成功。需檢查終止事件、退出狀態、媒體契約及最終發布。停止後不能為後續批次項目建立工作；原生崩潰隔離不等於修復驅動根因。

## 驗證與重現

```bash
python -m pytest -q tests/test_batch_resume_output.py tests/test_frozen_patch_entrypoints.py tests/test_gui_job_ordering.py tests/test_gui_preserve_input_structure.py tests/test_gui_processor_stop.py tests/test_gui_video_job_isolation.py tests/test_isolated_failure_diagnostics.py tests/test_main_entry.py tests/test_pre_scan_processor.py tests/test_video_job_process.py tests/test_windows_gpu_recovery.py tests/test_windows_guarded_attempt.py tests/test_windows_native_logs.py tests/test_windows_video_worker.py
```

先前精確源碼整合集合在 Linux 隔離 CPU 環境通過 **3141 項測試、225 項跳過、178 項子測試**。此次文件更新不改執行行為。這是整合集合結果，不代表 26 個無前置獨立分支各自全部通過。

歷史 Linux 原生驗收涵蓋 13 個任務（9427.04 秒），在記錄的處理基準上完成嚴格解碼／時間軸／接縫檢查，不能自動認證後來新增的 Windows 變更。此次文件更新不執行真實影片／GPU 負載。Windows SDK／原生資產重建、全卡遙測、實機 A/B 為 **WAIVED_BY_USER_NOT_RUN**，不是 PASS；其他未測 Windows／NVIDIA／LTX 與付費模型 AMD 路線仍未認證。

## 實作與詳細記錄

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
