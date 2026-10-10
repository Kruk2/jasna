# 共用流水線資源與失敗安全

[English (default)](../../en/changes/12-pipeline-resource-safety.md)

功能: `12-pipeline-resource-safety`. 源碼基準: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## 目的

共用修復 session 建立與解碼／檢測、主修復、二次修復、混合／編碼流水線。加入容量比例 GPU／主機預算、有界佇列所有權、原生卡住檢測與首錯傳遞。

## 使用方式與預設行為

記憶體回收使用容量比例水位和壓力／去抖策略，不是所有顯卡共用剩餘 4 GiB 觸發。Worker 失敗／取消時釋放阻塞生產者，僅取消路線清空佇列；健康佇列不受影響。

## 直接前置

- [共用原生任務與診斷契約](00-shared-native-job-contracts.md)
- [AMF 原生解碼與幀所有權](06-amf-native-decode.md)
- [有界 Windows D3D11–HIP 常駐媒體](07-windows-resident-media.md)
- [AMD 編碼契約與匹配源碼率 Peak VBR](08-encoder-source-rate.md)
- [有界 Linux HEVC 雙 GOP 編碼](09-dual-gop.md)
- [經驗證 BasicVSR++ MIGraphX B1 修復](11-basicvsrpp-migraphx.md)
- [Smart Render 接縫與持久續跑](13-smart-render-resume.md)
- [公開源碼可選授權邊界](19-public-source-license.md)

依賴描述整合實作；PR 必須保持單功能範圍。未合併的前置不等於原生驗收。

## 安全與支援限制

清理過程要保留第一個真實例外。使用者停止不應被捏造成 worker 失敗。資源上限是安全控制，不代表每張顯卡或格式均已效能認證。

## 驗證與重現

```bash
python -m pytest -q tests/test_dual_gop_encoder.py tests/test_frame_queue.py tests/test_main.py tests/test_main_validation.py tests/test_owned_vram_reader.py tests/test_pipeline_run.py tests/test_pipeline_run_sync.py tests/test_pipeline_segments.py tests/test_pipeline_threads.py tests/test_progressbar.py tests/test_session_config.py tests/test_session_factory.py tests/test_shared_pipeline_cleanup.py tests/test_streaming.py tests/test_tvai_secondary_restorer.py tests/test_video_session.py tests/test_vram_offloader.py tests/test_windows_vram_reader_factory.py
```

先前精確源碼整合集合在 Linux 隔離 CPU 環境通過 **3141 項測試、225 項跳過、178 項子測試**。此次文件更新不改執行行為。這是整合集合結果，不代表 26 個無前置獨立分支各自全部通過。

歷史 Linux 原生驗收涵蓋 13 個任務（9427.04 秒），在記錄的處理基準上完成嚴格解碼／時間軸／接縫檢查，不能自動認證後來新增的 Windows 變更。此次文件更新不執行真實影片／GPU 負載。Windows SDK／原生資產重建、全卡遙測、實機 A/B 為 **WAIVED_BY_USER_NOT_RUN**，不是 PASS；其他未測 Windows／NVIDIA／LTX 與付費模型 AMD 路線仍未認證。

## 實作與詳細記錄

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
