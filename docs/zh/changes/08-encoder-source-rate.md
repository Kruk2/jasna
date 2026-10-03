# AMD 編碼契約與匹配源碼率 Peak VBR

[English (default)](../../en/changes/08-encoder-source-rate.md)

功能: `08-encoder-source-rate`. 源碼基準: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## 目的

維持 codec／profile／位深、緩衝幀／PTS／LUT 所有權及 AMF 選項範圍。提供 GUI 自動匹配源碼率 HEVC Peak VBR 與手動固定 QP／CQ 的選擇。

## 使用方式與預設行為

受支援的 Linux AMD HEVC GUI 路線中，自動模式在 full 與 Smart Render 均使用源碼率派生 vbr_peak；手動模式使用 cqp 與使用者 CQ。Full 可把 H.264 輸入轉成 HEVC；Smart Render 複製封包還要求源流相容。

## 直接前置

- [HIP 色彩轉換核心](04-hip-colour.md)
- [有界 Windows D3D11–HIP 常駐媒體](07-windows-resident-media.md)

依賴描述整合實作；PR 必須保持單功能範圍。未合併的前置不等於原生驗收。

## 安全與支援限制

碼率控制不保證檔案大小完全相同，修復內容和複製區間會影響大小。AV1 Main10／P010 有獨立 preanalysis／碼率規則；Linux HEVC 效能與 CPU 測試均不能認證未測 Windows 格式。

## 驗證與重現

```bash
python -m pytest -q tests/test_amd_support.py tests/test_hevc_smart_render_encoder.py tests/test_media_init.py tests/test_video_encoder_mux.py tests/test_video_encoder_unit.py
```

先前精確源碼整合集合在 Linux 隔離 CPU 環境通過 **3141 項測試、225 項跳過、178 項子測試**。此次文件更新不改執行行為。這是整合集合結果，不代表 26 個無前置獨立分支各自全部通過。

歷史 Linux 原生驗收涵蓋 13 個任務（9427.04 秒），在記錄的處理基準上完成嚴格解碼／時間軸／接縫檢查，不能自動認證後來新增的 Windows 變更。此次文件更新不執行真實影片／GPU 負載。Windows SDK／原生資產重建、全卡遙測、實機 A/B 為 **WAIVED_BY_USER_NOT_RUN**，不是 PASS；其他未測 Windows／NVIDIA／LTX 與付費模型 AMD 路線仍未認證。

## 實作與詳細記錄

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
