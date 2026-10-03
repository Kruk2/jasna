# 特定 Windows AMF 傳輸失敗後隔離

[English (default)](../../en/changes/24-native-context-quarantine.md)

功能: `24-native-context-quarantine`. 源碼基準: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## 目的

以型別化錯誤分類已觀察到的特定 Windows AMD AMF 硬體到主機傳輸失敗，保留根因、隔離目前佇列，讓待處理任務保持可恢復。

## 使用方式與預設行為

分類要求目前是 Windows AMD AMF 解碼路線，且 errno／原生傳輸證據匹配。隔離後需重新啟動新進程，不在可能已污染的原生 GPU 上下文中繼續。

## 直接前置

- [AMF 原生解碼與幀所有權](06-amf-native-decode.md)
- [共用流水線資源與失敗安全](12-pipeline-resource-safety.md)
- [原生視訊任務隔離與持久輸出](16-isolated-video-jobs.md)

依賴描述整合實作；PR 必須保持單功能範圍。未合併的前置不等於原生驗收。

## 安全與支援限制

不能隱藏不相關媒體／I/O 失敗，也不能把每個錯誤都分類成上下文故障。這是故障隔離與可診斷性，不代表已消除 TDR、驅動重置或非法記憶體存取。

## 驗證與重現

```bash
python -m pytest -q tests/test_windows_amf_context_quarantine.py
```

先前精確源碼整合集合在 Linux 隔離 CPU 環境通過 **3141 項測試、225 項跳過、178 項子測試**。此次文件更新不改執行行為。這是整合集合結果，不代表 26 個無前置獨立分支各自全部通過。

歷史 Linux 原生驗收涵蓋 13 個任務（9427.04 秒），在記錄的處理基準上完成嚴格解碼／時間軸／接縫檢查，不能自動認證後來新增的 Windows 變更。此次文件更新不執行真實影片／GPU 負載。Windows SDK／原生資產重建、全卡遙測、實機 A/B 為 **WAIVED_BY_USER_NOT_RUN**，不是 PASS；其他未測 Windows／NVIDIA／LTX 與付費模型 AMD 路線仍未認證。

## 實作與詳細記錄

- [jasna/gpu_context_errors.py](../../../jasna/gpu_context_errors.py)
- [jasna/gui/processor.py](../../../jasna/gui/processor.py)
- [jasna/media/video_decoder.py](../../../jasna/media/video_decoder.py)
- [tests/test_windows_amf_context_quarantine.py](../../../tests/test_windows_amf_context_quarantine.py)
