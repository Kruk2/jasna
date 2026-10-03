# 排列感知 RF-DETR 精度診斷

[English (default)](../../en/changes/25-rfdetr-precision-probe.md)

功能: `25-rfdetr-precision-probe`. 源碼基準: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## 目的

使用 Hungarian 提案匹配比較合成 Windows B1／B4 FP32／FP16 RF-DETR 輸出，避免將提案重排誤判為精度錯誤，並記錄身分及證據限制。

## 使用方式與預設行為

執行 scripts/probe_windows_rfdetr_precision.py --weights <weights> --output <report>。可選 --batches 接受 1、4。CPU 執行是宣告的數值參考，不是 GPU 推論失敗後的靜默回退。

## 直接前置

- [身分限定 Windows AMD Math SDPA](23-windows-sdpa-compat.md)

依賴描述整合實作；PR 必須保持單功能範圍。未合併的前置不等於原生驗收。

## 安全與支援限制

合成比較不認證真實檢測／修復畫質。未另行實機測量時，畫質為 NOT_CERTIFIED，吞吐量／效能為 NOT_RUN。

## 驗證與重現

```bash
python -m pytest -q tests/test_rfdetr_precision_matching.py
```

先前精確源碼整合集合在 Linux 隔離 CPU 環境通過 **3141 項測試、225 項跳過、178 項子測試**。此次文件更新不改執行行為。這是整合集合結果，不代表 26 個無前置獨立分支各自全部通過。

歷史 Linux 原生驗收涵蓋 13 個任務（9427.04 秒），在記錄的處理基準上完成嚴格解碼／時間軸／接縫檢查，不能自動認證後來新增的 Windows 變更。此次文件更新不執行真實影片／GPU 負載。Windows SDK／原生資產重建、全卡遙測、實機 A/B 為 **WAIVED_BY_USER_NOT_RUN**，不是 PASS；其他未測 Windows／NVIDIA／LTX 與付費模型 AMD 路線仍未認證。

## 實作與詳細記錄

- [scripts/probe_windows_rfdetr_precision.py](../../../scripts/probe_windows_rfdetr_precision.py)
- [tests/test_rfdetr_precision_matching.py](../../../tests/test_rfdetr_precision_matching.py)
