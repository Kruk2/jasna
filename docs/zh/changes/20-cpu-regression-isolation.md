# 純 CPU SD 1.5 回歸隔離

[English (default)](../../en/changes/20-cpu-regression-isolation.md)

功能: `20-cpu-regression-isolation`. 源碼基準: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## 目的

讓 SD 1.5 單元測試 fixture 明確宣告模擬硬體／匯入狀態，使 CPU 測試主機不依賴實體 GPU 廠商也可重現。

## 使用方式與預設行為

在隔離 CPU 測試環境執行 tests/test_sd15_inpaint_restorer.py。合成／模擬模組是測試 fixture，不是產品推論後端。

## 直接前置

- [GUI 控制、診斷與可靠進度](17-gui-settings-diagnostics.md)

依賴描述整合實作；PR 必須保持單功能範圍。未合併的前置不等於原生驗收。

## 安全與支援限制

Fixture 通過不等於安裝私有模型、啟用付費功能或證明 SD 1.5 AMD GPU 相容性。本功能只改測試。

## 驗證與重現

```bash
python -m pytest -q tests/test_sd15_inpaint_restorer.py
```

先前精確源碼整合集合在 Linux 隔離 CPU 環境通過 **3141 項測試、225 項跳過、178 項子測試**。此次文件更新不改執行行為。這是整合集合結果，不代表 26 個無前置獨立分支各自全部通過。

歷史 Linux 原生驗收涵蓋 13 個任務（9427.04 秒），在記錄的處理基準上完成嚴格解碼／時間軸／接縫檢查，不能自動認證後來新增的 Windows 變更。此次文件更新不執行真實影片／GPU 負載。Windows SDK／原生資產重建、全卡遙測、實機 A/B 為 **WAIVED_BY_USER_NOT_RUN**，不是 PASS；其他未測 Windows／NVIDIA／LTX 與付費模型 AMD 路線仍未認證。

## 實作與詳細記錄

- [tests/test_sd15_inpaint_restorer.py](../../../tests/test_sd15_inpaint_restorer.py)
