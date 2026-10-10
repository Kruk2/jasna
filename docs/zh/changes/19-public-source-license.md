# 公開源碼可選授權邊界

[English (default)](../../en/changes/19-public-source-license.md)

功能: `19-public-source-license`. 源碼基準: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## 目的

私有 jasna.protection 缺失時，免費模型流程仍可匯入。編譯／圖像入口使用同一邊界；私有套件存在時保留官方 store 行為。

## 使用方式與預設行為

缺件 shim 返回無授權、判定未授權，並明確拒絕啟用。已安裝私有套件內部缺失依賴是實際錯誤，不能隱藏。

編譯器的純路徑測試直接匯入既有共用引擎路徑工具，使 CPU 檢查不需要 TensorRT；測試斷言與產品實作不變。

## 直接前置

無。

依賴描述整合實作；PR 必須保持單功能範圍。未合併的前置不等於原生驗收。

## 安全與支援限制

不實作付費模組、不繞過啟用、不解密 supporter 權重，也不認證付費模型 AMD 相容性；那些功能仍需要官方可匯入私有元件。

## 驗證與重現

```bash
python -m pytest -q tests/test_engine_compiler.py tests/test_license_api.py
```

先前精確源碼整合集合在 Linux 隔離 CPU 環境通過 **3141 項測試、225 項跳過、178 項子測試**。此次文件更新不改執行行為。這是整合集合結果，不代表 26 個無前置獨立分支各自全部通過。

歷史 Linux 原生驗收涵蓋 13 個任務（9427.04 秒），在記錄的處理基準上完成嚴格解碼／時間軸／接縫檢查，不能自動認證後來新增的 Windows 變更。此次文件更新不執行真實影片／GPU 負載。Windows SDK／原生資產重建、全卡遙測、實機 A/B 為 **WAIVED_BY_USER_NOT_RUN**，不是 PASS；其他未測 Windows／NVIDIA／LTX 與付費模型 AMD 路線仍未認證。

## 實作與詳細記錄

- [docs/PUBLIC_SOURCE_LICENSE_BOUNDARY_CN.md](../../../docs/PUBLIC_SOURCE_LICENSE_BOUNDARY_CN.md)
- [jasna/engine_compiler.py](../../../jasna/engine_compiler.py)
- [jasna/image_restore.py](../../../jasna/image_restore.py)
- [jasna/license_api.py](../../../jasna/license_api.py)
- [tests/test_engine_compiler.py](../../../tests/test_engine_compiler.py)
- [tests/test_license_api.py](../../../tests/test_license_api.py)
