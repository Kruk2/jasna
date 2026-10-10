# 固定版本統一媒體 runtime 與安裝器

[English (default)](../../en/changes/01-runtime-contract.md)

功能: `01-runtime-contract`. 源碼基準: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## 目的

將 PyAV 與其 FFmpeg 動態函式庫視為同一固定 ABI 單元。加入 runtime／源碼／檔案身分驗證、原子安裝及子進程啟動器，不另建一套處理流水線。

## 使用方式與預設行為

Linux 執行 scripts/run_jasna_unified.sh --preflight-only；Windows 執行 scripts/run_jasna_unified_windows.ps1 -PreflightOnly。JASNA_UNIFIED_RUNTIME_ROOT 指定已安裝 runtime；安裝選項見 scripts/install_unified_runtime.py --help。

## 直接前置

無。

依賴描述整合實作；PR 必須保持單功能範圍。未合併的前置不等於原生驗收。

## 安全與支援限制

只修改被啟動的子進程環境。雜湊／ABI 失敗會在媒體匯入前終止，不會靜默使用系統 PyAV／FFmpeg 或 CPU 回退。Runtime 資產需另行建置／安裝，本 PR 不附帶整套二進位 runtime。

## 驗證與重現

```bash
python -m pytest -q tests/test_runtime_contract.py tests/test_unified_runtime_installer.py
```

先前精確源碼整合集合在 Linux 隔離 CPU 環境通過 **3141 項測試、225 項跳過、178 項子測試**。此次文件更新不改執行行為。這是整合集合結果，不代表 26 個無前置獨立分支各自全部通過。

歷史 Linux 原生驗收涵蓋 13 個任務（9427.04 秒），在記錄的處理基準上完成嚴格解碼／時間軸／接縫檢查，不能自動認證後來新增的 Windows 變更。此次文件更新不執行真實影片／GPU 負載。Windows SDK／原生資產重建、全卡遙測、實機 A/B 為 **WAIVED_BY_USER_NOT_RUN**，不是 PASS；其他未測 Windows／NVIDIA／LTX 與付費模型 AMD 路線仍未認證。

## 實作與詳細記錄

- [docs/UNIFIED_RUNTIME_CN.md](../../../docs/UNIFIED_RUNTIME_CN.md)
- [docs/UNIFIED_RUNTIME_INSTALL_CN.md](../../../docs/UNIFIED_RUNTIME_INSTALL_CN.md)
- [jasna/runtime_contract.py](../../../jasna/runtime_contract.py)
- [pyproject.toml](../../../pyproject.toml)
- [scripts/install_unified_runtime.py](../../../scripts/install_unified_runtime.py)
- [scripts/run_jasna_unified.py](../../../scripts/run_jasna_unified.py)
- [scripts/run_jasna_unified.sh](../../../scripts/run_jasna_unified.sh)
- [scripts/run_jasna_unified_windows.ps1](../../../scripts/run_jasna_unified_windows.ps1)
- [tests/test_runtime_contract.py](../../../tests/test_runtime_contract.py)
- [tests/test_unified_runtime_installer.py](../../../tests/test_unified_runtime_installer.py)
