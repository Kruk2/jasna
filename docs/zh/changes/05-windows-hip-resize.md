# 可選 Windows HIP 縮放與正規化

[English (default)](../../en/changes/05-windows-hip-resize.md)

功能: `05-windows-hip-resize`. 源碼基準: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## 目的

實作預編譯 Windows AMD ResizeNormalizer 後端，保留共用前後處理介面、核心 ABI、串流選擇及 NVIDIA／Linux 預設。

## 使用方式與預設行為

JASNA_WINDOWS_HIP_RESIZE=1 會請求此後端。准入限定已記錄的 gfx1100 runtime／程式物件身分及 B=1..4、C=3 幾何範圍；既有契約文件列出精確版本／雜湊與步長限制。

## 直接前置

- [HIP 色彩轉換核心](04-hip-colour.md)

依賴描述整合實作；PR 必須保持單功能範圍。未合併的前置不等於原生驗收。

## 安全與支援限制

開關預設關閉。範圍外幾何沿用原 Torch 表達式；明確選用的 bundle 不匹配則報錯。歷史元件測量不等於新 SDK 或全部 Windows GUI 路線已認證。

## 驗證與重現

```bash
python -m pytest -q tests/test_windows_hip_resize_contract.py
```

先前精確源碼整合集合在 Linux 隔離 CPU 環境通過 **3141 項測試、225 項跳過、178 項子測試**。此次文件更新不改執行行為。這是整合集合結果，不代表 26 個無前置獨立分支各自全部通過。

歷史 Linux 原生驗收涵蓋 13 個任務（9427.04 秒），在記錄的處理基準上完成嚴格解碼／時間軸／接縫檢查，不能自動認證後來新增的 Windows 變更。此次文件更新不執行真實影片／GPU 負載。Windows SDK／原生資產重建、全卡遙測、實機 A/B 為 **WAIVED_BY_USER_NOT_RUN**，不是 PASS；其他未測 Windows／NVIDIA／LTX 與付費模型 AMD 路線仍未認證。

## 實作與詳細記錄

- [docs/WINDOWS_HIP_RESIZE_ACCEPTANCE_CN.md](../../../docs/WINDOWS_HIP_RESIZE_ACCEPTANCE_CN.md)
- [jasna/media/hip_resize_normalize.gfx1100.windows.json](../../../jasna/media/hip_resize_normalize.gfx1100.windows.json)
- [jasna/media/resize_normalize.gfx1100.windows.co](../../../jasna/media/resize_normalize.gfx1100.windows.co)
- [jasna/media/resize_normalize.py](../../../jasna/media/resize_normalize.py)
- [jasna/media/windows_hip_resize_contract.py](../../../jasna/media/windows_hip_resize_contract.py)
- [scripts/build_windows_hip_resize.py](../../../scripts/build_windows_hip_resize.py)
- [tests/test_windows_hip_resize_contract.py](../../../tests/test_windows_hip_resize_contract.py)
