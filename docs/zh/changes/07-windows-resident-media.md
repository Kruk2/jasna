# 有界 Windows D3D11–HIP 常駐媒體

[English (default)](../../en/changes/07-windows-resident-media.md)

功能: `07-windows-resident-media`. 源碼基準: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## 目的

加入明確使用 D3D11–HIP 常駐解編碼 surface 的 Python 協調器、Cython 包裝、原生實作、建置器及產品探測，維持有界所有權／fence 契約。

## 使用方式與預設行為

JASNA_WINDOWS_D3D11_HIP_RESIDENT=1 請求此元件，預設關閉。既有准入涵蓋 1920×1080 與 3840×2160。編碼輸出池包含四個 surface，必須與 AMF 設定匹配。

## 直接前置

- [固定版本統一媒體 runtime 與安裝器](01-runtime-contract.md)
- [HIP 色彩轉換核心](04-hip-colour.md)

依賴描述整合實作；PR 必須保持單功能範圍。未合併的前置不等於原生驗收。

## 安全與支援限制

即使單 reader 探測成功，8192×4096 仍因雙 reader 產品記憶體保護而拒絕。這不是 Windows 8K 預設優化；凍結打包／分發及更廣硬體驗收仍未認證。

## 驗證與重現

```bash
python -m pytest -q tests/test_windows_d3d11_hip_resident.py
```

先前精確源碼整合集合在 Linux 隔離 CPU 環境通過 **3141 項測試、225 項跳過、178 項子測試**。此次文件更新不改執行行為。這是整合集合結果，不代表 26 個無前置獨立分支各自全部通過。

歷史 Linux 原生驗收涵蓋 13 個任務（9427.04 秒），在記錄的處理基準上完成嚴格解碼／時間軸／接縫檢查，不能自動認證後來新增的 Windows 變更。此次文件更新不執行真實影片／GPU 負載。Windows SDK／原生資產重建、全卡遙測、實機 A/B 為 **WAIVED_BY_USER_NOT_RUN**，不是 PASS；其他未測 Windows／NVIDIA／LTX 與付費模型 AMD 路線仍未認證。

## 實作與詳細記錄

- [docs/WINDOWS_RESIDENT_MEDIA_CN.md](../../../docs/WINDOWS_RESIDENT_MEDIA_CN.md)
- [jasna/media/windows_d3d11_hip_resident.py](../../../jasna/media/windows_d3d11_hip_resident.py)
- [scripts/amf_d3d11_hip_resident.pyx](../../../scripts/amf_d3d11_hip_resident.pyx)
- [scripts/build_amf_d3d11_hip_resident.py](../../../scripts/build_amf_d3d11_hip_resident.py)
- [scripts/native/amf_d3d11_hip_resident_native.hpp](../../../scripts/native/amf_d3d11_hip_resident_native.hpp)
- [scripts/probe_windows_d3d11_hip_resident_product.py](../../../scripts/probe_windows_d3d11_hip_resident_product.py)
- [tests/test_windows_d3d11_hip_resident.py](../../../tests/test_windows_d3d11_hip_resident.py)
