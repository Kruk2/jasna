# 經驗證 BasicVSR++ MIGraphX B1 修復

[English (default)](../../en/changes/11-basicvsrpp-migraphx.md)

功能: `11-basicvsrpp-migraphx`. 源碼基準: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## 目的

加入有界 B1 修復實作、產物建置／探測，以及模型／源碼／runtime／擴充庫身分嚴格驗證。區分選定的 Torch-MIGraphX 二進位與已載入的其他擴充庫。

## 使用方式與預設行為

JASNA_BASICVSRPP_MIGRAPHX_B1 請求此路線，JASNA_BASICVSRPP_MIGRAPHX_B1_DIR 指定產物目錄。附帶建置／探測工具必須使用匹配的模型與 runtime 身分。

## 直接前置

- [Windows ROCm 匯入與顯卡廠商相容](03-windows-rocm-compat.md)
- [經驗證 RF-DETR MIGraphX 選擇](10-rfdetr-migraphx.md)

依賴描述整合實作；PR 必須保持單功能範圍。未合併的前置不等於原生驗收。

## 安全與支援限制

原生事件／非法記憶體存取失敗必須保留根因並傳遞。不能靜默切到 CPU、跳過修復，或僅憑檔名相同就重用產物。

## 驗證與重現

```bash
python -m pytest -q tests/test_basicvsrpp_migraphx_b1_product.py tests/test_basicvsrpp_mosaic_restorer.py tests/test_build_basicvsrpp_migraphx_b1.py
```

先前精確源碼整合集合在 Linux 隔離 CPU 環境通過 **3141 項測試、225 項跳過、178 項子測試**。此次文件更新不改執行行為。這是整合集合結果，不代表 26 個無前置獨立分支各自全部通過。

歷史 Linux 原生驗收涵蓋 13 個任務（9427.04 秒），在記錄的處理基準上完成嚴格解碼／時間軸／接縫檢查，不能自動認證後來新增的 Windows 變更。此次文件更新不執行真實影片／GPU 負載。Windows SDK／原生資產重建、全卡遙測、實機 A/B 為 **WAIVED_BY_USER_NOT_RUN**，不是 PASS；其他未測 Windows／NVIDIA／LTX 與付費模型 AMD 路線仍未認證。

## 實作與詳細記錄

- [docs/LINUX_AMD_HIP_MIGRAPHX_OPTIMIZATION_CN.md](../../../docs/LINUX_AMD_HIP_MIGRAPHX_OPTIMIZATION_CN.md)
- [jasna/restorer/basicvsrpp_migraphx_b1.py](../../../jasna/restorer/basicvsrpp_migraphx_b1.py)
- [jasna/restorer/basicvsrpp_mosaic_restorer.py](../../../jasna/restorer/basicvsrpp_mosaic_restorer.py)
- [scripts/build_basicvsrpp_migraphx_b1.py](../../../scripts/build_basicvsrpp_migraphx_b1.py)
- [scripts/probe_basicvsrpp_migraphx_b1.py](../../../scripts/probe_basicvsrpp_migraphx_b1.py)
- [tests/test_basicvsrpp_migraphx_b1_product.py](../../../tests/test_basicvsrpp_migraphx_b1_product.py)
- [tests/test_basicvsrpp_mosaic_restorer.py](../../../tests/test_basicvsrpp_mosaic_restorer.py)
- [tests/test_build_basicvsrpp_migraphx_b1.py](../../../tests/test_build_basicvsrpp_migraphx_b1.py)
