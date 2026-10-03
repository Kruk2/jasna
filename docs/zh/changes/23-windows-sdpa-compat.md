# 身分限定 Windows AMD Math SDPA

[English (default)](../../en/changes/23-windows-sdpa-compat.md)

功能: `23-windows-sdpa-compat`. 源碼基準: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## 目的

在建立 RF-DETR 前套用既有 Windows gfx1100 Math SDPA 策略，驗證 Torch 2.12.0+rocm10.0.0、HIP 7.15.26333、架構、實際 runtime API 與 DLL SHA-256，而非只相信套件標籤。

## 使用方式與預設行為

JASNA_WINDOWS_AMD_SDPA_POLICY 接受 auto、math、default。Auto 只修改精確驗證組合；math 拒絕未驗證組合；default 保留既有旗標。FP16 選擇獨立，Math SDPA 仍在 GPU 執行。

## 直接前置

- [Windows ROCm 匯入與顯卡廠商相容](03-windows-rocm-compat.md)
- [經驗證 RF-DETR MIGraphX 選擇](10-rfdetr-migraphx.md)

依賴描述整合實作；PR 必須保持單功能範圍。未合併的前置不等於原生驗收。

## 安全與支援限制

Linux、CPU、NVIDIA 不會被探測或修改。進程預設不覆蓋明確 sdpa_kernel 上下文；不是 LTX attention 認證，也不是通用 Windows ROCm 解法。

## 驗證與重現

```bash
python -m pytest -q tests/test_windows_sdpa_policy.py
```

先前精確源碼整合集合在 Linux 隔離 CPU 環境通過 **3141 項測試、225 項跳過、178 項子測試**。此次文件更新不改執行行為。這是整合集合結果，不代表 26 個無前置獨立分支各自全部通過。

歷史 Linux 原生驗收涵蓋 13 個任務（9427.04 秒），在記錄的處理基準上完成嚴格解碼／時間軸／接縫檢查，不能自動認證後來新增的 Windows 變更。此次文件更新不執行真實影片／GPU 負載。Windows SDK／原生資產重建、全卡遙測、實機 A/B 為 **WAIVED_BY_USER_NOT_RUN**，不是 PASS；其他未測 Windows／NVIDIA／LTX 與付費模型 AMD 路線仍未認證。

## 實作與詳細記錄

- [jasna/mosaic/rfdetr_torch_runner.py](../../../jasna/mosaic/rfdetr_torch_runner.py)
- [jasna/mosaic/windows_sdpa_policy.py](../../../jasna/mosaic/windows_sdpa_policy.py)
- [tests/test_windows_sdpa_policy.py](../../../tests/test_windows_sdpa_policy.py)
