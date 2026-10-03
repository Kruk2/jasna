# HIP 色彩轉換核心

[English (default)](../../en/changes/04-hip-colour.md)

功能: `04-hip-colour`. 源碼基準: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## 目的

加入 AMD HIP RGB／YUV 轉換、架構固定的程式物件、身分驗證、有界暫存重用與明確串流所有權。保留 NVIDIA 實作及既有色彩介面。

## 使用方式與預設行為

Linux 在准入範圍使用已驗證的 HIP 產品路線。Windows HIP 色彩路線仍需明確啟用，且 manifest、程式物件、架構及實際載入 runtime 必須匹配；建置／探測腳本隨本功能提供。

## 直接前置

- [Windows ROCm 匯入與顯卡廠商相容](03-windows-rocm-compat.md)

依賴描述整合實作；PR 必須保持單功能範圍。未合併的前置不等於原生驗收。

## 安全與支援限制

NV12 與 P010 各自有版面／位深契約。核心身分或同步失敗必須報錯，不能據此靜默改走 CPU 中轉。

## 驗證與重現

```bash
python -m pytest -q tests/test_hip_colour_kernel_product.py tests/test_hip_kernel.py tests/test_lut_kernel.py tests/test_rgb_to_yuv_kernel.py tests/test_yuv_scratch_reuse.py tests/test_yuv_to_rgb.py
```

先前精確源碼整合集合在 Linux 隔離 CPU 環境通過 **3141 項測試、225 項跳過、178 項子測試**。此次文件更新不改執行行為。這是整合集合結果，不代表 26 個無前置獨立分支各自全部通過。

歷史 Linux 原生驗收涵蓋 13 個任務（9427.04 秒），在記錄的處理基準上完成嚴格解碼／時間軸／接縫檢查，不能自動認證後來新增的 Windows 變更。此次文件更新不執行真實影片／GPU 負載。Windows SDK／原生資產重建、全卡遙測、實機 A/B 為 **WAIVED_BY_USER_NOT_RUN**，不是 PASS；其他未測 Windows／NVIDIA／LTX 與付費模型 AMD 路線仍未認證。

## 實作與詳細記錄

- [docs/WINDOWS_AMD_HIP_COLOR_KERNELS_CN.md](../../../docs/WINDOWS_AMD_HIP_COLOR_KERNELS_CN.md)
- [jasna/media/hip_color_kernels.gfx1100.windows.json](../../../jasna/media/hip_color_kernels.gfx1100.windows.json)
- [jasna/media/hip_kernel.py](../../../jasna/media/hip_kernel.py)
- [jasna/media/rgb_to_yuv.gfx1100.hsaco](../../../jasna/media/rgb_to_yuv.gfx1100.hsaco)
- [jasna/media/rgb_to_yuv.gfx1100.windows.co](../../../jasna/media/rgb_to_yuv.gfx1100.windows.co)
- [jasna/media/rgb_to_yuv.py](../../../jasna/media/rgb_to_yuv.py)
- [jasna/media/yuv_to_rgb.gfx1100.hsaco](../../../jasna/media/yuv_to_rgb.gfx1100.hsaco)
- [jasna/media/yuv_to_rgb.gfx1100.windows.co](../../../jasna/media/yuv_to_rgb.gfx1100.windows.co)
- [jasna/media/yuv_to_rgb.py](../../../jasna/media/yuv_to_rgb.py)
- [scripts/build_hip_code_objects.sh](../../../scripts/build_hip_code_objects.sh)
- [scripts/build_hip_code_objects_windows.ps1](../../../scripts/build_hip_code_objects_windows.ps1)
- [scripts/probe_amd_hip_color_kernels.py](../../../scripts/probe_amd_hip_color_kernels.py)
- [tests/test_hip_colour_kernel_product.py](../../../tests/test_hip_colour_kernel_product.py)
- [tests/test_hip_kernel.py](../../../tests/test_hip_kernel.py)
- [tests/test_lut_kernel.py](../../../tests/test_lut_kernel.py)
- [tests/test_rgb_to_yuv_kernel.py](../../../tests/test_rgb_to_yuv_kernel.py)
- [tests/test_yuv_scratch_reuse.py](../../../tests/test_yuv_scratch_reuse.py)
- [tests/test_yuv_to_rgb.py](../../../tests/test_yuv_to_rgb.py)
