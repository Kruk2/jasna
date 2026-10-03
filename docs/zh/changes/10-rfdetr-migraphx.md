# 經驗證 RF-DETR MIGraphX 選擇

[English (default)](../../en/changes/10-rfdetr-migraphx.md)

功能: `10-rfdetr-migraphx`. 源碼基準: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## 目的

在共用檢測註冊邊界選擇已驗證的 AMD RF-DETR 產物，驗證源權重、runtime、架構、檔案雜湊與 tensor ABI；掃描及正式處理共用同一選擇。

## 使用方式與預設行為

Linux AMD gfx1100、rfdetr-v6 FP16 安裝匹配 sidecar 後准入直接 MIGraphX 路線。沒有 sidecar 時仍選正常產品 Torch 路線；已安裝但無效的 sidecar 必須報錯。

## 直接前置

- [可選 Windows HIP 縮放與正規化](05-windows-hip-resize.md)

依賴描述整合實作；PR 必須保持單功能範圍。未合併的前置不等於原生驗收。

## 安全與支援限制

不修改檢測閾值、追蹤器、修復模型、產品 batch 或 NVIDIA 後端。Runtime 升級後不能跳過身分驗證沿用舊產物驗收。

## 驗證與重現

```bash
python -m pytest -q tests/test_detection_registry.py tests/test_migraphx_artifact.py tests/test_model_weights_dir.py tests/test_rfdetr_migraphx_product.py tests/test_rfdetr_postprocess.py tests/test_windows_hip_resize_integration.py tests/test_yolo_call.py
```

先前精確源碼整合集合在 Linux 隔離 CPU 環境通過 **3141 項測試、225 項跳過、178 項子測試**。此次文件更新不改執行行為。這是整合集合結果，不代表 26 個無前置獨立分支各自全部通過。

歷史 Linux 原生驗收涵蓋 13 個任務（9427.04 秒），在記錄的處理基準上完成嚴格解碼／時間軸／接縫檢查，不能自動認證後來新增的 Windows 變更。此次文件更新不執行真實影片／GPU 負載。Windows SDK／原生資產重建、全卡遙測、實機 A/B 為 **WAIVED_BY_USER_NOT_RUN**，不是 PASS；其他未測 Windows／NVIDIA／LTX 與付費模型 AMD 路線仍未認證。

## 實作與詳細記錄

- [docs/RFDETR_MIGRAPHX_CORE_CN.md](../../../docs/RFDETR_MIGRAPHX_CORE_CN.md)
- [docs/RFDETR_MIGRAPHX_PRODUCT_CN.md](../../../docs/RFDETR_MIGRAPHX_PRODUCT_CN.md)
- [jasna/gui/engine_preflight.py](../../../jasna/gui/engine_preflight.py)
- [jasna/migraphx_artifact.py](../../../jasna/migraphx_artifact.py)
- [jasna/mosaic/detection_registry.py](../../../jasna/mosaic/detection_registry.py)
- [jasna/mosaic/rfdetr.py](../../../jasna/mosaic/rfdetr.py)
- [jasna/mosaic/rfdetr_migraphx_runner.py](../../../jasna/mosaic/rfdetr_migraphx_runner.py)
- [jasna/mosaic/yolo.py](../../../jasna/mosaic/yolo.py)
- [tests/test_detection_registry.py](../../../tests/test_detection_registry.py)
- [tests/test_migraphx_artifact.py](../../../tests/test_migraphx_artifact.py)
- [tests/test_model_weights_dir.py](../../../tests/test_model_weights_dir.py)
- [tests/test_rfdetr_migraphx_product.py](../../../tests/test_rfdetr_migraphx_product.py)
- [tests/test_rfdetr_postprocess.py](../../../tests/test_rfdetr_postprocess.py)
- [tests/test_windows_hip_resize_integration.py](../../../tests/test_windows_hip_resize_integration.py)
- [tests/test_yolo_call.py](../../../tests/test_yolo_call.py)
