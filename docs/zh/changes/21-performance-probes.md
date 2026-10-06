# 明確啟用 AMD 效能與容量探測

[English (default)](../../en/changes/21-performance-probes.md)

功能: `21-performance-probes`. 源碼基準: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## 目的

加入明確啟用的非同步 D2H 雙 GOP、原生 YUV 往返、RF-DETR B2、單次解碼容量探測，記錄可比較計時、資源／生命週期證據及輸入輸出身分。

## 使用方式與預設行為

明確執行對應 scripts/probe_* 入口並查閱 --help。比較需使用相同源片、區間、ROI、格式、硬體與預熱策略；產品化前先使用有界樣本。

## 直接前置

- [共用流水線資源與失敗安全](12-pipeline-resource-safety.md)

依賴描述整合實作；PR 必須保持單功能範圍。未合併的前置不等於原生驗收。

## 安全與支援限制

這些腳本不會在 GUI 自動啟用 B2、單次解碼、原生往返或新佇列深度。容量及合成微測試不能認證整片速度或畫質。

## 驗證與重現

```bash
python -m pytest -q tests/test_probe_amd_dual_gop_async_d2h.py tests/test_probe_amd_native_yuv_roundtrip.py tests/test_probe_rfdetr_migraphx_b2.py tests/test_probe_single_decode_capacity.py
```

先前精確源碼整合集合在 Linux 隔離 CPU 環境通過 **3141 項測試、225 項跳過、178 項子測試**。此次文件更新不改執行行為。這是整合集合結果，不代表 26 個無前置獨立分支各自全部通過。

歷史 Linux 原生驗收涵蓋 13 個任務（9427.04 秒），在記錄的處理基準上完成嚴格解碼／時間軸／接縫檢查，不能自動認證後來新增的 Windows 變更。此次文件更新不執行真實影片／GPU 負載。Windows SDK／原生資產重建、全卡遙測、實機 A/B 為 **WAIVED_BY_USER_NOT_RUN**，不是 PASS；其他未測 Windows／NVIDIA／LTX 與付費模型 AMD 路線仍未認證。

## 實作與詳細記錄

- [docs/AMD_8K_PIPELINE_OPTIMIZATION_AUDIT_20260905_CN.md](../../../docs/AMD_8K_PIPELINE_OPTIMIZATION_AUDIT_20260905_CN.md)
- [scripts/probe_amd_dual_gop_async_d2h.py](../../../scripts/probe_amd_dual_gop_async_d2h.py)
- [scripts/probe_amd_native_yuv_roundtrip.py](../../../scripts/probe_amd_native_yuv_roundtrip.py)
- [scripts/probe_rfdetr_migraphx_b2.py](../../../scripts/probe_rfdetr_migraphx_b2.py)
- [scripts/probe_rfdetr_migraphx_b2_product.py](../../../scripts/probe_rfdetr_migraphx_b2_product.py)
- [scripts/probe_single_decode_capacity.py](../../../scripts/probe_single_decode_capacity.py)
- [tests/test_probe_amd_dual_gop_async_d2h.py](../../../tests/test_probe_amd_dual_gop_async_d2h.py)
- [tests/test_probe_amd_native_yuv_roundtrip.py](../../../tests/test_probe_amd_native_yuv_roundtrip.py)
- [tests/test_probe_rfdetr_migraphx_b2.py](../../../tests/test_probe_rfdetr_migraphx_b2.py)
- [tests/test_probe_single_decode_capacity.py](../../../tests/test_probe_single_decode_capacity.py)
