# Windows ROCm 匯入與顯卡廠商相容

[English (default)](../../en/changes/03-windows-rocm-compat.md)

功能: `03-windows-rocm-compat`. 源碼基準: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## 目的

讓公開源碼的匯入與系統檢查區分 AMD／ROCm 和 NVIDIA／CUDA。提供限定範圍的 MMEngine 相容層，並將 TensorRT 保持為可選元件，避免 AMD 執行時強制匯入 NVIDIA 專用模組。

## 使用方式與預設行為

沿用現有源碼啟動器與模型介面。檢測／媒體測試明確宣告顯卡廠商假設；匯入模組不應要求另一廠商不可用的函式庫。

## 直接前置

無。

依賴描述整合實作；PR 必須保持單功能範圍。未合併的前置不等於原生驗收。

## 安全與支援限制

這是相容基礎設施，不是新的推論後端，也不是 Windows 完整流程的效能證據。NVIDIA TensorRT 行為仍由對應後端負責。

## 驗證與重現

```bash
python -m pytest -q tests/test_mmengine_windows_rocm_compat.py tests/test_os_utils.py tests/test_trt_utils.py
```

先前精確源碼整合集合在 Linux 隔離 CPU 環境通過 **3141 項測試、225 項跳過、178 項子測試**。此次文件更新不改執行行為。這是整合集合結果，不代表 26 個無前置獨立分支各自全部通過。

歷史 Linux 原生驗收涵蓋 13 個任務（9427.04 秒），在記錄的處理基準上完成嚴格解碼／時間軸／接縫檢查，不能自動認證後來新增的 Windows 變更。此次文件更新不執行真實影片／GPU 負載。Windows SDK／原生資產重建、全卡遙測、實機 A/B 為 **WAIVED_BY_USER_NOT_RUN**，不是 PASS；其他未測 Windows／NVIDIA／LTX 與付費模型 AMD 路線仍未認證。

## 實作與詳細記錄

- [docs/DETECTION_TEST_VENDOR_ISOLATION_CN.md](../../../docs/DETECTION_TEST_VENDOR_ISOLATION_CN.md)
- [docs/MEDIA_TEST_ENVIRONMENT_ISOLATION_CN.md](../../../docs/MEDIA_TEST_ENVIRONMENT_ISOLATION_CN.md)
- [docs/WINDOWS_MATCHED_GUI_ACCEPTANCE_CN.md](../../../docs/WINDOWS_MATCHED_GUI_ACCEPTANCE_CN.md)
- [jasna/models/basicvsrpp/__init__.py](../../../jasna/models/basicvsrpp/__init__.py)
- [jasna/models/basicvsrpp/mmengine_compat.py](../../../jasna/models/basicvsrpp/mmengine_compat.py)
- [jasna/os_utils.py](../../../jasna/os_utils.py)
- [jasna/trt/__init__.py](../../../jasna/trt/__init__.py)
- [tests/conftest.py](../../../tests/conftest.py)
- [tests/test_mmengine_windows_rocm_compat.py](../../../tests/test_mmengine_windows_rocm_compat.py)
- [tests/test_os_utils.py](../../../tests/test_os_utils.py)
- [tests/test_trt_utils.py](../../../tests/test_trt_utils.py)
