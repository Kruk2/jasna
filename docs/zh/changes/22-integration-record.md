# 整合、依賴與驗收記錄

[English (default)](../../en/changes/22-integration-record.md)

功能: `22-integration-record`. 源碼基準: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## 目的

記錄以最新上游 main 為基準的完整功能集合、共用流程／後端分層、依賴順序、精確 CPU 證據、歷史 Linux 原生證據及 Windows 限制。

## 使用方式與預設行為

先閱讀英文功能索引，需要時使用中文對應版本。按依賴順序審查／合併單功能，更新 main 後 rebase 並重跑受影響檢查。本地完整集合不是單功能 PR 的替代品。

## 直接前置

- [Windows ROCm 匯入與顯卡廠商相容](03-windows-rocm-compat.md)
- [HIP 色彩轉換核心](04-hip-colour.md)
- [共用原生任務與診斷契約](00-shared-native-job-contracts.md)
- [固定版本統一媒體 runtime 與安裝器](01-runtime-contract.md)
- [可重現 FFmpeg 與 PyAV 建置](02-runtime-build.md)
- [可選 Windows HIP 縮放與正規化](05-windows-hip-resize.md)
- [有界 Windows D3D11–HIP 常駐媒體](07-windows-resident-media.md)
- [AMF 原生解碼與幀所有權](06-amf-native-decode.md)
- [AMD 編碼契約與匹配源碼率 Peak VBR](08-encoder-source-rate.md)
- [Smart Render 接縫與持久續跑](13-smart-render-resume.md)
- [有界 Linux HEVC 雙 GOP 編碼](09-dual-gop.md)
- [經驗證 RF-DETR MIGraphX 選擇](10-rfdetr-migraphx.md)
- [經驗證 BasicVSR++ MIGraphX B1 修復](11-basicvsrpp-migraphx.md)
- [公開源碼可選授權邊界](19-public-source-license.md)
- [共用流水線資源與失敗安全](12-pipeline-resource-safety.md)
- [自適應自動預掃描與漏區間修復](14-automatic-prescan.md)
- [保持輸入資料夾與輸出續跑驗證](15-preserved-folder-outputs.md)
- [原生視訊任務隔離與持久輸出](16-isolated-video-jobs.md)
- [GUI 控制、診斷與可靠進度](17-gui-settings-diagnostics.md)
- [精確 VR 片商與投影路由](18-vr-projection-studios.md)
- [純 CPU SD 1.5 回歸隔離](20-cpu-regression-isolation.md)
- [明確啟用 AMD 效能與容量探測](21-performance-probes.md)
- [身分限定 Windows AMD Math SDPA](23-windows-sdpa-compat.md)
- [特定 Windows AMF 傳輸失敗後隔離](24-native-context-quarantine.md)
- [排列感知 RF-DETR 精度診斷](25-rfdetr-precision-probe.md)

依賴描述整合實作；PR 必須保持單功能範圍。未合併的前置不等於原生驗收。

## 安全與支援限制

將歷史 v0.10 記錄明確標為歷史。不能把豁免／跳過的 Windows／NVIDIA／LTX 檢查稱為 PASS，也不能把缺失官方付費元件描述成已實作。

## 驗證與重現

本功能沒有獨立所屬測試模組；需檢查完整測試收集與整合回歸。

先前精確源碼整合集合在 Linux 隔離 CPU 環境通過 **3141 項測試、225 項跳過、178 項子測試**。此次文件更新不改執行行為。這是整合集合結果，不代表 26 個無前置獨立分支各自全部通過。

歷史 Linux 原生驗收涵蓋 13 個任務（9427.04 秒），在記錄的處理基準上完成嚴格解碼／時間軸／接縫檢查，不能自動認證後來新增的 Windows 變更。此次文件更新不執行真實影片／GPU 負載。Windows SDK／原生資產重建、全卡遙測、實機 A/B 為 **WAIVED_BY_USER_NOT_RUN**，不是 PASS；其他未測 Windows／NVIDIA／LTX 與付費模型 AMD 路線仍未認證。

## 實作與詳細記錄

- [docs/MAIN_INTEGRATION_ROCM10_20261001_CN.md](../../../docs/MAIN_INTEGRATION_ROCM10_20261001_CN.md)
- [docs/STACKED_PR_LINUX_ACCEPTANCE_CN.md](../../../docs/STACKED_PR_LINUX_ACCEPTANCE_CN.md)
- [docs/V010_PR_REBUILD_20261001_CN.md](../../../docs/V010_PR_REBUILD_20261001_CN.md)
- [docs/WINDOWS_ROCM10_COMPATIBILITY_CN.md](../../../docs/WINDOWS_ROCM10_COMPATIBILITY_CN.md)
- [docs/en/development.md](../../../docs/en/development.md)
