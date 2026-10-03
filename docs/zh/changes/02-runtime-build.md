# 可重現 FFmpeg 與 PyAV 建置

[English (default)](../../en/changes/02-runtime-build.md)

功能: `02-runtime-build`. 源碼基準: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## 目的

提供固定 FFmpeg／PyAV／AMF 源碼的 Linux、Windows 建置腳本，以及可審查的 FFmpeg 補丁，處理傳輸格式、舊幀上下文、解析度變更、關鍵幀重置、投影標籤和連續主機輸入。

## 使用方式與預設行為

使用 scripts/build_unified_ffmpeg_pyav_ubuntu.sh 或 scripts/build_unified_ffmpeg_pyav_windows.ps1，再透過 runtime 安裝器安裝驗證後的產物。單純建置不得切換桌面啟動器。

## 直接前置

- [固定版本統一媒體 runtime 與安裝器](01-runtime-contract.md)

依賴描述整合實作；PR 必須保持單功能範圍。未合併的前置不等於原生驗收。

## 安全與支援限制

可重現源碼版本不代表尚未建置的 SDK 組合一定可用。本次交付明確豁免 Windows SDK／原生資產重建，狀態為 NOT_RUN。

## 驗證與重現

```bash
python -m pytest -q tests/test_unified_build_scripts.py
```

先前精確源碼整合集合在 Linux 隔離 CPU 環境通過 **3141 項測試、225 項跳過、178 項子測試**。此次文件更新不改執行行為。這是整合集合結果，不代表 26 個無前置獨立分支各自全部通過。

歷史 Linux 原生驗收涵蓋 13 個任務（9427.04 秒），在記錄的處理基準上完成嚴格解碼／時間軸／接縫檢查，不能自動認證後來新增的 Windows 變更。此次文件更新不執行真實影片／GPU 負載。Windows SDK／原生資產重建、全卡遙測、實機 A/B 為 **WAIVED_BY_USER_NOT_RUN**，不是 PASS；其他未測 Windows／NVIDIA／LTX 與付費模型 AMD 路線仍未認證。

## 實作與詳細記錄

- [docs/UNIFIED_BUILD_PIPELINE_CN.md](../../../docs/UNIFIED_BUILD_PIPELINE_CN.md)
- [patches/ffmpeg/0001-amf-transfer-use-context-sw-format.patch](../../../patches/ffmpeg/0001-amf-transfer-use-context-sw-format.patch)
- [patches/ffmpeg/0002-amfdec-replace-stale-frames-context.patch](../../../patches/ffmpeg/0002-amfdec-replace-stale-frames-context.patch)
- [patches/ffmpeg/0003-amfdec-fix-dynamic-resolution-reinit.patch](../../../patches/ffmpeg/0003-amfdec-fix-dynamic-resolution-reinit.patch)
- [patches/ffmpeg/0004-matroska-projection-tag-spherical.patch](../../../patches/ffmpeg/0004-matroska-projection-tag-spherical.patch)
- [patches/ffmpeg/0005-amfdec-reset-state-at-keyframes.patch](../../../patches/ffmpeg/0005-amfdec-reset-state-at-keyframes.patch)
- [patches/ffmpeg/0006-amfenc-wrap-contiguous-host-input.patch](../../../patches/ffmpeg/0006-amfenc-wrap-contiguous-host-input.patch)
- [scripts/build_unified_ffmpeg_pyav_ubuntu.sh](../../../scripts/build_unified_ffmpeg_pyav_ubuntu.sh)
- [scripts/build_unified_ffmpeg_pyav_windows.ps1](../../../scripts/build_unified_ffmpeg_pyav_windows.ps1)
- [tests/test_unified_build_scripts.py](../../../tests/test_unified_build_scripts.py)
