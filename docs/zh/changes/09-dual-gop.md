# 有界 Linux HEVC 雙 GOP 編碼

[English (default)](../../en/changes/09-dual-gop.md)

功能: `09-dual-gop`. 源碼基準: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## 目的

將獨立 closed GOP 交替送入兩個持久 AMF 編碼會話，按時間軸順序組裝結果；限制 staging、surface、待處理 GOP 及關閉所有權。

## 使用方式與預設行為

GUI 預設會請求雙 GOP；CLI 必須明確傳入 --amd-dual-gop-encode。最終 HEVC 輸出准入要求 Main10／P010 至少 3840×2160 像素，或 Main／NV12 至少 5760×2880 像素，尺寸為正偶數且源碼率資訊可用。

## 直接前置

- [AMD 編碼契約與匹配源碼率 Peak VBR](08-encoder-source-rate.md)
- [Smart Render 接縫與持久續跑](13-smart-render-resume.md)

依賴描述整合實作；PR 必須保持單功能範圍。未合併的前置不等於原生驗收。

## 安全與支援限制

Smart Render 還要求 HEVC 源封包相容。不准入的 GUI 任務沿用單會話編碼。這是 Linux GOP 級並行，不是 Windows split-frame；提速必須與匹配基準測量比較。

## 驗證與重現

```bash
python -m pytest -q tests/test_dual_gop_encoder.py
```

先前精確源碼整合集合在 Linux 隔離 CPU 環境通過 **3141 項測試、225 項跳過、178 項子測試**。此次文件更新不改執行行為。這是整合集合結果，不代表 26 個無前置獨立分支各自全部通過。

歷史 Linux 原生驗收涵蓋 13 個任務（9427.04 秒），在記錄的處理基準上完成嚴格解碼／時間軸／接縫檢查，不能自動認證後來新增的 Windows 變更。此次文件更新不執行真實影片／GPU 負載。Windows SDK／原生資產重建、全卡遙測、實機 A/B 為 **WAIVED_BY_USER_NOT_RUN**，不是 PASS；其他未測 Windows／NVIDIA／LTX 與付費模型 AMD 路線仍未認證。

## 實作與詳細記錄

- [docs/AMD_DUAL_GOP_ENCODER_CN.md](../../../docs/AMD_DUAL_GOP_ENCODER_CN.md)
- [jasna/media/dual_gop_encoder.py](../../../jasna/media/dual_gop_encoder.py)
- [tests/test_dual_gop_encoder.py](../../../tests/test_dual_gop_encoder.py)
