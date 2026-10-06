# 自適應自動預掃描與漏區間修復

[English (default)](../../en/changes/14-automatic-prescan.md)

功能: `14-automatic-prescan`. 源碼基準: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## 目的

共用 reader／檢測器進行自適應粗精掃、簽名驗證 checkpoint、依覆蓋率選路及時間戳抖動感知的命中合併，避免長高解析輸入反覆建立解碼 epoch。

## 使用方式與預設行為

GUI 自動模式通常約每 4 秒粗掃、候選約每 0.5 秒精掃。沒有可信命中時可驗證後複製；預設覆蓋率 85% 選 full；其餘走 Smart Render。明確手動區間／整片處理優先。

## 直接前置

- [AMF 原生解碼與幀所有權](06-amf-native-decode.md)
- [經驗證 RF-DETR MIGraphX 選擇](10-rfdetr-migraphx.md)

依賴描述整合實作；PR 必須保持單功能範圍。未合併的前置不等於原生驗收。

## 安全與支援限制

抖動容差不能跨越真正缺失的採樣。舊無效簽名需重新掃描。源 GOP 不可重現時，自動改 full 仍只在確認的 PTS 區間執行修復；明確 Smart Render 請求不能靜默更改。

## 驗證與重現

```bash
python -m pytest -q tests/test_mosaic_scan.py tests/test_pre_scan_routing.py
```

先前精確源碼整合集合在 Linux 隔離 CPU 環境通過 **3141 項測試、225 項跳過、178 項子測試**。此次文件更新不改執行行為。這是整合集合結果，不代表 26 個無前置獨立分支各自全部通過。

歷史 Linux 原生驗收涵蓋 13 個任務（9427.04 秒），在記錄的處理基準上完成嚴格解碼／時間軸／接縫檢查，不能自動認證後來新增的 Windows 變更。此次文件更新不執行真實影片／GPU 負載。Windows SDK／原生資產重建、全卡遙測、實機 A/B 為 **WAIVED_BY_USER_NOT_RUN**，不是 PASS；其他未測 Windows／NVIDIA／LTX 與付費模型 AMD 路線仍未認證。

## 實作與詳細記錄

- [docs/AUTOMATIC_PRE_SCAN_ROUTING_CN.md](../../../docs/AUTOMATIC_PRE_SCAN_ROUTING_CN.md)
- [docs/MOSAIC_SCAN_UNIFIED_AMD_CN.md](../../../docs/MOSAIC_SCAN_UNIFIED_AMD_CN.md)
- [jasna/gui/mosaic_scan.py](../../../jasna/gui/mosaic_scan.py)
- [jasna/gui/pre_scan_routing.py](../../../jasna/gui/pre_scan_routing.py)
- [tests/test_mosaic_scan.py](../../../tests/test_mosaic_scan.py)
- [tests/test_pre_scan_routing.py](../../../tests/test_pre_scan_routing.py)
