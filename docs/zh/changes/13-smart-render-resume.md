# Smart Render 接縫與持久續跑

[English (default)](../../en/changes/13-smart-render-resume.md)

功能: `13-smart-render-resume`. 源碼基準: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## 目的

拼接複製與修復區間，驗證參數集、時間戳連續性、幀數／時長契約及區間邊界。保存持久分段 manifest，並使不相容的快取工作失效。

## 使用方式與預設行為

沿用既有區間／Smart Render 路線。可重用已完成且相容的分段；不能僅因檔案存在就接受未完成或不匹配分段。Full 區間的空 effect ranges 表示全幀處理，不是零修復。

## 直接前置

- [AMF 原生解碼與幀所有權](06-amf-native-decode.md)
- [AMD 編碼契約與匹配源碼率 Peak VBR](08-encoder-source-rate.md)

依賴描述整合實作；PR 必須保持單功能範圍。未合併的前置不等於原生驗收。

## 安全與支援限制

Smart Render 要維持源封包契約，因此不能任意改變輸出 codec／profile。結構檢查、嚴格解碼、接縫檢查與實際修復證據各自驗證不同問題。

## 驗證與重現

```bash
python -m pytest -q tests/test_smart_render_workspace.py tests/test_splice.py tests/test_splice_media.py
```

先前精確源碼整合集合在 Linux 隔離 CPU 環境通過 **3141 項測試、225 項跳過、178 項子測試**。此次文件更新不改執行行為。這是整合集合結果，不代表 26 個無前置獨立分支各自全部通過。

歷史 Linux 原生驗收涵蓋 13 個任務（9427.04 秒），在記錄的處理基準上完成嚴格解碼／時間軸／接縫檢查，不能自動認證後來新增的 Windows 變更。此次文件更新不執行真實影片／GPU 負載。Windows SDK／原生資產重建、全卡遙測、實機 A/B 為 **WAIVED_BY_USER_NOT_RUN**，不是 PASS；其他未測 Windows／NVIDIA／LTX 與付費模型 AMD 路線仍未認證。

## 實作與詳細記錄

- [docs/SMART_RENDER_RESUME_WORKSPACE_CN.md](../../../docs/SMART_RENDER_RESUME_WORKSPACE_CN.md)
- [docs/SMART_RENDER_TIMESTAMP_SEAM_CN.md](../../../docs/SMART_RENDER_TIMESTAMP_SEAM_CN.md)
- [docs/en/segments.md](../../../docs/en/segments.md)
- [docs/ja/segments.md](../../../docs/ja/segments.md)
- [docs/zh/segments.md](../../../docs/zh/segments.md)
- [jasna/media/splice.py](../../../jasna/media/splice.py)
- [jasna/smart_render_workspace.py](../../../jasna/smart_render_workspace.py)
- [tests/test_smart_render_workspace.py](../../../tests/test_smart_render_workspace.py)
- [tests/test_splice.py](../../../tests/test_splice.py)
- [tests/test_splice_media.py](../../../tests/test_splice_media.py)
