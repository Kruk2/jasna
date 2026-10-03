# 保持輸入資料夾與輸出續跑驗證

[English (default)](../../en/changes/15-preserved-folder-outputs.md)

功能: `15-preserved-folder-outputs`. 源碼基準: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## 目的

相對每個任務選定輸入根目錄解析輸出、保留要求的子資料夾、發布前建立目錄，並在批次續跑認定完成之前驗證既有輸出。

## 使用方式與預設行為

啟用 GUI 保持輸入子資料夾選項並選定輸出根目錄。Full、Smart Render、源流複製及續跑使用共用輸出路徑規則，Linux／Windows 不各自複製一套。

## 直接前置

- [Smart Render 接縫與持久續跑](13-smart-render-resume.md)

依賴描述整合實作；PR 必須保持單功能範圍。未合併的前置不等於原生驗收。

## 安全與支援限制

拒絕路徑逃逸、絕對路徑注入、符號連結逃逸及輸出碰撞。檔案存在不等於完成；取消／失敗需保留有用診斷，不能將待處理任務標為成功。

## 驗證與重現

```bash
python -m pytest -q tests/test_batch_resume_output.py
```

先前精確源碼整合集合在 Linux 隔離 CPU 環境通過 **3141 項測試、225 項跳過、178 項子測試**。此次文件更新不改執行行為。這是整合集合結果，不代表 26 個無前置獨立分支各自全部通過。

歷史 Linux 原生驗收涵蓋 13 個任務（9427.04 秒），在記錄的處理基準上完成嚴格解碼／時間軸／接縫檢查，不能自動認證後來新增的 Windows 變更。此次文件更新不執行真實影片／GPU 負載。Windows SDK／原生資產重建、全卡遙測、實機 A/B 為 **WAIVED_BY_USER_NOT_RUN**，不是 PASS；其他未測 Windows／NVIDIA／LTX 與付費模型 AMD 路線仍未認證。

## 實作與詳細記錄

- [docs/PRESERVED_FOLDER_OUTPUTS_CN.md](../../../docs/PRESERVED_FOLDER_OUTPUTS_CN.md)
- [jasna/gui/output_paths.py](../../../jasna/gui/output_paths.py)
- [jasna/gui/queue_panel.py](../../../jasna/gui/queue_panel.py)
- [jasna/gui/resume_validation.py](../../../jasna/gui/resume_validation.py)
- [tests/test_batch_resume_output.py](../../../tests/test_batch_resume_output.py)
