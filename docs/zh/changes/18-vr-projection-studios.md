# 精確 VR 片商與投影路由

[English (default)](../../en/changes/18-vr-projection-studios.md)

功能: `18-vr-projection-studios`. 源碼基準: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## 目的

以精確代碼識別擴充共用 VR 片商投影規則，保留使用者明確投影選擇，避免相似片商前綴碰撞。

## 使用方式與預設行為

自動路由包含新增 CCVR、KBVR、KMVR、DSVR、MAXVR、JPSVR 魚眼映射。檔名無法可靠描述投影時，明確選 raw／fisheye，並參考 VR 指南。

## 直接前置

無。

依賴描述整合實作；PR 必須保持單功能範圍。未合併的前置不等於原生驗收。

## 安全與支援限制

片商命名是路由啟發式，不是投影或必然檢測成功的證明。檢測閾值／後端需與投影判斷分開。

## 驗證與重現

```bash
python -m pytest -q tests/test_vr180.py
```

先前精確源碼整合集合在 Linux 隔離 CPU 環境通過 **3141 項測試、225 項跳過、178 項子測試**。此次文件更新不改執行行為。這是整合集合結果，不代表 26 個無前置獨立分支各自全部通過。

歷史 Linux 原生驗收涵蓋 13 個任務（9427.04 秒），在記錄的處理基準上完成嚴格解碼／時間軸／接縫檢查，不能自動認證後來新增的 Windows 變更。此次文件更新不執行真實影片／GPU 負載。Windows SDK／原生資產重建、全卡遙測、實機 A/B 為 **WAIVED_BY_USER_NOT_RUN**，不是 PASS；其他未測 Windows／NVIDIA／LTX 與付費模型 AMD 路線仍未認證。

## 實作與詳細記錄

- [docs/zh/vr180.md](../../../docs/zh/vr180.md)
- [jasna/vr180.py](../../../jasna/vr180.py)
- [tests/test_vr180.py](../../../tests/test_vr180.py)
