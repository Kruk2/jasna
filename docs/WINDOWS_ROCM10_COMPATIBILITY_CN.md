# Windows ROCm 10 兼容收斂（2026-10-03，本地審查）

基於上游 main 0.11.0 / `81dc8b0` 與本地 23 個已驗證功能集合
`e0566eb`。最新 26 項本地審查栈包含下列兼容增量；Linux 集合已獲用戶授權
替換桌面 GUI，Windows 新產品入口仍未冒稱真機通過。不發布上游 PR。

本輪用戶明確豁免 Windows SDK 原生資產重建、全卡遙測及實機 A/B：
狀態為 `WAIVED_BY_USER_NOT_RUN`，不是 PASS，也不繞過現有原生資產身份門。
本次只拆分已部署處理代碼和刷新文檔，不重新測 GPU，不改桌面入口。

## 證據邊界

- Windows 交接的三份 `evidence.zip` 外層 SHA256 與交接相符，ZIP CRC 通過。
  GPU 診斷包 84 項及超時診斷包 10 項成員 SHA256 另行通過；歷史短測包
  的工具 manifest 標註 provenance-only，不冒稱它涵蓋 ZIP 所有證據成員。
- 原 Windows 合成實測：GPU Math SDPA 的 B4 FP32/FP16 執行與已記錄 HIP
  狀態通過；B1 FP32 CPU/GPU 數值對照通過。這是啟動層實測，不是本次
  產品入口已在 Windows 重跑，也不是 FP16 畫質或性能 A/B 通過。
- 使用者後續反饋「長測沒什麼問題」保留為人工反饋。尚缺該次完整
  日誌/evidence、運行時 FP16/隊列配置，不能代替完整驗收矩陣。
- 2026-10-02 18:37 的 AMF host transfer UnknownError 與同次 Windows
  LiveKernelEvent 141 保留為歷史失敗；不能直接歸因於 FP16/解碼/編碼/
  驅動，也不能把同秒重報的舊轉儲當成新增崩潰。Torch 的 VRAM 數字漏記
  部分原生分配，因此不能排除全卡壓力。

## 產品 SDPA 策略

`jasna/mosaic/windows_sdpa_policy.py` 在 `RfDetrTorchRunner` 匯入及構造
RF-DETR wrapper **之前**執行；共享 CLI、GUI、粗/精掃與恢復檢測入口。

預設 `JASNA_WINDOWS_AMD_SDPA_POLICY=auto`，只有以下身份全部相符才選 Math：

- Windows / AMD HIP CUDA device、gfx1100（可帶 arch suffix）；
- Torch `2.12.0+rocm10.0.0`、實際 Torch HIP `7.15.26333`；
- 實際載入 HIP runtime API `71526333`；
- HIP DLL SHA256 `546fb3d6e2d2194a9526fb94ec2fd3aa5b92a48a7595f04efece80162047ef69`。

Math 保留 GPU 運行。關閉 Flash、memory-efficient、cuDNN SDPA 並啟用 Math，
回讀四個旗標；回讀失敗拒絕 forward。INFO JSON 記錄 requested/resolved、
版本、實際 DLL 身份、之前/之後旗標與獨立的 FP16 設定。

`math` 嚴格要求同一已驗證 Windows AMD 身份；不相符拒絕。
`default` 不改現有旗標，也不重設外部 bootstrap 已配置的 Math。
auto 的其他 Windows AMD 版本/架構保留原旗標並記錄未匹配原因。
Linux、CPU、NVIDIA 不載入 HIP 身份、不探測 GPU、不更改後端。

這是已驗證啟動層的**進程預設策略**。LTX 的顯式 `sdpa_kernel` 上下文及
其他模型的強制 backend 未驗收，不能據此聲稱受此策略保護。
沒有改 Linux MIGraphX、雙 GOP、AMF Vulkan/HIP、模型精度或 rocDecode 策略。

## FP16 排序感知診斷

`scripts/probe_windows_rfdetr_precision.py` 只用 seed 固定的合成張量，走
產品 Torch runner；預設 B1/B4，各測 GPU FP32/FP16，CPU FP32 只作明確
數值參考，並非 GPU 出错回退。使用非清除式 `hipPeekAtLastError`。

以 box 與 sigmoid 類別分數做一對一 Hungarian 匹配，再報框/分類/mask
數值誤差、重排數、有活動 proposal 的高重疊匹配/漏匹配數。
即使合成診斷全部相符，仍標 `quality_acceptance=NOT_CERTIFIED`；
沒有實際馬賽克的輸入不能證明漏檢率。同步診斷也不能用作性能 A/B。

Windows 在已完成 native preflight 的隔離候選環境執行，需 scipy：

```powershell
python .\scripts\probe_windows_rfdetr_precision.py --weights .\model_weights\rfdetr-v6.pt --output .\precision-new.json
```

輸出不得覆蓋既有證據。發生 native 錯誤停止，不在同一 GPU 上重試。

## AMF 失效上下文隔離

只在 Windows AMD、實際 `_amf` 解碼上下文，同時遇到 FFmpeg errno
`±1313558101` 與具體 `AVHWFramesContext Convert(amf::AMF_MEMORY_HOST) failed`
日誌，解碼器才拋出型別化 `NativeGpuContextUnusableError`。
通用 UnknownError、損壞包或 Linux/NVIDIA/軟解不會被當成該故障。

共用 Processor 保留原始 cause，失敗項標 ERROR，下一項維持 PENDING，
拒絕同進程再次 Start。保留錯誤/工作區；不重試失效上下文，不執行 Torch
synchronize/reset，session 關閉失敗也不覆蓋原始故障及 completion(False)。
**退出並重開 GUI**才重建解碼、編碼及模型資源；本次没有默認啟用尚未
真機驗證的 Windows guarded worker，也沒有宣稱已自動恢復 driver/TDR。
這是防止故障擴散，並非已找到或修好歷史 GPU timeout 的觸發源。

## 新驗收工具（kit-02）

- 舊 kit-01、Windows kit-short 與歷史 evidence 保持不變。
- 納入已驗證私有修正：Quick 可指定 10–30 秒；Full 仍 30/300 秒。
  GPU 停止後 A/B 不再索引缺失的 wall_seconds，缺實測標 INCOMPLETE/NOT_RUN。
- 「Device-side assertion tracking was not enabled by user」不再誤報；
  真正 assert triggered/failed、hipErrorAssert、非法訪問、AV/fail-fast 仍停止。
- `fp16_mode` 必須為 JSON boolean，CLI/Processor 均明確傳遞並在報告記錄
  運行參數，不用磁盤保存的預設推斷實際精度。
- `source_kind` 須聲明 synthetic/public_nonexplicit；是輸入方聲明，不是
  自動內容識別。本輪只使用合成或非露骨公開受控素材，不截圖。
- 新 ZIP 的 manifest 記錄實際源樹/源碼 SHA 與本次離線結果，不把歷史
  Windows 啟動包測試冒充新入口的 PowerShell 5.1 或 native 驗收。

## 未測與豁免邊界

以下保留後續研發路線，不是要求本輪重新補測。第 1、2 項及第 5 項的
新舊實機 A/B 已獲用戶豁免；其餘未測範圍仍標 `NOT_RUN`，不能被豁免改寫為 PASS。

1. 按實際 HIP 7.15.26333 SDK 重建色彩、resize 二進制、manifest 與契約。
   舊資產 7.2.53211 / 7.16.26354 保持嚴格拒絕，沒有只改版本/hash 放行。
   resize 舊 builder 只允許舊 SDK byte-identical 重建，不能冒稱可更新新 SDK。
2. 當前 DLL 沒有 hipDeviceGetLuid。實際 SDK `hip_runtime_api.h` 定義
   `hipDeviceProp_tR0600.luid[8]` / `luidDeviceNodeMask` 與 R0600 API。
   下一步用**同 SDK 編譯的 C ABI 小橋**呼叫已載入的 HIP module、輸出
   8 bytes LUID/node-mask；驗證非空身份/單位元 mask，與 DXGI/PDH 精確
   對應。不在 Python 猜 struct offset、不按 GPU 名稱挑卡。橋目前未重建
   及真機驗證，whole-card telemetry 仍 NOT_RUN，不冒稱 Torch 數字完整。
3. 產品入口 Math 原生重跑、FP16 任務級畫質對照（公開受控素材）、LTX。
4. AMF 雙 reader、獨立編碼、Torch 合成負載的分組並發，含全卡峰值、
   driver 事件、取消/回收/重開新進程；區分 timeout 觸發源。
5. full/smart/copy，Main8/Main10/H264，五分鐘與長队列，严格软件解码、
   帧数/时长/PTS/DTS/音频/接缝，及匹配新舊環境的熱身+三對 A/B。
   不將已有後續人工長測正常反饋變成整個矩陣 PASS。

原生資產門仍未滿足時，kit 將如實記錄缺項，**不是可無錯完整驗收包**。
不要反覆長跑來代替未具備的資產/硬件條件；豁免不代表原生路線已認證。
歷史 kit-02 及部署源碼包保持不變；新審查文檔不改寫原有證據 SHA。
最新 PR 已按「Windows SDPA 兼容」「失效上下文隊列隔離」「檢測精度診斷」
分開；SDK/遙測橋未實現，沒有為其建立虛構實作 PR。先本地整理及組合復驗，
推送或發布仍需用戶後續確認。

參考：本機結論來自交接實測；[TheRock #7992](https://github.com/ROCm/TheRock/issues/7992)
是另一架構的參考，不是本機根因證明；[Microsoft 0x141](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0x141---video-engine-timeout-detected)
說明 video engine timeout，不指出其實際觸發組件。
