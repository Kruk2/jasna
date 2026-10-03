# Linux 與 Windows 功能審查索引

[English (default)](../en/feature_reviews.md) | 中文

本索引涵蓋以上游 main `81dc8b053fb317c063390daab1dab8289c2094df` 為基準的 26 項單功能變更。每項都有英文指南及中文對應版本；英文是預設審查／文件入口。歷史中文工程記錄保留作補充證據，不替代這些指南。

## 功能與直接前置

前置數字對應此表，不是 GitHub PR 編號。沒有依賴不代表硬體已認證。

| 順序 | 功能 | 直接前置 |
| --- | --- | --- |
| 1 | [Windows ROCm 匯入與顯卡廠商相容](changes/03-windows-rocm-compat.md) | 無 |
| 2 | [HIP 色彩轉換核心](changes/04-hip-colour.md) | 1 |
| 3 | [共用原生任務與診斷契約](changes/00-shared-native-job-contracts.md) | 1、2 |
| 4 | [固定版本統一媒體 runtime 與安裝器](changes/01-runtime-contract.md) | 無 |
| 5 | [可重現 FFmpeg 與 PyAV 建置](changes/02-runtime-build.md) | 4 |
| 6 | [可選 Windows HIP 縮放與正規化](changes/05-windows-hip-resize.md) | 2 |
| 7 | [有界 Windows D3D11–HIP 常駐媒體](changes/07-windows-resident-media.md) | 4、2 |
| 8 | [AMF 原生解碼與幀所有權](changes/06-amf-native-decode.md) | 3、4、2、7 |
| 9 | [AMD 編碼契約與匹配源碼率 Peak VBR](changes/08-encoder-source-rate.md) | 2、7 |
| 10 | [Smart Render 接縫與持久續跑](changes/13-smart-render-resume.md) | 8、9 |
| 11 | [有界 Linux HEVC 雙 GOP 編碼](changes/09-dual-gop.md) | 9、10 |
| 12 | [經驗證 RF-DETR MIGraphX 選擇](changes/10-rfdetr-migraphx.md) | 6 |
| 13 | [經驗證 BasicVSR++ MIGraphX B1 修復](changes/11-basicvsrpp-migraphx.md) | 1、12 |
| 14 | [公開源碼可選授權邊界](changes/19-public-source-license.md) | 無 |
| 15 | [共用流水線資源與失敗安全](changes/12-pipeline-resource-safety.md) | 3、8、7、9、11、13、10、14 |
| 16 | [自適應自動預掃描與漏區間修復](changes/14-automatic-prescan.md) | 8、12 |
| 17 | [保持輸入資料夾與輸出續跑驗證](changes/15-preserved-folder-outputs.md) | 10 |
| 18 | [原生視訊任務隔離與持久輸出](changes/16-isolated-video-jobs.md) | 3、4、2、8、11、15、10、16、17 |
| 19 | [GUI 控制、診斷與可靠進度](changes/17-gui-settings-diagnostics.md) | 3、18、14 |
| 20 | [精確 VR 片商與投影路由](changes/18-vr-projection-studios.md) | 無 |
| 21 | [純 CPU SD 1.5 回歸隔離](changes/20-cpu-regression-isolation.md) | 19 |
| 22 | [明確啟用 AMD 效能與容量探測](changes/21-performance-probes.md) | 15 |
| 23 | [身分限定 Windows AMD Math SDPA](changes/23-windows-sdpa-compat.md) | 1、12 |
| 24 | [特定 Windows AMF 傳輸失敗後隔離](changes/24-native-context-quarantine.md) | 8、15、18 |
| 25 | [排列感知 RF-DETR 精度診斷](changes/25-rfdetr-precision-probe.md) | 23 |
| 26 | [整合、依賴與驗收記錄](changes/22-integration-record.md) | 1、2、3、4、5、6、7、8、9、10、11、12、13、14、15、16、17、18、19、20、21、22、23、24、25 |

## 提交方式與共用架構

任務／GUI／掃描／修復／輸出協調流程跨廠商與作業系統共用；原生解編碼、GPU 核心、身分驗證及故障恢復由各自介接器負責。不恢復 rocDecode。

源碼審查分支栈是累計的。跨 fork 的上游 PR 不能使用只存在 fork 中的基底分支。應按依賴順序對上游 main 提交單功能差異，前置合併後 rebase 並重測；不能靜默將累計後綴當成獨立功能提交。無前置的根功能為 Windows ROCm 相容、runtime 契約、公開源碼授權邊界及 VR 片商路由。

## 證據與限制

先前精確源碼整合集合在 Linux 隔離 CPU 環境通過 **3141 項測試、225 項跳過、178 項子測試**。此次文件更新不改執行行為。這是整合集合結果，不代表 26 個無前置獨立分支各自全部通過。

歷史 Linux 原生驗收涵蓋 13 個任務（9427.04 秒），在記錄的處理基準上完成嚴格解碼／時間軸／接縫檢查，不能自動認證後來新增的 Windows 變更。此次文件更新不執行真實影片／GPU 負載。Windows SDK／原生資產重建、全卡遙測、實機 A/B 為 **WAIVED_BY_USER_NOT_RUN**，不是 PASS；其他未測 Windows／NVIDIA／LTX 與付費模型 AMD 路線仍未認證。

私有 protection 源碼、憑證、權重、生成媒體、已安裝 runtime 及使用者設定均不屬公開 PR。模擬授權測試通過不等於解鎖付費功能。文件／提交工作不切換現用 GUI 啟動器。
