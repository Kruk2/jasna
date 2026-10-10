# AMF 原生解碼與幀所有權

[English (default)](../../en/changes/06-amf-native-decode.md)

功能: `06-amf-native-decode`. 源碼基準: upstream main `81dc8b053fb317c063390daab1dab8289c2094df`.

## 目的

使用 Linux AMF Vulkan→HIP 裝置直傳路線，明確管理 surface／快取／檔案描述符生命週期及同步。解碼 seek／源幀生命週期正確性與 Windows 保護留在後端邊界。

## 使用方式與預設行為

使用共用 reader／後端選擇，而非掃描專用解碼器。核對日誌中的 D2D 計數、幀順序、原生關閉錯誤及格式准入。明確選用的原生後端違反契約時必須顯式失敗。

## 直接前置

- [共用原生任務與診斷契約](00-shared-native-job-contracts.md)
- [固定版本統一媒體 runtime 與安裝器](01-runtime-contract.md)
- [HIP 色彩轉換核心](04-hip-colour.md)
- [有界 Windows D3D11–HIP 常駐媒體](07-windows-resident-media.md)

依賴描述整合實作；PR 必須保持單功能範圍。未合併的前置不等於原生驗收。

## 安全與支援限制

rocDecode 已移除，不得重新引入。Linux Vulkan／HIP 和 Windows D3D11／HIP 是不同介接器；Linux 驗收不等於 Windows 組合已認證。

## 驗證與重現

```bash
python -m pytest -q tests/test_amf_interop_core.py tests/test_decoder_source_lifetime.py tests/test_video_decoder_backends.py tests/test_video_decoder_seek.py tests/test_video_decoder_software.py
```

先前精確源碼整合集合在 Linux 隔離 CPU 環境通過 **3141 項測試、225 項跳過、178 項子測試**。此次文件更新不改執行行為。這是整合集合結果，不代表 26 個無前置獨立分支各自全部通過。

歷史 Linux 原生驗收涵蓋 13 個任務（9427.04 秒），在記錄的處理基準上完成嚴格解碼／時間軸／接縫檢查，不能自動認證後來新增的 Windows 變更。此次文件更新不執行真實影片／GPU 負載。Windows SDK／原生資產重建、全卡遙測、實機 A/B 為 **WAIVED_BY_USER_NOT_RUN**，不是 PASS；其他未測 Windows／NVIDIA／LTX 與付費模型 AMD 路線仍未認證。

## 實作與詳細記錄

- [docs/AMF_AV1_NATIVE_CN.md](../../../docs/AMF_AV1_NATIVE_CN.md)
- [docs/AMF_INTEROP_CORE_CN.md](../../../docs/AMF_INTEROP_CORE_CN.md)
- [docs/AMF_INTEROP_EVENT_POOL_CN.md](../../../docs/AMF_INTEROP_EVENT_POOL_CN.md)
- [docs/LINUX_AMD_AMF_CACHE_FD_LIFECYCLE_CN.md](../../../docs/LINUX_AMD_AMF_CACHE_FD_LIFECYCLE_CN.md)
- [docs/LINUX_AMD_AUTO_DECODE_CN.md](../../../docs/LINUX_AMD_AUTO_DECODE_CN.md)
- [docs/ROCDECODE_REMOVAL_CN.md](../../../docs/ROCDECODE_REMOVAL_CN.md)
- [docs/SHARED_DECODER_SOURCE_LIFETIME_ACCEPTANCE_CN.md](../../../docs/SHARED_DECODER_SOURCE_LIFETIME_ACCEPTANCE_CN.md)
- [docs/WINDOWS_AMD_CORRECTNESS_CN.md](../../../docs/WINDOWS_AMD_CORRECTNESS_CN.md)
- [jasna/media/video_decoder.py](../../../jasna/media/video_decoder.py)
- [scripts/amf_surface_probe.pyx](../../../scripts/amf_surface_probe.pyx)
- [scripts/build_amf_surface_probe.py](../../../scripts/build_amf_surface_probe.py)
- [tests/test_amf_interop_core.py](../../../tests/test_amf_interop_core.py)
- [tests/test_decoder_source_lifetime.py](../../../tests/test_decoder_source_lifetime.py)
- [tests/test_video_decoder_backends.py](../../../tests/test_video_decoder_backends.py)
- [tests/test_video_decoder_seek.py](../../../tests/test_video_decoder_seek.py)
- [tests/test_video_decoder_software.py](../../../tests/test_video_decoder_software.py)
