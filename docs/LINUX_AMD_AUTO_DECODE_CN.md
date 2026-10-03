# Linux AMD 产品自动解码路由

更新时间：2026-08-30

## 决策范围

`JASNA_DECODE_BACKEND=auto` 只在以下条件全部满足时，把既有 Linux AMD PyAV/AMF
host 路径替换为 AMF Vulkan → HIP D2D：

- 平台是 Linux，设备 vendor 是 AMD；
- codec 是 H.264 Main/High 8-bit、HEVC Main 8-bit、HEVC Main10 10-bit，或
  AV1 Main 8-bit NV12 / Main 10-bit P010；
- reader batch 是既有产品允许的 B1/B2/B4/B8；
- unified runtime 中存在通过 source/binary hash、Python SOABI 与 entry-point preflight
  的 AMF bridge。

H.264/HEVC 使用既有 `private-deferred`；AV1 使用单 reader epoch 的稳定 dma-buf
identity cache。它是一项 backend substitution，不调整共享 `auto` 判定顺序、VR/2D 识别、Tracker、
Pipeline、检测/修复阈值、NVIDIA VALI/NVDEC 或 Windows AMD policy。产品默认仍是 B4；
B8 仍只由 `--batch-size 8` 显式选择。

Linux AMD AV1 在完成 8/10-bit、4K/8K、10,000 帧生命周期、PTS/hash、stop/close 与
资源配平后进入自动 D2D。产品 cache 的长稳态吞吐为 `924.696 fps`，已知同口径
旧 rocDecode 基准为 `1137.729 fps`；用户明确接受 `18.72%` 差距。后续独立删除阶段
已经移除旧后端、运行时 bridge 和自动 fallback。native gate 外 AV1 保持普通 PyAV
产品路线；PyAV 打开或读取失败时直接保留原错误，不再切换第二套 AMD 解码实现。

## 失败与资源边界

一旦合格 H.264/HEVC 被 `auto` 选入 native route，bridge 缺失、dependency probe 失败、
native surface/format/context 改变、HIP/Vulkan copy 失败或 teardown 不配平都会作为终止
错误上报。不会静默改走软件解码、CPU Map、host transfer、staging、D2H 或其他后端。
合格 AV1 同样 fail closed，并固定启用稳定 dma-buf identity cache；H.264/HEVC 不接受
cache-on。runtime preflight 还必须确认 bridge session 暴露 cache copy/close/stats API，
避免旧 bridge 预检通过后到首次 AV1 解码才失败。

显式 `JASNA_DECODE_BACKEND=amf-interop` 仍用于诊断全部已支持格式；它继续遵守
`JASNA_AMF_INTEROP_DECODE_COPY_STREAM`，默认 `null`。产品 `auto` 对 H.264/HEVC 则固定
使用已验收的 `private-deferred`，不依赖 shell 中遗留的实验变量。

## Linux 验收

接入时焦点回归覆盖自动选择矩阵、AV1/unsupported 保持原顺序、native failure terminal、
Windows/NVIDIA isolation 和当时的兼容 fallback，结果为 `152 passed, 1 skipped`。最终
删除阶段另行验证旧环境值 fail-closed、PyAV 错误直返和既有 Linux/Windows/NVIDIA 路由。

在 RX 7900 XTX 上通过独立安装且 preflight 成功的 unified runtime，以不设置任何 decode
或 copy-stream 环境变量的产品 `auto`、B4 完整读取两份静态 fixture：

| 输入 | 结果 | PTS | native lifecycle |
|---|---:|---|---|
| H.264 High 8-bit 4096×2048 | 120/120 | 与 FFprobe 完全相同且严格递增 | stream 1/1、event 6/6、source 120/120 |
| HEVC Main 8-bit 4096×2048 | 120/120 | 与 FFprobe 完全相同且严格递增 | stream 1/1、event 6/6、source 120/120 |
| HEVC Main10 8192×4096 | 120/120 | 与 FFprobe 完全相同且严格递增 | stream 1/1、event 6/6、source 120/120 |

两次运行 final in-flight 都为 0，cache hit/miss 为 0/0，host/CPU Map/staging/D2H/failed
bridge 全为 0。运行窗口没有 GPU reset、ring timeout、page fault 或 OOM；结束时 junction
65°C、memory 66°C，D 盘仍高于 25 GiB。

AV1 cache 的完整产品接入与 Windows 边界见 `docs/AMF_AV1_NATIVE_CN.md` 最后一节。
