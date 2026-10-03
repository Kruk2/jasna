# Windows AMD HEVC 源码率与 GUI 待办

状态：`Windows 完整编码诊断开关已通过真机短视频验收；Smart Render 在固定 runtime 上禁用，GUI 默认仍为 TODO`

Linux AMD HEVC 的编码设置提供 `encoder_rate_mode` 上级选项。`auto_source` 是默认值，Smart
Render 自动使用源码率 `vbr_peak` 合同并隐藏 CQ；`manual_cq` 显示 CQ，并显式传递
`rc=cqp`，保证 Smart Render 不进入自动源码率路线。Linux GUI 的完整视频编码也已贯通
`auto_source`；CLI 未开启实验开关时的完整视频编码仍使用既有 CQ/CQP 行为。
该选项目前只在 Linux AMD HEVC 组合显示；其他编码器、编码格式和平台继续直接显示 CQ。

## Windows VBR Peak 诊断开关（默认关闭）

Windows AMD HEVC 自动源码率仍不开放。为进行真机短视频验收，只有显式设置
`JASNA_AMF_HEVC_VBR_PEAK=1` 时，Windows AMD HEVC **完整编码**可以试用源码率
`vbr_peak`；未设置、`auto` 及 GUI 的 `auto_source_rate` 仍保持既有 CQP 行为，不会改变
产品默认。

该诊断路线要求正的源码视频码率。它将 target 精确设为源码率、peak 设为 target 的
1.25 倍、buffer 设为 target 的 2 倍，并强制 `rc=vbr_peak`、`preanalysis=0`、`vbaq=0`，移除
QP 设置；自定义 `maxrate`、`bufsize` 或冲突的 `rc` 会明确失败，避免混合码率合同。

Windows Smart Render 不能使用该显式开关：固定 runtime 上三次 Main8 8K/600 帧 CQP
基线均完成正确的 3 fragment copy/render/copy 组装后，在 window 1/2 的产品内 FFmpeg
`framemd5` 子进程无限空闲。限制 seek 范围和把 `framemd5` 重定向到具名文件均未修复；
相关 splice 实验已完整回退。故在该 runtime 上即使设置
`JASNA_AMF_HEVC_VBR_PEAK=1` 也会明确失败，不能把未完成的 CQP 接缝基线用于 VBR 比较。
临时组装和日志在 `runs/m8-600-smart-cqp-win`、`runs/m8-600-smart-cqp-win-v2`、
`runs/m8-600-smart-cqp-win-v3`、`runs/m8-600-smart-cqp-win-v4`；不改变 Windows GUI 或
默认 CQP，也不将完整编码结果外推到 Smart Render。

### 2026-09-05 完整编码真机结果

RX 7900 XTX / gfx1100、HIP 7.2、固定 Windows unified runtime 上，Main/NV12 与
Main10/P010 均先通过 60 帧功能探针，再完成 600 帧产品路线。每个正式输出均通过 600
帧、profile/pix_fmt、PTS/DTS、3 个 IDR、VPS/SPS/PPS 和独立 FFmpeg 严格软件解码验证；
Main10 的 AAC 音频结构保持，样本均无字幕流。

| 600 帧路线 | CQP 两次均值 | VBR Peak | 墙钟变化 | CQP 大小 | VBR 大小 | 大小变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Main/NV12 | 118.28 s | 119.58 s | +1.10% | 23,944,833 B | 17,047,732 B | -28.80% |
| Main10/P010 | 161.68 s | 166.47 s | +2.96% | 40,188,492 B | 18,198,719 B | -54.72% |

Main/NV12 日志合同为 target/peak/buffer = 14,180,827 / 17,726,033 /
28,361,654 bit；输出平均码率 13,621,688 bit/s，最大 1 秒 packet window
16,487,192 bit/s。Main10/P010 为 14,773,026 / 18,466,282 / 29,546,052 bit；
输出平均码率 14,277,400 bit/s，最大 1 秒 packet window 17,409,304 bit/s。
两者均未出现 AMF 码控、InitialVBVBufferFullness 或 PreAnalysis 警告。

相对同源 CQP 解码像素，Main/NV12 为 PSNR 50.317 dB、SSIM 0.995629；Main10/P010
为 PSNR 47.906 dB、SSIM 0.993864。该结果支持保留默认关闭的完整编码诊断开关，但不能
外推到 Smart Render，且 VBR 增加约 1.1%–3.0% 墙钟，不是编码性能优化。

机器可读证据：
`D:\AI\amf-unified-work\transactions\jasna-windows-amf-vbr-peak-acceptance-20260905\REPORT.json`

Windows Smart Render 需先修复或更换固定 runtime 并完成 CQP 接缝基线验证，才可重新评估
fragment VBR；在此之前保持 fail-closed。

全部通过后才能在 Windows AMD HEVC 复用同一 GUI 自动选项；不能只依据 AMF 声明支持
`vbr_peak` 就改变产品默认。

## Windows AMD 原生双 VCN split-frame 待办

Windows 真机还应单独验证 AMF 1.4.35 增加的 HEVC 原生 split-frame。它通过
`AMF_VIDEO_ENCODER_HEVC_MULTI_HW_INSTANCE_ENCODE` 把同一帧横向分片交给多个 VCN，
与 Linux 候选的“两个独立编码会话按封闭 GOP 分工”不是同一种实现，不能共用结论。

AMD 官方工程师给出的当前约束是：

- 只支持 DX11 memory mode，Linux Host/Vulkan/P010 路线不能用该能力；
- 该属性只是给驱动的建议，驱动检查不通过时可以忽略；
- `HIGH_MOTION_QUALITY_BOOST_ENABLE`、`PREENCODE_ENABLE` 或 filler data 开启时禁用；
- 输出必须是 frame mode，picture-transfer mode 必须关闭；
- 码控仅限 CQP、Peak-Constrained VBR 或 CBR；
- 需 4K 以上等驱动条件；RX 7900 XTX 的 AV1 只有一个 VCN，故本待办先限 HEVC。

参考：

- [AMD AMF 1.4.35 版本记录](https://github.com/GPUOpen-LibrariesAndSDKs/AMF#version-history)
- [AMD 关于 split-frame 启用条件的说明](https://github.com/GPUOpen-LibrariesAndSDKs/AMF/issues/585#issuecomment-4165598416)
- [AMD HEVC 编码 API](https://github.com/GPUOpen-LibrariesAndSDKs/AMF/blob/master/amf/doc/AMF_Video_Encode_HEVC_API.md)

Windows 验收步骤：

1. 在固定 AMF/FFmpeg runtime 中补充并验证显式 `multi_hw_instance` 开关；默认关闭，
   且 FFmpeg/PyAV 未接受该参数时必须失败，不能静默继续。当前 upstream FFmpeg 的
   `hevc_amf` 选项表尚未公开该属性，需先实现受版本约束的 FFmpeg 补丁或直接 AMF 接口，
   不能把一个未知参数当作已启用。
2. 使用原生 DX11 输入面，分别运行明确关闭和请求开启的相同 4K/8K HEVC Main、
   Main10 素材；沿用自动源码率时使用 `vbr_peak`，禁止 PreAnalysis/preencode、
   high-motion boost 和 filler data。
3. 不以任务管理器的一条 Video Codec 曲线判断是否使用双 VCN。记录 AMF 日志、GPUView
   或等价底层证据，并检查输出 slice 结构，证明驱动实际接受了 split-frame 提示。
4. 比较墙钟时间、首帧/平均延迟、文件体积、码率、质量指标和显存；还要验证帧数、
   持续时间、PTS/DTS、VPS/SPS/PPS、严格软件解码和 Smart Render copy/render 接缝。
5. 只有在两种 bit depth 和真实 Smart Render 路线都正确、速度收益稳定且画质/体积无
   明显回退时，才考虑 Windows AMD HEVC 自动启用；否则只保留诊断开关。
