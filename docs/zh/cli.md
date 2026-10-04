# CLI 参考

Jasna 的 CLI 与 GUI 功能一致。`jasna --help` 始终显示完整的最新选项列表；本页补充说明和示例。

官方发行版会在处理前向标准错误输出显示启动检查。`--no-progress` 只保留
简短的启动消息。安装或更新后的首次启动需要更长时间。

```bash
# Single video
jasna --input input.mp4 --output output.mkv

# Still image (routes to SD 1.5 automatically)
jasna --input photo.png --output restored.png

# Whole folder (images first, then videos)
jasna --input input_folder --output output_folder
```

在 Windows 上，CLI 与应用是同一个文件: `jasna.exe --input ...`。

## 通用

| 选项 | 默认值 | 说明 |
| ------ | ------- | ----- |
| `--version` | — | 打印 Jasna 版本并退出。 |
| `--input` | — | 视频、图像或文件夹。 |
| `--output` | — | 输出文件；当 `--input` 是文件夹时为输出文件夹。 |
| `--output-pattern` | `{original}_out` | 文件夹输入的文件名模板。`{original}` 是输入文件名主干。图像保留源扩展名；视频在模板提供扩展名时使用该扩展名。Jasna 会在处理前检查计划输出路径，如果两个输入映射到同一个文件则报错退出。 |
| `--device` | `cuda:0` | GPU 选择。AMD 显卡通过 ROCm 也使用相同的 `cuda:N` 名称。 |
| `--batch-size` | `4` | 检测批处理大小。旧版 `rfdetr-v5` 始终使用 4。 |
| `--fp16` / `--no-fp16` | 开启 | 在支持的环节使用 FP16（修复 + TensorRT）。降低 VRAM，可能提升速度。 |
| `--log-level` | `error` | `debug`、`info`、`warning`、`error`。 |
| `--no-progress` | 关闭 | 禁用进度条。 |

## 修复

| 选项 | 默认值 | 说明 |
| ------ | ------- | ----- |
| `--restoration-model-name` | `basicvsrpp` | 视频修复模型：`basicvsrpp`（快）、`ltx`（细节更多，慢很多）或 `ltx-undistilled`（最佳 LTX 模型：比 `ltx` 稍好，约慢 1.5 倍）。见[选择模型](models.md#修复模型)。 |
| `--restoration-model-path` | `model_weights/lada_mosaic_restoration_model_generic_v1.2.pth` | 修复模型权重（LTX 为文件夹）。 |
| `--ltx-fast` / `--no-ltx-fast` | 关闭 | LTX：快约 1.4 倍，细节略少。仅限 RTX 50 系列。 |
| `--ltx-large-canvas` / `--no-ltx-large-canvas` | 关闭 | LTX：大马赛克细节更多，这部分约慢 3 倍。需要 10 GB 空闲显存。 |
| `--ltx-seed` | 模型默认值 | LTX：换一个种子，结果会略有不同。 |
| `--ltx-trial` / `--no-ltx-trial` | 关闭 | LTX：无需模型文件即可测试速度。视频不会被修复。 |
| `--ltx-auto-split` | — | LTX：大马赛克（达到视频高度的这个百分比，例如 `12`）用 LTX，较小的用 BasicVSR++。可用于文件夹；拆分后的视频保持原编码，无法拆分的视频整段用 LTX 修复。见[区间](segments.md)。 |
| `--compile-basicvsrpp` / `--no-compile-basicvsrpp` | 开启 | TensorRT 编译: 大幅提速，占用更多 VRAM。见[调优](tuning.md)。 |
| `--max-clip-size` | `90` | 跟踪片段的最大帧数。VRAM 的主要调节手段。 |
| `--temporal-overlap` | `8` | 片段拼接处的重叠和丢弃余量；减少边界闪烁。 |
| `--enable-crossfade` / `--no-enable-crossfade` | 开启 | 使用已处理的帧交叉淡化片段边界；没有额外 GPU 开销。 |
| `--denoise` | `none` | 对修复裁切图的空间降噪: `low`、`medium`、`high`。 |
| `--denoise-step` | `after_primary` | 在二级修复之前（`after_primary`）或混合前（`after_secondary`）应用降噪。 |

## 检测

| 选项 | 默认值 | 说明 |
| ------ | ------- | ----- |
| `--detection-model` | `rfdetr-v6` | 已安装模型从 `model_weights/` 中发现；`rfdetr-v6`（快速）和 `rfdetr-vr-v1`（VR180）已内置，`rfdetr-v6-large` 和 `zelefans-vr-yolo-v2` 为可选下载。见[模型](models.md)。 |
| `--detection-model-path` | 自动 | 默认为 `model_weights/<detection-model>`，并使用适合你显卡的文件类型：RF-DETR 在 NVIDIA 上用 `.onnx`，在 AMD 上用 `.pt`；YOLO 始终用 `.pt`。 |
| `--detection-score-threshold` | 自动 | 默认使用所选模型的推荐值（`rfdetr-v6`：0.35，`rfdetr-v6-large`：0.40）。漏检马赛克时调低；正常区域被误检时调高。 |
| `--max-detection-gap` | `2` | 当马赛克在相同位置重新出现时，填补最多 N 帧的检测中断。`0` 表示禁用。 |
| `--min-detection-duration` | `2` | 丢弃持续少于 N 帧的检测（视为误检，相应帧保持原样）。`0` 表示禁用。 |
| `--scene-detection` | 开 | 检测硬切镜头并在切换点结束所有跟踪中的马赛克片段，使片段不会跨越两个镜头。使用 `--no-scene-detection` 关闭。 |

## 二级修复

| 选项 | 默认值 | 说明 |
| ------ | ------- | ----- |
| `--secondary-restoration` | `none` | `unet-4x`、`tvai` 或 `rtx-super-res`。见[模型](models.md)。 |
| `--rtx-scale` | `4` | RTX Super Res 放大倍数（`2` 或 `4`）。 |
| `--rtx-quality` | `high` | `low`–`ultra`。 |
| `--rtx-denoise` | `medium` | `none` 表示禁用。 |
| `--rtx-deblur` | `none` | `none` 表示禁用。 |
| `--tvai-ffmpeg-path` | Topaz 默认安装路径 | Topaz Video `ffmpeg.exe` 的路径。 |
| `--tvai-model` | `iris-2` | 例如 `iris-2`、`prob-4`、`iris-3`。 |
| `--tvai-scale` | `4` | 输出尺寸为 `256*scale`；`1` = 不放大。 |
| `--tvai-args` | 见 `--help` | 额外的 `tvai_up` 参数。 |
| `--tvai-workers` | `2` | 并行的 TVAI ffmpeg 工作进程数。 |
| `--tvai-denoise` | 关闭 | 在 TVAI 增强处理前应用降噪。 |

## SD 1.5 图像修复

静态图像会自动路由到这里；`--restoration-model-name` 仅用于视频。

| 选项 | 默认值 | 说明 |
| ------ | ------- | ----- |
| `--image-restoration-model-name` | `sd-15-jav` | 目前唯一的值。 |
| `--sd15-steps` | `25` | 扩散步数。 |
| `--sd15-strength` | `0.6` | SDEdit 去噪强度，限制为 `<= 0.7`。 |
| `--sd15-freeu` / `--no-sd15-freeu` | 开启 | FreeU UNet 调整。 |
| `--sd15-seed` | `0` | 基础种子。 |
| `--sd15-variants` | `1` | 使用种子 `seed..seed+N-1` 生成 N 个变体；保留最好的。 |

## VR

| 选项 | 默认值 | 说明 |
| ------ | ------- | ----- |
| `--vr-mode` | `auto` | `auto`、`off`、`sbs`、`sbs-fisheye`。见 [VR180](vr180.md)。 |

## 编码

| 选项 | 默认值 | 说明 |
| ------ | ------- | ----- |
| `--codec` | `hevc` | 离线输出可选 `hevc`、`h264` 或 `av1`。HLS 流媒体始终使用 H.264。 |
| `--cq` | 根据 GPU/编解码器 | 原样传给编码器的质量目标。越低质量越好、文件越大。NVIDIA 默认值：H.264 25、HEVC 28、AV1 35；AMD 默认值：H.264 24、HEVC 25、AV1 32。 |
| `--encoder-settings` | — | JSON 对象或逗号分隔的高级 `key=value` 设置，例如 `{"rc-lookahead":32}` 或 `rc-lookahead=32,bf=4`。见下文。 |
| `--lut` | — | `.cube` 色彩 LUT（1D 或 3D），编码前由 GPU 应用。也可在 GUI 的编码设置部分设置。 |
| `--sharpen` | `0` | 编码前锐化画面，取值 `0`（关闭）到 `1`（最强）。与 ffmpeg 的 `cas` 滤镜一致，无需二次转码。见[高级处理](advanced_processing.md)。 |
| `--retarget-high-fps` | 关闭 | 通过每两帧处理一帧实现 60 → 30 FPS（以及 59.94 → 29.97）。其他帧率不变；音频时序保持不变。 |
| `--fmp4` | 关闭 | `.mp4`/`.mov` 输出在生成过程中即可播放，任务中断后仍可播放。不能与 `--stream` 或 `--segments` 同时使用。见[高级处理](advanced_processing.md)。 |
| `--segments` | — | 只修复选定区间，例如 `10-25,01:10-01:30.5`。不能与 `--stream`、`--retarget-high-fps` 或 `--fmp4` 同时使用。见[区间](segments.md)。 |
| `--working-directory` | 输出目录 | 区间临时文件的写入位置。见[区间](segments.md)。 |

### 选择编解码器

- **`hevc`**（默认）: 质量和文件大小的最佳平衡，以 10-bit 编码。所有
  现代设备和播放器都能播放。除非有特殊原因，建议使用它。
- **`h264`**: 兼容性最强（老电视、浏览器、剪辑软件），仅支持 8-bit，
  相同质量下文件更大。也是流媒体使用的编解码器。
- **`av1`**: 压缩率最高 — 相同质量下文件最小，10-bit。需要支持 AV1
  编码的 GPU（NVIDIA RTX 40 系列或更新）和较新的播放器。

使用 `--segments` 时，编解码器锁定为输入视频的编解码器，`--codec` 不
生效。

### 编码器设置

`--cq` 是主要的质量控制。GUI 显示或命令行输入的数值会原样传给当前编码器（AMD AV1 会乘以 5）；
切换编解码器不会转换数值。数值越低，质量越好，文件越大。

| GPU | H.264 默认值 | HEVC 默认值 | AV1 默认值 | 允许范围 |
| --- | ---: | ---: | ---: | --- |
| NVIDIA | 25 | 28 | 35 | H.264/HEVC 为 1–51；AV1 为 1–63 |
| AMD | 24 | 25 | 32 | H.264/HEVC 为 0–51；AV1 为 1–51 |

NVIDIA 将 CQ 0 保留为自动值，因此 Jasna 要求显式质量目标从 1 开始。在编辑当前
任务期间，GUI 会分别记住每个编解码器的原始数值。

`--encoder-settings` 用于微调其他硬件编码器设置。参数会根据当前编码器进行
校验 — 不支持的参数会失败，并给出清晰的错误提示，列出该编码器接受的参数:

```bash
# 要提高质量（并增大文件），请降低 CQ。
jasna --input in.mp4 --output out.mkv --cq 22

# CQ 与高级设置
jasna --input in.mp4 --output out.mkv --cq 22 --encoder-settings "rc-lookahead=32,bf=4"
```

为保持兼容，省略 `--cq` 时仍可在 `--encoder-settings` 中使用 `cq=22`。若通过
两个入口同时指定 CQ，Jasna 会报错，而不会静默选择其中一个。GUI 中的 CQ 控件是
唯一入口，因此**自定义参数**不接受 CQ 别名。

#### NVIDIA (NVENC) 参数 — 所有编解码器

| 参数 | 作用 |
| --- | ------------ |
| `cq` | VBR 的目标质量。越低 = 质量越好、文件越大。H.264/HEVC 的原值范围为 1–51（默认 25/28），AV1 为 1–63（默认 35）。输出体积自动上限可能让相近的数值得到相同结果。 |
| `preset` | 速度/质量权衡，从 `p1`（最快）到 `p7`（最佳）。默认 `p5`。 |
| `tune` | `hq`（默认）、`ll`、`ull` 或 `lossless`。 |
| `rc` | 码率控制模式: `vbr`（默认）、`cbr`、`constqp`。 |
| `qmin` / `qmax` | VBR 的质量下限/上限。默认 17/34（仅 H.264/HEVC；AV1 使用不同的 0–255 QP 范围，不设置这两项）。 |
| `init_qpI` / `init_qpP` / `init_qpB` | 各帧类型的初始量化值。默认 17（H.264/HEVC）。 |
| `g` | 关键帧间隔（帧数）。默认 250。越小 = 跳转越流畅，文件越大。 |
| `bf` | 最大连续 B 帧数。默认 4。 |
| `b_ref_mode` | 把 B 帧用作参考帧: `disabled`、`each`、`middle`（默认）。 |
| `b_adapt` | 自适应 B 帧放置。 |
| `nonref_p` | 非参考 P 帧，默认开启。 |
| `spatial_aq` / `spatial-aq` | 空间自适应量化 — 把码率花在肉眼更敏感的区域。默认开启。AV1 只接受带连字符的写法。 |
| `temporal-aq` | 时间自适应量化。默认开启。 |
| `aq-strength` | AQ 强度，1–15。默认 8。 |
| `rc-lookahead` | 码率控制的前瞻分析帧数。默认 32。 |
| `lookahead_level` | 前瞻质量，0–3。仅 HEVC/AV1 — 在 H.264 上会被忽略并警告（编码器无法使用它）。 |
| `maxrate` / `bufsize` | 码率上限和 VBV 缓冲区大小（比特/秒）。Jasna 会根据源码率自动设置（见下文）；自行指定 `maxrate` 则以你的值为准。 |
| `multipass` | 两遍编码: `disabled`、`qres`、`fullres`。 |
| `weighted_pred` | 加权预测。NVENC 仅在 `bf=0` 时支持；否则（以及 AV1 上始终）会被忽略并警告。 |
| `tf_level` | 时间滤波级别。 |

#### 输出体积自动上限

`cq` 只针对固定质量，不考虑源文件的存储方式，因此低码率保存的源会被重新编码到远高于
其自身质量的水平，体积膨胀数倍。为此，Jasna 会根据源视频码率推导 `maxrate`，并将
`bufsize` 设为其两倍：

| 情况 | 上限 |
| ---- | ---- |
| NVIDIA H.264 输出 | 源视频码率的 2.0 倍 |
| 其他输出，HEVC 源 | 源视频码率的 1.25 倍 |
| 其他组合 | 源视频码率的 1.0 倍 |

NVIDIA H.264 获得更多余量，因为保留修复后的细节需要更多码率。该上限只对低码率保存的源
生效；码率充足的源不受影响。达到上限时，CQ 仍是目标质量，但相近的数值可能得到相同的
码率和文件大小。

自行指定 `maxrate` 即可替换；设为很大的值可实际停用。若源完全没有报告码率，Jasna 会
记录警告并在无上限的情况下编码。
AMD 的 10-bit HEVC/AV1 输出使用固定 QP，因此没有自动体积上限。


各编解码器额外参数:

| 编解码器 | 额外参数 |
| ----- | ---------- |
| `hevc` | `profile`（`main`、`main10` — 默认 `main10`）、`tier` |
| `h264` | `profile`（`baseline`、`main`、`high` — 默认 `high`）、`coder`（`cabac`/`cavlc`） |
| `av1` | `tier`、`tile-rows`、`tile-columns`（并行解码大分辨率画面） |

#### AMD (AMF) 参数 — 所有编解码器

| 参数 | 作用 |
| --- | ------------ |
| `cq` | AMF 的质量值：H.264 使用 QVBR，10-bit HEVC/AV1 使用固定 QP（AV1 内部刻度更细，CQ 会乘以 5 后传入）。越低越好；默认值依次为 24、25、32。 |
| `qvbr_quality_level` | 旧版 CQ 别名。省略 `--cq` 时可用于 CLI 高级设置；GUI 自定义参数中不接受。 |
| `usage` | 编码器用途配置。默认 `high_quality`。 |
| `quality` | 速度/质量预设: `speed`、`balanced`、`quality`（默认）。 |
| `rc` | 码率控制模式。H.264 默认 `qvbr`；10-bit HEVC/AV1 默认 `cqp`。 |
| `preset` | AMF 预设。 |
| `g` | 关键帧间隔（帧数）。默认 250。 |
| `bf` | 最大连续 B 帧数。 |
| `preanalysis` | 预分析。H.264 默认开启，10-bit HEVC/AV1 默认关闭。 |
| `maxrate` / `bufsize` | 码率上限和 VBV 缓冲区大小（比特/秒）。除 CQP 模式外，会根据源码率自动设置。 |
| `profile` / `level` | 编解码器 profile 和 level。 |

各编解码器额外参数:

| 编解码器 | 额外参数 |
| ----- | ---------- |
| `hevc` | `tier`、`bitdepth`（默认 10）、`vbaq`（默认关闭） |
| `h264` | `coder`、`bf_ref`（B 帧引用）、`pa_adaptive_mini_gop`（自适应 B 帧排列）、`vbaq`（默认开启） |
| `av1` | `bitdepth`（默认 10）、`aq_mode`（默认关闭） |

## 流媒体

| 选项 | 默认值 | 说明 |
| ------ | ------- | ----- |
| `--stream` | 关闭 | HLS 流媒体模式，不输出文件。见[流媒体](streaming.md)。 |
| `--stream-port` | `8765` | HTTP 端口。 |
| `--stream-segment-duration` | `4.0` | HLS 分段长度（秒）。 |
| `--no-browser` | 关闭 | 不打开浏览器窗口。 |

## 导出后操作

| 选项 | 默认值 | 说明 |
| ------ | ------- | ----- |
| `--post-export-action` | `none` | `shutdown` 或 `command`，在所有导出完成后运行。 |
| `--post-export-command` | — | `--post-export-action command` 使用的 shell 命令。 |
| `--post-export-video-command` | — | 每个视频成功导出后运行的 shell 命令。支持 `{input}`、`{output}`、`{output_dir}`、`{output_stem}` 和 `{output_suffix}`。 |

```bash
jasna --input input.mp4 --output output.mkv --post-export-action shutdown
jasna --input folder_in --output folder_out --post-export-action command --post-export-command "echo done"
jasna --input folder_in --output folder_out --post-export-video-command "ffmpeg -i {output} -map 0 -map_metadata 0 -map_chapters 0 -c copy -movflags +faststart {output_dir}/{output_stem}_remuxed{output_suffix}"
```

## 激活

| 选项 | 说明 |
| ------ | ----- |
| `--machine-key` | 显示机器密钥后退出，无需启动 GPU 或 GUI。 |
| `--license-code CODE` | 在 GPU 初始化之前，使用官方卖家的许可证码在线激活。未指定处理选项时，激活后退出。 |
| `--cloud-workspace` | 与 `--machine-key` 或 `--license-code` 一起使用，创建或复用云工作区身份。卖家须为许可证码开通云权限。 |
| `--activation-key KEY` | 保存本机的激活密钥。未指定处理选项时，激活后退出。 |

在线激活使用 `--license-code`，仅通过 HTTPS 向[激活服务](https://jasna-activation.kruk2.workers.dev)发送许可证码和本机的机器密钥。Jasna 确认返回的**激活密钥**适用于本机后，只保存此密钥，不保存许可证码。之后启动和修复时离线验证激活，不会定期联网检查。

Jasna 语言设为中文时优先使用[香港服务](https://47.238.135.110)，其他语言优先使用原服务。无法连接时会尝试另一个地址。两个地址共用许可证码和 3 个设备名额。

```bash
jasna --license-code "YOUR_LICENSE_CODE"
jasna --license-code "YOUR_LICENSE_CODE" --input input.mp4 --output output.mkv
```

手动激活时，复制机器密钥，在任意联网设备上打开同一页面，输入机器密钥和许可证码，或向官方卖家申请。保存返回的激活密钥：

```bash
jasna --machine-key
jasna --activation-key "YOUR_ACTIVATION_KEY"
```

`--license-code` 不能与 `--machine-key` 或 `--activation-key` 同时使用。`--license-code` 和 `--activation-key` 均可与 `--input`、`--output` 一起使用，激活后直接处理。

在 RunPod 或 Vast 上，请先向官方卖家申请云权限，并在激活前将设置目录保存在持久卷上。Linux 的 `XDG_CONFIG_HOME` 指定用于存放 `jasna/` 的目录。选择在线或手动激活：

```bash
# 在线
XDG_CONFIG_HOME=/workspace/config jasna --license-code "YOUR_LICENSE_CODE" --cloud-workspace

# 手动
XDG_CONFIG_HOME=/workspace/config jasna --machine-key --cloud-workspace
XDG_CONFIG_HOME=/workspace/config jasna --activation-key "YOUR_ACTIVATION_KEY"
```

之后运行也须使用同一设置目录。主机或 GPU 更换后仍复用同一名额；将设置目录和激活密钥一起复制，也会转移身份。

两种方式共用每个许可证码的 **3 台机器或工作区**名额，相同机器密钥的重复请求不占新名额。只有所有者可以释放名额或批准更换。激活密钥只包含已购买的模型版本；旧版激活格式的密钥须向官方卖家申请替换。

在线激活失败时，请从另一台联网设备手动激活，或将许可证码和机器密钥发送给官方卖家申请。服务无法保证在中国大陆等地区可用；同样适用 3 个名额的限制。

## 基准测试

| 选项 | 说明 |
| ------ | ----- |
| `--benchmark` | 运行基准测试而不是处理。 |
| `--benchmark-filter` | 只运行名称包含此字符串的基准测试。 |
| `--benchmark-video` | 基准测试使用的视频路径；可重复指定。 |
