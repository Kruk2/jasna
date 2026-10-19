# CLI 参考

命令行几乎能做 GUI 能做的所有事。只有视频播放器、LTX 种子预览、遮罩建议和交互式图片处理
仅限 GUI。`jasna --help` 总是显示所有选项。本页解释这些选项并给出示例。

安装或更新后的首次启动需要更长时间，因为 Jasna 要先检查它的文件。

```bash
# 一个视频
jasna --input input.mp4 --output output.mkv

# 静态图片（自动使用 SD 1.5）
jasna --input photo.png --output restored.png

# 整个文件夹（先处理图片，再处理视频）
jasna --input input_folder --output output_folder
```

在 Windows 上，CLI 与应用是同一个文件：`jasna.exe --input ...`。

## Windows 临时文件

Jasna 可以把临时文件放在内存盘上，例如 ImDisk（NTFS）。这样可以减少 SSD 的写入。
启动 Jasna 前设置 `TMP` 和 `TEMP`：

```powershell
$env:TMP = 'Z:\Temp'
$env:TEMP = 'Z:\Temp'
.\jasna.exe --version
```

Jasna 优先使用 `TMP`，其次使用 `TEMP`。不使用 `TMPDIR`。
`--working-directory` 单独设置区间和 LTX 临时文件的文件夹。

## 通用

| 选项 | 默认值 | 说明 |
| ------ | ------- | ----- |
| `--version` | — | 打印 Jasna 版本并退出。 |
| `--input` | — | 视频、图片或文件夹。 |
| `--output` | — | 输出文件；当 `--input` 是文件夹时为输出文件夹。 |
| `--output-pattern` | `{original}_out` | 文件夹输入的文件名模板。`{original}` 是输入文件名主干。图片保留源扩展名；视频在模板提供扩展名时使用该扩展名。Jasna 会在处理前检查计划输出路径，如果两个输入映射到同一个文件则报错退出。 |
| `--device` | `cuda:0` | GPU 选择。AMD 显卡通过 ROCm 也使用相同的 `cuda:N` 名称。 |
| `--batch-size` | `4` | 检测批处理大小。旧版 `rfdetr-v5` 始终使用 4。 |
| `--fp16` / `--no-fp16` | 开启 | 半精度：占用更少显存，通常更快，看不出画质损失。 |
| `--log-level` | `error` | `debug`、`info`、`warning`、`error`。 |
| `--no-progress` | 关闭 | 禁用进度条。 |

## 修复

| 选项 | 默认值 | 说明 |
| ------ | ------- | ----- |
| `--restoration-model-name` | `basicvsrpp` | 视频修复模型：`basicvsrpp`（快）、`ltx`（标准画质 LTX：细节更多，慢很多；推荐）、`ltx-4step`（良好画质：细节比 `ltx` 稍多，慢约 1.5 倍）、`ltx-8step`（高画质：细节比 `ltx` 更多，慢约 2.5 倍）或 `ltx-undistilled`（顶级画质：稍好，比 `ltx` 慢约 5 倍）。见[选择模型](models.md#修复模型)。 |
| `--restoration-model-path` | `model_weights/lada_mosaic_restoration_model_generic_v1.2.pth`，LTX 为 `model_weights/ltx-restore` | 修复模型权重（LTX 为文件夹）。 |
| `--ltx-fast` / `--no-ltx-fast` | 关闭 | LTX：快约 1.4 倍，细节略少。仅限 RTX 50 系列；其他 GPU 会报错停止。 |
| `--ltx-large-canvas` / `--no-ltx-large-canvas` | 关闭 | LTX：大马赛克细节更多，这部分约慢 3 倍。需要约 12 GB 空闲显存；不够时，这些马赛克按普通方式修复。 |
| `--experimental-ltx-less-overlap` / `--no-experimental-ltx-less-overlap` | 开启 | LTX（实验性）：马赛克持续超过 4 秒的场景约快 13%。关闭后使用重叠更多的旧模式。 |
| `--ltx-seed` | `20260923` | LTX：换一个种子，结果会略有不同；同一个种子会重复同样的结果。 |
| `--ltx-trial` / `--no-ltx-trial` | 关闭 | LTX：不需要模型和许可证即可测试速度。视频不会被修复。 |
| `--ltx-auto-split` | — | LTX：大马赛克（达到视频高度的这个百分比，例如 `12`）用 LTX，较小的用 BasicVSR++。可用于文件夹。不能与 `--segments` 或 `--fmp4` 一起使用。见[区间](segments.md#cli---ltx-auto-split)。 |
| `--compile-basicvsrpp` / `--no-compile-basicvsrpp` | 开启 | 为你的 GPU 准备 BasicVSR++：快约 3 倍，几乎不多占显存。首次运行需要 15–60 分钟。仅限 NVIDIA。见[调优](tuning.md)。 |
| `--max-clip-size` | `90` | 一起修复的帧数。越大越稳定、越快，但占用更多显存。在显存允许的范围内用最大的值。 |
| `--temporal-overlap` | `8` | 相邻片段共享的帧数，用于隐藏拼接处的闪烁。`8`–`20` 效果好；必须小于 `--max-clip-size` 的一半。 |
| `--enable-crossfade` / `--no-enable-crossfade` | 开启 | 使用已处理的帧交叉淡化片段边界；没有额外 GPU 开销。 |
| `--denoise` | `none` | 去除修复区域中的噪点：`low`、`medium`、`high`。 |
| `--denoise-step` | `after_primary` | 在二次修复之前（`after_primary`）或之后（`after_secondary`）去除噪点。 |

`--ltx-*` 选项需要 `--restoration-model-name ltx`、`ltx-4step`、`ltx-8step` 或 `ltx-undistilled`。
`--no-compile-basicvsrpp` 只能与 `basicvsrpp` 一起使用。

## 检测

| 选项 | 默认值 | 说明 |
| ------ | ------- | ----- |
| `--detection-model` | `rfdetr-v6` | 已安装模型从 `model_weights/` 中发现；`rfdetr-v6`（快速）、`rfdetr-vr-v1`（VR180）和 `lada-yolo-v4` 已内置，`rfdetr-v6-large` 和 `zelefans-vr-yolo-v2` 为可选下载。见[模型](models.md)。 |
| `--detection-model-path` | 自动 | 默认为 `model_weights/<detection-model>`，并使用适合你显卡的文件类型：RF-DETR 在 NVIDIA 上用 `.onnx`，在 AMD 上用 `.pt`；YOLO 始终用 `.pt`。 |
| `--detection-score-threshold` | 自动 | 取值 `0` 到 `1`。默认使用所选模型的推荐值（`rfdetr-v6`：0.35，`rfdetr-v6-large` 和 `rfdetr-vr-v1`：0.40，YOLO 模型：0.25）。漏检马赛克时调低；正常区域被误检时调高。 |
| `--max-detection-gap` | `2` | 当马赛克在相同位置重新出现时，填补最多 N 帧的检测中断。`0` 表示禁用。 |
| `--min-detection-duration` | `2` | 丢弃持续少于 N 帧的检测（视为误检，相应帧保持原样）。`0` 表示禁用。 |
| `--scene-detection` | 开 | 检测硬切镜头并在切换点结束所有跟踪中的马赛克片段，使片段不会跨越两个镜头。使用 `--no-scene-detection` 关闭。 |

## 二次修复

| 选项 | 默认值 | 说明 |
| ------ | ------- | ----- |
| `--secondary-restoration` | `none` | `unet-4x`、`tvai` 或 `rtx-super-res`。见[模型](models.md)。 |
| `--rtx-scale` | `4` | RTX Super Res 放大倍数（`2` 或 `4`）。 |
| `--rtx-quality` | `high` | `low`、`medium`、`high`、`ultra`。 |
| `--rtx-denoise` | `medium` | `none`、`low`、`medium`、`high`、`ultra`。 |
| `--rtx-deblur` | `none` | `none`、`low`、`medium`、`high`、`ultra`。 |
| `--tvai-ffmpeg-path` | Topaz 默认安装路径 | Topaz Video `ffmpeg.exe` 的路径（仅限 Windows）。 |
| `--tvai-model` | `iris-2` | 例如 `iris-2`、`prob-4`、`iris-3`。 |
| `--tvai-scale` | `4` | 输出尺寸为 `256*scale`；`1` = 不放大。 |
| `--tvai-args` | 见 `--help` | 额外的 `tvai_up` 参数。 |
| `--tvai-workers` | `2` | 并行的 TVAI ffmpeg 工作进程数。 |
| `--tvai-denoise` | 关闭 | 在 Topaz 模型运行前去除噪点。更慢，占用更多显存。 |

## SD 1.5 图片修复

静态图片会自动使用这里的设置；`--restoration-model-name` 仅用于视频。

| 选项 | 默认值 | 说明 |
| ------ | ------- | ----- |
| `--image-restoration-model-name` | `sd-15-jav` | 目前唯一的值。 |
| `--sd15-steps` | `25` | 图片被细化的次数。越多细节可能越多，但更慢。 |
| `--sd15-strength` | `0.6` | 马赛克区域被重绘的程度，最大 `0.7`。越高越能处理重马赛克，但可能不太像原图。 |
| `--sd15-freeu` / `--no-sd15-freeu` | 开启 | 通常能让形状更清晰。出现生硬边缘时关闭。 |
| `--sd15-seed` | `0` | 种子。同样的种子和设置会得到同样的结果。 |
| `--sd15-variants` | `1` | 保存 N 个版本，命名为 `name_v1` … `name_vN`（种子 `seed` … `seed+N-1`）。由你自己挑选最好的。 |

## VR

| 选项 | 默认值 | 说明 |
| ------ | ------- | ----- |
| `--vr-mode` | `auto` | `auto`、`off`、`sbs`、`sbs-fisheye`。见 [VR180](vr180.md)。 |

## 编码

| 选项 | 默认值 | 说明 |
| ------ | ------- | ----- |
| `--codec` | `hevc` | 离线输出可选 `hevc`、`h264` 或 `av1`。HLS 流媒体始终使用 H.264。 |
| `--cq` | 根据 GPU/编解码器 | 视频质量。越低质量越好、文件越大。NVIDIA 默认值：H.264 25、HEVC 28、AV1 35；AMD 默认值：H.264 24、HEVC 25、AV1 32。 |
| `--encoder-settings` | — | JSON 对象或逗号分隔的高级 `key=value` 设置，例如 `{"rc-lookahead":32}` 或 `rc-lookahead=32,bf=4`。见下文。 |
| `--lut` | — | `.cube` 色彩 LUT（1D 或 3D），在写入视频时应用。也可在 GUI 的编码设置部分设置。 |
| `--sharpen` | `0` | 编码前锐化画面，取值 `0`（关闭）到 `1`（最强）。与 ffmpeg 的 `cas` 滤镜一致，无需二次转码。见[高级处理](advanced_processing.md)。 |
| `--retarget-high-fps` | 关闭 | 每两帧保留一帧，实现 60 → 30 FPS（以及 59.94 → 29.97）。其他帧率不变；音频保持同步。不能与 `--stream` 或 `--segments` 同时使用。 |
| `--fmp4` | 关闭 | `.mp4`/`.mov` 输出在生成过程中即可播放，任务中断后仍可播放。不能与 `--stream`、`--segments` 或 `--ltx-auto-split` 同时使用。见[高级处理](advanced_processing.md)。 |
| `--segments` | — | 只修复一个视频中的这些区间，单位为秒或 `mm:ss`，例如 `10-25,01:10-01:30.5`。不能与 `--stream`、`--retarget-high-fps`、`--fmp4` 或 `--ltx-auto-split` 同时使用，也不能用于图片或文件夹输入。见[区间](segments.md)。 |
| `--working-directory` | 输出目录 | `--segments` 和 LTX 临时文件的写入位置。见[区间](segments.md)。 |

### 选择编解码器

- **`hevc`**（默认）：相对文件大小质量好，10-bit。大多数现代设备都能播放。
  不确定时就用它。
- **`h264`**：几乎到处都能播放（老电视、浏览器、视频剪辑软件）。仅 8-bit，
  相同质量下文件更大。流媒体总是使用它。
- **`av1`**：相同质量下文件最小，10-bit。需要能编码 AV1 的 GPU（NVIDIA RTX 40
  系列、AMD RX 7000 或更新）和较新的播放器。

使用 `--segments` 时，输出总是使用输入的编解码器。指定不同的 `--codec` 会报错停止。

### 编码器设置

`--cq` 是主要的质量设置。数值越低，质量越好，文件越大。Jasna 会把数值原样传给
编码器（只有 AMD AV1 会乘以 5）。更换编解码器时，数值不会被换算。

| GPU | H.264 默认值 | HEVC 默认值 | AV1 默认值 | 允许范围 |
| --- | ---: | ---: | ---: | --- |
| NVIDIA | 25 | 28 | 35 | H.264/HEVC 为 1–51；AV1 为 1–63 |
| AMD | 24 | 25 | 32 | H.264/HEVC 为 0–51；AV1 为 1–51 |

在 NVIDIA 上，CQ 0 表示“自动”，所以 Jasna 需要 1 或更大的值。GUI 会为每个
编解码器分别保存 CQ 值。

`--encoder-settings` 用于修改其他编码器选项。Jasna 会检查每个参数。编码器不认识的
参数会让任务报错停止，错误信息会列出它接受的参数：

```bash
# 更高质量（文件更大）：降低 CQ。
jasna --input in.mp4 --output out.mkv --cq 22

# CQ 加高级参数
jasna --input in.mp4 --output out.mkv --cq 22 --encoder-settings "rc-lookahead=32,bf=4"
```

不使用 `--cq` 时，`--encoder-settings` 中的 `cq=22` 仍然有效。两处同时设置 CQ 会
报错。在 GUI 中，CQ 只能通过 CQ 控件设置，不能写在**自定义参数**中。

#### NVIDIA (NVENC) 参数，所有编解码器

| 参数 | 作用 |
| --- | ------------ |
| `cq` | VBR 的目标质量。越低 = 质量越好、文件越大。H.264/HEVC 的范围为 1–51（默认 25/28），AV1 为 1–63（默认 35）。输出体积自动上限可能让相近的数值得到相同结果。 |
| `preset` | 速度/质量权衡，从 `p1`（最快）到 `p7`（最佳）。默认 `p5`。 |
| `tune` | `hq`（默认）、`ll`、`ull` 或 `lossless`。 |
| `rc` | 码率控制模式：`vbr`（默认）、`cbr`、`constqp`。 |
| `qmin` / `qmax` | VBR 的质量下限/上限。默认 17/34（仅 H.264/HEVC；AV1 使用不同的 0–255 QP 范围，不设置这两项）。 |
| `init_qpI` / `init_qpP` / `init_qpB` | 各帧类型的初始量化值。默认 17（H.264/HEVC）。 |
| `g` | 关键帧间隔（帧数）。默认 250。越小 = 跳转越流畅，文件越大。 |
| `bf` | 最大连续 B 帧数。默认 4。 |
| `b_ref_mode` | 把 B 帧用作参考帧：`disabled`、`each`、`middle`（默认）。 |
| `b_adapt` | 自适应 B 帧放置。 |
| `nonref_p` | 非参考 P 帧，默认开启。 |
| `spatial_aq` / `spatial-aq` | 空间自适应量化：把更多码率花在肉眼更敏感的区域。默认开启。AV1 只接受带连字符的写法。 |
| `temporal-aq` | 时间自适应量化。默认开启。 |
| `aq-strength` | AQ 强度，1–15。默认 8。 |
| `rc-lookahead` | 码率控制的前瞻分析帧数。默认 32。 |
| `lookahead_level` | 前瞻质量，0–3。仅 HEVC/AV1。在 H.264 上会被忽略并警告。 |
| `maxrate` / `bufsize` | 码率上限和 VBV 缓冲区大小（比特/秒）。Jasna 会根据源码率自动设置（见下文）；自行指定 `maxrate` 则以你的值为准。 |
| `multipass` | 两遍编码：`disabled`、`qres`、`fullres`。 |
| `weighted_pred` | 加权预测。NVENC 仅在 `bf=0` 时支持；否则（以及 AV1 上始终）会被忽略并警告。 |
| `tf_level` | 时间滤波级别。 |

各编解码器额外参数：

| 编解码器 | 额外参数 |
| ----- | ---------- |
| `hevc` | `profile`（`main`、`main10` — 默认 `main10`）、`tier` |
| `h264` | `profile`（`baseline`、`main`、`high` — 默认 `high`）、`coder`（`cabac`/`cavlc`） |
| `av1` | `tier`、`tile-rows`、`tile-columns`（并行解码大分辨率画面） |

#### 输出体积自动上限

`cq` 追求固定的质量，不管源视频是怎么编码的。这样，低码率的源输出后会变大好几倍。
为避免这种情况，Jasna 会根据源视频的码率设置 `maxrate`，并把 `bufsize` 设为它的
两倍：

| 情况 | 上限 |
| ---- | ---- |
| NVIDIA H.264 输出 | 源视频码率的 2.0 倍 |
| 其他输出，HEVC 源 | 源视频码率的 1.25 倍 |
| 其他组合 | 源视频码率的 1.0 倍 |

NVIDIA H.264 的余量更大，因为它需要更多码率来保留修复后的细节。这个上限只对低码率的
源有影响。高码率的源本来就低于上限。达到上限时，相近的 CQ 值可能得到相同的文件大小。

自己设置 `maxrate` 可替换这个上限，设一个很大的值则相当于关闭。如果源没有码率信息，
Jasna 会记录警告，并且不设上限。AMD HEVC 和 AV1（10-bit）使用固定 QP，没有上限。

#### AMD (AMF) 参数，所有编解码器

| 参数 | 作用 |
| --- | ------------ |
| `cq` | AMF 的质量值：H.264 使用 QVBR，10-bit HEVC/AV1 使用固定 QP（AV1 内部刻度更细，CQ 会乘以 5 后传入）。越低越好；默认值依次为 24、25、32。 |
| `qvbr_quality_level` | 旧版 CQ 别名。省略 `--cq` 时可用于 CLI 高级设置；GUI 自定义参数中不接受。 |
| `usage` | 编码器用途配置。默认 `high_quality`。 |
| `quality` | 速度/质量预设：`speed`、`balanced`、`quality`（默认）。 |
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
| `--post-export-action` | `none` | 所有视频完成后运行。`shutdown` 会立即关机（与 GUI 不同，没有倒计时）；`command` 运行 `--post-export-command`。 |
| `--post-export-command` | — | `--post-export-action command` 使用的 shell 命令。 |
| `--post-export-video-command` | — | 每个视频成功导出后运行的 shell 命令。支持 `{input}`、`{output}`、`{output_dir}`、`{output_stem}` 和 `{output_suffix}`。 |

```bash
jasna --input input.mp4 --output output.mkv --post-export-action shutdown
jasna --input folder_in --output folder_out --post-export-action command --post-export-command "echo done"
jasna --input folder_in --output folder_out --post-export-video-command "ffmpeg -i {output} -map 0 -map_metadata 0 -map_chapters 0 -c copy -movflags +faststart {output_dir}/{output_stem}_remuxed{output_suffix}"
```

最后一个示例会保留原输出，并另写一个重新封装的文件。要替换原文件，请用脚本配合临时文件。
`ffmpeg` 必须在 `PATH` 中，或写出它的完整路径。

## 激活

| 选项 | 说明 |
| ------ | ----- |
| `--machine-key` | 显示机器密钥后退出，无需启动 GPU 或 GUI。 |
| `--license-code CODE` | 在 GPU 初始化之前，使用你的许可证码在线激活。未指定处理选项时，激活后退出。 |
| `--cloud-workspace` | 与 `--machine-key` 或 `--license-code` 一起使用，创建或复用云身份。仅限受支持的容器内的 Linux 官方版 CLI，且许可证码须已开通云权限。 |
| `--activation-key KEY` | 保存本机的激活密钥。未指定处理选项时，激活后退出。 |

在线激活：

```bash
jasna --license-code "YOUR_LICENSE_CODE"
jasna --license-code "YOUR_LICENSE_CODE" --input input.mp4 --output output.mkv
```

手动激活：显示机器密钥，在任意设备上打开[激活页面](https://jasna-activation.kruk2.workers.dev)，
输入机器密钥和许可证码，然后保存返回的激活密钥：

```bash
jasna --machine-key
jasna --activation-key "YOUR_ACTIVATION_KEY"
```

`--license-code` 不能与 `--machine-key` 或 `--activation-key` 一起使用。

- 租用的 GPU（RunPod、Vast）：加上 `--cloud-workspace`，并把 `XDG_CONFIG_HOME`
  设为一个你会备份的私有文件夹。它保存你的激活，所以每次运行都要用同一个文件夹：

  ```bash
  export XDG_CONFIG_HOME=/root/jasna-config
  ./jasna --license-code "YOUR_LICENSE_CODE" --cloud-workspace
  ```

  不要使用 RunPod 的 `/workspace`，它会忽略私有文件权限。完整步骤：[云 GPU](cloud.md)。
- 设备名额、隐私，以及激活失败时怎么办：[激活支持者模型](gui.md#激活支持者模型)。

## 基准测试

| 选项 | 说明 |
| ------ | ----- |
| `--benchmark` | 运行基准测试而不是处理。 |
| `--benchmark-filter` | 只运行名称包含此字符串的基准测试。 |
| `--benchmark-video` | 基准测试使用的视频路径；可重复指定。 |
