# 高级处理

用于特殊情况的额外功能。这里的所有功能在 GUI（每个设置都有提示）和 CLI 中都能用。

## 降噪

修复区域可能会有一些噪点。**降噪强度**（`--denoise low|medium|high`）只清理
修复区域。画面的其余部分不会改变。请从 `low` 开始，噪点仍然明显时再调高。

使用 LTX 时不会用到。

## 检测稳定性

检测在逐帧之间并不完美。马赛克可能消失一两帧。这时片段会被切成几段，留下明显的
接缝和一帧没有修复。单帧误检会触发不需要的修复。

- **最大检测间隙**（`--max-detection-gap`，默认 `2`）：马赛克在同一位置重新出现时，
  填补最多 N 帧的中断。LTX 始终填补不超过 3 帧的中断；大于 3 的值对 LTX 同样生效。
- **最短检测持续帧数**（`--min-detection-duration`，默认 `2`）：丢弃短于 N 帧的
  检测。这些帧保持原样。LTX 也同样适用。

两者都保持较小的值，这样不会影响真正短暂出现的马赛克。`0` 表示关闭。

## 镜头切换检测

没有它时，马赛克可能被跨过硬切镜头持续跟踪，修复时会混合两个不同的镜头。
**镜头切换检测**（`--scene-detection`，默认开启）会找到硬切，并在那里结束每个
片段。它在 GPU 上运行，几乎没有开销。

只有当片段在没有真正切换的地方被切开时，才关闭它（`--no-scene-detection`，或
**专家**视图中**检测**标签页的开关）。

## 60 FPS 降至 30 FPS

对于 60（或 59.94）FPS 的视频，**将 60 FPS 降至 30 FPS**（`--retarget-high-fps`）
每两帧处理一帧，并保存为 30（或 29.97）FPS。工作量减半。音频和速度保持正确。
其他帧率不会改变。

```bash
jasna --input input.mp4 --output output.mp4 --retarget-high-fps
```

不能与[选定区间](segments.md)或流媒体一起使用。

## 处理过程中即可播放（fMP4）

普通 MP4 只有在任务完成后才能打开。使用**处理过程中即可播放（fMP4）**（`--fmp4`），
输出还在生成时就可以播放，例如尽早检查质量。如果任务中途停止，文件也仍然可以播放。

```bash
jasna --input input.mp4 --output output.mp4 --fmp4
```

视频每次增长几秒，任务结束前播放器可能显示错误的时长。只适用于 `.mp4` 和 `.mov`
输出。不能用于流媒体或[选定区间](segments.md)。

## 色彩 LUT

对输出应用 `.cube` 色彩 LUT（1D 或 3D），用于调色。在**输出**标签页设置
**色彩 LUT (.cube)**，或使用 `--lut path/to/look.cube`。它在写入视频时应用，
几乎没有开销。有[选定区间](segments.md)时，区间之间的部分会重新编码而不是复制，
让 LUT 覆盖整个视频。

## 锐化

修复后的视频可能看起来有点软。**输出**标签页中的**锐化**（`--sharpen`）会在保存视频时
让边缘和细节更清晰。不需要其他工具。

```bash
jasna --input in.mp4 --output out.mkv --sharpen 0.5
```

`0` 表示关闭，`0.2`–`0.5` 是轻微增强，`1` 最强，可能显得太生硬。画面越锐利，需要的
码率越多。如果结果变差，请同时调低 CQ。预览中不显示锐化。有[选定区间](segments.md)时，
区间之间的部分会重新编码而不是复制，让锐化覆盖整个视频。

## 放大整个视频

**放大**标签页中的**放大视频**（`--upscale`）会把整个保存的视频放大并变清晰，而不只是
修复区域。选择输出高度（`--upscale-height`：1080p、1440p 或 4K），宽度按画面比例变化。
已经达到这个高度的视频不会放大。

- **SPAN**（`span`）：快速的 AI 放大。支持 NVIDIA 和 AMD。在 NVIDIA 上，每种新的视频
  尺寸会先准备一次，约需半分钟。
- **RealPLKSR**（`realplksr`）：较慢的 AI 放大，保留更多细节并清除压缩痕迹。支持 NVIDIA 和
  AMD。在 RTX 5090 上从 1080p 到 4K 约 5 FPS，所以长视频需要一些时间；准备约需一分钟。
- **RTX Super Res**（`rtx-super-res`）：仅限 NVIDIA，非常快。还可以先去除噪点或模糊
  （`--upscale-rtx-denoise`、`--upscale-rtx-deblur`）。
- **DLSS 5**（`dlss5`，仅命令行，需要 Windows 和 RTX 30 系列或更新的 GPU；RTX 30 较慢）：
  NVIDIA DLSS 5 还会重绘光照和材质，所以颜色和皮肤可能改变。Jasna 不包含它：首次使用时，
  Jasna 会询问是否从 GitHub 下载非官方的 DLSS 5 Visual Enhancer v3.0（467 MB），并把其运行库
  保存在 `model_weights/dlss5`。`--upscale-dlss5-preset` 选择放大模型（`l` 和 `m` 重建更多纹理），
  `--upscale-dlss5-style` 选择风格，`--upscale-dlss5-intensity`（0–2）控制重绘强度。

```bash
jasna --input in.mp4 --output out.mp4 --restoration-model-name ltx --upscale span --upscale-height 2160
```

**强度**（`--upscale-strength`，0–1）调低后会混入普通缩放，画面更柔和自然。4K 需要更大的
文件，所以 Jasna 会相应提高码率上限。BasicVSR++ 和 LTX 都可以使用。会处理整个视频，
所以不能与[选定区间](segments.md)一起使用，并跳过**仅大马赛克用 LTX**。不支持流媒体。
预览中不显示。

## 编码器质量与自定义设置

**CQ**（`--cq`）设置质量。数值越低，质量越好，文件越大。Jasna 还会限制输出大小，
达到上限时，相近的 CQ 值可能得到相同的文件。

```bash
jasna --input in.mp4 --output out.mkv --cq 22
```

**自定义参数**（**专家**视图，`--encoder-settings`）用于其他编码器选项，例如码率上限或关键帧间隔。
在 GUI 中不能在这里设置 CQ，请使用 CQ 控件。所有参数、范围和默认值见
[CLI 参考](cli.md#编码)。

## 视频中保存的设置

Jasna 会把所用的设置写入输出文件的 **Comment** 标签：模型、LTX 种子、检测和编码器设置等。
它是一行文本。可以在 MediaInfo、VLC（**工具 → 媒体信息**）中查看，或使用：

```bash
ffprobe -v error -show_entries format_tags=comment out.mkv
```

源视频原有的注释会保留在这一行的 `source_comment` 中。

## 队列完成后

整个队列完成后执行一个操作：**关闭电脑**或**自定义命令**（例如给你发消息的脚本，仅**专家**视图）。
在**输出**标签页的**队列完成后**设置，或在 CLI 中：

```bash
jasna --input input.mp4 --output output.mkv --post-export-action shutdown
jasna --input folder_in --output folder_out --post-export-action command --post-export-command "echo done"
```

任务运行时也可以更改或关闭这个操作。**关闭电脑**前，GUI 会倒计时 60 秒。按**取消**
可以不关机。CLI 会立即关机。

要在每个视频完成后运行命令，请使用**每个视频完成后的命令**（**专家**视图）或
`--post-export-video-command`。Jasna 会等它结束。如果命令失败，该视频会被标记为
失败。命令在输出文件夹中运行。

占位符：`{input}`、`{output}`、`{output_dir}`、`{output_stem}`、`{output_suffix}`。
它们已经加好引号。示例见 [CLI 参考](cli.md#导出后操作)。

在 GUI 中还可以使用 `{log}`：这是该视频的日志文件，保存在输出文件旁边，名为
`<输出文件名>.log`。可以把它复制或移动到任意位置，例如 `copy {log} D:\logs`。
