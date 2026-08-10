# 高级处理

针对特殊场景的可选功能。这里的所有功能在 GUI（找到对应设置，每个都
有提示）和 CLI 中都可用。

## 降噪

修复区域可能带有噪点伪影。降噪设置（`--denoise low|medium|high`）
只对修复区域应用温和的空间降噪 — 画面的其余部分不受影响。请从 `low`
开始，仅在伪影仍然存在时再提高。

默认在二级修复之前运行；`--denoise-step after_secondary` 会把它移到
混合回原视频之前。

## 检测稳定性过滤

检测在逐帧层面并不完美：马赛克可能短暂消失一两帧（导致一个片段被切成
多段，留下明显接缝和未修复的帧），而单帧误检会触发不必要的修复。

- **最大检测间隙**（`--max-detection-gap`，默认 `2`）：当马赛克在相同位置
  重新出现时，填补最多 N 帧的检测中断，保持片段连续。
- **最短检测持续帧数**（`--min-detection-duration`，默认 `2`）：丢弃持续
  少于 N 帧的检测（视为误检），相应帧保持原样。

两者都应保持较小的数值，以免影响真正快速出现/消失的画面。`0` 表示禁用。

## 镜头切换检测

如果没有此功能，跨越硬切镜头被持续跟踪的马赛克会进入同一个片段，导致修复时
混合切换前后两个镜头的画面。**镜头切换检测**（`--scene-detection`，默认开启）
会检测硬切并在切换点结束所有跟踪中的片段，使每个片段都保持在单个镜头内。
它在 GPU 上运行，开销可以忽略不计。

仅当你发现片段在没有真实镜头切换的位置被切分时，才使用 `--no-scene-detection`
（或 GUI 高级设置中的开关）将其关闭。

## 60 FPS 降至 30 FPS 导出

对于 60（或 59.94）FPS 的输入，**将 60 FPS 降至 30 FPS**
（`--retarget-high-fps`）会每两帧处理一帧，并输出 30（或 29.97）FPS —
处理量减半。音频时序和播放速度保持不变。其他帧率不受影响:

```bash
jasna --input input.mp4 --output output.mp4 --retarget-high-fps
```

不能与[区间处理](segments.md)同时使用。

## 处理过程中即可播放（分片 MP4）

普通 MP4 只有在任务完成后才能打开。**处理过程中即可播放**（`--fmp4`）让你在
输出还在生成时就能播放它 — 不用等待即可检查质量，任务中断后文件也仍可播放:

```bash
jasna --input input.mp4 --output output.mp4 --fmp4
```

视频每隔几秒增长一段，完成前播放器可能显示错误的时长。只影响 `.mp4` 和 `.mov`
输出。不能用于流媒体或[区间处理](segments.md)。

## 色彩 LUT

对输出应用 `.cube` 色彩 LUT（1D 或 3D）— 用于调色或统一画面风格。
在 GUI 的编码设置部分设置，或使用 `--lut path/to/look.cube`。LUT 在
编码前由 GPU 应用，几乎没有额外开销。

## 锐化

修复后的画面有时会显得偏软。GUI 编码设置中的**锐化**（`--sharpen`）会在导出时让
边缘和细节更清晰，无需再用其他工具转码一次。

```bash
jasna --input in.mp4 --output out.mkv --sharpen 0.5
```

`0` 表示关闭，`0.2`–`0.5` 为轻微增强，`1` 最强且可能显得生硬。画面越锐利所需
文件越大，如果结果反而变差，请同时调低 CQ 值。预览中不显示该效果。

## 编码器质量与自定义设置

使用 GUI 的 **CQ** 控件或 `--cq` 调节编码器质量。显示或输入的数值会原样传给
编码器；数值越低，质量越好，文件越大。Jasna 还会限制输出大小，因此达到上限时，
相近的 CQ 值可能得到相同结果:

```bash
jasna --input in.mp4 --output out.mkv --cq 22
```

**自定义参数**输入框（`--encoder-settings`）用于码率上限、关键帧间隔等其他高级
设置。为确保 CQ 控件显示的数值始终与编码器收到的数值一致，GUI 自定义参数不接受
CQ 别名。各编解码器支持的参数、原生范围和默认值均记录在 [CLI 参考](cli.md)中。

## 导出后操作

在整个队列完成后执行操作: **关闭电脑**或**自定义命令**（例如一个
通知脚本）。在 GUI 的导出后操作部分设置，或通过 CLI:

```bash
jasna --input input.mp4 --output output.mkv --post-export-action shutdown
jasna --input folder_in --output folder_out --post-export-action command --post-export-command "echo done"
```

要在每个视频成功导出后运行命令，请填写**每个视频完成后的命令**，或使用
`--post-export-video-command`。Jasna 会等待命令完成；如果命令失败，该视频会被
标记为错误。路径占位符已自动添加引号:

```bash
jasna --input folder_in --output folder_out --post-export-video-command "ffmpeg -i {output} -map 0 -map_metadata 0 -map_chapters 0 -c copy -movflags +faststart {output_dir}/{output_stem}_remuxed{output_suffix}"
```

可用占位符为 `{input}`、`{output}`、`{output_dir}`、`{output_stem}` 和
`{output_suffix}`。此示例会保留原始输出；如需安全替换，请使用先写入临时文件的
脚本。`ffmpeg` 必须位于 `PATH` 中，也可以改用完整路径。
命令的工作目录是输出文件夹。
