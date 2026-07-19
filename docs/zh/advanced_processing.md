# 高级处理

针对特殊场景的可选功能。这里的所有功能在 GUI（找到对应设置，每个都
有提示）和 CLI 中都可用。

## 降噪

修复区域可能带有噪点伪影。降噪设置（`--denoise low|medium|high`）
只对修复区域应用温和的空间降噪 — 画面的其余部分不受影响。请从 `low`
开始，仅在伪影仍然存在时再提高。

默认在二级修复之前运行；`--denoise-step after_secondary` 会把它移到
混合回原视频之前。

## 60 FPS 降至 30 FPS 导出

对于 60（或 59.94）FPS 的输入，**将 60 FPS 降至 30 FPS**
（`--retarget-high-fps`）会每两帧处理一帧，并输出 30（或 29.97）FPS —
处理量减半。音频时序和播放速度保持不变。其他帧率不受影响:

```bash
jasna --input input.mp4 --output output.mp4 --retarget-high-fps
```

不能与[区间处理](segments.md)同时使用。

## 色彩 LUT

对输出应用 `.cube` 色彩 LUT（1D 或 3D）— 用于调色或统一画面风格。
在 GUI 的编码设置部分设置，或使用 `--lut path/to/look.cube`。LUT 在
编码前由 GPU 应用，几乎没有额外开销。

## 自定义编码器设置

**自定义参数**输入框（`--encoder-settings`）可微调硬件视频编码器 —
质量等级、码率上限、关键帧间隔等。最主要的参数是 `cq`（越低 = 质量
越好，文件越大）:

```bash
jasna --input in.mp4 --output out.mkv --encoder-settings "cq=22"
```

每个编解码器支持的所有参数都记录在 [CLI 参考](cli.md)中。

## 导出后操作

在整个队列完成后执行操作: **关闭电脑**或**自定义命令**（例如一个
通知脚本）。在 GUI 的导出后操作部分设置，或通过 CLI:

```bash
jasna --input input.mp4 --output output.mkv --post-export-action shutdown
jasna --input folder_in --output folder_out --post-export-action command --post-export-command "echo done"
```
