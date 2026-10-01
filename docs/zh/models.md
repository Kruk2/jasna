# 选择模型

## 检测模型

检测模型负责在每一帧中找到马赛克。

- **使用最新的 RF-DETR 模型**（`rfdetr-v6`）— 它是默认值、速度快，也是综合
  表现最好的选择。已随 Jasna 内置。
- **`rfdetr-v6-large`** 是质量更高、速度更慢的变体，4K 视频可能更适合用它。
  它是可选的单独下载，
  请按显卡选择对应文件，放入 `model_weights/`，Jasna 会自动检测：
  - NVIDIA：
    [rfdetr-v6-large.onnx](https://github.com/Kruk2/jasna/releases/download/0.1/rfdetr-v6-large.onnx)
  - AMD：
    [rfdetr-v6-large.pt](https://github.com/Kruk2/jasna/releases/download/0.1/rfdetr-v6-large.pt)
- **Lada YOLO** 模型在 2D 动画上可能效果更好。
- **rfdetr-vr-v1**（已内置）是用于 VR180 的 RF-DETR 检测模型，最适合 VR180 视频。
- **zelefans-vr-yolo-v2**（可选下载）是 VR180 的备用检测模型。
- **在 AMD 上**，RF-DETR 在显卡上运行，使用 `.pt` 模型文件（NVIDIA 使用
  `.onnx`）。它比在 NVIDIA 上慢，若速度比检测质量更重要，请选择
  `lada-yolo-v4`。

每个模型默认应用其推荐的检测阈值（`rfdetr-v6`：0.35，`rfdetr-v6-large`：0.40）；
可用 `--detection-score-threshold` 覆盖。

旧版 `rfdetr-v5` 模型仍受支持。

```bash
jasna --input input.mp4 --output output.mkv --detection-model rfdetr-v6
```

你也可以在[区间编辑器](segments.md)中为每个视频单独设置不同的检测模型。

## 修复模型

**BasicVSR++** 是默认模型：速度快，所有支持的显卡都能运行。
**LTX** 慢很多，但效果更细腻、更自然。适合你最在意的视频或场景。在
**修复模型**区域选择它，或使用：

```bash
jasna --input input.mp4 --output output.mkv --restoration-model-name ltx
```

- 仅限 NVIDIA RTX。不支持 VR180 左右并排视频和流媒体。使用 LTX 时
  二级修复和降噪会关闭。
- 视频播放器和**修复预览**始终使用 BasicVSR++。要预览 LTX，请在
  [区间编辑器](segments.md)中使用**试试种子**。
- 耗时取决于马赛克出现的时长，而不是视频长度。首次运行时 Jasna 要为你的
  显卡准备 LTX，会更慢。
- 模型不随 Jasna 附带：约 15 GB（快速模式约 9 GB）。下载开放后，
  Jasna 会在下载前询问你。

### LTX 选项

- **模型**：**蒸馏**（默认）或**未蒸馏**。未蒸馏效果稍好，约慢 2 倍
  （需另外下载 13 GB）。CLI：`--restoration-model-name ltx-undistilled`。
- **快速模式**：仅限 RTX 50 系列。快约 1.4 倍，细节略少，下载也更小。
  CLI：`--ltx-fast`。
- **大马赛克更清晰**：让大马赛克（特写、4K）细节更多，这部分约慢 3 倍。
  需要 10 GB 空闲显存，不够时不生效。CLI：`--ltx-large-canvas`。
- **种子**：换一个种子，结果会略有不同；同一个种子会得到同样的结果。
  CLI：`--ltx-seed 7`。
- **试用模式**：无需模型文件，以真实速度运行 LTX，看看它在你的
  电脑上有多快。视频不会被修复。CLI：`--ltx-trial`。

### 显卡与耗时

**1 分钟 1080p、马赛克一直在画面中的视频**的大致耗时（蒸馏模型）。
没有马赛克的场景快得多；未蒸馏约需 2 倍时间。

| 显卡 | LTX | 耗时 | 快速模式 |
| --- | --- | ---: | ---: |
| RTX 5090 | 最快 | 约 5 分钟 | 约 3 分钟 |
| RTX 4090 | 流畅 | 约 7 分钟 | — |
| RTX 5080 | 流畅 | 约 10 分钟 | 约 6 分钟 |
| RTX 3090 | 能用，较慢 | 约 15 分钟 | — |
| RTX 2080 Ti | 能用，很慢 | 约 45 分钟 | — |
| GTX 16 系列 | 未测试 | — | — |
| AMD | 不支持 | — | — |

- **显存**：24 GB 可把整个模型放在显卡上，16 GB 也没问题，11–12 GB
  能用但更慢。低于 11 GB 未测试。
- **内存**：建议 32 GB（加载模型时约占用 19 GB）。只有 16 GB 时，请使用约需 10 GB 的 Fast 模式。

## 二级修复

Jasna 会修复每个马赛克区域的 256x256 裁切图。因此，大面积马赛克、
特写和 4K 视频在主修复后可能看起来模糊。二级模型会先把修复后的裁切图
放大到 512x512 或 1024x1024，再混合回原视频，让画面明显更清晰。

目前，二级修复仅适用于 NVIDIA GPU；AMD 版本只支持主修复。

- **unet-4x**: 支持者模型。当前测试中比 TVAI 更快，质量相近。它在
  视频修复数据集上训练，视觉效果接近 TVAI `iris-2`。可以查看
  SLS Discord 上的示例。
  使用支持者密钥解锁 — 见[支持本项目](../../README.zh.md)。
- **RTX Super Resolution**: 非常快、免费、没有额外依赖。质量尚可。
  部分视频可能会闪烁，请先用短片段测试。
- **TVAI**: 质量优于 RTX Super Resolution，并与 unet-4x 接近，但非常
  慢。需要 [Topaz Video](https://www.topazlabs.com/topaz-video)，这是
  付费软件且仅支持 Windows。推荐模型: `iris-2`。

```bash
jasna --input input.mp4 --output output.mkv --secondary-restoration unet-4x
```

对于 TVAI，请按下图所示，把 `TVAI_MODEL_DATA_DIR` 和 `TVAI_MODEL_DIR`
环境变量设置为你的 Topaz Video 模型文件夹（`--tvai-args` 可以进一步
自定义 Topaz 模型参数）:

<img width="505" height="37" alt="Topaz Video environment variables" src="https://github.com/user-attachments/assets/e19ced9d-d549-4e85-b20f-888e42466f1d" />

### 速度与 VRAM 对比

| 二级类型 | CAWD 1080p | K109 1080p |
| ------------------------ | -----------------:| -----------------:|
| 无二级修复 | 22秒 / 10.0 GB VRAM | 11秒 / 10.7 GB VRAM |
| unet-4x                  | 29秒 / 12.5 GB VRAM | 14秒 / 12.6 GB VRAM |
| RTX Super-Res            | 25秒 / 11.7 GB VRAM | 13秒 / 11.4 GB VRAM |
| TVAI (2 workers, Iris-2) | 52秒 / 12.1 GB VRAM | 24秒 / 12.4 GB VRAM |

## 静态图像修复（SD 1.5）

对于静态图像，Jasna 使用微调过的 Stable Diffusion 1.5 inpaint 模型，
而不是视频流水线。只需把图像加入 GUI 队列（或通过 CLI 传入）— 图像
任务会自动路由到 SD 1.5:

```bash
jasna --input photo.png --output restored.png
```

- 该模型**未随程序打包**，大小约 **6.9 GB**。Jasna 会在从
  [huggingface.co/Kruk2/sd-15-jav](https://huggingface.co/Kruk2/sd-15-jav)
  下载前先询问你。
- 目前仅面向支持者提供，并使用与 unet-4x 相同的密钥 —
  见[支持本项目](../../README.zh.md)。
- 推理期间大约需要 **7 GB VRAM**，较大的 4K 图像会再多一些。

SD 1.5 路径是实验性的。结果因场景而异，但有些图像可能效果非常好。
可以生成多个变体，保留最好的那个:

```bash
jasna --input photo.png --output restored.png --sd15-variants 4
```

所有可调参数（`--sd15-steps`、`--sd15-strength`、`--sd15-seed` 等）
都列在 [CLI 参考](cli.md)中。

示例:
SLS Discord 上的 SD 1.5 示例
和更多示例。
