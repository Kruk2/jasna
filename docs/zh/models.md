# 选择模型

## 检测模型

检测模型负责在每一帧中找到马赛克。

- **`rfdetr-v6`**（默认，已内置）：速度快，适合大多数视频。不确定时就用它。
- **`rfdetr-v6-large`**：检测更好，但更慢。4K 视频可能更适合。
  它需要单独下载。请按显卡选择文件，放入 `model_weights/`。Jasna 会自动找到它。
  - NVIDIA：
    [rfdetr-v6-large.onnx](https://github.com/Kruk2/jasna/releases/download/0.1/rfdetr-v6-large.onnx)
  - AMD：
    [rfdetr-v6-large.pt](https://github.com/Kruk2/jasna/releases/download/0.1/rfdetr-v6-large.pt)
- **`rfdetr-vr-v1`**（已内置）：最适合 VR180 视频。
- **`zelefans-vr-yolo-v2`**（单独下载）：另一个 VR180 模型。
- **Lada YOLO** 模型：在 2D 动画上可能更好。
- 旧的 `rfdetr-v5` 仍然可用。

在 AMD 上，RF-DETR 比在 NVIDIA 上慢。如果速度比检测质量更重要，请用
`lada-yolo-v4`。

每个模型都有自己的默认检测阈值（`rfdetr-v6`：0.35，`rfdetr-v6-large` 和 `rfdetr-vr-v1`：0.40）。
在**基本处理**中的**检测阈值**修改，或用 `--detection-score-threshold`。

```bash
jasna --input input.mp4 --output output.mkv --detection-model rfdetr-v6
```

在[区间编辑器](segments.md)中，可以为每个视频选择不同的检测模型。

## 修复模型

| | **BasicVSR++**（默认） | **LTX** |
| --- | --- | --- |
| 速度 | 快 | 慢很多（每分钟视频需要几分钟） |
| 质量 | 好 | 细节更多，更自然 |
| GPU | 所有支持的 GPU | 仅 NVIDIA |
| 费用 | 免费 | 支持者专享（[支持本项目](../../README.zh.md#支持本项目)） |

把 LTX 用在你最在意的视频或场景上。选择之前，先看看
这些示例：1、2、3、4、5、6、7、8
（想看更多示例，可以浏览 SLS Discord 的新频道和
旧频道（已归档）），并运行**试用模式（速度测试）**，看看你的电脑能不能运行它、需要多长时间。
在**修复模型**部分选择它，或使用：

```bash
jasna --input input.mp4 --output output.mkv --restoration-model-name ltx
```

### LTX 需要什么

- **许可证**：包含 LTX 的许可证码（25 美元，同时包含 UNet 4x 和 SD 1.5，见
  [支持本项目](../../README.zh.md#支持本项目)）。没有的话，应用会改为提供
  **试用模式（速度测试）**（见下文）；CLI 会报错停止。
- **GPU**：NVIDIA RTX 30、40 或 50 系列。RTX 20 系列应该可以用，但只测试过
  RTX 2080 Ti。GTX 16 系列未测试。不支持 AMD。
- **显存**：至少 8 GB。显存越大越快：显存较小的显卡会把模型的一部分放在系统
  内存中。
- **系统内存**：至少 16 GB（LTX 约占用 10 GB）。
- **磁盘**：模型不随 Jasna 附带。第一次开始 LTX 任务时，Jasna 会先询问再下载，只下载你选的模型：

  | 模型 | 下载大小 |
  | --- | ---: |
  | 标准 | 约 15 GB |
  | 快速模式 | 约 9 GB |
  | 最佳，在标准之外 | 再加约 13 GB |

  如果无法访问 Hugging Face（例如在中国），Jasna 会自动使用 hf-mirror.com。

LTX 不支持 VR180 左右并排视频，也不支持流媒体。使用 LTX 时，二次修复和降噪会关闭。

### LTX 选项

- **模型**：**标准**（默认）或**最佳**。最佳效果稍好，约慢 1.5 倍。
  CLI：`--restoration-model-name ltx-undistilled`。
- **快速模式**（仅限 RTX 50 系列）：快约 1.4 倍，细节略少，下载更小。
  CLI：`--ltx-fast`。
- **大马赛克更清晰**：大马赛克（特写、4K）细节更多，这部分约慢 3 倍。
  需要约 12 GB 空闲显存（区间编辑器中的区间需要 16 GB 显卡）；否则 Jasna 会
  按普通方式修复它们。CLI：`--ltx-large-canvas`。
- **种子**：换一个种子，结果会略有不同；同一个种子得到同样的结果。
  CLI：`--ltx-seed 7`。
- **试用模式（速度测试）**：不需要模型和许可证，以全速运行 LTX，让你看看它在
  你的电脑上有多快。视频**不会**被修复。CLI：`--ltx-trial`。
- **仅大马赛克用 LTX**：大马赛克用 LTX，其余用更快的 BasicVSR++。长视频能省很多
  时间。见[只修复视频的一部分](segments.md#cli---ltx-auto-split)。

要预览 LTX，请打开[区间编辑器](segments.md)，选择一个区间，选 LTX，再打开
**修复预览**。视频播放器总是使用 BasicVSR++。

### 需要多长时间

**1 分钟 1080p、马赛克一直在画面中的视频**的大致耗时（标准模型）。没有马赛克的
场景快得多。最佳约需 1.5 倍时间。

| GPU | 耗时 | 快速模式 |
| --- | ---: | ---: |
| RTX 5090 | 约 5 分钟 | 约 3 分钟 |
| RTX 4090 | 约 7 分钟 | 不可用 |
| RTX 5080 | 约 10 分钟 | 约 7 分钟 |
| RTX 3090 | 约 15 分钟 | 不可用 |
| RTX 2080 Ti | 约 45 分钟 | 不可用 |

- 第一次运行 LTX 时，Jasna 要为你的 GPU 做准备，所以更慢。
- 进度显示四个步骤：查找马赛克、准备、修复、保存。在你的 GPU 上第一次（以及每次更新后第一次）
  处理 LTX 视频时，会先做一次简短的速度测试。之后从第 1 步结束起，剩余时间就是准确的。
- 其他应用占用 GPU 时，LTX 会变慢。请关闭游戏和浏览器。
- **Windows**：Windhawk 在 Jasna 中运行时，LTX 会因“拒绝访问”而失败。请退出
  Windhawk，或把 `jasna.exe` 加入它的排除进程列表。Jasna 启动时会提示这一点。

## 二次修复

Jasna 把每个马赛克修复成一张 256x256 的图。在大马赛克、特写和 4K 视频上，这可能
看起来模糊。二次模型会先把这张图放大（512x512 或 1024x1024），再放回画面，
所以结果更清晰。

仅限 NVIDIA。

- **UNet 4x**（支持者专享）：针对修复任务训练，质量接近 TVAI `iris-2`，比 TVAI 快。见
  SLS Discord 上的示例
  和[支持本项目](../../README.zh.md#支持本项目)。
- **RTX Super Res**（免费）：非常快，无需安装任何东西。质量尚可。部分视频
  会闪烁，请先用短片段测试。
- **Topaz TVAI**：和 UNet 4x 相近，但非常慢。需要
  [Topaz Video](https://www.topazlabs.com/topaz-video)，它是付费软件，且仅支持
  Windows。请使用 `iris-2` 模型。

使用 LTX 时不会用到。

```bash
jasna --input input.mp4 --output output.mkv --secondary-restoration unet-4x
```

使用 TVAI 时，请把环境变量 `TVAI_MODEL_DATA_DIR` 和 `TVAI_MODEL_DIR` 设置为你的
Topaz Video 模型文件夹，如下图。`--tvai-args` 可修改 Topaz 的其他设置。

<img width="505" height="37" alt="Topaz Video environment variables" src="https://github.com/user-attachments/assets/e19ced9d-d549-4e85-b20f-888e42466f1d" />

### 速度与显存对比

两个 1080p 视频的耗时和显存：

| 二次修复类型 | CAWD 1080p | K109 1080p |
| ------------------------ | -----------------:| -----------------:|
| 无二次修复 | 22秒 / 10.0 GB VRAM | 11秒 / 10.7 GB VRAM |
| UNet 4x                  | 29秒 / 12.5 GB VRAM | 14秒 / 12.6 GB VRAM |
| RTX Super Res            | 25秒 / 11.7 GB VRAM | 13秒 / 11.4 GB VRAM |
| TVAI (2 workers, Iris-2) | 52秒 / 12.1 GB VRAM | 24秒 / 12.4 GB VRAM |

## 静态图片修复（SD 1.5）

对于图片，Jasna 使用另一个模型：训练过的 Stable Diffusion 1.5 模型。把图片加入
队列（或交给 CLI），Jasna 会自动使用这个模型：

```bash
jasna --input photo.png --output restored.png
```

- 支持者专享。一个许可证码可以同时包含 SD 1.5、UNet 4x 和 LTX。见
  [支持本项目](../../README.zh.md#支持本项目)。
- 不包含在下载包中。大小约 **6.9 GB**。Jasna 会先询问，再从
  [huggingface.co/Kruk2/sd-15-jav](https://huggingface.co/Kruk2/sd-15-jav) 下载。
- 需要约 **7 GB 显存**，较大的 4K 图片需要更多一点。

这个模型是实验性的。不同图片的效果差别很大，但有些非常好。多生成几个版本，
保留最好的：

```bash
jasna --input photo.png --output restored.png --sd15-variants 4
```

要在 GUI 中逐个尝试种子，请使用[交互式处理图片](gui.md#逐张处理图片)。所有选项
（`--sd15-steps`、`--sd15-strength`、`--sd15-seed` 等）都在
[CLI 参考](cli.md#sd-15-图片修复)中。

示例：
SLS Discord 上的 SD 1.5 示例
和更多。
