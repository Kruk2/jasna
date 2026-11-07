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
在**检测**标签页中的**检测阈值**修改，或用 `--detection-score-threshold`。

```bash
jasna --input input.mp4 --output output.mkv --detection-model rfdetr-v6
```

在[区间编辑器](segments.md)中，可以为每个视频选择不同的检测模型。

## 修复模型

| | **BasicVSR++**（默认） | **LTX** |
| --- | --- | --- |
| 速度 | 快 | 慢很多（每分钟视频需要几分钟） |
| 质量 | 好 | 细节更多，更自然 |
| GPU | 所有支持的 GPU | NVIDIA 或兼容的 AMD 显卡（Windows AMD LTX 为测试版） |
| 费用 | 免费 | 支持者专享（[支持本项目](../../README.zh.md#支持本项目)） |

把 LTX 用在你最在意的视频或场景上。选择之前，先看看
这些示例：[1](https://discord.com/channels/1196376491815092265/1199059436199759943/1556346988486856826)、[2](https://discord.com/channels/1196376491815092265/1199059436199759943/1556337300814958684)、[3](https://discord.com/channels/1196376491815092265/1199059436199759943/1555612574643388598)、[4](https://discord.com/channels/1196376491815092265/1556960822578909244/1559341521575084063)、[5](https://discord.com/channels/1196376491815092265/1556960822578909244/1559295317529202780)、[6](https://discord.com/channels/1196376491815092265/1556960822578909244/1559205894959403121)、[7](https://discord.com/channels/1196376491815092265/1556960822578909244/1559197341473574973)、[8](https://discord.com/channels/1196376491815092265/1556960822578909244/1558798127195226112)
（想看更多示例，可以浏览 SLS Discord 的[新频道](https://discord.com/channels/1196376491815092265/1556960822578909244)和
[旧频道（已归档）](https://discord.com/channels/1196376491815092265/1199059436199759943)），并运行**试用模式（速度测试）**，看看你的电脑能不能运行它、需要多长时间。
在**模型**标签页选择它，或使用：

```bash
jasna --input input.mp4 --output output.mkv --restoration-model-name ltx
```

### LTX 需要什么

- **许可证**：包含 LTX 的许可证码（25 美元，同时包含 UNet 4x 和 SD 1.5，见
  [支持本项目](../../README.zh.md#支持本项目)）。没有的话，应用会改为提供
  **试用模式（速度测试）**（见下文）；CLI 会报错停止。
- **GPU**：NVIDIA RTX 30、40 或 50 系列。RTX 20 系列应该可以用，但只测试过
  RTX 2080 Ti。GTX 16 系列未测试。AMD Radeon RX 7000/9000 和 Radeon AI PRO R9700
  可在 Linux 上使用；Windows AMD 为**测试版**。AMD 核显未测试。
  见 [AMD 显卡表](../../README.zh.md#要求)。
- **显存**：至少 8 GB。显存越大越快：显存较小的显卡会把模型的一部分放在系统
  内存中。
- **系统内存**：至少 16 GB（LTX 约占用 10 GB）。
- **磁盘**：模型不随 Jasna 附带。第一次开始 LTX 任务时，Jasna 会先询问再下载，只下载你选的模型：

  | 模型 | 下载大小 |
  | --- | ---: |
  | 标准画质（默认） | 约 15 GB |
  | 快速模式 | 约 9 GB |
  | 基础、良好、高画质或顶级画质，额外 | 每个再加约 13 GB |

  如果无法访问 Hugging Face（例如在中国），Jasna 会自动使用 hf-mirror.com。
  下载中断后再次开始，会从中断处继续。
  仍然下载失败？见[手动下载 LTX](#手动下载-ltx)。

LTX 不支持流媒体。使用 LTX 时降噪会关闭，二次修复只能用 RTX Super Res。使用**仅大马赛克用 LTX**（`--ltx-auto-split`）时，由 BasicVSR++ 修复的较小马赛克仍会使用所有二次修复和降噪。

使用 ESET？如果 LTX 启动不了，并且消息里提到 `ebehmonl.dll` 或 `ebehmoni.dll`，
请在 ESET 的深度行为检查（Deep Behavioral Inspection）排除项里加入 Jasna
文件夹。把新版 Jasna 放到新文件夹后，需要再加一次。

### 暂停与继续

LTX 任务和带区间的任务会边处理边保存进度。用 **Ⅱ** 暂停，用 **▶** 继续。
关闭 Jasna 后，请重新添加同一视频，使用相同的输出路径和设置；命令行中重新运行同一命令即可。

最后未完成的部分可能会重新处理。如果改了输入或设置，或更新了 Jasna，
重新开始前会先询问（应用中请参阅[队列](gui.md#队列)里的**过时的进度**）。只改编码、画质、颜色 LUT 或锐化会保留已保存的 LTX 进度，其余部分重新处理。
没有区间的 BasicVSR++ 任务不支持续跑。

### 手动下载 LTX

如果在 Jasna 里下载一直失败，可以用浏览器从
[hf-mirror.com](https://hf-mirror.com/Kruk2/ltx-restore/tree/0.12.1)
（或 [huggingface.co/Kruk2/ltx-restore](https://huggingface.co/Kruk2/ltx-restore/tree/0.12.1)）
下载。需要你所选模型的文件和它的 VAE 文件（基础画质共 4 个文件，其他画质共 3 个）。
Jasna 0.11.3 或更早版本请使用 `main` 分支。

| 你选的模型 | 文件 |
| --- | --- |
| 基础画质 | `ltx-restore-alpha1-distill1-int8.safetensors.enc` |
| 基础画质 + 快速模式 | `ltx-restore-alpha1-distill1-nvfp4.safetensors.enc` |
| 标准画质 | `ltx-restore-alpha1-distill2-int8.safetensors.enc` |
| 标准画质 + 快速模式 | `ltx-restore-alpha1-distill2-nvfp4.safetensors.enc` |
| 良好画质 | `ltx-restore-alpha1-distill4-int8.safetensors.enc` |
| 良好画质 + 快速模式 | `ltx-restore-alpha1-distill4-nvfp4.safetensors.enc` |
| 高画质 | `ltx-restore-alpha1-distill8-int8.safetensors.enc` |
| 高画质 + 快速模式 | `ltx-restore-alpha1-distill8-nvfp4.safetensors.enc` |
| 顶级画质 | `ltx-restore-alpha1-int8.safetensors.enc` |
| 顶级画质 + 快速模式 | `ltx-restore-alpha1-nvfp4.safetensors.enc` |
| VAE（基础画质） | `vae.safetensors.enc`、`vae-encoder-half.safetensors.enc` 和 `vae-decoder-max.safetensors.enc` |
| VAE（标准画质、良好画质、高画质、顶级画质） | `vae.safetensors.enc` 和 `vae-decoder-full.safetensors.enc` |

选了别的 **LTX 编码器**或 **LTX 解码器**？请下载对应的 `vae-encoder-half`、`vae-encoder-max`
（完整是 `vae.safetensors.enc`）或 `vae-decoder-full`、`vae-decoder-half`、`vae-decoder-max`
（`.safetensors.enc`）。

把这些文件都放到 Jasna 文件夹里的 `model_weights/ltx-restore`（没有
`ltx-restore` 文件夹就新建一个）。文件名保持原样，包括 `.enc`。重新开始 LTX
任务，Jasna 就不会再要求下载。

### LTX 选项

- **模型**：**基础画质**、**标准画质**（默认，推荐）、**良好画质**、**高画质**或**顶级画质**。基础画质比标准画质快约 1.7 倍，画面稍柔和；
  良好画质细节稍多，慢约 1.5 倍；高画质细节更多，慢约 2.5 倍；
  顶级画质比高画质稍好，慢约 5 倍，通常不值得。在 RTX 5090 上，标准画质耗时约为视频长度的 1–2 倍。
  CLI：`--restoration-model-name ltx-basic`（基础）、`ltx`（标准）、`ltx-4step`（良好）、`ltx-8step`（高画质）或 `ltx-undistilled`（顶级）。
- **快速模式**（仅限 RTX 50 系列）：快约 1.4 倍，细节略少，下载更小。
  CLI：`--ltx-fast`。
- **大马赛克更清晰**：大马赛克（特写、4K）细节更多。大多数视频都有大马赛克，整个视频可能慢到约 3 倍。
  需要约 12 GB 空闲显存（区间编辑器中的区间需要 16 GB 显卡）；否则 Jasna 会
  按普通方式修复它们。CLI：`--ltx-large-canvas`。
- **LTX 重叠**（**专家**视图）：LTX 修复的各段之间共享的帧数，默认 `17`。马赛克持续超过 4 秒时，
  越少越快；`33` 是重叠更多的旧模式。CLI：`--ltx-overlap 33`。
- **种子**（**专家**视图）：换一个种子，结果会略有不同；同一个种子得到同样的结果。
  CLI：`--ltx-seed 7`。
- **试用模式（速度测试）**：不需要模型和许可证，以全速运行 LTX，让你看看它在
  你的电脑上有多快。视频**不会**被修复。CLI：`--ltx-trial`。
- **仅大马赛克用 LTX**：大马赛克用 LTX，其余用更快的 BasicVSR++。长视频能省很多
  时间。见[只修复视频的一部分](segments.md#cli---ltx-auto-split)。
- **LTX 编码器**和 **LTX 解码器**（**专家**视图）：**完整**最清晰也最慢，**半量**更快，**极速**最快，
  画面稍柔和。**自动**（默认）在基础画质下用半量编码器和极速解码器，其他画质用完整。
  如果基础画质看起来太柔和，两个都选完整。还没下载的选项会在点击开始时下载。
  CLI：`--ltx-vae-encoder`、`--ltx-vae-decoder`。

要预览 LTX，请打开[区间编辑器](segments.md)，选择一个区间，选 LTX，再打开
**修复预览**。视频播放器总是使用 BasicVSR++。

### 需要多长时间

**1 分钟 1080p、马赛克一直在画面中的视频**的大致耗时（默认的标准画质模型）。没有马赛克的
场景快得多。良好画质慢约 1.5 倍，高画质慢约 2.5 倍，顶级画质慢约 5 倍。开启**大马赛克更清晰**后，大马赛克约需 2.5 倍时间。

| GPU | 耗时 | 快速模式 |
| --- | ---: | ---: |
| RTX 5090 | 约 1 分钟 | 不到 1 分钟 |
| RTX 4090 | 约 1.5 分钟 | 不可用 |
| RTX 5080 | 约 2 分钟 | 约 1.5 分钟 |
| RTX 3090 | 约 3 分钟 | 不可用 |
| RTX 2080 Ti | 约 10 分钟 | 不可用 |

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

UNet 4x 支持 NVIDIA 和兼容的 AMD 显卡。RTX Super Res 和 Topaz TVAI 需要 NVIDIA。

- **UNet 4x**（支持者专享）：质量接近 TVAI `iris-2`，比 TVAI 快。见
  [SLS Discord 上的示例](https://discord.com/channels/1196376491815092265/1199059436199759943/1516497879684874260)
  和[支持本项目](../../README.zh.md#支持本项目)。
- **RTX Super Res**（免费）：非常快，无需安装任何东西。质量尚可。部分视频
  会闪烁，请先用短片段测试。
- **Topaz TVAI**：和 UNet 4x 相近，但非常慢。需要
  [Topaz Video](https://www.topazlabs.com/topaz-video)，它是付费软件，且仅支持
  Windows。请使用 `iris-2` 模型。

使用 LTX 时只能用 RTX Super Res，且只用于大马赛克。示例：
`jasna --input input.mp4 --output output.mkv --restoration-model-name ltx --secondary-restoration rtx-super-res`

```bash
jasna --input input.mp4 --output output.mkv --secondary-restoration unet-4x
```

使用 TVAI 时，请把环境变量 `TVAI_MODEL_DATA_DIR` 和 `TVAI_MODEL_DIR` 设置为你的
Topaz Video 模型文件夹，如下图。`--tvai-args` 可修改 Topaz 的其他设置。

<img width="505" height="37" alt="Topaz Video environment variables" src="https://github.com/user-attachments/assets/e19ced9d-d549-4e85-b20f-888e42466f1d" />

### 速度与显存对比

两个 1080p 视频的耗时和显存：

| 二次修复类型 | CAWD 1080p | KV-109 1080p |
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
[SLS Discord 上的 SD 1.5 示例](https://discord.com/channels/1196376491815092265/1199059436199759943/1492139124348420106)
和[更多](https://discord.com/channels/1196376491815092265/1199059436199759943/1516571355317800990)。
