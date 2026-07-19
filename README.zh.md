[English](README.md) | [日本語](README.ja.md) | [**中文**](README.zh.md)

# <img width="32" src="https://github.com/Kruk2/jasna/blob/main/assets/jasna-logo.png?raw=true" /> Jasna

Jasna 是一个 JAV 马赛克修复工具，提供简洁 GUI、CLI、纯 GPU 处理流水线、NVIDIA TensorRT 与实验性 AMD ROCm 支持、可选二级修复模型、静态图像修复以及流媒体支持。

它受 [Lada](https://codeberg.org/ladaapp/lada) 启发，部分代码也基于 Lada。Jasna 使用的 `mosaic_restoration_1.2` 修复模型由 Lada 作者 ladaapp 训练。

Jasna 是免费的。支持者会获得一个密钥，用于解锁为本项目训练的额外模型: **unet-4x** 二级放大模型，以及实验性的 **SD 1.5 图像修复**模型。详情见[支持本项目](#支持本项目)。

<img width="1200" height="907" alt="image" src="https://github.com/user-attachments/assets/d59a914b-482d-4f37-ae72-5c59eb5dc9bb" />


## 目录

- [Jasna 能做什么](#jasna-能做什么)
- [社区](#社区)
- [要求](#要求)
- [快速开始](#快速开始)
- [首次运行](#首次运行)
- [了解更多](#了解更多)
- [基准测试](#基准测试)
- [支持本项目](#支持本项目)
- [TODO](#todo)

## Jasna 能做什么

- 修复视频文件中的马赛克。
- 使用实验性 SD 1.5 图像模型修复静态图像中的马赛克。
- 默认使用 RF-DETR 模型检测马赛克；也提供 Lada 和 ZeLeFans YOLO 模型。
- 可逐眼处理并排 VR180 视频，并可在检测和修复时使用鱼眼重投影。
- 通过时间重叠和交叉淡化减少片段边界闪烁。
- 可使用可选的[二级修复模型](docs/zh/models.md) — **unet-4x**、**RTX Super Resolution** 或 **Topaz Video AI** — 进一步提升质量，让修复区域更清晰，尤其是大面积马赛克、特写和 4K 视频。
- 可将修复后的视频串流到内置浏览器播放器，或支持的 Stash 分支。

## 社区

加入 [SLS Discord](https://discord.gg/uNwQ4mHqgv) 查看示例、获取支持，并讨论设置。请不要表现得太奇怪。

## 要求

- NVIDIA **GTX 16 系列 / RTX 20 系列或更新**的 GPU。GTX 10 系列及更旧的显卡（GTX 1050/1060/1070/1080）无法使用。不确定自己的显卡？请查看 NVIDIA 的 [GPU 表格](https://developer.nvidia.com/cuda/gpus) — 需要计算能力 7.5 或更高。
- Nvidia 驱动: Windows 需 **610 或更新**，Linux 需 **580 或更新**。
- AMD 支持是实验性的，需要 ROCm 支持的 GPU。
- 请把 Jasna 安装到路径只包含英文字母和数字的文件夹中。

Jasna 会自动管理 VRAM: 显存不足时，等待中的帧会临时移动到系统内存。无需任何配置。

## 快速开始

1. 下载与你的操作系统和 GPU 厂商匹配的发行包。
2. 解压到路径只包含英文字符的文件夹。
3. 启动应用:
   - Windows: 双击 `jasna.exe`。
   - Linux NVIDIA: 运行 `jasna` 文件。
   - Linux AMD: 运行 `run_jasna_amd.sh`。
4. 添加视频或图像，选择设置，然后开始处理。

GUI 中的每个设置都有提示 — 把鼠标悬停在旁边的 ⓘ 图标上即可查看。[GUI 指南](docs/zh/gui.md)会带你了解其余功能: 队列排序、预设、输出文件名模板等等。

更喜欢命令行？

```bash
# Single video
jasna --input input.mp4 --output output.mkv

# Still image
jasna --input photo.png --output restored.png

# Whole folder
jasna --input input_folder --output output_folder
```

运行 `jasna --help` 查看全部选项，或阅读 [CLI 参考](docs/zh/cli.md)。

## 首次运行

首次运行会比较慢，因为 Jasna 需要针对你的显卡准备 GPU 专用文件。NVIDIA 通常需要 **15-60 分钟**；AMD 的准备时间要短得多。这只会发生一次 — 结果会缓存在 `model_weights` 中，之后每次运行都会复用。你也可以把它们从旧版本 Jasna 复制到新版本中。

请关闭其他应用程序（包括浏览器），并在此期间避免使用电脑。

如果处理时显存不足，请先降低**最大片段大小**，例如从 `180` 降到 `60`。见[调整 VRAM 和 GPU 占用](docs/zh/tuning.md)。

## 了解更多

- **[使用 GUI](docs/zh/gui.md)** — 队列（拖放、排序）、预设、输出文件名模板与文件冲突，以及其他容易错过的功能。
- **[选择模型](docs/zh/models.md)** — 该选哪个检测模型、用二级修复（unet-4x / RTX Super Resolution / Topaz）获得更清晰的结果，以及 SD 1.5 静态图像修复。
- **[只修复视频的一部分](docs/zh/segments.md)** — 区间编辑器、内置马赛克扫描、提交更好的遮罩，以及 `--segments` CLI 参数。
- **[VR180 视频](docs/zh/vr180.md)** — Jasna 如何处理并排 VR，以及何时使用鱼眼模式。
- **[调整 VRAM 和 GPU 占用](docs/zh/tuning.md)** — 片段大小、时间重叠、模型编译，以及显存不足时该怎么办。
- **[高级处理](docs/zh/advanced_processing.md)** — 降噪、60→30 FPS 导出、色彩 LUT、自定义编码器设置和导出后操作。
- **[流媒体](docs/zh/streaming.md)** — 在浏览器中或通过 Stash 实时观看修复后的视频。
- **[CLI 参考](docs/zh/cli.md)** — 所有命令行选项，包括输出模板、各编解码器的编码器设置和导出后操作。
- **[从源代码运行](docs/en/development.md)** — 开发者环境搭建和构建说明。

## 基准测试

RTX 5090 + i9 13900k:

| 文件 | 片段（秒） | lada 0.10.1 | jasna 0.3.0 | jasna 0.5.0 | **jasna 0.6.2** |
| --- | ---: | ---: | ---: | ---: | ---: |
| **ABF-017** (4k，2小时25分) | 60 | 02:56:26 | 01:20:49 (快 2.2 倍) | 01:10:00 (快 2.5 倍) | — |
| **HUBLK-063** (1080p，3小时10分) | 180 | 01:34:51 | 44:21 (快 2.1 倍) | 37:57 (快 2.5 倍) | **30:58 (快 3.1 倍)** |
| **DASS-570_2m** | 30 | 01:08 | 00:30 (快 2.3 倍) | 00:24 (快 2.8 倍) | **00:20 (快 3.4 倍)** |
| **NASK-223_Test** | 30 | 03:12 | 01:18 (快 2.5 倍) | 01:02 (快 3.1 倍) | **00:58 (快 3.3 倍)** |
| **test-007** | 30 | 01:16 | 00:41 (快 1.9 倍) | 00:28 (快 2.7 倍) | **00:22 (快 3.5 倍)** |
| **厚码测试2** | 30 | 01:52 | 00:43 (快 2.6 倍) | 00:36 (快 3.1 倍) | **00:34 (快 3.3 倍)** |

## 支持本项目

支持用于训练额外模型，主要是租用 GPU，以及在更大数据集上训练所需的计算时间。支持者会获得一个密钥，用于解锁:

- **unet-4x** 二级放大模型，用于更清晰的 256->1024 修复。
- **SD 1.5 图像修复**，实验性静态图像模型。

结果示例:

- [unet-4x / 二级修复示例（SLS Discord）](https://discord.com/channels/1196376491815092265/1199059436199759943/1516497879684874260)
- [SD 1.5 图像修复示例（SLS Discord）](https://discord.com/channels/1196376491815092265/1199059436199759943/1492139124348420106) 和 [更多 SD 1.5 示例](https://discord.com/channels/1196376491815092265/1199059436199759943/1516571355317800990)

如何获取密钥:

1. 累计贡献 **15 美元或以上**，不限次数、不限时间。
2. 贡献处理完成后，支持者密钥会自动发送:
   - **[Unifans](https://app.unifans.io/c/kruk2)**: 通过平台消息发送，可能会有轻微延迟。
   - **[Buy Me a Coffee](https://buymeacoffee.com/kruk2)**，包括**加密货币**: 发送到贡献时使用的邮箱或账号。密钥与该邮箱或账号绑定。

## TODO

当前 TODO:

- SeedVR 支持？
- 持续改善性能和 VRAM 使用。
- 更好的修复模型。
- 更好的检测模型。
