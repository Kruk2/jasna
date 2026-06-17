[English](README.md) | [日本語](README.ja.md) | [**中文**](README.zh.md)

# Jasna

Jasna 是一个 JAV 马赛克修复工具，提供简洁 GUI、CLI、纯 GPU 处理流水线、TensorRT 支持、可选二级修复模型、静态图像修复以及流媒体支持。

它受 [Lada](https://codeberg.org/ladaapp/lada) 启发，部分代码也基于 Lada。Jasna 使用的 `mosaic_restoration_1.2` 修复模型由 Lada 作者 ladaapp 训练。

Jasna 是免费的。支持者会获得一个密钥，用于解锁为本项目训练的额外模型: **unet-4x** 二级放大模型，以及实验性的 **SD 1.5 图像修复**模型。详情见[支持本项目](#支持本项目)。

![Jasna GUI](https://github.com/user-attachments/assets/ae5d9b73-ea22-4263-8203-0ff89bbbcc51)

## 目录

- [Jasna 能做什么](#jasna-能做什么)
- [社区](#社区)
- [要求](#要求)
- [快速开始](#快速开始)
- [首次运行](#首次运行)
- [模型选择](#模型选择)
- [调整质量和 VRAM](#调整质量和-vram)
- [流媒体](#流媒体)
- [基准测试](#基准测试)
- [支持本项目](#支持本项目)
- [当前限制和 TODO](#当前限制和-todo)
- [从源代码运行](#从源代码运行)

## Jasna 能做什么

- 修复视频文件中的马赛克。
- 使用实验性 SD 1.5 图像模型修复静态图像中的马赛克。
- 默认使用 RF-DETR 模型检测马赛克；也提供 Lada YOLO 模型。
- 通过时间重叠和交叉淡化减少片段边界闪烁。
- 可使用 **unet-4x**、**RTX Super Resolution** 或 **Topaz Video AI** 进行二级修复。
- 可将修复后的视频串流到内置浏览器播放器，或支持的 Stash 分支。

## 社区

加入 [SLS Discord](https://discord.gg/5R2Rx5nBH) 查看示例、获取支持，并讨论设置。请不要表现得太奇怪。

## 要求

- 计算能力 **7.5 或更新**的现代 Nvidia GPU。
- GPU 粗略判断: **GTX 16 系列**、**RTX 20 系列**、**RTX 30 系列**、**RTX 40 系列**、**RTX 50 系列**，以及更新的工作站/数据中心显卡。
- 太旧: **GTX 10 系列**，包括 GTX 1050/1060/1070/1080。
- 精确 GPU 查询请查看 NVIDIA 的 [CUDA GPU compute capability table](https://developer.nvidia.com/cuda/gpus)。
- 最新 Nvidia 驱动。已测试驱动: **591.67**。59x 驱动系列是最低预期系列。
- 安装路径只能包含 ASCII 字符。
- Windows 发行包: 已包含 `ffmpeg`、`ffprobe` 和 `mkvmerge`。
- Linux 发行包: 系统中需要有 `ffmpeg`、`ffprobe` 和 `mkvmerge`。`ffmpeg` 主版本号必须为 **8**。`mkvmerge` 是 [MKVToolNix](https://mkvtoolnix.download/downloads.html) 的一部分。

Jasna 会自动管理 VRAM。当 GPU 显存不足时，处理队列中等待的帧会临时移动到系统内存，并在需要时移回。无需配置。

## 快速开始

1. 下载最新的 Windows 或 Linux 发行包。
2. 解压到只包含 ASCII 字符的路径。
3. 启动应用:
   - Windows: 双击 `jasna.exe`。
   - Linux: 运行 `jasna` 文件。
4. 在 GUI 中添加视频或图像，选择设置，然后开始处理。

也可以通过命令行使用 Jasna:

```bash
jasna --input input.mp4 --output output.mkv
```

静态图像不需要额外的图像专用参数:

```bash
jasna --input photo.png --output restored.png
```

使用文件夹输入时，`--input` 和 `--output` 都必须是文件夹。Jasna 会先处理图像，再处理视频，显示整体 `[current/total]` 文件计数，并默认将 `<name>_out<ext>` 写入输出文件夹。

```bash
jasna --input input_folder --output output_folder
```

文件夹批处理也可以使用与 GUI 相同的 `{original}` 文件名模板:

```bash
jasna --input input_folder --output output_folder --output-pattern "{original}_restored.mp4"
```

图像会保留源文件扩展名；视频会在模板提供扩展名时使用该扩展名。Jasna 会在处理前检查计划输出路径，如果模板让多个输入映射到同一个输出文件，则会报错退出。

## 首次运行

首次运行会比较慢，因为 TensorRT 引擎会针对你的 GPU 编译。通常需要 **15-60 分钟**。

编译期间请关闭其他应用，包括浏览器，并避免使用电脑。引擎会缓存在 `model_weights` 中，后续运行会复用。你可以把旧 Jasna 版本中的引擎文件和文件夹复制到新版本中。

如果处理时显存不足，请先降低 **max clip size**，例如从 `180` 降到 `60`。禁用 BasicVSR++ 编译也会降低峰值 VRAM，但处理速度会变慢。

## 模型选择

### 检测模型

通常建议使用最新的 RF-DETR 模型。Lada YOLO 模型也可用，并且在 2D 动画上可能效果更好。

CLI 选项:

```bash
jasna --input input.mp4 --output output.mkv --detection-model rfdetr-v5
```

### 二级修复

Jasna 和 Lada 会修复每个马赛克区域的 256x256 裁切图。因此，大马赛克区域、特写和 4K 视频在一次修复后可能看起来模糊。二级修复模型可以先将修复后的裁切图放大到 512x512 或 1024x1024，再混合回原视频。

支持的二级模型:

- **unet-4x**: 支持者模型。当前测试中比 TVAI 更快，质量相近。它在 JAV 领域数据集上训练，视觉效果接近 TVAI `iris-2`。可以查看 [unet-4x / 二级修复示例（SLS Discord）](https://discord.com/channels/1196376491815092265/1199059436199759943/1516497879684874260)。使用支持者密钥解锁；见[支持本项目](#支持本项目)。如果遇到质量问题，请提交 [GitHub issue](https://github.com/Kruk2/jasna/issues)。
- **RTX Super Resolution**: 非常快、免费、没有额外依赖。质量尚可。部分视频可能会闪烁，请先用短片段测试。
- **TVAI**: 当前测试中质量优于 RTX Super Resolution，并与 unet-4x 接近，但非常慢。需要 [Topaz Video](https://www.topazlabs.com/topaz-video)，这是付费软件且仅支持 Windows。推荐模型: `iris-2`。

CLI 选项:

```bash
jasna --input input.mp4 --output output.mkv --secondary-restoration unet-4x
```

对于 TVAI，`--tvai-args` 可以自定义 Topaz 模型参数。默认模型是 `iris-2`。请为 Topaz Video 设置这些环境变量:

<img width="505" height="37" alt="Topaz Video environment variables" src="https://github.com/user-attachments/assets/e19ced9d-d549-4e85-b20f-888e42466f1d" />

VRAM 和处理时间:

| 二级类型 | CAWD 1080p | KV-109 1080p |
| --- | ---: | ---: |
| 无二级修复 | 22秒 / 10.0 GB VRAM | 11秒 / 10.7 GB VRAM |
| unet-4x | 29秒 / 12.5 GB VRAM | 14秒 / 12.6 GB VRAM |
| RTX Super-Res | 25秒 / 11.7 GB VRAM | 13秒 / 11.4 GB VRAM |
| TVAI (2 workers, Iris-2) | 52秒 / 12.1 GB VRAM | 24秒 / 12.4 GB VRAM |

修复示例可在 [SLS Discord](https://discord.com/channels/1196376491815092265/1199059436199759943/1516497879684874260) 查看。

### 静态图像修复

对于静态图像，Jasna 可以使用微调过的 Stable Diffusion 1.5 inpaint 模型，而不是视频流水线。它会检测马赛克，在 512x512 下对每个区域进行 inpaint，再把结果混合回原图。

- CLI: `jasna --input photo.png --output out.png`
- GUI: 将图像加入队列。图像任务会自动路由到 SD 1.5。
- 调整选项: `--sd15-steps`、`--sd15-strength`（限制为 `<= 0.7`）、`--sd15-freeu` / `--no-sd15-freeu`、`--sd15-seed`、`--sd15-variants N`。
- 图像模型通过 `--image-restoration-model-name` 选择。当前默认且唯一的值是 `sd-15-jav`。
- `--restoration-model-name` 仅用于视频。

SD 1.5 模型**未随程序打包**，大小约 **6.9 GB**。它应放在 `model_weights/sd-15-jav/`。你可以自己把模型包放到那里，或让 Jasna 从 [huggingface.co/Kruk2/sd-15-jav](https://huggingface.co/Kruk2/sd-15-jav) 获取。Jasna 会在下载前询问，可以通过 CLI 提示或 GUI 的 **Download model** 按钮确认。

checkpoint 目前仅面向支持者提供，并使用与 unet-4x 相同的密钥。详情见[支持本项目](#支持本项目)。

SD 1.5 路径是实验性的。结果因场景而异，但合适的图像可能效果很好。可以尝试多个 `--sd15-variants`，保留最好的结果。推理期间大约需要 **7 GB VRAM**，较大的 4K 图像会再多一些。

示例可在 [SLS Discord](https://discord.com/channels/1196376491815092265/1199059436199759943/1492139124348420106) 和[更多 SD 1.5 示例](https://discord.com/channels/1196376491815092265/1199059436199759943/1516571355317800990)查看。

## 调整质量和 VRAM

### Max Clip Size 和 Temporal Overlap

时间重叠用于减少片段边界闪烁。数值越大，处理时间越长，但可能减少闪烁。超过 `20` 通常没有太大帮助。

推荐起点:

- 使用 GPU 能承受的最大 **max clip size**。
- 将 **temporal overlap** 设置在 `8` 到 `20` 之间。
- 保持 `--enable-crossfade` 启用。

有限测试中的参考:

| Max clip size | Temporal overlap | 说明 |
| ---: | ---: | --- |
| 60 | 6 | 较低 VRAM 选择。 |
| 90 | 8 | 接近当前默认设置的平衡点。 |
| 180 | 15 | 启用 BasicVSR++ 编译时需要 12 GB+ VRAM；禁用编译时需求更低。 |

4K 视频使用更多 VRAM。较低的片段大小可能产生类似质量，并且处理更快。低于 `60` 的片段大小在部分视频上可用，但即使需要禁用模型编译，也更推荐 `60`。

CLI 示例:

```bash
jasna --input input.mp4 --output output.mkv --max-clip-size 90 --temporal-overlap 8 --enable-crossfade
```

### 修复模型编译

修复模型会编译为 TensorRT 子引擎。编译会提高速度，但会使用更多 VRAM。你可以禁用它，以性能换取更低 VRAM:

```bash
jasna --input input.mp4 --output output.mkv --no-compile-basicvsrpp
```

下面仅为编译引擎本身占用的 VRAM，不是总处理 VRAM:

| | Clip 60 | Clip 180 |
| --- | ---: | ---: |
| Engine VRAM, compiled | 约 1.9 GB | 约 5.4 GB |
| Engine VRAM, no compilation | 约 1.2 GB | 约 1.2 GB |

## 流媒体

流媒体可以让你不必先处理完整文件，就实时观看修复后的视频。

### 浏览器播放器

流媒体模式目前仅支持 CLI。它会在浏览器中打开 HLS 播放器。选择视频文件即可开始观看。支持跳转。

```bash
jasna --stream
```

### Stash 集成

Jasna 可以通过自定义 Stash 分支在 [Stash](https://github.com/stashapp/stash) 内使用。播放场景时，Stash 会自动启动 Jasna，并在观看时实时处理。支持跳转。

自定义分支: **[Stash v0.30.1-jasna](https://github.com/Kruk2/stash/releases/tag/v0.30.1-jasna)**

设置:

1. 从上方链接下载 Stash 分支。
2. 启动 Stash 前设置环境变量:
   - `JASNA_CLI_PATH`: `jasna-cli.exe` 的完整路径
   - `JASNA_WORKING_DIR`: 包含 `jasna-cli.exe` 的文件夹完整路径
3. 启动 Stash 并播放场景。

## 基准测试

RTX 5090 + i9 13900k:

| 文件 | 片段（秒） | lada 0.10.1 | jasna 0.3.0 | jasna 0.5.0 | **jasna 0.6.2** |
| --- | ---: | ---: | ---: | ---: | ---: |
| **ABF-017** (4k，2小时25分) | 60 | 02:56:26 | 01:20:49 (快 2.2 倍) | 01:10:00 (快 2.5 倍) | xx |
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

1. 累计贡献 **15 USD 或以上**，不限次数、不限时间。
2. 贡献处理完成后，支持者密钥会自动发送:
   - **[Unifans](https://app.unifans.io/c/kruk2)**: 通过平台消息发送，可能会有轻微延迟。
   - **[Buy Me a Coffee](https://buymeacoffee.com/kruk2)**，包括**加密货币**: 发送到贡献时使用的邮箱或账号。密钥与该邮箱或账号绑定。

## 当前限制和 TODO

Jasna 仍处于早期开发阶段。主要目标是按顺序改善修复质量、马赛克检测、速度和 VRAM 使用。当前项目更面向技术用户，因此部分流程可能仍然粗糙。欢迎 Pull Request。

当前 TODO:

- 完善 VR 支持。
- SeedVR 支持。
- 持续改善性能和 VRAM 使用。

## 从源代码运行

`pyproject.toml` 中的 Python 要求: **Python 3.13 或更新**。

公开源代码检出不包含 protection module。它可以用于开发和免费模型，但普通源代码检出无法使用 **unet-4x** 和 **SD 1.5 图像修复** 等支持者专用模型。

安装运行时依赖:

```bash
uv pip install . --no-build-isolation
```

构建 Nvidia 库还需要:

- 带 C++ 支持的 VS Build Tools 2022。
- 系统中安装 CUDA 13.0。
- `cmake` 和 `ninja`:

```bash
uv pip install cmake ninja
```

开发者设置还需要:

- `ffmpeg` 和 `ffprobe` 位于 `PATH` 中；`ffmpeg` 主版本号必须为 **8**。
- [MKVToolNix](https://mkvtoolnix.download/downloads.html) 中的 `mkvmerge`。
- 将以下两个库安装到 Python 环境:
  - https://codeberg.org/Kruk2/vali
  - https://codeberg.org/Kruk2/PyNvVideoCodec

然后以 editable mode 安装 Jasna:

```bash
uv pip install -e .[dev]
```
