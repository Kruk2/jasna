[English](README.md) | [日本語](README.ja.md) | [**中文**](README.zh.md)

# Jasna

Jasna 是一个 JAV 马赛克修复工具，提供简洁 GUI、CLI、纯 GPU 处理流水线、NVIDIA TensorRT 与实验性 AMD ROCm 支持、可选二级修复模型、静态图像修复以及流媒体支持。

它受 [Lada](https://codeberg.org/ladaapp/lada) 启发，部分代码也基于 Lada。Jasna 使用的 `mosaic_restoration_1.2` 修复模型由 Lada 作者 ladaapp 训练。

Jasna 是免费的。支持者会获得一个密钥，用于解锁为本项目训练的额外模型: **unet-4x** 二级放大模型，以及实验性的 **SD 1.5 图像修复**模型。详情见[支持本项目](#支持本项目)。

![Jasna GUI](https://github.com/user-attachments/assets/ae5d9b73-ea22-4263-8203-0ff89bbbcc51)

## 目录

- [Jasna 能做什么](#jasna-能做什么)
- [社区](#社区)
- [要求](#要求)
- [快速开始](#快速开始)
- [区间编辑器](#区间编辑器)
- [提交更好的遮罩](#提交更好的遮罩)
- [VR180 视频](#vr180-视频)
- [导出后操作](#导出后操作)
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
- 默认使用 RF-DETR 模型检测马赛克；也提供 Lada 和 ZeLeFans YOLO 模型。
- 可逐眼处理并排 VR180 视频，并可在检测和修复时使用鱼眼重投影。
- 通过时间重叠和交叉淡化减少片段边界闪烁。
- 可使用 **unet-4x**、**RTX Super Resolution** 或 **Topaz Video AI** 进行二级修复。
- 可将修复后的视频串流到内置浏览器播放器，或支持的 Stash 分支。

## 社区

加入 [SLS Discord](https://discord.gg/uNwQ4mHqgv) 查看示例、获取支持，并讨论设置。请不要表现得太奇怪。

## 要求

发行包按 GPU 厂商区分:

- NVIDIA: 计算能力 **7.5 或更新**的现代 Nvidia GPU。
- GPU 粗略判断: **GTX 16 系列**、**RTX 20 系列**、**RTX 30 系列**、**RTX 40 系列**、**RTX 50 系列**，以及更新的工作站/数据中心显卡。
- 太旧: **GTX 10 系列**，包括 GTX 1050/1060/1070/1080。
- 精确 GPU 查询请查看 NVIDIA 的 [CUDA GPU compute capability table](https://developer.nvidia.com/cuda/gpus)。
- Nvidia 驱动: Windows 需 **610.00 或更高版本**，Linux 需 **580.xx 或更高版本**。内置的
  FFmpeg 8.1.2 库使用 NVENC API 13.1，旧版驱动不提供此 API。旧版驱动可能仍能初始化
  CUDA 推理，但硬件视频编码会失败。
- AMD（实验性，仅 Linux）: ROCm 支持的 AMD GPU 和 ROCm 7.2。当前发行目标为
  Ubuntu 24.04、PyTorch 2.9.1 和 Python 3.12。
- 安装路径只能包含 ASCII 字符。
- Windows 发行包: 已包含 `ffmpeg` 和 `ffprobe`。
- Linux 发行包: 已包含 `ffmpeg` 和 `ffprobe`。

Jasna 会自动管理 VRAM。当 GPU 显存不足时，处理队列中等待的帧会临时移动到系统内存，并在需要时移回。无需配置。

## 快速开始

1. 下载与你的操作系统和 GPU 厂商匹配的发行包。
2. 解压到只包含 ASCII 字符的路径。
3. 启动应用:
   - Windows: 双击 `jasna.exe`。
   - Linux NVIDIA: 运行 `jasna` 文件。
   - Linux AMD: 运行 `run_jasna_amd.sh`。
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

## 区间编辑器

<!-- 截图占位符：在此添加最终的区间编辑器截图。 -->

区间编辑器可预览队列中的视频，并按帧精确选择需要修复的区间；不选择任何区间时会修复完整视频。**修复预览**可在正式处理前，用当前修复设置显示当前帧或短片段的效果。

选择一个或多个区间后，编辑器会提示：为直接流复制未选择的部分，导出将使用源视频编解码器；主界面的 **编码** 编解码器选项不会生效。

编辑器内置马赛克扫描:

- 可逐帧扫描，或按 0.25–2 秒间隔扫描。扫描完全在 GPU 上运行，在 **RTX 5090 上约为 2,000 FPS**；实际速度取决于视频、模型和设置。
- 扫描后调整置信度会立即更新琥珀色检测区间，然后可将其加入紫色修复选择。
- 采样帧显示已保存蒙版，未采样帧会按需精确检测。显存不足时会自动通过系统内存循环存储结果。

检测模型和置信度会为队列中的每个视频单独保存，并用于最终处理，因此不同视频可以使用不同设置。

## 提交更好的遮罩

当马赛克检测结果不正确时，你可以提交修正后的遮罩，帮助训练更好的检测模型。在区间编辑器中暂停到该帧并点击**提交更好的遮罩**：

- 点击在每个马赛克区域周围添加顶点 — 每个区域一个形状，需要几个就画几个。点击画面外会自动吸附到边缘。
- 点击第一个顶点、双击或按 Enter 闭合形状。
- 滚轮缩放以便精确勾画，右键拖动平移，按 H 暂时隐藏形状，并可用不透明度滑块调整遮罩显示。
- 请尽量准确：如果马赛克边缘模糊或渐变，也请把这部分柔和区域包含进来。

提交时帧和遮罩会**匿名**上传。数据在你的电脑上加密后才上传，且仅附带应用版本、检测模型名称和帧分辨率 — 绝不包含文件名、时间戳或任何可识别你的信息。

## VR180 视频

VR180 文件通常会把左眼和右眼画面并排放在一个宽画面中。Jasna 会拆分画面，分别修复双眼，再把双眼重新组合到输出视频中。

### 快速设置

1. 像添加普通视频一样添加 VR 视频。
2. 将 **VR180 模式** 保持为 **自动（推荐）**。
3. 为了获得最佳 VR 马赛克检测效果，请选择内置的 **zelefans-vr-yolo-v2** 检测模型。该模型已包含在 Jasna 中，无需另行下载。
4. 正常开始处理。如果只想修复视频的一部分，也可以使用区间编辑器。

自动模式始终会把高度超过 1080 像素的严格 2:1 视频视为并排 VR。例如，3840x1920 和 8192x4096 都会被自动识别。兼容的 VR 元数据和已知 VR 片商名称也可以启用 VR 处理。

### 模式说明

- **自动（推荐）**：让 Jasna 自动判断。建议先使用此模式。
- **关闭**：把文件当作普通平面视频处理。
- **SBS — 分眼处理**：如果自动模式未识别视频，强制使用并排 VR 处理。
- **SBS + 鱼眼**：当马赛克在 VR 画面边缘被严重拉伸，或普通 SBS 模式漏检时使用。此模式会临时校正镜头畸变，以改善检测和修复。

### 预览和输出

为了让画面足够大、便于检查，区间编辑器有意只显示左眼。这并不表示右眼会被忽略：马赛克扫描、选定区间、修复预览和最终导出都会处理双眼。

为了获得最好的 VR 播放器兼容性，建议导出为 MP4 或 MOV。Jasna 会保留兼容的源 VR 元数据；如果没有，则添加标准并排 VR180 元数据。其他容器仍包含修复后的双眼画面，但部分 VR 播放器可能无法自动将其识别为 VR。

**SBS + 鱼眼** 只是内部处理选项，不会把完成的视频转换为另一种投影格式。输出会保持与源视频相同的画面布局。流媒体行为不变。

命令行用户可以通过 `--vr-mode` 选择相同模式:

```bash
jasna --input input.mp4 --output output.mp4 --vr-mode auto
jasna --input input.mp4 --output output.mp4 --vr-mode sbs-fisheye
```

## 导出后操作

GUI 可以在整个队列完成后执行操作: **无**、**关闭电脑** 或 **自定义命令**。Windows 和 Linux 的 CLI 也支持同一功能:

```bash
jasna --input input.mp4 --output output.mkv --post-export-action shutdown
```

自定义命令会在所有导出完成后通过系统 shell 运行:

```bash
jasna --input input_folder --output output_folder --post-export-action command --post-export-command "echo done"
```

## 首次运行

首次运行会比较慢，因为需要准备 GPU 专用检测产物。NVIDIA 会编译 TensorRT 引擎（通常 **15-60 分钟**）；AMD 会生成 RF-DETR MIGraphX 缓存。

编译期间请关闭其他应用，包括浏览器，并避免使用电脑。引擎会缓存在 `model_weights` 中，后续运行会复用。你可以把旧 Jasna 版本中的引擎文件和文件夹复制到新版本中。

如果处理时显存不足，请先降低 **max clip size**，例如从 `180` 降到 `60`。禁用 BasicVSR++ 编译也会降低峰值 VRAM，但处理速度会变慢。

## 模型选择

### 检测模型

通常建议使用最新的 RF-DETR 模型。Lada YOLO 模型也可用，并且在 2D 动画上可能效果更好。VR180 可使用 [ZeLeFans VR Mosaic Remover](https://huggingface.co/zelefans/vrmr) 的高精度检测模型 `zelefans-vr-yolo-v2`，该模型已内置于 Jasna。

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

在 Windows 上，串流也使用应用本身的文件: `jasna.exe --stream`。可能没有单独的 `jasna-cli.exe`。

### Stash 集成

Jasna 可以通过自定义 Stash 分支在 [Stash](https://github.com/stashapp/stash) 内使用。播放场景时，Stash 会自动启动 Jasna，并在观看时实时处理。支持跳转。

自定义分支: **[Stash v0.30.1-jasna](https://github.com/Kruk2/stash/releases/tag/v0.30.1-jasna)**

设置:

1. 从上方链接下载 Stash 分支。
2. 启动 Stash 前设置环境变量:
   - `JASNA_CLI_PATH`: `jasna.exe` 的完整路径，除非你自己重命名了它。
   - `JASNA_WORKING_DIR`: 包含该可执行文件的文件夹完整路径。
3. **重要:** 使用 Stash 前，先用同样的设置在短视频上串流一次。这样可以预编译 TensorRT 引擎，避免第一次健康检查超时。
4. 启动 Stash 并播放场景。

如果 Stash 日志出现 `timeout waiting for jasna-cli to become healthy`，先检查 `JASNA_CLI_PATH`，然后按上面的方法预编译。

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

- SeedVR 支持。
- 持续改善性能和 VRAM 使用。
- 更好的修复模型。
- 更好的检测模型。

## 从源代码运行

`pyproject.toml` 中的 Python 要求: **Python 3.12 或更新**。

公开源代码检出不包含 protection module。它可以用于开发和免费模型，但普通源代码检出无法使用 **unet-4x** 和 **SD 1.5 图像修复** 等支持者专用模型。

安装对应 GPU 厂商的运行时依赖:

```bash
# NVIDIA
uv pip install ".[nvidia]" --extra-index-url https://download.pytorch.org/whl/cu130

# AMD Linux（在 ROCm 7.2 环境中）
uv pip install ".[amd]" \
  --find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/
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

然后以 editable mode 安装 Jasna:

```bash
uv pip install -e ".[nvidia,dev]"  # 或 .[amd,dev]
```

AMD 发行版使用专用 Docker 脚本构建，不会修改主机 Python 环境:

```bash
jasna/protection/keytool/build_linux_amd.sh
jasna/protection/keytool/validate_amd_ssh.sh user@amd-host
```

AMD 版使用 PyTorch ROCm 运行 BasicVSR++/YOLO，使用 MIGraphX 运行 RF-DETR，并使用
AMF 对 H.264/HEVC/AV1 进行硬件解码和编码。AMF 无法解码时会回退到 FFmpeg 软件解码。
二级修复和区间智能渲染目前仍仅支持 NVIDIA。为避免动态卷积形状反复分析性能，AMD
默认设置 `MIOPEN_FIND_MODE=FAST`。

`--device cuda:N` 会选择 PyTorch/MIGraphX GPU，也会传递给 NVIDIA 视频 I/O。
FFmpeg 8 的 Linux AMF 设备上下文目前会忽略适配器参数，因此在多 GPU AMD 主机上，
AMF 解码/编码可能使用默认 Vulkan 适配器。如果必须确定 AMF 使用哪张卡，请在容器或
主机层面只暴露目标 GPU。
