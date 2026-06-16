[**English**](README.md) | [日本語](README.ja.md) | [中文](README.zh.md)

# Jasna
### 🚀 Jasna 是免费的。[支持它](#支持本项目)，让我能训练更好的模型 🚀
支持者将获得一个密钥，解锁我额外训练的模型——**unet-4x** 二级放大模型和实验性的
**SD 1.5 图像修复**模型。详情见[支持本项目](#支持本项目)。

受 [Lada](https://codeberg.org/ladaapp/lada) 启发（部分基于其代码）的 JAV 模型修复工具。\
Jasna 使用的修复模型（mosaic_restoration_1.2）由 ladaapp（Lada 作者）训练。

全新马赛克检测模型 & **超快速纯 GPU 流水线** & TVAI 支持 & 简洁 GUI。\
请查看下方的基准测试和使用说明。
![slop_gui](https://github.com/user-attachments/assets/ae5d9b73-ea22-4263-8203-0ff89bbbcc51)


### 差异:
- **改进的马赛克检测模型。**
- **减少闪烁的时间重叠。**
- **二级修复模型（Topaz TVAI 和 RTX Super Res）。提升画质和锐度。**
- **静态照片的实验性 SD 1.5 图像修复**——有时能产生非常好的结果（支持者模型）。
- 纯 GPU 处理。初步测试表明速度可提升 2 倍。无马赛克区域的原始处理在 RTX 5090 上约 250fps <img width="860" height="56" alt="image" src="https://github.com/user-attachments/assets/a80ecaee-e36d-4c91-93e4-8bdd75048ac3" />
- GPU 上精确的色彩转换（输入与输出匹配，无色带）。
- 仅支持较新的 Nvidia GPU。
- TensorRT 支持。

### 待办:
- 完善 VR 支持
- ~~TVAI~~ 和 SeedVR
- ~~可在 Stash（及其他工具）中播放的流媒体~~
- ~~提升性能（当前版本非常简单）~~
- ~~改善 VRAM 使用~~

### VRAM 管理
Jasna 自动管理 VRAM。当 GPU 显存不足时，处理队列中等待的帧会被临时移动到系统内存，需要时再移回。这在后台自动完成，无需任何配置。

如果仍然遇到内存不足错误，请减小**片段大小**（例如从 180 减到 60）或禁用模型编译——两者都能显著降低峰值 VRAM（见下表）。

### 基准测试
RTX 5090 + i9 13900k
| 文件 | 片段（秒） | lada 0.10.1 | jasna 0.3.0 | jasna 0.5.0 | **jasna 0.6.2** |
|---|---|---|---|---|---|
| **ABF-017**（4k，2小时25分） | 60 | 02:56:26 | 01:20:49（快 2.2 倍） | 01:10:00（快 2.5 倍） | xx |
| **HUBLK-063**（1080p，3小时10分） | 180 | 01:34:51 | 44:21（快 2.1 倍） | 37:57（快 2.5 倍） | **30:58（快 3.1 倍）** |
| **DASS-570_2m** | 30 | 01:08 | 00:30（快 2.3 倍） | 00:24（快 2.8 倍） | **00:20（快 3.4 倍）** |
| **NASK-223_Test** | 30 | 03:12 | 01:18（快 2.5 倍） | 01:02（快 3.1 倍） | **00:58（快 3.3 倍）** |
| **test-007** | 30 | 01:16 | 00:41（快 1.9 倍） | 00:28（快 2.7 倍） | **00:22（快 3.5 倍）** |
| **厚码测试2** | 30 | 01:52 | 00:43（快 2.6 倍） | 00:36（快 3.1 倍） | **00:34（快 3.3 倍）** |



## 支持本项目
支持用于训练额外的模型——主要是租用更大的 GPU 以及在更大数据集上训练所需的算力。支持者将获得一个密钥，解锁我以此方式训练的模型：
- **unet-4x** 二级放大模型（更清晰的 256→1024 修复）。
- **SD 1.5 图像修复**（实验性静态图像模型）。

如何获取密钥：
1. 累计贡献 **15 USD 或以上**——不限次数、任何时间均可。
2. 如何拿到密钥取决于你在哪里贡献：
   - **[Unifans](https://app.unifans.io/c/kruk2)**——我会直接把密钥私信给你。如果我忘了，戳我一下。
   - **[请我喝杯咖啡](https://buymeacoffee.com/kruk2)**（也支持**加密货币**）——用你贡献时使用的邮箱/账号发邮件到 **myprotonmailkekw@proton.me**，我会把密钥发给你。密钥与该邮箱绑定。

### 使用方法
下载最新的发行包（Windows/Linux）。\
解压发行包，然后双击 jasna.exe（Windows）或 jasna 文件（Linux）。这会打开控制台。\
你也可以把它作为命令行程序使用。

- **如果你下载了应用程序（推荐）**：
  - **Windows**：开箱即用——Jasna 已包含所需的一切（`ffmpeg`、`ffprobe` 和 `mkvmerge`）。
  - **Linux**：需要系统中有 `ffmpeg`、`ffprobe`（**主版本号必须为 8**）和 `mkvmerge`。通过包管理器安装。MKVToolNix：[下载](https://mkvtoolnix.download/downloads.html)。

**首次运行会较慢**——TensorRT 引擎会针对你的 GPU 进行编译，需要 **15-60 分钟**。\
编译期间请关闭所有其他应用程序（包括浏览器），不要使用电脑。\
引擎缓存在 `model_weights` 文件夹中，后续运行会自动复用（你可以将引擎文件和文件夹复制到新版本中）。

**请确保 Nvidia 驱动程序为最新版。**\
已测试驱动：**591.67**（但 59x 系列的任何版本应该都可以，这是最低要求版本）。\
**Jasna 要求 GPU 最低计算能力：7.5**

### 检测模型
通常建议选择最新的 rf-detr。\
Lada Yolo 模型可用，因为它们更擅长处理 2D 动画。

### 二级修复模型
Jasna/Lada 提取马赛克区域的 256×256 像素裁切并以 256×256 分辨率修复。当马赛克区域较大时（特写、4K 视频等），结果会模糊。\
为缓解此问题，可以使用第二个修复模型将 256×256 放大到 512×512 或 1024×1024，以获得更清晰的结果。
目前支持：
- **unet-4x**（支持者模型，比 TVAI 快很多，画质相近）。在领域内（JAV）数据集上训练，视觉效果相当接近 TVAI iris-2，但可在本地运行，无需额外设置。如果遇到画质问题，请提交 [GitHub issue](https://github.com/Kruk2/jasna/issues)。使用支持者密钥解锁——见[支持本项目](#支持本项目)。
- **RTX Super-resolution**（非常快，画质尚可）。非常快，免费，零依赖。某些视频中可能产生闪烁效果——请先在短片段上测试。将 Jasna 放在仅含英文字符的文件夹中。
- **TVAI**（优于 RTX Super Resolution，接近 unet-4x，非常慢）。需要 [Topaz Video](https://www.topazlabs.com/topaz-video)（付费，仅限 Windows）。推荐模型：**iris-2**。\
  ```--tvai-args``` 允许你自定义模型和其他参数。默认为 iris-2。\
  为 "Topaz Video" 设置以下环境变量：\
  <img width="505" height="37" alt="image" src="https://github.com/user-attachments/assets/e19ced9d-d549-4e85-b20f-888e42466f1d" />

VRAM + 处理时间：
| 二级类型 | CAWD 1080p | KV-109 1080p |
|---|---:|---:|
| 无二级修复 | 22秒 / 10.0 GB VRAM | 11秒 / 10.7 GB VRAM |
| unet-4x | 29秒 / 12.5 GB VRAM | 14秒 / 12.6 GB VRAM |
| RTX Super-Res | 25秒 / 11.7 GB VRAM | 13秒 / 11.4 GB VRAM |
| TVAI（2 workers, Iris-2） | 52秒 / 12.1 GB VRAM | 24秒 / 12.4 GB VRAM |

修复示例可在 [SLS discord](https://discord.com/channels/1196376491815092265/1199059436199759943/1516497879684874260) 查看。

### 图像修复（SD 1.5）
对于**静态图像**，Jasna 可以使用微调过的 Stable Diffusion 1.5 inpaint 模型，而不是视频流水线。它会检测马赛克，在 512×512 下对每个区域进行 inpaint，再把结果混合回原图。

修复示例可以查看 [SLS discord](https://discord.com/channels/1196376491815092265/1199059436199759943/1492139124348420106) 和[这里](https://discord.com/channels/1196376491815092265/1199059436199759943/1516571355317800990)。

- CLI：`jasna --input photo.png --output out.png`——图像输入会**自动路由**到 SD 1.5 模型，无需任何标志。图像模型通过 `--image-restoration-model-name` 选择（默认且唯一的值：`sd-15-jav`）；`--restoration-model-name` 仅用于视频。可调参数：`--sd15-steps`、`--sd15-strength`（≤ 0.7）、`--sd15-freeu/--no-sd15-freeu`、`--sd15-seed`、`--sd15-variants N`。
- GUI：只需把图像加入队列——图像任务会自动路由到 SD 1.5 模型。在 **Image Restoration**（图像修复）设置区进行调节。
- 文件夹输入（CLI）：`--input <文件夹> --output <文件夹>` 会处理文件夹中的每个媒体文件——**先图像，后视频**——将 `<name>_out<ext>` 写入输出文件夹（此时 `--output` 必须是文件夹）。GUI 队列本身就可以自由混合图像和视频。
- 该模型**未随程序打包**（约 6.9 GB），位于 `model_weights/sd-15-jav/`。你可以自己把模型包放进去，或让 Jasna 从 [huggingface.co/Kruk2/sd-15-jav](https://huggingface.co/Kruk2/sd-15-jav) 下载——下载前会先询问（CLI 提示，或 GUI 中的 **Download model** 按钮）。目前 checkpoint 仅面向支持者提供——与 unet-4x 使用同一个密钥——见[支持本项目](#支持本项目)。
- 它是**实验性的**：效果因场景而异，但在合适的图像上可以看起来相当不错。尝试几个 `--sd15-variants`（不同的种子），保留最好的那个。
- 推理期间大约需要 **7 GB 显存**（较大的 4K 图像会再多一些）。

### 最大片段大小 + 时间重叠
时间重叠的主要目的是减少片段边缘的闪烁。\
超过 20 效果提升不大。重叠越大，处理时间越长，但闪烁越少。\
**选择你能使用的最大片段大小，将重叠设置为 8-20 之间。**\
来自有限测试的参考：
- 片段大小 60 + 时间重叠 6
- 片段大小 90 + 时间重叠 8
- 片段大小 180 + 时间重叠 15（启用 BasicVSR++ 编译时需要 12 GB 以上显存，禁用时需要更少）。

4K 视频使用更多显存——较小的片段大小可能产生类似的画质但处理速度更快。可以多试试。

片段大小低于 60 可能看起来没问题，取决于视频，但建议即使禁用模型编译也使用 60。
```--enable-crossfade``` 可减少闪烁，建议始终启用。

### 修复模型编译
修复模型被编译为 TensorRT 子引擎。\
首次编译需要 **15-60 分钟**——请关闭所有其他应用程序（包括浏览器），不要使用电脑。\
引擎会被缓存并自动复用。你可以选择不编译，但会牺牲性能。

下表显示**仅编译引擎**占用的显存（非总处理显存）：

| | 片段 60 | 片段 180 |
|---|---|---|
| **引擎显存（已编译）** | 约 1.9 GB | 约 5.4 GB |
| **引擎显存（未编译）** | 约 1.2 GB | 约 1.2 GB |

### 流媒体

无需处理整个文件即可实时观看修复后的视频。

**浏览器播放器（内置）**\
使用 `--stream` 参数运行 jasna（目前仅限 CLI）。浏览器窗口将打开 HLS 播放器。选择视频文件开始观看。支持跳转。
```
jasna --stream
```

**Stash 集成**\
在 [Stash](https://github.com/stashapp/stash) 中直接使用 Jasna——播放任何场景，Jasna 会实时处理。Stash 会自动启动 Jasna。提供了支持 Jasna 的自定义 Stash 分支：\
👉 **[Stash v0.30.1-jasna](https://github.com/Kruk2/stash/releases/tag/v0.30.1-jasna)**

设置：
1. 从上方链接下载 Stash 分支。
2. 启动 Stash 前设置环境变量：
   - `JASNA_CLI_PATH` — `jasna-cli.exe` 的完整路径
   - `JASNA_WORKING_DIR` — `jasna-cli.exe` 所在文件夹的完整路径
3. 启动 Stash 并播放任何场景——Jasna 会在你观看时实时处理。支持跳转。

### 免责声明
Jasna 处于早期开发阶段，主要目标是改善：修复质量、马赛克检测、速度和显存消耗（按此优先级）。
因此，当前项目面向技术用户，程序的易用性较低，可能对部分用户有门槛。
这样做是为了将更多时间投入到重要功能上。如果你想帮忙，欢迎提交 Pull Request。

## 构建
通过 ```uv pip install . --no-build-isolation``` 安装以下库。\
构建下方的 Nvidia 库还需要 VS Build Tools 2022（C++）。
确保有 cmake 和 ninja ```uv pip install cmake ninja```\
以及系统中的 CUDA 13.0。

### 从源代码运行（开发者设置）
- 安装 `ffmpeg` + `ffprobe` 并确保它们在 PATH 中（**ffmpeg 主版本号必须为 8**）。
- 安装 `mkvmerge`（MKVToolNix 的一部分）：[下载](https://mkvtoolnix.download/downloads.html)。

https://codeberg.org/Kruk2/vali

https://codeberg.org/Kruk2/PyNvVideoCodec

将上述两个库安装到你的 Python 环境后，在 jasna 仓库中运行：
```uv pip install -e .[dev]```
