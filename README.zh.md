> **支持者模型的正版许可证码只能通过 [Buy Me a Coffee](https://buymeacoffee.com/kruk2)、[Unifans](https://app.unifans.io/c/kruk2) 或 [闲鱼](#在闲鱼购买) 获取。**

[English](README.md) | [日本語](README.ja.md) | [**中文**](README.zh.md)

# <img width="32" src="https://github.com/Kruk2/jasna/blob/main/assets/jasna-logo.png?raw=true" /> Jasna

Jasna 修复 JAV 视频和图片中的马赛克。Jasna 免费，提供简单的应用和命令行，在你的 GPU
上运行。

<img width="1200" height="907" alt="image" src="https://github.com/user-attachments/assets/d59a914b-482d-4f37-ae72-5c59eb5dc9bb" />

## 目录

**从这里开始**

- [Jasna 能做什么](#jasna-能做什么)
- [要求](#要求)
- [快速开始](#快速开始)
- [帮助和社区](#帮助和社区)
- [支持本项目](#支持本项目)（支持者模型）

**高级**

- [Jasna 如何工作](#jasna-如何工作)
- [命令行](#命令行)
- [所有指南](#所有指南)
- [基准测试](#基准测试)
- [许可证](#许可证)

- [支持本项目](#支持本项目)
- [致谢](#致谢)

## Jasna 能做什么

- 找出视频中的马赛克并修复。画面的其他部分不会被改动。
- 两个修复模型：快速的 **BasicVSR++**（免费），以及慢但细节更多的 **LTX**（支持者专享，仅限 NVIDIA）。
- 在**区间编辑器**中只修复你选择的部分，并可预览。
- 可用于 VR180 视频和静态图片（支持者专享的 SD 1.5 模型）。
- 在**视频播放器**中实时观看修复后的视频，不用生成文件。

## 要求

- **NVIDIA** GTX 16 系列 / RTX 20 系列或更新，并使用较新的驱动
  （Windows 610 或更新，Linux 580 或更新）。GTX 10 系列及更旧的显卡无法使用。
- 或 **AMD** Radeon RX 7000/9000（实验性）。
- 本地磁盘上的一个文件夹，路径中不能有中文、日文、带重音的字母或其他非英文字符。

<details>
<summary>更多细节</summary>

- NVIDIA：你的 GPU 在 NVIDIA 的 [GPU 表格](https://developer.nvidia.com/cuda/gpus)
  中需要计算能力 7.5 或更高。
- AMD：ROCm 支持的 GPU（Radeon RX 7000/9000、Radeon 780M/760M/740M、Ryzen AI）。
  Windows 上需要 AMD Adrenalin 驱动 26.8.1 或更新。二次修复和 LTX 仅限 NVIDIA。
- 磁盘格式：Windows 用 NTFS；Linux 用 ext4、Btrfs 或 XFS。不支持网络磁盘。
- LTX 需要 NVIDIA RTX GPU，至少 8 GB 显存和 16 GB 内存。见
  [LTX 需要什么](docs/zh/models.md#ltx-需要什么)。
- NVIDIA **GTX 16 系列 / RTX 20 系列或更新**的 GPU。GTX 10 系列及更旧的显卡（GTX 1050/1060/1070/1080）无法使用。不确定自己的显卡？请查看 NVIDIA 的 [GPU 表格](https://developer.nvidia.com/cuda/gpus) — 需要计算能力 7.5 或更高。
- Nvidia 驱动: Windows 需 **610 或更新**，Linux 需 **580 或更新**。
- AMD 支持是实验性的，需要 ROCm 支持的 GPU（Radeon RX 7000/9000、Radeon 780M/760M/740M、Ryzen AI 核显）。Windows 上请把 AMD Adrenalin 驱动更新到 26.8.1 或更高版本。
- 请把 Jasna 安装到路径只包含英文字母和数字的文件夹中。

</details>

## 快速开始

1. 下载适合你的系统和 GPU 的发行包。
2. 解压到一个文件夹，路径中不能有中文、日文、带重音的字母或其他非英文字符。
3. 启动应用：
   - Windows：双击 `jasna.exe`。
   - Linux NVIDIA：运行 `jasna` 文件。
   - Linux AMD：运行 `run_jasna_amd.sh`。
4. 添加视频，选择设置，点击界面底部的 **▶** 按钮。

**第一个任务会很慢。** 在 NVIDIA 上，Jasna 会先为你的 GPU 准备模型。这大约需要
**15–60 分钟**，只会发生一次。在此期间请关闭其他应用，也不要使用电脑。更新 Jasna 时，
把 `model_weights` 文件夹复制到新版本中，这样就不用再等一次。

每个设置都有提示：把鼠标移到 ⓘ 图标上即可查看。GPU 显存不足？降低
**最大片段大小**，例如从 `90` 降到 `60`。见[调整显存和 GPU 占用](docs/zh/tuning.md)。

## 帮助和社区

- [应用指南](docs/zh/gui.md)：队列、预设、输出文件名、视频播放器和激活。
- [SLS Discord](https://discord.gg/FgrSuwZS3s)：示例、帮助和设置。
  Jasna 有一个[新频道](https://discord.com/channels/1196376491815092265/1556960822578909244)；
  以前的帖子保留在[旧频道（已归档）](https://discord.com/channels/1196376491815092265/1199059436199759943)。
  请友善交流，别太离谱。
- 遇到问题？点击应用底部的**日志**，再点击**导出**保存日志文件。求助时请附上它。

## 支持本项目

Jasna 是免费的。你的支持用于训练新模型，主要是租用 GPU。支持者会获得支持者模型的
许可证码：

| 支持 | 模型 |
| --- | --- |
| 累计 **15 美元**或以上 | **UNet 4x**：让修复区域更清晰。<br>**SD 1.5**：用于静态图片的实验性模型。 |
| **25 美元** | **UNet 4x** 和 **SD 1.5**，再加上<br>**LTX**：比 BasicVSR++ 慢很多，但细节更多。仅限 NVIDIA。包含这个 LTX 模型以后的更新。 |

**截至 2026 年 10 月 14 日（含当天）**累计支持 **15 美元或以上**的支持者可免费获得 LTX。

**选择 LTX 之前：**

- 看看这些 LTX 示例：[1](https://discord.com/channels/1196376491815092265/1199059436199759943/1556346988486856826)、[2](https://discord.com/channels/1196376491815092265/1199059436199759943/1556337300814958684)、[3](https://discord.com/channels/1196376491815092265/1199059436199759943/1555612574643388598)、[4](https://discord.com/channels/1196376491815092265/1556960822578909244/1559341521575084063)、[5](https://discord.com/channels/1196376491815092265/1556960822578909244/1559295317529202780)、[6](https://discord.com/channels/1196376491815092265/1556960822578909244/1559205894959403121)、[7](https://discord.com/channels/1196376491815092265/1556960822578909244/1559197341473574973)、[8](https://discord.com/channels/1196376491815092265/1556960822578909244/1558798127195226112)。
  想看更多示例，可以浏览 SLS Discord 的[新频道](https://discord.com/channels/1196376491815092265/1556960822578909244)和
  [旧频道（已归档）](https://discord.com/channels/1196376491815092265/1199059436199759943)。
- 在你的电脑上用**试用模式（速度测试）**运行 LTX。它是免费的，可以看出你的 GPU 能否运行 LTX、
  需要多长时间。视频不会被修复。见
  [LTX 需要什么](docs/zh/models.md#ltx-需要什么)。

更多示例：
[UNet 4x](https://discord.com/channels/1196376491815092265/1199059436199759943/1516497879684874260)、
[SD 1.5](https://discord.com/channels/1196376491815092265/1199059436199759943/1492139124348420106)、
[更多 SD 1.5](https://discord.com/channels/1196376491815092265/1199059436199759943/1516571355317800990)。

### 在哪里支持

许可证码**只**能从这里获得：

- [Buy Me a Coffee](https://buymeacoffee.com/kruk2)（也可以用加密货币）
- [Unifans](https://app.unifans.io/c/kruk2)
- 中国用户：[闲鱼 (Xianyu)](#在闲鱼购买)，见下文

#### 在闲鱼购买

国内用户可以直接在闲鱼购买，得到的许可证码和其他渠道一样。闲鱼使用单独的人民币本地价格，以商品页面为准。

1. 用闲鱼 App 扫描下面的二维码，会打开卖家的闲鱼主页，在主页上购买。
2. 下单后，卖家会发给你**许可证码**。
3. 按照下面的[如何激活](#如何激活)在 Jasna 中激活。

<img width="200" alt="闲鱼卖家二维码" src="assets/xianyu-qr.png" />

只有这个闲鱼卖家是官方的。闲鱼上其他卖 Jasna 的人都不是官方卖家。

其他出售 Jasna、许可证码或修改版文件的人都是骗子：他们的密钥无法使用，他们的文件
可能在你的电脑上运行隐藏代码。请只从 GitHub 下载 Jasna。

### 如何激活
## 许可证

Jasna 的公开应用程序源代码采用 AGPL-3.0。模型和捆绑依赖项保留各自的许可条款。
请参阅[许可与源代码范围](LICENSING.md)、
[模型声明](assets/THIRD_PARTY_MODELS.md)和
[发行版源代码对应表](RELEASE_SOURCES.md)。

## 支持本项目

1. 在上面的任一页面支持。你会收到一个**许可证码**。
2. 在 Jasna 中打开**激活 Jasna**，输入许可证码，点击**在线激活**。
3. 这台电脑没有网络，或激活失败？选择**手动**，见
   [激活支持者模型](docs/zh/gui.md#激活支持者模型)。

一个许可证码可用于 **3 台电脑或云工作区**。激活后，Jasna 可以离线使用。租用的 GPU：
见[云 GPU](docs/zh/cloud.md)。

## 高级

### Jasna 如何工作

从读取视频到保存，几乎所有步骤都在 GPU 上运行。

1. **解码**：在 GPU 上解码视频（NVIDIA 上用 NVDEC，AMD 上用 AMF）。
2. **检测**：在每一帧中查找马赛克（默认模型 `rfdetr-v6`）。
3. **跟踪**：跟踪每个马赛克，把它的帧分成片段（最多**最大片段大小**帧）。
   片段在镜头切换处结束，所以不会混合两个不同的镜头。
4. **修复**：用 BasicVSR++ 把每个片段修复成 256x256 的画面。片段之间有一点重叠，
   所以片段之间不会有明显的跳变。
5. 可选的**二次修复**（UNet 4x、RTX Super Res 或 Topaz TVAI）让修复后的画面
   更大、更清晰。
6. **融合**：把修复后的画面放回原始帧。只有马赛克区域会改变。
7. **编码**：用 GPU 编码器编码（NVIDIA 上用 NVENC，AMD 上用 AMF）。音频直接复制（只有输出格式
   不支持该音频时才会重新编码）。

第一个任务很慢，因为第 2 步和第 4 步的模型（如果使用 UNet 4x，也包括它）要为你的 GPU
做准备（仅限 NVIDIA）。结果保存在 `model_weights` 中，之后会重复使用。你可以把它复制到
新版本的 Jasna 中。
密钥**仅**在以上两个页面出售。其他任何出售 Jasna、密钥或修改版文件的人都是骗子：他们的密钥无法用于官方版本，他们的文件可能会在你的电脑上运行隐藏代码。请只从 GitHub 下载 Jasna。

若激活失败，请核对购买时使用的邮箱或账号，并重新复制完整密钥。
密钥无效本身并不能证明欺诈；Jasna 会单独提示来自已知非官方卖家的密钥。

## 致谢

LTX 的工作方式不同。它会把视频处理 4 遍：查找马赛克、准备、用扩散模型修复、保存。它更慢，但能加入更多真实细节。见
[选择模型](docs/zh/models.md#修复模型)。

### 命令行

命令行可以做应用能做的大部分事情（视频播放器、LTX 种子预览、遮罩建议和交互式图片处理除外）。

```bash
# 一个视频
jasna --input input.mp4 --output output.mkv

# 一张图片
jasna --input photo.png --output restored.png

# 整个文件夹
jasna --input input_folder --output output_folder
```

`jasna --help` 会列出所有选项。见 [CLI 参考](docs/zh/cli.md)。

### 所有指南

- [使用 GUI](docs/zh/gui.md)：队列、预设、输出文件名、视频播放器、激活。
- [选择模型](docs/zh/models.md)：检测模型、BasicVSR++ 还是 LTX、二次修复、
  用于图片的 SD 1.5。
- [只修复视频的一部分](docs/zh/segments.md)：区间编辑器、马赛克扫描、
  仅大马赛克用 LTX。
- [VR180 视频](docs/zh/vr180.md)：如何处理左右并排 VR。
- [调整显存和 GPU 占用](docs/zh/tuning.md)：片段大小、时间重叠、显存不足时
  怎么办。
- [高级处理](docs/zh/advanced_processing.md)：降噪、60→30 FPS、色彩 LUT、
  锐化、编码器质量、队列完成后的操作。
- [流媒体](docs/zh/streaming.md)：在浏览器或 Stash 中观看。
- [云 GPU](docs/zh/cloud.md)：在租用的 GPU（RunPod、Vast）上运行 Jasna。
- [CLI 参考](docs/zh/cli.md)：所有命令行选项。
- [从源代码运行](docs/en/development.md)（英文）：面向开发者。从源代码运行时可以使用免费的
  BasicVSR++；激活和支持者模型需要官方发行版。

### 基准测试

用 BasicVSR++ 修复一整段 3:22 视频所需的时间，RTX 5090（Linux）。越低越好。

| 输入 | Lada 0.11.0 | Jasna v0.9.1 |
| --- | ---: | ---: |
| 720p H.264 | 01:47 | **00:32**（快 3.4 倍） |
| 1080p H.264 | 02:02 | **00:34**（快 3.6 倍） |
| 2160p H.264 | 04:56 | **01:03**（快 4.7 倍） |

<details>
<summary>GPU 显存、旧版本和测试方法</summary>

GPU 显存占用（典型/峰值，单位 GB）。`—` 表示未测量。Lada 占用的 GPU 显存比 Jasna 少。

| 输入 | Lada | v0.4.1 | v0.5.0 | v0.9.0 | **v0.9.1** |
| --- | ---: | ---: | ---: | ---: | ---: |
| 720p H.264 8-bit | 1.8/3.1 | 17.6/19.9 | 8.6/9.0 | 8.6/9.1 | **4.2/4.6** |
| 1080p H.264 8-bit | 2.0/3.3 | 18.9/21.4 | 9.5/10.2 | 9.3/10.1 | **4.8/5.6** |
| 2160p H.264 8-bit | 2.6/4.1 | 26.0/30.5 | 12.9/15.0 | 11.4/15.4 | **7.2/10.9** |
| 8K VR HEVC 60 fps（15 秒） | — | — | — | 17.1/18.3 | — |

旧版本的时间（720p / 1080p / 2160p：v0.4.1 为 01:35 / 01:46 / 03:23；v0.5.0 为
00:45 / 00:47 / 01:22；v0.9.0 为 00:34 / 00:39 / 01:15；8K VR：00:34）、测试设置、
内存占用以及所有编码，见[完整基准测试报告](benchmarks/2026-07-26_release_matrix.md)。
较早的完整长视频：[旧版报告](benchmarks/2026-07-26_legacy_videos.md)。

</details>

### 许可证

Jasna 的公开应用程序源代码采用 AGPL-3.0。模型和捆绑依赖项保留各自的许可条款。
请参阅[许可与源代码范围](LICENSING.md)、
[模型声明](assets/THIRD_PARTY_MODELS.md)和
[发行版源代码对应表](RELEASE_SOURCES.md)。

### 致谢

- **[Lada](https://codeberg.org/ladaapp/lada)（Codeberg）**：Jasna 受 Lada 启发，
  部分地方也基于 Lada。Jasna 使用的 `mosaic_restoration_1.2` 模型由 Lada 作者
  ladaapp 训练。
- **ZeLeFans**：感谢提供 VR 检测模型，以及对马赛克形状和 VR 投影的详细分析。
