> **支持者模型的正版许可证码只能通过 [Ko-fi](https://ko-fi.com/kruk2)、[Unifans](https://app.unifans.io/c/kruk2) 或 [闲鱼](#在闲鱼购买) 获取。**

[English](README.md) | [日本語](README.ja.md) | [**中文**](README.zh.md)

# <img width="32" src="https://github.com/Kruk2/jasna/blob/main/assets/jasna-logo.png?raw=true" /> Jasna

Jasna 修复视频和图片中的马赛克。Jasna 免费，提供简单的应用和命令行，在你的 GPU
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
- [致谢](#致谢)

## Jasna 能做什么

- **修复视频中的马赛克。** 只有马赛克区域会改变。[快速开始](#快速开始)
- **快速或细节，任你选。** BasicVSR++ 快速且免费；LTX 较慢但细节更多（支持者专享）。[选择模型](docs/zh/models.md#修复模型)
- **免费试用 LTX。** 试用模式能看出 LTX 在你的电脑上有多快。[LTX 选项](docs/zh/models.md#ltx-选项)
- **只在需要的地方用 LTX。** 大马赛克用 LTX，其余用 BasicVSR++。[马赛克扫描](docs/zh/segments.md#马赛克扫描)
- **只修复部分场景。** 在区间编辑器中边预览边选择区间。[区间编辑器](docs/zh/segments.md#区间编辑器gui)
- **暂停与继续。** 较长的 LTX 任务会保留进度。[暂停与继续](docs/zh/models.md#暂停与继续)
- **让修复区域更清晰。** UNet 4x、RTX Super Res 或 Topaz。[二次修复](docs/zh/models.md#二次修复)
- **放大整个视频**到 1080p、1440p 或 4K。[放大整个视频](docs/zh/advanced_processing.md#放大整个视频)
- **画面更流畅。** 在 RTX 40 和 50 上把帧率翻倍。[补帧](docs/zh/gui.md#补帧)
- **VR180 视频**，左右并排或上下。[VR180](docs/zh/vr180.md)
- **静态图片**用 SD 1.5 修复（支持者专享）。[图片](docs/zh/models.md#静态图片修复sd-15)
- **边修复边观看**，在应用中或浏览器中。[视频播放器](docs/zh/gui.md#视频播放器)、[流媒体](docs/zh/streaming.md)
- **一次处理很多视频。** 队列、预设和队列完成后的操作。[使用 GUI](docs/zh/gui.md#队列)
- **输出附加功能。** 色彩 LUT、锐化、60→30 FPS、处理过程中即可播放。[高级处理](docs/zh/advanced_processing.md)
- **命令行和租用 GPU。** [CLI](docs/zh/cli.md)、[云 GPU](docs/zh/cloud.md)

## 要求

- **NVIDIA** GTX 16 系列 / RTX 20 系列或更新，并使用较新的驱动
  （Windows 610 或更新，Linux 580 或更新）。GTX 10 系列及更旧的显卡无法使用。
- 或 **AMD** Radeon RX 6000 或更新（实验性）。已测试 RX 7000/9000；
  RX 6000 未测试。
- 本地磁盘上的一个文件夹，路径中不能有中文、日文、带重音的字母或其他非英文字符。

<details>
<summary>更多细节</summary>

- NVIDIA：你的 GPU 在 NVIDIA 的 [GPU 表格](https://developer.nvidia.com/cuda/gpus)
  中需要计算能力 7.5 或更高。A100、H100、B200 等数据中心 GPU 无法使用。
- AMD：见下表。支持 UNet 4x 二次修复。
- 磁盘格式：Windows 用 NTFS；Linux 用 ext4、Btrfs、XFS 或 ZFS。不支持网络磁盘。请使用
  固态硬盘（SSD）；在机械硬盘（HDD）上 Jasna 启动和加载模型都很慢。
- LTX 需要 NVIDIA RTX 或 AMD Radeon RX 7000/9000 显卡（AMD 核显未测试），至少 8 GB 显存和 16 GB 内存。见
  [LTX 需要什么](docs/zh/models.md#ltx-需要什么)。

| AMD 显卡 | Jasna | LTX |
| --- | --- | --- |
| Radeon RX 9060/9070、Radeon AI PRO R9700 | 支持 | 支持（Windows 为测试版） |
| Radeon RX 7600–7900、Radeon PRO W7000 | 支持 | 支持（Windows 为测试版） |
| Radeon 740M/760M/780M/880M/890M、Ryzen AI Max（8050S/8060S） | 支持（未测试） | 支持（未测试） |
| Radeon RX 6000、Radeon PRO W6000 | 支持（未测试） | 不支持 |
| Radeon 610M/660M/680M/820M/840M/860M | 支持（未测试） | 不支持 |
| Radeon RX 5000、Radeon PRO W5000、RX Vega、RX 500 及更早 | 不支持 | 不支持 |

</details>

## 快速开始

1. 下载适合你的系统和 GPU 的发行包。
2. 解压到一个文件夹，路径中不能有中文、日文、带重音的字母或其他非英文字符。
3. 启动应用：
   - Windows：双击 `jasna.exe`。AMD 用户请先安装
     [适用于 ROCm 10.1 的 AMD 驱动](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-ROCM-10-01.html)
     （**26.10.41.05**，[下载 .exe](https://drivers.amd.com/drivers/amd-software-adrenalin-edition-26.10.41.05-win11-b-204865.exe)），
     然后重启电脑。无需安装 ROCm。
   - Linux NVIDIA：运行 `jasna` 文件。
   - Linux AMD：先安装一次 AMD 的 AMF 视频运行库。我们测试过
     [AMF 26.20](https://repo.radeon.com/amf/26.20/)：

     ```bash
     curl -fLO https://repo.radeon.com/amf/26.20/amf_installer_26.20.zip
     unzip amf_installer_26.20.zip
     sudo bash amf_installer_26.20.sh
     ```

     按提示同意 AMD 的许可协议，然后运行 `run_jasna_amd.sh`。
     如果使用 Distrobox，请在运行 Jasna 的同一个容器内安装 AMF。
4. 添加视频，选择设置，点击界面底部的 **▶** 按钮。

**第一个任务会很慢。** 在 NVIDIA 上，Jasna 会先为你的 GPU 准备模型。这大约需要
**15–60 分钟**，只会发生一次。在此期间请关闭其他应用，也不要使用电脑。更新 Jasna 时，
把 `model_weights` 文件夹复制到新版本中，这样就不用再等一次。更新后，LTX 仍可能要求重新下载模型。
在 AMD 上，只有首次使用 UNet 4x 时需要这样准备。

每个设置都有提示：把鼠标移到 ⓘ 图标上即可查看。GPU 显存不足？降低
**最大片段大小**，例如从 `90` 降到 `60`。见[调整显存和 GPU 占用](docs/zh/tuning.md)。

## 帮助和社区

- [应用指南](docs/zh/gui.md)：队列、预设、输出文件名、视频播放器和激活。
- 遇到问题？点击应用底部的**日志**，再点击**导出**保存日志文件。求助时请附上它。

## 支持本项目

Jasna 是免费的。你的支持用于训练新模型，主要是租用 GPU。支持者会获得支持者模型的
许可证码：

| 支持 | 模型 |
| --- | --- |
| 累计 **15 美元**（Ko-fi：**14 欧元**）或以上 | **UNet 4x**：让修复区域更清晰。<br>**SD 1.5**：用于静态图片的实验性模型。 |
| **25 美元**（Ko-fi：**23 欧元**） | **UNet 4x** 和 **SD 1.5**，再加上<br>**LTX**：比 BasicVSR++ 慢很多，但细节更多。包含这个 LTX 模型以后的更新。 |

**截至 2026 年 10 月 14 日（含当天）**累计支持 **15 美元或以上**的支持者可免费获得 LTX。

**选择 LTX 之前：**

- 在你的电脑上用**试用模式（速度测试）**运行 LTX。它是免费的，可以看出你的 GPU 能否运行 LTX、
  需要多长时间。视频不会被修复。见
  [LTX 需要什么](docs/zh/models.md#ltx-需要什么)。

### 在哪里支持

许可证码**只**能从这里获得：

- [Ko-fi](https://ko-fi.com/kruk2)
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

1. 在上面的任一页面支持。你会收到一个**许可证码**。
2. 在 Jasna 中打开**激活 Jasna**，输入许可证码，选择**我的电脑**或**租用或共用的电脑**，点击**激活**。

一个许可证码可**同时在 3 台设备上**使用。在你自己的、有 TPM 2.0 安全芯片的电脑上
（大多数 Windows 11 电脑），之后 Jasna 可以离线使用。其他情况下，使用支持者模型时需要联网。
详细说明和问题处理：[激活支持者模型](docs/zh/gui.md#激活支持者模型)。租用 GPU：见
[云 GPU](docs/zh/cloud.md)。

## 高级

### Jasna 如何工作

Jasna 找到每个马赛克，逐帧跟踪，只修复这块区域，再放回原处。画面的其他部分不会被改动。
几乎所有步骤都在 GPU 上运行。LTX 更慢，但能加入更多真实细节；见
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
- [VR180 视频](docs/zh/vr180.md)：如何处理左右并排和上下 VR。
- [调整显存和 GPU 占用](docs/zh/tuning.md)：片段大小、时间重叠、显存不足时
  怎么办。
- [高级处理](docs/zh/advanced_processing.md)：降噪、60→30 FPS、色彩 LUT、
  锐化、放大整个视频、编码器质量、视频中保存的设置、队列完成后的操作。
- [流媒体](docs/zh/streaming.md)：在浏览器或 Stash 中观看。
- [云 GPU](docs/zh/cloud.md)：在租用的 GPU（RunPod、Vast）上运行 Jasna。
- [CLI 参考](docs/zh/cli.md)：所有命令行选项。
- [从源代码运行](docs/en/development.md)（英文）：面向开发者。从源代码运行时可以使用免费的
  BasicVSR++；激活和支持者模型需要官方发行版。

### 基准测试

在 RTX 5090（Linux）上，Jasna 0.9.1 用 BasicVSR++ 修复一段 3:22 的 1080p 视频用了 34 秒。
测试设置、GPU 显存和旧版本见[完整基准测试报告](benchmarks/2026-07-26_release_matrix.md)；
较新的结果见 [benchmarks](benchmarks/)。

### 许可证

Jasna 的公开应用程序源代码采用 AGPL-3.0。模型和捆绑依赖项保留各自的许可条款。
请参阅[许可与源代码范围](LICENSING.md)、
[模型声明](assets/THIRD_PARTY_MODELS.md)和
[发行版源代码对应表](RELEASE_SOURCES.md)。

### 致谢

- **Lada**：Jasna 受 Lada 启发，
  部分地方也基于 Lada。Jasna 使用的 `mosaic_restoration_1.2` 模型由 Lada 作者
  ladaapp 训练。
- **ZeLeFans**：感谢提供 VR 检测模型，以及对马赛克形状和 VR 投影的详细分析。
