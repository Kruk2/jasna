# 调整显存和 GPU 占用

显存不够时，Jasna 会把等待中的帧移到系统内存，需要时再移回来。这一点不需要
任何设置。下面的设置会改变一个任务占用多少显存和 GPU 时间。

片段和编译设置用于 BasicVSR++。LTX 会自己管理内存，见
[LTX 需要什么](models.md#ltx-需要什么)。

## 最大片段大小和时间重叠

Jasna 按片段修复马赛克：片段是一起处理的一组帧。

- **最大片段大小**（**模型**标签页，**高级**视图）：最长的片段，单位是帧。片段越长，结果越稳定，但显存占用
  越多。
- **时间重叠**：让两个片段的衔接处更平滑。重叠越多，耗时越长。超过 `20` 通常
  没有帮助。

推荐起点：

- 在显存放得下的范围内，用最大的**最大片段大小**。片段越大越稳定，*而且*越快。
- 把**时间重叠**设在 `8` 到 `20` 之间。

一段 1080p 视频，重叠 8，RTX 5090。你的时间会不同，但趋势一样：

| 最大片段大小 | 时间 | VRAM |
| -------------:| ----:| ----:|
| 30            | 51 秒 | 3.4 GB |
| 60            | 40 秒 | 3.7 GB |
| 90            | 34 秒 | 4.1 GB |
| 180           | 35 秒 | 5.0 GB |
| 360           | 32 秒 | 6.3 GB |
| 720           | 33 秒 | 8.0 GB |

小片段**更慢**，而不是更快。重叠的帧在每个片段里都要再处理一次，所以片段越多，
重复的工作越多。超过大约 `180` 后，时间不再变好，只有显存上升。同样的片段
大小下，4K 视频大约需要 1.5 倍的显存。

```bash
jasna --input input.mp4 --output output.mkv --max-clip-size 90 --temporal-overlap 8
```

## 显存不足？

1. 降低**最大片段大小**。上面的表格显示了每一档的代价。
2. 关闭二次修复，或换一个更轻的（见[对比](models.md#速度与显存对比)）。
3. 关闭其他使用 GPU 的应用（浏览器、游戏、视频播放器）。

4–6 GB 显存的显卡：1080p 视频请从最大片段大小 `30` 开始。4K 需要更多。

关闭**编译 BasicVSR++** 没有帮助：它只能省下不到 0.5 GB，却让处理慢约 3 倍。

## 模型编译

在 NVIDIA 上，Jasna 会为你的 GPU 编译检测模型和修复模型（TensorRT）；如果使用
UNet 4x，也会编译它。这样处理速度快得多。AMD 从不编译。

- 编译在第一个任务时进行，大约需要 **15–60 分钟**。在此期间请关闭其他应用，也不要
  使用电脑。
- 结果保存在 `model_weights` 中，之后的每个任务都会重复使用。
- 你可以把 `model_weights` 从旧版本 Jasna 复制到新版本中，这样就不用再等。

你可以关闭修复模型的编译，但几乎没有理由这么做。

```bash
jasna --input input.mp4 --output output.mkv --no-compile-basicvsrpp
```

同一段 1080p 视频，片段大小 180：

|                    | 时间 | VRAM |
| ------------------ | ----:| ----:|
| 编译（默认）        | 33 秒 | 5.0 GB |
| 不编译              | 107 秒 | 4.6 GB |

更改最大片段大小不会重新开始编译。

## 高级环境变量

这些用于排查问题和测试。大多数人用默认值就是最好的。

| 变量 | 取值 | 作用 |
| --- | --- | --- |
| `JASNA_LTX_VAE_ENCODER` | `auto`（默认）、`full`、`half`、`max` | 仅限命令行，应用里用 **LTX 编码器**（见 [LTX 选项](models.md#ltx-选项)）。选择 LTX 视频编码器：`half` 和 `max` 更快，画面稍柔和。`auto`：基础画质用 `half`，其他用 `full`。 |
| `JASNA_LTX_VAE_DECODER` | `auto`（默认）、`full`、`half`、`max` | 仅限命令行，应用里用 **LTX 解码器**。选择 LTX 视频解码器：`half` 和 `max` 更快，画面稍柔和。`auto`：基础画质用 `max`，其他用 `full`。 |
| `JASNA_DECODE_BACKEND` | `auto`（默认）、`vali`、`pyav-hw`、`pyav-sw` | 仅限命令行，应用里用 **视频解码器**（**模型**标签页，**专家**视图）。选择 Jasna 读取输入视频的方式。视频打不开或画面损坏时，试试 `pyav-sw`（更慢）。`vali` 仅限 NVIDIA。 |
| `JASNA_TRT_CUDAGRAPHS` | `1`（默认）、`0` | NVIDIA：`0` 关闭已编译 BasicVSR++ 的一项加速。遇到 GPU 错误时可以试试。 |
| `JASNA_AMD_GRAPHS` | `1`（默认）、`0` | AMD：`0` 关闭 BasicVSR++ 的一项加速。遇到 GPU 错误或显存不足时可以试试。 |
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` | `1`（默认）、`0` | AMD：`0` 关闭一种更快的注意力模式。遇到 GPU 错误时，和 `JASNA_AMD_GRAPHS=0` 一起试试。 |
| `TVAI_MODEL_DATA_DIR`、`TVAI_MODEL_DIR` | 文件夹路径 | Topaz Video 的模型文件夹，TVAI 需要（见[二次修复](models.md#二次修复)）。 |

Windows：在 Jasna 文件夹中打开命令提示符，设置变量后在同一个窗口里启动 Jasna
（也可以把这两行存成该文件夹里的 `.bat` 文件）：

```bat
set JASNA_TRT_CUDAGRAPHS=0
jasna.exe
```

Linux（AMD 上用 `./run_jasna_amd.sh` 代替 `./jasna`）：

```bash
JASNA_TRT_CUDAGRAPHS=0 ./jasna
```

此设置在关闭该窗口前有效。
