# 选择模型

## 检测模型

检测模型负责在每一帧中找到马赛克。

- **使用最新的 RF-DETR 模型**（`rfdetr-v5`）— 它是默认值，也是综合
  表现最好的选择。
- **Lada YOLO** 模型在 2D 动画上可能效果更好。
- **zelefans-vr-yolo-v2**（已内置）在 VR180 视频上可能更准确。
- **在 AMD 上**，RF-DETR 非常慢（在 Windows 上甚至只能用 CPU 运行）—
  除非特别需要 RF-DETR，否则请改用 `lada-yolo-v4`。

```bash
jasna --input input.mp4 --output output.mkv --detection-model rfdetr-v5
```

你也可以在[区间编辑器](segments.md)中为每个视频单独设置不同的检测模型。

## 二级修复

Jasna 会修复每个马赛克区域的 256x256 裁切图。因此，大面积马赛克、
特写和 4K 视频在主修复后可能看起来模糊。二级模型会先把修复后的裁切图
放大到 512x512 或 1024x1024，再混合回原视频，让画面明显更清晰。

- **unet-4x**: 支持者模型。当前测试中比 TVAI 更快，质量相近。它在
  JAV 领域数据集上训练，视觉效果接近 TVAI `iris-2`。可以查看
  [SLS Discord 上的示例](https://discord.com/channels/1196376491815092265/1199059436199759943/1516497879684874260)。
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

| 二级类型 | CAWD 1080p | KV-109 1080p |
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
[SLS Discord 上的 SD 1.5 示例](https://discord.com/channels/1196376491815092265/1199059436199759943/1492139124348420106)
和[更多示例](https://discord.com/channels/1196376491815092265/1199059436199759943/1516571355317800990)。
