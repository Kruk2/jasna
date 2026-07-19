# Choosing Models

## Detection model

The detection model finds mosaics in each frame.

- **Use the latest RF-DETR model** (`rfdetr-v5`) — it's the default and the
  best all-rounder.
- **Lada YOLO** models can work better for 2D animations.
- **zelefans-vr-yolo-v2** (bundled) can be more accurate for VR180 videos.
- **On AMD**, RF-DETR is very slow (on Windows it even runs on the CPU) —
  use `lada-yolo-v4` instead unless you specifically need RF-DETR.

```bash
jasna --input input.mp4 --output output.mkv --detection-model rfdetr-v5
```

You can also set a different detection model per video inside the
[segment editor](segments.md).

## Secondary restoration

Jasna restores a 256x256 crop of each mosaic region. Large mosaic regions,
close-ups, and 4K videos can therefore look blurry after the primary
restoration. A secondary model upscales the restored crop to 512x512 or
1024x1024 before blending it back, making it noticeably sharper.

- **unet-4x**: supporter model. Faster than TVAI with similar quality in
  current testing. Trained on an in-domain JAV dataset and visually close to
  TVAI `iris-2`. See
  [examples on SLS Discord](https://discord.com/channels/1196376491815092265/1199059436199759943/1516497879684874260).
  Unlock it with a supporter key — see
  [Supporting the project](../../README.md#supporting-the-project).
- **RTX Super Resolution**: very fast, free, and needs nothing extra.
  Quality is okay. Some videos may flicker, so test on a short clip first.
- **TVAI**: better than RTX Super Resolution and comparable to unet-4x, but
  very slow. Requires [Topaz Video](https://www.topazlabs.com/topaz-video),
  which is paid and Windows-only. Recommended model: `iris-2`.

```bash
jasna --input input.mp4 --output output.mkv --secondary-restoration unet-4x
```

For TVAI, set the `TVAI_MODEL_DATA_DIR` and `TVAI_MODEL_DIR` environment
variables to your Topaz Video model folders, as shown below
(`--tvai-args` can further customize the Topaz model parameters):

<img width="505" height="37" alt="Topaz Video environment variables" src="https://github.com/user-attachments/assets/e19ced9d-d549-4e85-b20f-888e42466f1d" />

### Speed and VRAM comparison

| Secondary type           | CAWD 1080p        | KV-109 1080p      |
| ------------------------ | -----------------:| -----------------:|
| No secondary             | 22s / 10.0 GB VRAM | 11s / 10.7 GB VRAM |
| unet-4x                  | 29s / 12.5 GB VRAM | 14s / 12.6 GB VRAM |
| RTX Super-Res            | 25s / 11.7 GB VRAM | 13s / 11.4 GB VRAM |
| TVAI (2 workers, Iris-2) | 52s / 12.1 GB VRAM | 24s / 12.4 GB VRAM |

## Still-image restoration (SD 1.5)

For still images, Jasna uses a fine-tuned Stable Diffusion 1.5 inpaint model
instead of the video pipeline. Just add an image to the GUI queue (or pass it
on the CLI) — image jobs route to SD 1.5 automatically:

```bash
jasna --input photo.png --output restored.png
```

- The model is **not bundled** and is about **6.9 GB**. Jasna asks before
  downloading it from
  [huggingface.co/Kruk2/sd-15-jav](https://huggingface.co/Kruk2/sd-15-jav).
- It is currently available only to supporters and uses the same key as
  unet-4x — see
  [Supporting the project](../../README.md#supporting-the-project).
- Expect about **7 GB VRAM** during inference, a bit more for large 4K
  images.

The SD 1.5 path is experimental. Results vary by scene, but some images can
work very well. Generate several variants and keep the best one:

```bash
jasna --input photo.png --output restored.png --sd15-variants 4
```

Every knob (`--sd15-steps`, `--sd15-strength`, `--sd15-seed`, ...) is listed
in the [CLI reference](cli.md#sd-15-image-restoration).

Examples:
[SD 1.5 examples on SLS Discord](https://discord.com/channels/1196376491815092265/1199059436199759943/1492139124348420106)
and [more](https://discord.com/channels/1196376491815092265/1199059436199759943/1516571355317800990).
