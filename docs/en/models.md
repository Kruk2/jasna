# Choosing Models

## Detection model

The detection model finds mosaics in each frame.

- **Use the latest RF-DETR model** (`rfdetr-v6`) — it's the default, fast, and
  the best all-rounder. Bundled with Jasna.
- **`rfdetr-v6-large`** is a higher-quality, slower variant, and can be the
  better pick for 4K videos. It is an optional
  separate download — pick the file for your graphics card, drop it into
  `model_weights/`, and Jasna detects it automatically:
  - NVIDIA:
    [rfdetr-v6-large.onnx](https://github.com/Kruk2/jasna/releases/download/0.1/rfdetr-v6-large.onnx)
  - AMD:
    [rfdetr-v6-large.pt](https://github.com/Kruk2/jasna/releases/download/0.1/rfdetr-v6-large.pt)
- **Lada YOLO** models can work better for 2D animations.
- **rfdetr-vr-v1** (bundled) is the RF-DETR VR180 detection model — best for VR180 videos.
- **zelefans-vr-yolo-v2** (optional download) is an alternative VR180 detector.
- **On AMD**, RF-DETR runs on your graphics card and uses the `.pt` model
  files (NVIDIA uses `.onnx`). It is slower than on NVIDIA, so pick
  `lada-yolo-v4` when speed matters more than detection quality.

Each model applies its own recommended detection threshold by default
(`rfdetr-v6`: 0.35, `rfdetr-v6-large`: 0.40); override with
`--detection-score-threshold`.

The legacy `rfdetr-v5` model remains supported.

```bash
jasna --input input.mp4 --output output.mkv --detection-model rfdetr-v6
```

You can also set a different detection model per video inside the
[segment editor](segments.md).

## Restoration model

**BasicVSR++** is the default: fast, and it runs on every supported GPU.
**LTX** is much slower but gives more detailed, more natural results. Use it
for the videos or scenes you care about most. Pick it in the **Restoration
Model** section, or:

```bash
jasna --input input.mp4 --output output.mkv --restoration-model-name ltx
```

- NVIDIA RTX only. Not for VR180 side-by-side videos or streaming.
  Secondary restoration and denoise are off with LTX.
- The Video Player always uses BasicVSR++. To preview LTX, turn on
  **Restore preview** on an LTX range in the [Segment Editor](segments.md).
- The time depends on how long mosaics are on screen, not on the length of
  the video. The first run takes longer while Jasna prepares LTX for your GPU.
- The model is not bundled: about 15 GB (about 9 GB for Fast mode). Jasna
  asks before downloading it. If Hugging Face is blocked (for example in
  China), it downloads from hf-mirror.com.

### LTX options

- **Model**: **Distilled** (default) or **Undistilled**, which is slightly
  better and about 2x slower (another 13 GB download). CLI:
  `--restoration-model-name ltx-undistilled`.
- **Fast mode**: RTX 50-series only. About 1.4x faster with slightly less
  detail, and a smaller download. CLI: `--ltx-fast`.
- **Sharper large mosaics**: more detail on big mosaics (close-ups, 4K),
  about 3x slower on those. Needs 10 GB of free VRAM, otherwise it has no
  effect. CLI: `--ltx-large-canvas`.
- **Seed**: another seed gives a slightly different result; the same seed
  repeats it. CLI: `--ltx-seed 7`.
- **Trial mode**: runs LTX at full speed without the model files, so
  you can see how fast it is on your PC. The video is not restored. CLI:
  `--ltx-trial`.
- **LTX only for big mosaics**: big mosaics get LTX, smaller ones the faster
  BasicVSR++. Each video is scanned first and parts without mosaics are
  copied. Videos with ranges you set keep them. CLI: `--ltx-auto-split 12`.

### Which GPU and how long

Rough time for **1 minute of 1080p video with mosaics on screen the whole
time** (Distilled model). Scenes without mosaics are much quicker;
Undistilled takes about twice as long.

| GPU | LTX | Time | With Fast mode |
| --- | --- | ---: | ---: |
| RTX 5090 | Fastest | about 5 min | about 3 min |
| RTX 4090 | Works well | about 7 min | — |
| RTX 5080 | Works well | about 10 min | about 6 min |
| RTX 3090 | Works, slower | about 15 min | — |
| RTX 2080 Ti | Works, very slow | about 45 min | — |
| GTX 16-series | Not tested | — | — |
| AMD | Not supported | — | — |

- **VRAM**: 24 GB keeps the whole model on the GPU, 16 GB is fine, and
  11–12 GB works but is slower. Less than 11 GB is untested.
- **System RAM**: 32 GB recommended (loading the model uses about 19 GB). With
  16 GB, use Fast mode, which needs about 10 GB.

## Secondary restoration

Jasna restores a 256x256 crop of each mosaic region. Large mosaic regions,
close-ups, and 4K videos can therefore look blurry after the primary
restoration. A secondary model upscales the restored crop to 512x512 or
1024x1024 before blending it back, making it noticeably sharper.

Secondary restoration is currently available only on NVIDIA GPUs. AMD builds
support primary restoration only.

- **unet-4x**: supporter model. Faster than TVAI with similar quality in
  current testing. Trained on an in-domain dataset and visually close to
  TVAI `iris-2`. See
  examples on SLS Discord.
  Unlock it with a machine Activation key — see
  [Supporting the project](../../README.md#supporting-the-project).
  Keys from earlier activation formats need a replacement from your official seller.
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

| Secondary type           | CAWD 1080p        | K109 1080p      |
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
- It is available to supporters. One machine Activation key can include SD 1.5,
  unet-4x and purchased LTX versions — see
  [Supporting the project](../../README.md#supporting-the-project).
  Keys from earlier activation formats need a replacement from your official seller.
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
SD 1.5 examples on SLS Discord
and more.
