# Choosing Models

## Detection model

The detection model finds the mosaics in each frame.

- **`rfdetr-v6`** (default, included): fast and good for most videos. Use
  this one if you are not sure.
- **`rfdetr-v6-large`**: better detection, but slower. Can be better for 4K.
  It is a separate download. Take the file for your GPU and put it in
  `model_weights/`. Jasna finds it automatically.
  - NVIDIA:
    [rfdetr-v6-large.onnx](https://github.com/Kruk2/jasna/releases/download/0.1/rfdetr-v6-large.onnx)
  - AMD:
    [rfdetr-v6-large.pt](https://github.com/Kruk2/jasna/releases/download/0.1/rfdetr-v6-large.pt)
- **`rfdetr-vr-v1`** (included): best for VR180 videos.
- **`zelefans-vr-yolo-v2`** (separate download): another VR180 model.
- **Lada YOLO** models: can be better for 2D animation.
- The old `rfdetr-v5` still works.

On AMD, RF-DETR is slower than on NVIDIA. If speed is more important than
detection quality, use `lada-yolo-v4`.

Each model has its own default detection threshold (`rfdetr-v6`: 0.35,
`rfdetr-v6-large` and `rfdetr-vr-v1`: 0.40). Change it with **Detection
Threshold** in **Basic Processing**, or with `--detection-score-threshold`.

```bash
jasna --input input.mp4 --output output.mkv --detection-model rfdetr-v6
```

In the [Segment Editor](segments.md) you can choose another detection model
for each video.

## Restoration model

| | **BasicVSR++** (default) | **LTX** |
| --- | --- | --- |
| Speed | Fast | Much slower (minutes per minute of video) |
| Quality | Good | More detail, more natural |
| GPU | Every supported GPU | NVIDIA only |
| Cost | Free | Paid ([buy separately](../../README.md#supporting-the-project)) |

Use LTX for the videos or scenes you care about most. Before you buy it, look
at the samples on the
SLS Discord channel
and run **Trial mode** to see if your PC can run it and how long it takes. Pick it in the
**Restoration Model** section, or:

```bash
jasna --input input.mp4 --output output.mkv --restoration-model-name ltx
```

### What LTX needs

- **License**: a License code that includes LTX ($25 USD, also gives UNet 4x and SD 1.5, see
  [Supporting the project](../../README.md#supporting-the-project)). Without
  one, the app offers **Trial mode** instead (see below); the CLI stops with
  an error.
- **GPU**: NVIDIA RTX 20-series or newer. GTX 16-series is untested. AMD is
  not supported.
- **VRAM**: 16 GB or more is best. Smaller cards work but are slower: parts of
  the model are kept in system RAM. 11 GB is the smallest card tested so far.
- **System RAM**: 32 GB recommended (loading the model uses about 19 GB).
- **Disk**: the model is not bundled. Jasna asks before downloading it the
  first time you start an LTX job, and downloads only the model you picked:

  | Model | Download |
  | --- | ---: |
  | Standard | about 15 GB |
  | Fast mode | about 9 GB |
  | Best, on top of Standard | about 13 GB more |

  If Hugging Face can't be reached (for example in China), Jasna uses
  hf-mirror.com automatically.

LTX does not work with VR180 side-by-side videos or streaming. Secondary
restoration and denoising are turned off with LTX.

### LTX options

- **Model**: **Standard** (default) or **Best**. Best is slightly better and about 1.5x
  slower. CLI: `--restoration-model-name ltx-undistilled`.
- **Fast mode** (RTX 50-series only): about 1.4x faster, slightly less detail,
  smaller download. CLI: `--ltx-fast`.
- **Sharper large mosaics**: more detail on big mosaics (close-ups, 4K),
  about 3x slower on those. Needs about 12 GB of free VRAM (16 GB cards for
  Segment Editor ranges); otherwise Jasna restores them normally. CLI:
  `--ltx-large-canvas`.
- **Seed**: a different seed gives a slightly different result; the same seed
  gives the same result. CLI: `--ltx-seed 7`.
- **Trial mode**: runs LTX at full speed without the model or a license, so
  you can see how fast it is on your PC. The video is **not** restored.
  CLI: `--ltx-trial`.
- **LTX only for big mosaics**: big mosaics get LTX, the rest the faster
  BasicVSR++. Saves a lot of time on long videos. See
  [Restoring only parts of a video](segments.md#cli---ltx-auto-split).

To preview LTX, open the [Segment Editor](segments.md), select a range,
choose LTX and turn on **Restore preview**. The Video Player always uses
BasicVSR++.

### How long it takes

Rough time for **1 minute of 1080p video with mosaics on screen the whole
time** (Standard model). Scenes without mosaics are much quicker. Best takes
about 1.5x as long.

| GPU | Time | With Fast mode |
| --- | ---: | ---: |
| RTX 5090 | about 5 min | about 3 min |
| RTX 4090 | about 7 min | not available |
| RTX 5080 | about 10 min | about 7 min |
| RTX 3090 | about 15 min | not available |
| RTX 2080 Ti | about 45 min | not available |

- The first LTX run takes longer while Jasna prepares it for your GPU.
- Progress shows four steps: finding mosaics, preparing, restoring, saving.
  The first LTX video on your GPU, and the first after each update, starts
  with a short speed test. After that, the time left is accurate from the end
  of step 1.
- LTX slows down when other apps use the GPU. Close games and browsers.
- **Windows**: LTX fails with "Access is denied" while Windhawk runs inside
  Jasna. Exit Windhawk or add `jasna.exe` to its excluded processes. Jasna
  warns about this when it starts.

## Secondary restoration

Jasna restores each mosaic as a 256x256 picture. On big mosaics, close-ups and
4K video this can look blurry. A secondary model makes this picture bigger
(512x512 or 1024x1024) before it goes back into the frame, so the result is
sharper.

NVIDIA only.

- **UNet 4x** (paid): trained for restoration, quality close to TVAI `iris-2`, and
  faster than TVAI. See the
  examples on SLS Discord
  and [Supporting the project](../../README.md#supporting-the-project).
- **RTX Super Res** (free): very fast, nothing to install. Quality is
  OK. Some videos flicker, so test a short clip first.
- **Topaz TVAI**: similar to UNet 4x, but very slow. Needs
  [Topaz Video](https://www.topazlabs.com/topaz-video), which is paid and
  Windows only. Use the `iris-2` model.

Not used with LTX.

```bash
jasna --input input.mp4 --output output.mkv --secondary-restoration unet-4x
```

For TVAI, set the environment variables `TVAI_MODEL_DATA_DIR` and
`TVAI_MODEL_DIR` to your Topaz Video model folders, like below. `--tvai-args`
changes other Topaz settings.

<img width="505" height="37" alt="Topaz Video environment variables" src="https://github.com/user-attachments/assets/e19ced9d-d549-4e85-b20f-888e42466f1d" />

### Speed and VRAM comparison

Time and VRAM for two 1080p videos:

| Secondary type           | CAWD 1080p        | K109 1080p      |
| ------------------------ | -----------------:| -----------------:|
| No secondary             | 22s / 10.0 GB VRAM | 11s / 10.7 GB VRAM |
| UNet 4x                  | 29s / 12.5 GB VRAM | 14s / 12.6 GB VRAM |
| RTX Super Res            | 25s / 11.7 GB VRAM | 13s / 11.4 GB VRAM |
| TVAI (2 workers, Iris-2) | 52s / 12.1 GB VRAM | 24s / 12.4 GB VRAM |

## Still-image restoration (SD 1.5)

For images, Jasna uses a different model: a trained Stable Diffusion 1.5
model. Add an image to the queue (or give it to the CLI) and Jasna uses this
model automatically:

```bash
jasna --input photo.png --output restored.png
```

- Paid. One License code can include SD 1.5, UNet 4x and LTX. See
  [Supporting the project](../../README.md#supporting-the-project).
- Not included in the download. It is about **6.9 GB**. Jasna asks before it
  downloads it from
  [huggingface.co/Kruk2/sd-15-jav](https://huggingface.co/Kruk2/sd-15-jav).
- Needs about **7 GB of VRAM**, a bit more for big 4K images.

This model is experimental. Results change a lot from image to image, but some
are very good. Make a few versions and keep the best one:

```bash
jasna --input photo.png --output restored.png --sd15-variants 4
```

To try seeds one by one in the GUI, use
[Process images interactively](gui.md#images-one-by-one). All options
(`--sd15-steps`, `--sd15-strength`, `--sd15-seed`, ...) are in the
[CLI reference](cli.md#sd-15-image-restoration).

Examples:
SD 1.5 examples on SLS Discord
and more.
