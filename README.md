[**English**](README.md) | [日本語](README.ja.md) | [中文](README.zh.md)

# Jasna

Jasna is a JAV mosaic restoration tool with a simple GUI, a CLI, a GPU-only processing pipeline, TensorRT support, optional secondary restoration models, still-image restoration, and streaming support.

It is inspired by, and in some places based on, [Lada](https://codeberg.org/ladaapp/lada). The `mosaic_restoration_1.2` restoration model used by Jasna was trained by ladaapp, the Lada author.

Jasna is free. Supporters get a key that unlocks the extra models trained for this project: the **unet-4x** secondary upscaler and the experimental **SD 1.5 image restoration** model. See [Supporting the project](#supporting-the-project).

![Jasna GUI](https://github.com/user-attachments/assets/ae5d9b73-ea22-4263-8203-0ff89bbbcc51)

## Contents

- [What Jasna Does](#what-jasna-does)
- [Community](#community)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [First Run](#first-run)
- [Choosing Models](#choosing-models)
- [Tuning Quality and VRAM](#tuning-quality-and-vram)
- [Streaming](#streaming)
- [Benchmarks](#benchmarks)
- [Supporting the Project](#supporting-the-project)
- [Current Limitations and TODO](#current-limitations-and-todo)
- [Running from Source](#running-from-source)

## What Jasna Does

- Restores mosaics in video files.
- Restores mosaics in still images with the experimental SD 1.5 image model.
- Detects mosaics with RF-DETR models by default; Lada YOLO models are also available.
- Reduces clip-boundary flicker with temporal overlap and crossfade.
- Can use secondary restoration through **unet-4x**, **RTX Super Resolution**, or **Topaz Video AI**.
- Can stream restored video to the built-in browser player or a supported Stash fork.

## Community

Join the [SLS Discord](https://discord.gg/5R2Rx5nBH) for examples, support, and settings discussion. Please don't be too weird.

## Requirements

- A modern Nvidia GPU with compute capability **7.5 or newer**.
- Rough GPU guide: **GTX 16-series**, **RTX 20-series**, **RTX 30-series**, **RTX 40-series**, **RTX 50-series**, and newer workstation/data-center cards.
- Too old: **GTX 10-series**, including GTX 1050/1060/1070/1080.
- For exact GPU lookup, check NVIDIA's [CUDA GPU compute capability table](https://developer.nvidia.com/cuda/gpus).
- Up-to-date Nvidia drivers. Tested driver: **591.67**. The 59x driver family is the minimum expected family.
- An install path that uses ASCII characters only.
- Windows release package: bundled with `ffmpeg`, `ffprobe`, and `mkvmerge`.
- Linux release package: requires `ffmpeg`, `ffprobe`, and `mkvmerge` on your system. `ffmpeg` major version must be **8**. `mkvmerge` is part of [MKVToolNix](https://mkvtoolnix.download/downloads.html).

Jasna automatically manages VRAM. When GPU VRAM runs low, frames waiting in the processing queue are temporarily moved to system RAM and moved back when needed. This requires no configuration.

## Quick Start

1. Download the latest Windows or Linux release package.
2. Unzip it into a folder with ASCII-only characters in the path.
3. Start the app:
   - Windows: double click `jasna.exe`.
   - Linux: run the `jasna` file.
4. Add a video or image in the GUI, choose settings, and start processing.

You can also use Jasna from the command line:

```bash
jasna --input input.mp4 --output output.mkv
```

For still images, no image-specific flag is needed:

```bash
jasna --input photo.png --output restored.png
```

For folder input, both `--input` and `--output` must be folders. Jasna processes images first, then videos, shows an overall `[current/total]` file counter, and writes `<name>_out<ext>` into the output folder by default.

```bash
jasna --input input_folder --output output_folder
```

Folder batches can also use the same `{original}` filename template style as the GUI:

```bash
jasna --input input_folder --output output_folder --output-pattern "{original}_restored.mp4"
```

Images keep their source extension, while videos use the template extension when one is provided. Jasna checks the planned folder outputs before processing and exits with an error if the template maps multiple inputs to the same output file.

## First Run

The first run is slow because TensorRT engines are compiled for your GPU. Compilation usually takes **15-60 minutes**.

Close other applications, including browsers, and avoid using the PC during compilation. Engines are cached in `model_weights` and reused on later runs. You can copy engine files and folders from an older Jasna version to a newer one.

If you run out of VRAM during processing, reduce **max clip size** first, for example from `180` to `60`. Disabling BasicVSR++ compilation also lowers peak VRAM, but processing will be slower.

## Choosing Models

### Detection Model

In general, use the latest RF-DETR model. Lada YOLO models are also available and can work better for 2D animations.

CLI option:

```bash
jasna --input input.mp4 --output output.mkv --detection-model rfdetr-v5
```

### Secondary Restoration

Jasna and Lada restore a 256x256 crop of each mosaic region. Large mosaic regions, close-ups, and 4K videos can therefore look blurry after the primary restoration model. A secondary restoration model can upscale the restored crop to 512x512 or 1024x1024 before blending it back.

Supported secondary models:

- **unet-4x**: supporter model. Faster than TVAI with similar quality in current testing. Trained on an in-domain JAV dataset and visually close to TVAI `iris-2`. See [unet-4x / secondary restoration examples on SLS Discord](https://discord.com/channels/1196376491815092265/1199059436199759943/1516497879684874260). Unlock it with a supporter key; see [Supporting the project](#supporting-the-project). If you hit quality problems, open a [GitHub issue](https://github.com/Kruk2/jasna/issues).
- **RTX Super Resolution**: very fast, free, and has no extra dependencies. Quality is okay. Some videos may flicker, so test on a short clip first.
- **TVAI**: better than RTX Super Resolution and comparable to unet-4x in current testing, but very slow. Requires [Topaz Video](https://www.topazlabs.com/topaz-video), which is paid and Windows-only. Recommended model: `iris-2`.

CLI option:

```bash
jasna --input input.mp4 --output output.mkv --secondary-restoration unet-4x
```

For TVAI, `--tvai-args` can customize the Topaz model parameters. The default model is `iris-2`. Configure these environment variables for Topaz Video:

<img width="505" height="37" alt="Topaz Video environment variables" src="https://github.com/user-attachments/assets/e19ced9d-d549-4e85-b20f-888e42466f1d" />

VRAM and time usage:

| Secondary type           | CAWD 1080p        | KV-109 1080p      |
| ------------------------ | -----------------:| -----------------:|
| No secondary             | 22s / 10.0 GB VRAM | 11s / 10.7 GB VRAM |
| unet-4x                  | 29s / 12.5 GB VRAM | 14s / 12.6 GB VRAM |
| RTX Super-Res            | 25s / 11.7 GB VRAM | 13s / 11.4 GB VRAM |
| TVAI (2 workers, Iris-2) | 52s / 12.1 GB VRAM | 24s / 12.4 GB VRAM |

Restoration examples are available on [SLS Discord](https://discord.com/channels/1196376491815092265/1199059436199759943/1516497879684874260).

### Still-Image Restoration

For still images, Jasna can use a fine-tuned Stable Diffusion 1.5 inpaint model instead of the video pipeline. It detects mosaics, inpaints each region at 512x512, and blends the result back.

- CLI: `jasna --input photo.png --output out.png`
- GUI: add an image to the queue. Image jobs route to SD 1.5 automatically.
- Tuning options: `--sd15-steps`, `--sd15-strength` (clamped to `<= 0.7`), `--sd15-freeu` / `--no-sd15-freeu`, `--sd15-seed`, and `--sd15-variants N`.
- The image model is selected with `--image-restoration-model-name`. The default and only current value is `sd-15-jav`.
- `--restoration-model-name` is for video only.

The SD 1.5 model is **not bundled** and is about **6.9 GB**. It belongs in `model_weights/sd-15-jav/`. You can place the bundle there yourself or let Jasna fetch it from [huggingface.co/Kruk2/sd-15-jav](https://huggingface.co/Kruk2/sd-15-jav). Jasna asks before downloading, either through the CLI prompt or the GUI **Download model** button.

The checkpoint is currently available only to supporters and uses the same key as unet-4x. See [Supporting the project](#supporting-the-project).

The SD 1.5 path is experimental. Results vary by scene, but some images can work very well. Try several `--sd15-variants` values and keep the best result. Expect about **7 GB VRAM** during inference, and a bit more for large 4K images.

Examples are available on [SLS Discord](https://discord.com/channels/1196376491815092265/1199059436199759943/1492139124348420106) and [more SD 1.5 examples](https://discord.com/channels/1196376491815092265/1199059436199759943/1516571355317800990).

## Tuning Quality and VRAM

### Max Clip Size and Temporal Overlap

Temporal overlap reduces flicker at clip boundaries. Larger overlap increases processing time but can reduce flicker. Going above `20` usually does not help much.

Recommended starting point:

- Use the highest **max clip size** your GPU can handle.
- Set **temporal overlap** between `8` and `20`.
- Keep crossfade enabled with `--enable-crossfade`.

Limited testing guidance:

| Max clip size | Temporal overlap | Notes |
| -------------:| ----------------:| ----- |
| 60            | 6                | Lower VRAM option. |
| 90            | 8                | Current default-style balance. |
| 180           | 15               | Needs 12 GB+ VRAM with BasicVSR++ compilation enabled; less with compilation disabled. |

4K videos use more VRAM. A lower clip size may produce similar quality and process faster. Clip sizes below `60` can work on some videos, but `60` is preferred even if you need to disable model compilation.

CLI example:

```bash
jasna --input input.mp4 --output output.mkv --max-clip-size 90 --temporal-overlap 8 --enable-crossfade
```

### Restoration Model Compilation

The restoration model is compiled into TensorRT sub-engines. Compilation improves speed but uses more VRAM. You can opt out at the cost of performance:

```bash
jasna --input input.mp4 --output output.mkv --no-compile-basicvsrpp
```

Compiled engine VRAM only, not total processing VRAM:

|                               | Clip 60 | Clip 180 |
| ----------------------------- | -------:| --------:|
| Engine VRAM, compiled         | ~1.9 GB | ~5.4 GB  |
| Engine VRAM, no compilation   | ~1.2 GB | ~1.2 GB  |

## Streaming

Streaming lets you watch restored video on the fly without processing the whole file first.

### Browser Player

Streaming mode is CLI-only for now. It opens an HLS player in a browser window. Pick a video file and start watching. Seeking is supported.

```bash
jasna --stream
```

### Stash Integration

Jasna can be used inside [Stash](https://github.com/stashapp/stash) through a custom Stash fork. Play a scene and Stash launches Jasna automatically, processing as you watch. Seeking works.

Custom fork: **[Stash v0.30.1-jasna](https://github.com/Kruk2/stash/releases/tag/v0.30.1-jasna)**

Setup:

1. Download the Stash fork from the link above.
2. Set environment variables before starting Stash:
   - `JASNA_CLI_PATH`: full path to `jasna-cli.exe`
   - `JASNA_WORKING_DIR`: full path to the folder containing `jasna-cli.exe`
3. Start Stash and play a scene.

## Benchmarks

RTX 5090 + i9 13900k:

| File                            | Clip (s) | lada 0.10.1 | jasna 0.3.0          | jasna 0.5.0          | **jasna 0.6.2**        |
| ------------------------------- | -------: | ----------: | --------------------:| --------------------:| ----------------------:|
| **ABF-017** (4k, 2h 25min)      | 60       | 02:56:26    | 01:20:49 (2.2x faster) | 01:10:00 (2.5x faster) | xx |
| **HUBLK-063** (1080p, 3h 10min) | 180      | 01:34:51    | 44:21 (2.1x faster)  | 37:57 (2.5x faster)  | **30:58 (3.1x faster)** |
| **DASS-570_2m**                 | 30       | 01:08       | 00:30 (2.3x faster)  | 00:24 (2.8x faster)  | **00:20 (3.4x faster)** |
| **NASK-223_Test**               | 30       | 03:12       | 01:18 (2.5x faster)  | 01:02 (3.1x faster)  | **00:58 (3.3x faster)** |
| **test-007**                    | 30       | 01:16       | 00:41 (1.9x faster)  | 00:28 (2.7x faster)  | **00:22 (3.5x faster)** |
| **厚码测试2**                   | 30       | 01:52       | 00:43 (2.6x faster)  | 00:36 (3.1x faster)  | **00:34 (3.3x faster)** |

## Supporting the Project

Support pays for training extra models, mainly GPU rental and compute time for larger datasets. Supporters get a key that unlocks:

- **unet-4x** secondary upscaler for sharper 256->1024 restoration.
- **SD 1.5 image restoration**, the experimental still-image model.

Example results:

- [unet-4x / secondary restoration examples on SLS Discord](https://discord.com/channels/1196376491815092265/1199059436199759943/1516497879684874260)
- [SD 1.5 image restoration examples on SLS Discord](https://discord.com/channels/1196376491815092265/1199059436199759943/1492139124348420106) and [more SD 1.5 examples](https://discord.com/channels/1196376491815092265/1199059436199759943/1516571355317800990)

How to get a key:

1. Contribute **$15 USD or more in total**, across any number of contributions and at any time.
2. After your contribution is processed, your supporter key is sent automatically:
   - **[Unifans](https://app.unifans.io/c/kruk2)**: sent by platform message. There might be a slight delay.
   - **[Buy Me a Coffee](https://buymeacoffee.com/kruk2)**, including **crypto**: sent to the email or handle used for the contribution. The key is tied to that email or handle.

## Current Limitations and TODO

Jasna is in early development. The main goals are improving restoration quality, mosaic detection, speed, and VRAM usage, in that order. The project is currently aimed at more technical users, so some workflows may still be rough. Pull requests are welcome.

Current TODO:

- Proper VR support.
- SeedVR support.
- Continued performance and VRAM improvements.

## Running from Source

Python requirement from `pyproject.toml`: **Python 3.13 or newer**.

The public source checkout does not include the protection module. Running from source is fine for development and free models, but supporter-only models such as **unet-4x** and **SD 1.5 image restoration** will not be available from a plain source checkout.

Install runtime dependencies:

```bash
uv pip install . --no-build-isolation
```

For Nvidia library builds, you also need:

- VS Build Tools 2022 with C++ support.
- CUDA 13.0 installed on the system.
- `cmake` and `ninja`:

```bash
uv pip install cmake ninja
```

Developer setup also requires:

- `ffmpeg` and `ffprobe` on `PATH`; `ffmpeg` major version must be **8**.
- `mkvmerge`, from [MKVToolNix](https://mkvtoolnix.download/downloads.html).
- The two libraries below installed into your Python environment:
  - https://codeberg.org/Kruk2/vali
  - https://codeberg.org/Kruk2/PyNvVideoCodec

Then install Jasna in editable mode:

```bash
uv pip install -e .[dev]
```
