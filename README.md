[**English**](README.md) | [日本語](README.ja.md) | [中文](README.zh.md)

# <img width="32" src="https://github.com/Kruk2/jasna/blob/main/assets/jasna-logo.png?raw=true" /> Jasna 

Jasna is a JAV mosaic restoration tool with a simple GUI, a CLI, a GPU-only processing pipeline, NVIDIA TensorRT and experimental AMD ROCm support, optional secondary restoration models, still-image restoration, and streaming support.

It is inspired by, and in some places based on, [Lada](https://codeberg.org/ladaapp/lada). The `mosaic_restoration_1.2` restoration model used by Jasna was trained by ladaapp, the Lada author.

Jasna is free. Supporters get a key that unlocks the extra models trained for this project: the **unet-4x** secondary upscaler and the experimental **SD 1.5 image restoration** model. See [Supporting the project](#supporting-the-project).

<img width="1200" height="907" alt="image" src="https://github.com/user-attachments/assets/d59a914b-482d-4f37-ae72-5c59eb5dc9bb" />


## Contents

- [What Jasna Does](#what-jasna-does)
- [Community](#community)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [First Run](#first-run)
- [Learn More](#learn-more)
- [Benchmarks](#benchmarks)
- [Supporting the Project](#supporting-the-project)
- [TODO](#todo)

## What Jasna Does

- Restores mosaics in video files.
- Restores mosaics in still images with the experimental SD 1.5 image model.
- Detects mosaics with RF-DETR models by default; Lada and ZeLeFans YOLO models are also available.
- Processes side-by-side VR180 videos per eye, with optional fisheye reprojection for detection and restoration.
- Reduces clip-boundary flicker with temporal overlap and crossfade.
- Can further improve quality with optional [secondary restoration models](docs/en/models.md#secondary-restoration) — **unet-4x**, **RTX Super Resolution**, or **Topaz Video AI** — which sharpen restored regions, especially large mosaics, close-ups, and 4K video.
- Can stream restored video to the built-in browser player or a supported Stash fork.

## Community

Join the [SLS Discord](https://discord.gg/uNwQ4mHqgv) for examples, support, and settings discussion. Please don't be too weird.

## Requirements

- An NVIDIA **GTX 16-series / RTX 20-series or newer** GPU. GTX 10-series and older cards (GTX 1050/1060/1070/1080) won't work. Not sure about yours? Check NVIDIA's [GPU table](https://developer.nvidia.com/cuda/gpus) — compute capability 7.5+ is required.
- Nvidia driver **610 or newer** on Windows, **580 or newer** on Linux.
- AMD support is experimental and needs a ROCm-supported GPU.
- Install Jasna into a folder whose path contains only English letters and numbers.

Jasna manages VRAM automatically: when it runs low, waiting frames are temporarily moved to system RAM. No configuration needed.

## Quick Start

1. Download the release package for your OS and GPU vendor.
2. Unzip it into a folder with only English characters in the path.
3. Start the app:
   - Windows: double click `jasna.exe`.
   - Linux NVIDIA: run the `jasna` file.
   - Linux AMD: run `run_jasna_amd.sh`.
4. Add a video or image, choose settings, and start processing.

Every setting in the GUI has a tooltip — hover the ⓘ icon next to it. The
[GUI guide](docs/en/gui.md) tours the rest: queue reordering, presets, output
patterns, and more.

Prefer the command line?

```bash
# Single video
jasna --input input.mp4 --output output.mkv

# Still image
jasna --input photo.png --output restored.png

# Whole folder
jasna --input input_folder --output output_folder
```

Run `jasna --help` for all options, or read the [CLI reference](docs/en/cli.md).

## First Run

The first run is slow because Jasna prepares GPU-specific files for your exact card. On NVIDIA this usually takes **15-60 minutes**; on AMD the preparation is much shorter. It only happens once — the results are cached in `model_weights` and reused on every later run. You can copy them from an older Jasna version to a newer one.

Close other applications, including browsers, and avoid using the PC while this runs.

If you run out of VRAM during processing, reduce **max clip size** first, for example from `180` to `60`. See [Tuning VRAM and GPU usage](docs/en/tuning.md).

## Learn More

- **[Using the GUI](docs/en/gui.md)** — the queue (drag & drop, reordering), presets, output patterns and file conflicts, and other easy-to-miss features.
- **[Choosing models](docs/en/models.md)** — which detection model to pick, sharper results with secondary restoration (unet-4x / RTX Super Resolution / Topaz), and SD 1.5 still-image restoration.
- **[Restoring only parts of a video](docs/en/segments.md)** — the Segment Editor, built-in mosaic scanning, suggesting better masks, and the `--segments` CLI flag.
- **[VR180 videos](docs/en/vr180.md)** — how Jasna handles side-by-side VR and when to use fisheye mode.
- **[Tuning VRAM and GPU usage](docs/en/tuning.md)** — clip size, temporal overlap, model compilation, and what to do when VRAM runs out.
- **[Advanced processing](docs/en/advanced_processing.md)** — denoising, 60→30 FPS export, color LUTs, custom encoder settings, and post-export actions.
- **[Streaming](docs/en/streaming.md)** — watch restored video on the fly in your browser or through Stash.
- **[CLI reference](docs/en/cli.md)** — every command-line option, including output templates, encoder settings per codec, and post-export actions.
- **[Running from source](docs/en/development.md)** — developer setup and build notes.

## Benchmarks

RTX 5090 + i9 13900k:

| File                            | Clip (s) | lada 0.10.1 | jasna 0.3.0          | jasna 0.5.0          | **jasna 0.6.2**        |
| ------------------------------- | -------: | ----------: | --------------------:| --------------------:| ----------------------:|
| **ABF-017** (4k, 2h 25min)      | 60       | 02:56:26    | 01:20:49 (2.2x faster) | 01:10:00 (2.5x faster) | — |
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

## TODO

Current TODO:

- SeedVR support?
- Continued performance and VRAM improvements.
- Better restoration model.
- Better detection model.
