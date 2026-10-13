> **The only places to get a genuine License code for the paid models are [Buy Me a Coffee](https://buymeacoffee.com/kruk2), [Unifans](https://app.unifans.io/c/kruk2), and for China the seller in the [Chinese README](README.zh.md#在闲鱼购买).**
>
> **付费模型的正版密钥只能通过 [Buy Me a Coffee](https://buymeacoffee.com/kruk2)、[Unifans](https://app.unifans.io/c/kruk2) 或 [闲鱼](README.zh.md#在闲鱼购买) 获取。**

[**English**](README.md) | [日本語](README.ja.md) | [中文](README.zh.md)

# <img width="32" src="https://github.com/Kruk2/jasna/blob/main/assets/jasna-logo.png?raw=true" /> Jasna 

Jasna restores mosaics in videos and images. It has a simple GUI and a
command line, and runs fully on the GPU (NVIDIA, or AMD as experimental).

Jasna is free. Supporters get a License code for extra models trained for this
project: the **unet-4x** upscaler and the experimental **SD 1.5** image model
for $15 USD, or both plus the **LTX** restoration model for $25 USD. See
[Supporting the project](#supporting-the-project).

<img width="1200" height="907" alt="image" src="https://github.com/user-attachments/assets/d59a914b-482d-4f37-ae72-5c59eb5dc9bb" />


## Contents

- [What Jasna Does](#what-jasna-does)
- [Community](#community)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [First Run](#first-run)
- [Learn More](#learn-more)
- [Benchmarks](#benchmarks)
- [Licensing](#licensing)
- [Supporting the Project](#supporting-the-project)
- [Acknowledgments](#acknowledgments)
- [TODO](#todo)

## What Jasna Does

- Restores mosaics in videos, and in images with the SD 1.5 model.
- Finds mosaics with the fast `rfdetr-v6` model. Other detection models are
  available too (bigger RF-DETR, Lada YOLO, ZeLeFans).
- Two restoration models: fast **BasicVSR++** (default, free) and slow but
  more detailed [LTX](docs/en/models.md#restoration-model) (paid, NVIDIA
  only).
- Optional [secondary restoration](docs/en/models.md#secondary-restoration)
  makes restored areas sharper, mostly on big mosaics and 4K (NVIDIA only):
  **unet-4x**, **RTX Super Resolution** or **Topaz Video AI**.
- Handles VR180 side-by-side videos per eye, with settings picked per studio.
- **Segment Editor**: restore only some parts of a video, scan for mosaics,
  preview the result, and choose a model per range.
- Smooth clip borders (overlap and crossfade) and no mixing across scene cuts.
- **Video Player**: watch a video restored live, without making a file.
- Streaming to your browser or to a [Stash](docs/en/streaming.md#stash)
  version made for Jasna.

## Community

Join the SLS Discord for examples, help and
settings. Please don't be too weird.

## Requirements

- NVIDIA **GTX 16-series / RTX 20-series or newer**. GTX 10-series and older
  (GTX 1050/1060/1070/1080) don't work. Not sure? Your GPU needs compute
  capability 7.5 or higher in NVIDIA's
  [GPU table](https://developer.nvidia.com/cuda/gpus).
- NVIDIA driver **610 or newer** on Windows, **580 or newer** on Linux.
- AMD (experimental): a GPU supported by ROCm (Radeon RX 7000/9000, Radeon
  780M/760M/740M, Ryzen AI). On Windows, AMD Adrenalin driver 26.8.1 or newer.
- Install Jasna in a folder whose path has only English letters and numbers,
  on a local disk: NTFS on Windows, ext4, Btrfs or XFS on Linux.
- LTX needs more. See [What LTX needs](docs/en/models.md#what-ltx-needs).

## Quick Start

1. Download the release for your system and GPU.
2. Unpack it into a folder with only English letters in the path.
3. Start the app:
   - Windows: double-click `jasna.exe`.
   - Linux NVIDIA: run the `jasna` file.
   - Linux AMD: run `run_jasna_amd.sh`.
4. Add a video or image, choose settings and click Start.

The first start takes longer. A loading window shows the installation check
before the main window opens.

On Windows, Jasna's temporary files can also be on an NTFS RAM disk such as
ImDisk. See [Windows temporary files](docs/en/cli.md#windows-temporary-files).

Every setting has a tooltip: move the mouse over the ⓘ icon. The
[GUI guide](docs/en/gui.md) explains the rest: the queue, presets, output
names, the Video Player and more.

Command line:

```bash
# One video
jasna --input input.mp4 --output output.mkv

# One image
jasna --input photo.png --output restored.png

# A whole folder
jasna --input input_folder --output output_folder
```

`jasna --help` lists all options. More in the [CLI reference](docs/en/cli.md).

## First Run

The first job is slow, because Jasna prepares files for your exact GPU. On
NVIDIA this takes about **15–60 minutes**, on AMD much less. It happens only
once. The files are saved in `model_weights` and used again later. You can
copy them from an older Jasna version to a newer one.

Close other apps, also the browser, and don't use the PC during this time.

Out of VRAM during a job? Lower **max clip size** first, for example from `90`
to `60`. See [Tuning VRAM and GPU usage](docs/en/tuning.md).

## Learn More

- **[Using the GUI](docs/en/gui.md)**: queue, presets, output names, Video
  Player, activation.
- **[Choosing models](docs/en/models.md)**: detection models, BasicVSR++ or
  LTX, secondary restoration, SD 1.5 for images.
- **[Restoring only parts of a video](docs/en/segments.md)**: Segment Editor,
  mosaic scan, LTX only for big mosaics, `--segments`.
- **[VR180 videos](docs/en/vr180.md)**: how side-by-side VR is handled.
- **[Tuning VRAM and GPU usage](docs/en/tuning.md)**: clip size, overlap, what
  to do when VRAM runs out.
- **[Advanced processing](docs/en/advanced_processing.md)**: denoise,
  60→30 FPS, color LUT, sharpening, encoder quality, actions after the queue.
- **[Streaming](docs/en/streaming.md)**: watch in the browser or in Stash.
- **[Cloud GPUs](docs/en/cloud.md)**: Jasna on a rented GPU (RunPod, Vast).
- **[CLI reference](docs/en/cli.md)**: all command line options.
- **[Running from source](docs/en/development.md)**: for developers.

From source you can use the free BasicVSR++. Activation and paid models need
the official release.

## Benchmarks

Time to restore a whole video, BasicVSR++ with default models, on an RTX 5090
(Linux), compared with Lada. Lower is better. The test clips are 3:22 long,
the 8K VR clip is 15 s.

| Input | Lada 0.11.0 v2 | v0.4.1 | v0.5.0 | v0.9.0 (4a171c9) | **v0.9.1 (18add8a)** |
| --- | ---: | ---: | ---: | ---: | ---: |
| 720p H.264 8-bit | 01:47 (baseline) | 01:35 (1.1x faster) | 00:45 (2.4x faster) | 00:34 (3.1x faster) | **00:32 (3.4x faster)** |
| 1080p H.264 8-bit | 02:02 (baseline) | 01:46 (1.2x faster) | 00:47 (2.6x faster) | 00:39 (3.2x faster) | **00:34 (3.6x faster)** |
| 2160p H.264 8-bit | 04:56 (baseline) | 03:23 (1.5x faster) | 01:22 (3.6x faster) | 01:15 (3.9x faster) | **01:03 (4.7x faster)** |
| 8K VR HEVC 8-bit 60 fps | — | — | — | 00:34 (8K baseline) | — |

GPU memory used (typical / peak, in GB). `—` means not measured.

| Input | Lada | v0.4.1 | v0.5.0 | v0.9.0 | **v0.9.1** |
| --- | ---: | ---: | ---: | ---: | ---: |
| 720p H.264 8-bit | 1.8/3.1 | 17.6/19.9 (baseline) | 8.6/9.0 (2.0x less) | 8.6/9.1 (2.0x less) | **4.2/4.6 (4.2x less)** |
| 1080p H.264 8-bit | 2.0/3.3 | 18.9/21.4 (baseline) | 9.5/10.2 (2.0x less) | 9.3/10.1 (2.0x less) | **4.8/5.6 (3.9x less)** |
| 2160p H.264 8-bit | 2.6/4.1 | 26.0/30.5 (baseline) | 12.9/15.0 (2.0x less) | 11.4/15.4 (2.3x less) | **7.2/10.9 (3.6x less)** |
| 8K VR HEVC 8-bit 60 fps | — | — | — | 17.1/18.3 (8K baseline) | — |

How it was measured, RAM use and every version and codec:
[full benchmark report](benchmarks/2026-07-26_release_matrix.md). Older
full-length videos: [legacy report](benchmarks/2026-07-26_legacy_videos.md).

## Licensing

Jasna's public application source is AGPL-3.0. Models and bundled dependencies
retain their own terms. See [licensing and source availability](LICENSING.md),
[model notices](assets/THIRD_PARTY_MODELS.md), and the
[release source correspondence](RELEASE_SOURCES.md).

## Supporting the Project

Your support pays for training new models, mostly GPU rental for bigger
datasets. Paid models:

| Support | Models |
| --- | --- |
| **$15 USD** or more in total | **unet-4x**: secondary upscaler, sharper results (256 → 1024).<br>**SD 1.5**: experimental model for still images. |
| **$25 USD** | **unet-4x** and **SD 1.5**, plus<br>**LTX**: much slower than BasicVSR++, but more detail. NVIDIA only. Includes future updates of this LTX model. |

Supporters who gave **$15 USD or more before [TBD]** get LTX for free.

**Before you buy LTX:**

- Look at the samples in the
  SLS Discord channel.
- Run LTX in **Trial mode** on your PC. It is free and shows if your GPU can
  run LTX and how long it takes. The video is not restored. See
  [What LTX needs](docs/en/models.md#what-ltx-needs).

More examples:

- unet-4x / secondary restoration on SLS Discord
- SD 1.5 image restoration on SLS Discord and more SD 1.5 examples

### Where to buy

License codes are sold **only** here:

- [Buy Me a Coffee](https://buymeacoffee.com/kruk2) (crypto works too)
- [Unifans](https://app.unifans.io/c/kruk2)
- For China: see the [Chinese README](README.zh.md#在闲鱼购买)

Anyone else selling Jasna, codes or modified files is a scam: their keys don't
work, and their files can run hidden code on your PC. Download Jasna only from
GitHub.

### How to activate

1. Buy on one of the pages above. You get a **License code**.
2. In Jasna, open **Activate Jasna**, enter the License code and press
   **Activate online**.
3. No internet on that PC, or it fails? Choose **Manual**, see
   [Activate paid models](docs/en/gui.md#activate-paid-models).

One License code works on **3 PCs or cloud workspaces**. Activation is checked
offline after that. Rented GPUs: see [Cloud GPUs](docs/en/cloud.md).

## Acknowledgments

- **Lada (Codeberg)**: Jasna is inspired
  by Lada and in some places based on it. The `mosaic_restoration_1.2` model
  used by Jasna was trained by ladaapp, the author of Lada.
- **ZeLeFans**: for the VR detection model and the detailed analysis of mosaic
  shapes and VR projection.

## TODO

Current TODO:

- SeedVR support?
- Continued performance and VRAM improvements.
- Better restoration model.
- Better detection model.
