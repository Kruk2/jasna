> **The only places to get a genuine License code for the supporter models are [Buy Me a Coffee](https://buymeacoffee.com/kruk2), [Unifans](https://app.unifans.io/c/kruk2), and for China the seller in the [Chinese README](README.zh.md#在闲鱼购买).**
>
> **支持者模型的正版许可证码只能通过 [Buy Me a Coffee](https://buymeacoffee.com/kruk2)、[Unifans](https://app.unifans.io/c/kruk2) 或 [闲鱼](README.zh.md#在闲鱼购买) 获取。**

[**English**](README.md) | [日本語](README.ja.md) | [中文](README.zh.md)

# <img width="32" src="https://github.com/Kruk2/jasna/blob/main/assets/jasna-logo.png?raw=true" /> Jasna 

Jasna restores mosaics in videos and images. It is free, has a simple app
and a command line, and runs on your GPU.

<img width="1200" height="907" alt="image" src="https://github.com/user-attachments/assets/d59a914b-482d-4f37-ae72-5c59eb5dc9bb" />

## Contents

**Start here**
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

- [What Jasna does](#what-jasna-does)
- [What you need](#what-you-need)
- [Quick start](#quick-start)
- [Help and community](#help-and-community)
- [Supporting the project](#supporting-the-project) (supporter models)

**Advanced**

- [How Jasna works](#how-jasna-works)
- [Command line](#command-line)
- [All guides](#all-guides)
- [Benchmarks](#benchmarks)
- [Licensing](#licensing)
- [Acknowledgments](#acknowledgments)

## What Jasna does

- Finds mosaics in a video and restores them. The rest of the picture is not
  touched.
- Two restoration models: fast **BasicVSR++** (free) and slow but more
  detailed **LTX** (for supporters, NVIDIA only).
- Restores only the parts you choose, with a preview, in the **Segment
  Editor**.
- Works with VR180 videos and still images (SD 1.5, for supporters).
- Lets you watch a video restored live in the **Video Player**, without making
  a file.

## What you need

- **NVIDIA** GTX 16-series / RTX 20-series or newer, with a recent driver
  (610+ on Windows, 580+ on Linux). GTX 10-series and older don't work.
- Or **AMD** Radeon RX 7000/9000 (experimental).
- A folder on a local disk, with no Chinese, Japanese, accented or other
  non-English characters in its path.

<details>
<summary>More details</summary>

- NVIDIA: your GPU needs compute capability 7.5 or higher in NVIDIA's
  [GPU table](https://developer.nvidia.com/cuda/gpus).
- AMD: a GPU supported by ROCm (Radeon RX 7000/9000, Radeon 780M/760M/740M,
  Ryzen AI). On Windows, AMD Adrenalin driver 26.8.1 or newer. Secondary
  restoration and LTX are NVIDIA only.
- Disk format: NTFS on Windows; ext4, Btrfs, XFS or ZFS on Linux. Network disks
  are not supported. Use an SSD; on a hard disk (HDD) Jasna starts and loads
  models slowly.
- LTX needs an NVIDIA RTX GPU, at least 8 GB VRAM and 16 GB RAM. See
  [What LTX needs](docs/en/models.md#what-ltx-needs).

</details>

## Quick start

1. Download the release for your system and GPU.
2. Unpack it into a folder with no Chinese, Japanese, accented or other
   non-English characters in the path.
3. Start the app:
   - Windows: double-click `jasna.exe`.
   - Linux NVIDIA: run the `jasna` file.
   - Linux AMD: run `run_jasna_amd.sh`.
4. Add a video, choose settings and click the **▶** button at the bottom.

**Your first job is slow.** On NVIDIA, Jasna first prepares the models for
your exact GPU. This takes about **15–60 minutes**, only once. Close other
apps and don't use the PC during this time. When you update Jasna, copy the
`model_weights` folder to the new version, so you don't wait again.

Every setting has a tooltip: move the mouse over the ⓘ icon. Out of GPU
memory (VRAM)? Lower **Max Clip Size**, for example from `90` to `60`. See
[Tuning](docs/en/tuning.md).
- An NVIDIA **GTX 16-series / RTX 20-series or newer** GPU. GTX 10-series and older cards (GTX 1050/1060/1070/1080) won't work. Not sure about yours? Check NVIDIA's [GPU table](https://developer.nvidia.com/cuda/gpus) — compute capability 7.5+ is required.
- Nvidia driver **610 or newer** on Windows, **580 or newer** on Linux.
- AMD support is experimental and needs a ROCm-supported GPU (Radeon RX 7000/9000, Radeon 780M/760M/740M, Ryzen AI iGPUs). On Windows, update the AMD Adrenalin driver to 26.8.1 or newer.
- Install Jasna into a folder whose path contains only English letters and numbers.

## Help and community

- [App guide](docs/en/gui.md): the queue, presets, output names, the Video
  Player and activation.
- [SLS Discord](https://discord.gg/FgrSuwZS3s): examples, help and settings.
  Jasna has a [new channel](https://discord.com/channels/1196376491815092265/1556960822578909244);
  older posts stay in the [old channel (archived)](https://discord.com/channels/1196376491815092265/1199059436199759943).
  Please don't be too weird.
- Problem? Click **Logs** at the bottom of the app, then **Export** to save a
  log file. Attach it when you ask for help.

## Supporting the project

Jasna is free. Your support pays for training new models, mostly GPU rental.
Supporters get a License code for the supporter models:

| Support | Models |
| --- | --- |
| **$15 USD** or more in total | **UNet 4x**: makes restored areas sharper.<br>**SD 1.5**: experimental model for still images. |
| **$25 USD** | **UNet 4x** and **SD 1.5**, plus<br>**LTX**: much slower than BasicVSR++, but more detail. NVIDIA only. Includes future updates of this LTX model. |

Supporters who gave **$15 USD or more on or before 14 October 2026** get LTX for free.

**Before you choose LTX:**

- Look at these LTX samples: [1](https://discord.com/channels/1196376491815092265/1199059436199759943/1556346988486856826), [2](https://discord.com/channels/1196376491815092265/1199059436199759943/1556337300814958684), [3](https://discord.com/channels/1196376491815092265/1199059436199759943/1555612574643388598), [4](https://discord.com/channels/1196376491815092265/1556960822578909244/1559341521575084063), [5](https://discord.com/channels/1196376491815092265/1556960822578909244/1559295317529202780), [6](https://discord.com/channels/1196376491815092265/1556960822578909244/1559205894959403121), [7](https://discord.com/channels/1196376491815092265/1556960822578909244/1559197341473574973), [8](https://discord.com/channels/1196376491815092265/1556960822578909244/1558798127195226112).
  Explore the [new channel](https://discord.com/channels/1196376491815092265/1556960822578909244) and the
  [old channel (archived)](https://discord.com/channels/1196376491815092265/1199059436199759943) on SLS Discord for more samples.
- Run LTX in **Trial mode (speed test)** on your PC. It is free and shows if your GPU can
  run LTX and how long it takes. The video is not restored. See
  [What LTX needs](docs/en/models.md#what-ltx-needs).

More examples:
[UNet 4x](https://discord.com/channels/1196376491815092265/1199059436199759943/1516497879684874260),
[SD 1.5](https://discord.com/channels/1196376491815092265/1199059436199759943/1492139124348420106),
[more SD 1.5](https://discord.com/channels/1196376491815092265/1199059436199759943/1516571355317800990).

### Where to support

License codes come **only** from here:

- [Buy Me a Coffee](https://buymeacoffee.com/kruk2) (crypto works too)
- [Unifans](https://app.unifans.io/c/kruk2)
- For China: see the [Chinese README](README.zh.md#在闲鱼购买)

Anyone else selling Jasna, codes or modified files is a scam: their keys don't
work, and their files can run hidden code on your PC. Download Jasna only from
GitHub.

### How to activate

1. Support on one of the pages above. You get a **License code**.
2. In Jasna, open **Activate Jasna**, enter the License code and press
   **Activate online**.
3. No internet on that PC, or it fails? Choose **Manual**, see
   [Activate supporter models](docs/en/gui.md#activate-supporter-models).

One License code works on **3 PCs or cloud workspaces**. After activation,
Jasna works offline. Rented GPUs: see [Cloud GPUs](docs/en/cloud.md).

What counts as a new PC: on Windows, a new motherboard; on Linux, a fresh
Linux install. Each uses one of your 3 slots. Reinstalling Windows or
changing the GPU, disk or RAM does not: just activate again with the same
License code. A few Windows PCs have no real motherboard ID; on those, a new
system disk also counts as a new PC.

Have an email and key from Jasna 0.10? They no longer work. Get a new License
code from where you bought it.

## Advanced

### How Jasna works

Almost everything runs on the GPU, from reading the video to saving it.

1. **Decode** the video on the GPU (NVDEC on NVIDIA, AMF on AMD).
2. **Detect** mosaics in each frame (default model `rfdetr-v6`).
3. **Track** each mosaic over time and group its frames into clips (up to
   **Max Clip Size** frames). Clips end at scene cuts, so two different shots
   are never mixed.
4. **Restore** each clip as a 256x256 picture with BasicVSR++. Clips overlap a
   little, so there are no visible jumps between them.
5. Optional **secondary restoration** (UNet 4x, RTX Super Res or Topaz TVAI)
   makes the restored picture bigger and sharper.
6. **Blend** the restored picture back into the original frame. Only the
   mosaic area changes.
7. **Encode** with the GPU encoder (NVENC on NVIDIA, AMF on AMD). Audio is
   copied (encoded again only if the output format can't hold it).

The first job is slow because the models for steps 2 and 4 (and UNet 4x, if
used) are prepared for your exact GPU (NVIDIA only). The result is saved in
`model_weights` and used again later. You can copy it to a newer Jasna
version.

LTX works differently. It goes over the video in 4 passes: find mosaics,
prepare, restore with a diffusion model, save. It is slower but can add more
real detail. See
[Choosing models](docs/en/models.md#restoration-model).

### Command line

The command line can do most of what the app does (not the Video Player,
LTX seed previews, mask suggestions or interactive image mode).
## Licensing

Jasna's public application source is AGPL-3.0. Models and bundled dependencies
retain their own terms. See [licensing and source availability](LICENSING.md),
[model notices](assets/THIRD_PARTY_MODELS.md), and the
[release source correspondence](RELEASE_SOURCES.md).

## Supporting the Project

```bash
# One video
jasna --input input.mp4 --output output.mkv

# One image
jasna --input photo.png --output restored.png

# A whole folder
jasna --input input_folder --output output_folder
```

`jasna --help` lists all options. See the [CLI reference](docs/en/cli.md).

### All guides

- [Using the GUI](docs/en/gui.md): queue, presets, output names, Video
  Player, activation.
- [Choosing models](docs/en/models.md): detection models, BasicVSR++ or LTX,
  secondary restoration, SD 1.5 for images.
- [Restoring only parts of a video](docs/en/segments.md): Segment Editor,
  mosaic scan, LTX only for big mosaics.
- [VR180 videos](docs/en/vr180.md): how side-by-side VR is handled.
- [Tuning VRAM and GPU usage](docs/en/tuning.md): clip size, overlap, what to
  do when VRAM runs out.
- [Advanced processing](docs/en/advanced_processing.md): denoise, 60→30 FPS,
  color LUT, sharpening, encoder quality, actions after the queue.
- [Streaming](docs/en/streaming.md): watch in the browser or in Stash.
- [Cloud GPUs](docs/en/cloud.md): Jasna on a rented GPU (RunPod, Vast).
- [CLI reference](docs/en/cli.md): all command line options.
- [Running from source](docs/en/development.md): for developers. From source
  you can use the free BasicVSR++; activation and supporter models need the
  official release.

### Benchmarks

Time to restore a whole 3:22 video with BasicVSR++ on an RTX 5090 (Linux).
Lower is better.

| Input | Lada 0.11.0 | Jasna v0.9.1 |
| --- | ---: | ---: |
| 720p H.264 | 01:47 | **00:32** (3.4x faster) |
| 1080p H.264 | 02:02 | **00:34** (3.6x faster) |
| 2160p H.264 | 04:56 | **01:03** (4.7x faster) |

<details>
<summary>GPU memory, older versions and how it was measured</summary>

GPU memory used (typical / peak, in GB). `—` means not measured. Lada uses
less GPU memory than Jasna.

| Input | Lada | v0.4.1 | v0.5.0 | v0.9.0 | **v0.9.1** |
| --- | ---: | ---: | ---: | ---: | ---: |
| 720p H.264 8-bit | 1.8/3.1 | 17.6/19.9 | 8.6/9.0 | 8.6/9.1 | **4.2/4.6** |
| 1080p H.264 8-bit | 2.0/3.3 | 18.9/21.4 | 9.5/10.2 | 9.3/10.1 | **4.8/5.6** |
| 2160p H.264 8-bit | 2.6/4.1 | 26.0/30.5 | 12.9/15.0 | 11.4/15.4 | **7.2/10.9** |
| 8K VR HEVC 60 fps (15 s) | — | — | — | 17.1/18.3 | — |

Times for older versions (v0.4.1: 01:35 / 01:46 / 03:23; v0.5.0: 00:45 /
00:47 / 01:22; v0.9.0: 00:34 / 00:39 / 01:15 for 720p / 1080p / 2160p; 8K VR:
00:34), the test settings, RAM use and every codec are in the
[full benchmark report](benchmarks/2026-07-26_release_matrix.md). Older
full-length videos: [legacy report](benchmarks/2026-07-26_legacy_videos.md).

</details>
Keys are sold **only** on these two pages. Anyone else selling Jasna, keys or modified files is a scam: their keys do not work with the official app, and their files can run hidden code on your PC. Download Jasna only from GitHub.

If activation fails, check the exact purchase email or handle and copy the full key again.
An invalid key alone does not prove fraud; Jasna identifies keys from known unofficial sellers separately.

## Acknowledgments

### Licensing

Jasna's public application source is AGPL-3.0. Models and bundled dependencies
keep their own terms. See [licensing and source availability](LICENSING.md),
[model notices](assets/THIRD_PARTY_MODELS.md) and the
[release source correspondence](RELEASE_SOURCES.md).

### Acknowledgments

- **[Lada](https://codeberg.org/ladaapp/lada) (Codeberg)**: Jasna is inspired
  by Lada and in some places based on it. The `mosaic_restoration_1.2` model
  used by Jasna was trained by ladaapp, the author of Lada.
- **ZeLeFans**: for the VR detection model and the detailed analysis of mosaic
  shapes and VR projection.
