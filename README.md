[**English**](README.md) | [日本語](README.ja.md) | [中文](README.zh.md)

# Jasna
### 🚀 Jasna is free. [Support it](#supporting-the-project) so I can train better models 🚀
Supporters get a key that unlocks the extra models I train — the **unet-4x** secondary upscaler and the
experimental **SD 1.5 image restoration** model. Details in [Supporting the project](#supporting-the-project).

JAV model restoration tool inspired (and in some places based on) by [Lada](https://codeberg.org/ladaapp/lada).\
Restoration model (mosaic_restoration_1.2) used in Jasna was trained by ladaapp (the lada author).

Features new mosaic detection model & **super fast GPU only pipeline** & TVAI support & simple GUI.\
Check benchmarks and usage below.
![slop_gui](https://github.com/user-attachments/assets/ae5d9b73-ea22-4263-8203-0ff89bbbcc51)


### Differences:
- **Improved mosaic detection model.**
- **Temporal overlap which reduces flickering.**
- **Secondary restoration model (Topaz TVAI and RTX Super Res). Improves quality and sharpnes**
- **Experimental SD 1.5 image restoration for still photos** — sometimes produces really good results (supporter model).
- GPU only processing. Intial tests show that it can be 2x faster. Raw processing for places without mosaic is ~250fps on RTX 5090 <img width="860" height="56" alt="image" src="https://github.com/user-attachments/assets/a80ecaee-e36d-4c91-93e4-8bdd75048ac3" />
- Accurate color conversions on gpu (input matches output and no banding).
- Only modern Nvidia gpu is supported.
- TensorRT support.

### TODO:
- proper VR support
- ~~TVAI~~ and SeedVR
- ~~Proper stream that can be played in Stash (and maybe others?)~~
- ~~improve performance (this version is very simple)~~
- ~~improve VRAM usage~~

### VRAM Management
Jasna automatically manages VRAM. When your GPU runs low on VRAM, frames waiting in the processing queue are temporarily moved to system RAM and moved back when needed. This happens in the background and requires no configuration.

If you still run into out-of-memory errors, reduce the **clip size** (e.g. from 180 to 60) or disable model compilation — both lower peak VRAM significantly (see table below).

### Benchmark
RTX 5090 + i9 13900k
| File                            | Clip (s) | lada 0.10.1 |            jasna 0.3.0 |            jasna 0.5.0 |         **jasna 0.6.2** |
| ------------------------------- | -------: | ----------: | ---------------------: | ---------------------: | ----------------------: |
| **ABF-017** (4k, 2h 25min)      |       60 |    02:56:26 | 01:20:49 (2.2× faster) | 01:10:00 (2.5× faster) |                      xx |
| **HUBLK-063** (1080p, 3h 10min) |      180 |    01:34:51 |    44:21 (2.1× faster) |    37:57 (2.5× faster) | **30:58 (3.1× faster)** |
| **DASS-570_2m**                 |       30 |       01:08 |    00:30 (2.3× faster) |    00:24 (2.8× faster) | **00:20 (3.4× faster)** |
| **NASK-223_Test**               |       30 |       03:12 |    01:18 (2.5× faster) |    01:02 (3.1× faster) | **00:58 (3.3× faster)** |
| **test-007**                    |       30 |       01:16 |    00:41 (1.9× faster) |    00:28 (2.7× faster) | **00:22 (3.5× faster)** |
| **厚码测试2**                       |       30 |       01:52 |    00:43 (2.6× faster) |    00:36 (3.1× faster) | **00:34 (3.3× faster)** |



## Supporting the project
Support pays for training the extra models — mainly renting bigger GPUs and the compute time to train
on larger datasets. Supporters get a key that unlocks the models I train this way:
- **unet-4x** secondary upscaler (sharper 256→1024 restoration).
- **SD 1.5 image restoration** (experimental still-image model).

How to get a key:
1. Contribute **$15 USD or more in total** — across any number of contributions, at any time.
2. After your contribution is processed, your supporter key is sent automatically:
   - **[Unifans](https://app.unifans.io/c/kruk2)** — the key is sent automatically by platform message. There might be a slight delay.
   - **[Buy Me a Coffee](https://buymeacoffee.com/kruk2)** (also handles **crypto**) — the key is sent automatically to the email/handle you used for the contribution. The key is tied to that email/handle.

### Usage
Download the latest release package (Windows/Linux).\
Unzip the release and double click on jasna.exe (Windows) or on jasna file (Linux). This will open console.\
You can also use it as command line program.

- **If you downloaded the app (recommended)**:
  - **Windows**: You’re good to go — Jasna ships with everything it needs (`ffmpeg`, `ffprobe`, and `mkvmerge`).
  - **Linux**: You need `ffmpeg`, `ffprobe` (**major version must be 8**), and `mkvmerge` available on your system. Install via your package manager. MKVToolNix: [downloads](https://mkvtoolnix.download/downloads.html).

**First run will be slow** — TensorRT engines are compiled for your GPU. This takes **15-60 minutes**.\
Close all other applications (including browsers) and do not use the PC during compilation.\
Engines are cached in the `model_weights` folder and reused on all future runs (you can copy engine files & folders to a new version).

**Remember to have up to date nvidia drivers.**\
Tested nvidia drivers: **591.67** (but anything from 59x family should be ok and it's minimum required).\
**Jasna requires GPU with minimum compute capability: 7.5**

### Detection Model
In general it's recommended to pick latest rf-detr.\
Lada Yolo models are available as they handle 2d animations better.

### Secondary Restoration Model
Jasna/Lada takes 256x256 pixel crop of mosaic region and restores it still in 256x256 resolution. This causes blurry results when mosaic area was bigger (close ups, 4k video etc).\
To eleviate that, you can use 2nd restoration model that upscales 256x256 to 512x512 or 1024x1024 which produces cleaner results.
Currently supported:
- **unet-4x** (supporter model, much faster than TVAI and similar quality). Trained on an in-domain (JAV) dataset and visually quite close to TVAI iris-2, but runs locally with no extra setup. If you run into quality issues, please open a [GitHub issue](https://github.com/Kruk2/jasna/issues). Unlock it with a supporter key — see [Supporting the project](#supporting-the-project).
- **RTX Super-resolution** (very fast, ok quality). Very fast, free, zero dependencies. In some videos it may produce a flickering effect — test on a short clip first. Place jasna in folder with english only characters.
- **TVAI** (better than RTX super resolution, comparable to unet-4x, very slow). Requires [Topaz Video](https://www.topazlabs.com/topaz-video) (paid, Windows only). Recommended model: **iris-2**.\
  ```--tvai-args``` allows you to customize model and other params. Defaults to iris-2.\
  Setup these as env variables for "Topaz Video":\
  <img width="505" height="37" alt="image" src="https://github.com/user-attachments/assets/e19ced9d-d549-4e85-b20f-888e42466f1d" />

VRAM + time usage:
| Secondary type           |         CAWD 1080p |       KV-109 1080p |
| ------------------------ | -----------------: | -----------------: |
| No secondary             | 22s / 10.0 GB VRAM | 11s / 10.7 GB VRAM |
| unet-4x                  | 29s / 12.5 GB VRAM | 14s / 12.6 GB VRAM |
| RTX Super-Res            | 25s / 11.7 GB VRAM | 13s / 11.4 GB VRAM |
| TVAI (2 workers, Iris-2) | 52s / 12.1 GB VRAM | 24s / 12.4 GB VRAM |

Restoration examples on [SLS discord](https://discord.com/channels/1196376491815092265/1199059436199759943/1516497879684874260)

### Image restoration (SD 1.5)
For **still images** Jasna can use a fine-tuned Stable Diffusion 1.5 inpaint model instead of the
video pipeline. It detects mosaics, inpaints each region at 512×512 and blends the result back.

For examples on restorations you can checkout [SLS discord](https://discord.com/channels/1196376491815092265/1199059436199759943/1492139124348420106) and [here](https://discord.com/channels/1196376491815092265/1199059436199759943/1516571355317800990).

- CLI: `jasna --input photo.png --output out.png` — image inputs **auto-route** to the SD 1.5 model,
  no flag needed. The image model is selected with `--image-restoration-model-name` (default and only
  value: `sd-15-jav`); `--restoration-model-name` stays video-only. Knobs: `--sd15-steps`,
  `--sd15-strength` (≤ 0.7), `--sd15-freeu/--no-sd15-freeu`, `--sd15-seed`, `--sd15-variants N`.
- GUI: just add an image to the queue — image jobs route to the SD 1.5 model automatically.
  Tune it in the **Image Restoration** settings section.
- Folder input (CLI): `--input <folder> --output <folder>` processes every media file in the folder —
  **images first, then videos** — writing `<name>_out<ext>` into the output folder (`--output` must be
  a folder here). The GUI queue already mixes images and videos freely.
- The model is **not bundled** (~6.9 GB). It lives in `model_weights/sd-15-jav/`. Either drop the
  bundle there yourself, or let Jasna fetch it from
  [huggingface.co/Kruk2/sd-15-jav](https://huggingface.co/Kruk2/sd-15-jav) — it asks before
  downloading (CLI prompt, or the **Download model** button in the GUI). The checkpoint is avilable only for supporters at the moment — same key as unet-4x — see
  [Supporting the project](#supporting-the-project).
- It's **experimental**: results vary by scene, but on the right image it can look genuinely good.
  Try a few `--sd15-variants` (different seeds) and keep the best one.
- Expect around **7 GB of VRAM** during inference (a bit more on large 4K images).

### Max clip + temporal overlap
Main goal for the temporal overlap is to reduce flickering on the edges of clips.\
Going above 20 doesn't bring much. The larger overlap the longer processing times but less flickering.\
**Pick highest clip size you can and set overlap to something between 8-20.**\
Some guidance from my limited testing:
- 60 clip size + temporal overlap 6
- 90 clip size + temporal overlap 8
- 180 clip size + temporal overlap 15 (needs 12 GB+ VRAM with Compile BasicVSR++ enabled, less with it disabled).

4K videos use more VRAM — a lower clip size may produce similar quality but process much faster. Experiment a bit.

Using clip size below 60 might look ok, depends on video but prefer using 60 even if it means disabling model compilation.
```--enable-crossfade``` should probably be always enabled as it reduces flickering .

### Restoration model compilation
The restoration model is compiled into TensorRT sub-engines.\
First compilation takes **15-60 minutes** — close all other applications (including browsers) and avoid using the PC.\
Engines are cached and reused automatically. You can opt out from compilation at the cost of performance.

The table below shows how much VRAM the **compiled engines alone** occupy (not total processing VRAM):

| | Clip 60 | Clip 180 |
|---|---|---|
| **Engine VRAM (compiled)** | ~1.9 GB | ~5.4 GB |
| **Engine VRAM (no compilation)** | ~1.2 GB | ~1.2 GB |

### Streaming

Watch restored videos on-the-fly without having to process the entire file first.

**Browser player (built-in)**\
Run jasna with `--stream` flag (CLI only for now). A browser window opens with an HLS player. Pick a video file and start watching. Seeking is supported.
```
jasna --stream
```

**Stash integration**\
Use Jasna directly inside [Stash](https://github.com/stashapp/stash) — play any scene and it gets processed through Jasna on-the-fly. Stash launches Jasna automatically. A custom Stash fork with Jasna support is available:\
👉 **[Stash v0.30.1-jasna](https://github.com/Kruk2/stash/releases/tag/v0.30.1-jasna)**

Setup:
1. Download the Stash fork from the link above.
2. Set environment variables before starting Stash:
   - `JASNA_CLI_PATH` — full path to `jasna-cli.exe`
   - `JASNA_WORKING_DIR` — full path to the folder where `jasna-cli.exe` is located
3. Start Stash and play any scene — Jasna processes it as you watch. Seeking works.

### Disclaimer
Jasna is in early development and the main goal is to improve: restoration quality, mosaic detection, speed & vram consumption (in this order).
Thats why currently this project is aimed at more technical users, meaning the program accessability is lower and might be gated for some users.
I do this to dedicate more time on important features and if you want to help Pull Requests are welcomed.

## Building
Install these libs via ```uv pip install . --no-build-isolation```\
To build nvidia libs below you need also VS Build Tools 2022 (c++)
make sure you have cmake and ninja ```uv pip install cmake ninja```\
and cuda 13.0 in your system.

### Running from source (developer setup)
- Install `ffmpeg` + `ffprobe` and make sure they are on PATH (**ffmpeg major version must be 8**).
- Install `mkvmerge` (part of MKVToolNix): [downloads](https://mkvtoolnix.download/downloads.html).

https://codeberg.org/Kruk2/vali

https://codeberg.org/Kruk2/PyNvVideoCodec

Once two libs above are installed to your python enviorment run:
```uv pip install -e .[dev]``` in jasna repository
