# Jasna
### 🚀 If you want to support this project [buy me a coffee](https://buymeacoffee.com/kruk2) 🚀

JAV model restoration tool inspired (and in some places based on) by [Lada](https://codeberg.org/ladaapp/lada).\
Restoration model (mosaic_restoration_1.2) used in Jasna was trained by ladaapp (the lada author).

Features new mosaic detection model & **super fast GPU only pipeline** & TVAI support & simple GUI.\
Check benchmarks and usage below.
![slop_gui](https://github.com/user-attachments/assets/ae5d9b73-ea22-4263-8203-0ff89bbbcc51)


### Differences:
- **Improved mosaic detection model.**
- **Temporal overlap which reduces flickering.**
- **Secondary restoration model (Topaz TVAI and RTX Super Res). Improves quality and sharpnes**
- GPU only processing. Intial tests show that it can be 2x faster. Raw processing for places without mosaic is ~250fps on RTX 5090 <img width="860" height="56" alt="image" src="https://github.com/user-attachments/assets/a80ecaee-e36d-4c91-93e4-8bdd75048ac3" />
- Accurate color conversions on gpu (input matches output and no banding).
- Only modern Nvidia gpu is supported.
- TensorRT support.

### TODO:
- proper VR support
- ~~TVAI~~ and SeedVR
- Proper stream that can be played in Stash (and maybe others?)
- ~~improve performance (this version is very simple)~~
- ~~improve VRAM usage~~

### VRAM Management
Jasna automatically manages VRAM. When your GPU runs low on VRAM, frames waiting in the processing queue are temporarily moved to system RAM and moved back when needed. This happens in the background and requires no configuration.

If you still run into out-of-memory errors, reduce the **clip size** (e.g. from 180 to 60) or disable model compilation — both lower peak VRAM significantly (see table below).

### Benchmark
RTX 5090 + i9 13900k
| File | Clip (s) | lada 0.10.1 | jasna 0.3.0 | jasna 0.5.0 |
|---|---|---|---|---|
| **ABF-017** (4k, 2h 25min) | 60 | 02:56:26 | 01:20:49 (2.2× faster) | -- |
| **HUBLK-063** (1080p, 3h 10min) | 180 | 01:34:51 | 44:21 (2.1× faster) | 37:57 (2.5× faster) |
| **DASS-570_2m** | 30 | 01:08 | 00:30 (2.3× faster) | 00:24 (2.8× faster) |
| **NASK-223_Test** | 30 | 03:12 | 01:18 (2.5× faster) | 01:02 (3.1× faster) |
| **test-007** | 30 | 01:16 | 00:41 (1.9× faster) | 00:28 (2.7× faster) |
| **厚码测试2** | 30 | 01:52 | 00:43 (2.6× faster) | 00:36 (3.1× faster) |




### Usage
Download the latest release package (Windows/Linux).

- **If you downloaded the app (recommended)**:
  - **Windows**: You’re good to go — Jasna ships with everything it needs (`ffmpeg`, `ffprobe`, and `mkvmerge`).
  - **Linux**: You need `ffmpeg`, `ffprobe` (**major version must be 8**), and `mkvmerge` available on your system. Install via your package manager. MKVToolNix: [downloads](https://mkvtoolnix.download/downloads.html).

**First run will be slow** — TensorRT engines are compiled for your GPU. This takes **15-60 minutes**.\
Close all other applications (including browsers) and do not use the PC during compilation.\
Engines are cached in the `model_weights` folder and reused on all future runs (you can copy engine files & folders to a new version).

Remember to have up to date nvidia drivers.\
Tested nvidia drivers: **591.67**

### Detection Model
In general it's recommended to pick latest rf-detr.\
Lada Yolo models are available as they handle 2d animations better.

### Secondary Restoration Model
Jasna/Lada takes 256x256 pixel crop of mosaic region and restores it still in 256x256 resolution. This causes blurry results when mosaic area was bigger (close ups, 4k video etc).\
To eleviate that, you can use 2nd restoration model that upscales 256x256 to 512x512 or 1024x1024 which produces cleaner results.
Currently supported:
- **RTX Super-resolution** (fast, may flicker). Very fast, free, zero dependencies. In some videos it may produce a flickering effect — test on a short clip first.
- **TVAI** (best quality, slowest). Requires [Topaz Video](https://www.topazlabs.com/topaz-video) (paid, Windows only). Recommended model: **iris-2**.\
  ```--tvai-args``` allows you to customize model and other params. Defaults to iris-2.\
  Setup these as env variables for "Topaz Video":\
  <img width="505" height="37" alt="image" src="https://github.com/user-attachments/assets/e19ced9d-d549-4e85-b20f-888e42466f1d" />

### Max clip + temporal overlap
Main goal for the temporal overlap is to reduce flickering on the edges of clips.\
Going above 20 doesn't bring much. The larger overlap the longer processing times but less flickering.\
**Pick highest clip size you can and set overlap to something between 8-20.**\
Some guidance from my limited testing:
- 60 clip size + temporal overlap 6
- 90 clip size + temporal overlap 8
- 180 clip size + temporal overlap 15 if you can fit the model.

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
