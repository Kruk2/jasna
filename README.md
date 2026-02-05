# Jasna
### 🚀 If you want to support this project [buy me a coffee](https://buymeacoffee.com/kruk2) 🚀

JAV model restoration tool inspired (and in some places based on) by [Lada](https://codeberg.org/ladaapp/lada).\
Restoration model (mosaic_restoration_1.2) used in Jasna was trained by ladaapp (the lada author).

Features new mosaic detection model & GPU only pipeline.\
Check benchmarks and usage below.

### Differences:
- **Improved mosaic detection model.**
- **Temporal overlap which reduces flickering.**
- ** Secondary restoration model (Topaz TVAI and Swin2SR).**
- GPU only processing. Intial tests show that it can be 2x faster. Raw processing for places without mosaic is ~250fps on RTX 5090 <img width="860" height="56" alt="image" src="https://github.com/user-attachments/assets/a80ecaee-e36d-4c91-93e4-8bdd75048ac3" />
- Accurate color conversions on gpu (input matches output and no banding).
- Only modern Nvidia gpu is supported.
- TensorRT support.
- CLI only

### TODO:
- proper VR support
- ~~TVAI~~ and SeedVR
- Proper stream that can be played in Stash (and maybe others?)
- improve performance (this version is very simple)

### Benchmark
RTX 5090 + i9 13900k
| File | Clip (s) | lada 0.10.1 | jasna 0.3.0 | Δ vs lada (0.3.0) |
|------|----------|-------------|-------------|-------------------|
| **ABF-017**<br>(4k 2h 25min) | 60 | **02:56:26** | **01:20:49** | **01:35:37 (-54.2%)** |
| **HUBLK-063**<br>(1080p 3h 10min) | 180 | **01:34:51** | **44:21** | **-50:30 (-53.2%)** |
| DASS-570_2m | 180 | 01:08 | 00:34 | -00:34 (-50.0%) |
| NASK-223_Test | 180 | 03:17 | 01:33 | -01:44 (-52.8%) |
| test-007 | 180 | 01:21 | 00:39 | -00:42 (-51.9%) |
| 厚码测试2 | 180 | 01:51 | 01:01 | -00:50 (-45.0%) |
| DASS-570_2m | 30 | 01:08 | 00:30 | -00:38 (-55.9%) |
| NASK-223_Test | 30 | 03:12 | 01:18 | -01:54 (-59.4%) |
| test-007 | 30 | 01:16 | 00:41 | -00:35 (-46.1%) |
| 厚码测试2 | 30 | 01:52 | 00:43 | -01:09 (-61.6%) |
 
### Usage
Go to releases page and download last package. Built for windows/linux on cuda 13.0.\
Make sure that ```ffmpeg```  ```ffprobe``` ```mkvmerge``` is in your path.\
You can download mkvmerge [here](https://mkvtoolnix.download/downloads.html).

**First run might be slow because models will be compiled for your hardware (you can copy .engine files from model_weights to a new version!)**

Remember to have up to date nvidia drivers.\
Tested nvidia drivers: **591.67** 

### Max clip + temporal overlap
Main goal for the temporal overlap is to reduce flickering on the edges of clips.\
Going above 15 doesn't bring much. The larger overlap the longer processing times.\
Pick highest clip size you can and set overlap to something between 8-15.\
Some guidance from my limited testing:\
- 60 clip size + temporal overlap 8
- 90 clip size + temporal overlap 10
- 180 clip size + temporal overlap 15 if you can fit the model.

Using clip size below 60 might look ok, depends on video but prefer using 60 even if it means disabling model compilation.

### Restoration model compilation.
Read [#6](https://github.com/Kruk2/jasna/issues/6) for more details.\
Compiled model takes a lot of vram. Rough estimate is around 2.5GB VRAM per 30 frames in clip size. If you plan to use 180 clip size you have to have 24gb vram+ (180/30 * 2.5).\
You can opt out from compiled model at the cost of performance.\

### Disclaimer
Jasna is in early development and the main goal is to improve: restoration quality, mosaic detection, speed & vram consumption (in this order).
Thats why currently this project is aimed at more technical users, meaning the program accessability is lower and might be gated for some users.
I do this to dedicate more time on important features and if you want to help Pull Requests are welcomed.

## Building
Install these libs via ```uv pip install . --no-build-isolation```\
To build nvidia libs below you need also VS Build Tools 2022 (c++)
make sure you have cmake and ninja ```uv pip install cmake ninja```\
and cuda 13.0 in your system.

https://codeberg.org/Kruk2/vali

https://codeberg.org/Kruk2/PyNvVideoCodec

Once two libs above are installed to your python enviorment run:
```uv pip install -e .[dev]``` in jasna repository
