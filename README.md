# Jasna
### 🚀 If you want to support this project [buy me a coffee](https://buymeacoffee.com/kruk2) 🚀

JAV model restoration tool inspired (and in some places based on) by [Lada](https://codeberg.org/ladaapp/lada).\
Restoration model (mosaic_restoration_1.2) used in Jasna was trained by ladaapp (the lada author).

### Benchmark
RTX 5090 + i9 13900k
| File | Clip (s) | lada 0.10.1 | jasna 0.2.0 | jasna 0.3.0 | Δ vs lada (0.2.0) | Δ vs lada (0.3.0) |
|------|----------|------|-------------|-------------|-------------------|-------------------|
| DASS-570_2m.mp4 | 180 | 01:08 | 00:56 | 00:34 | -00:12 (-17.6%) | -00:34 (-50.0%) |
| NASK-223_Test.mp4 | 180 | 03:17 | 02:37 | 01:33 | -00:40 (-20.3%) | -01:44 (-52.8%) |
| test-007.mp4 | 180 | 01:21 | 01:00 | 00:39 | -00:21 (-25.9%) | -00:42 (-51.9%) |
| 厚码测试2.mkv | 180 | 01:51 | 01:29 | 01:01 | -00:22 (-19.8%) | -00:50 (-45.0%) |
| DASS-570_2m.mp4 | 30 | 01:08 | 00:52 | 00:30 | -00:16 (-23.5%) | -00:38 (-55.9%) |
| NASK-223_Test.mp4 | 30 | 03:12 | 02:42 | 01:18 | -00:30 (-15.6%) | -01:54 (-59.4%) |
| test-007.mp4 | 30 | 01:16 | 01:04 | 00:41 | -00:12 (-15.8%) | -00:35 (-46.1%) |
| 厚码测试2.mkv | 30 | 01:52 | 01:34 | 00:43 | -00:18 (-16.1%) | -01:09 (-61.6%) |


### Differences:
- GPU only processing (benchmarks TBD). Intial tests show that it can be 2x faster. Raw processing for places without mosaic is ~250fps on RTX 5090
<img width="860" height="56" alt="image" src="https://github.com/user-attachments/assets/a80ecaee-e36d-4c91-93e4-8bdd75048ac3" />

- Improved mosaic detection model.
- Temporal overlap which reduces flickering (beta)
- Accurate color conversions on gpu (input matches output and no banding).
- Only modern Nvidia gpu is supported.
- TensorRT support.
- CLI only

### TODO:
- proper VR support
- TVAI and SeedVR
- Proper stream that can be played in Stash (and maybe others?)
- improve performance (this version is very simple)
 
### Usage
Go to releases page and download last package. Built for windows/linux on cuda 13.0.\
Make sure that ```ffmpeg``` and ```mkvmerge``` is in your path.\
You can download mkvmerge [here](https://mkvtoolnix.download/downloads.html).

**First run might be slow because models will be compiled for your hardware (you can copy .engine files from model_weights to a new version!)**

Remember to have up to date nvidia drivers.

### Max clip + temporal overlap
Main goal for the temporal overlap is to reduce flickering on the edges of clips.\
Initial testings shows that temporal overlap reduces flickering but might sometimes decrease quality of restoration. Feel free to test different values.\
Some guidance from my limited testing:\
- 30 clip size + temporal overlap 3 looks also ok but if you can fit higher clip size then go for it.
- 60 clip size + temporal overlap 4 is fine for most of my test clips
- 180 clip size + temporal overlap 8 looks very good in all my tests.

### Restoration model compilation.
Read [#6](https://github.com/Kruk2/jasna/issues/6) for more details.\
Compiled model takes a lot of vram. Rough estimate is around 2.5GB VRAM per 30 frames in clip size. If you plan to use 180 clip size you have to have 24gb vram+ (180/30 * 2.5).\
You can opt out from compiled model at the cost of performance.\
It's recommended to rather lower clip size and use temporal overlap with compiled model.

### Disclamer
This project is aimed at more technical users.

## Building
Install these libs via ```uv pip install . --no-build-isolation```\
make sure you have cmake and ninja ```uv pip install cmake ninja```\
and cuda 13.0 in your system.

https://codeberg.org/Kruk2/vali

https://codeberg.org/Kruk2/PyNvVideoCodec

Then:
```uv pip install -e .[dev]```
