## Jasna

JAV model restoration tool inspired (and in some places based on) by [Lada](https://codeberg.org/ladaapp/lada).
Restoration model (mosaic_restoration_1.2) used in Jasna was trained by ladaapp (the lada author).

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
- improve performance (this version is very simple)
- proper VR support
- TVAI and SeedVR
- Proper stream that can be played in Stash (and maybe others?)

### Usage
Go to releases page and download last package. Built for windows/linux on cuda 13.0.
Make sure that ```ffmpeg``` and ```mkvmerge``` is in your path.
You can download mkvmerge [here](https://mkvtoolnix.download/downloads.html).

**First run might be slow because models will be compiled for your hardware (you can copy .engine files from model_weights to a new version!)**

Remember to have up to date nvidia drivers.

### Install (editable)

```bash
python -m pip install -e .[dev]
```

### Run

```bash
python -m jasna --input path\to\in.mp4 --output path\to\out.mp4
```

Or, after install:

```bash
jasna --input path\to\in.mp4 --output path\to\out.mp4
```

### Build a self-bundled executable (PyInstaller)

```bash
pyinstaller jasna.spec
```

The output executable will be in `dist/`.
