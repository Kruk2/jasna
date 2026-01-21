## Jasna

JAV model restoration tool inspired by [Lada](https://codeberg.org/ladaapp/lada).
Restoration model (mosaic_restoration_1.2) used in Jasna was trained by ladaapp (the lada author).
Differences:
- GPU only processing (benchmarks TBD). Intial test for 10min: lada: 4min 12s and jasna: 3min 5s
- Improved mosaic detection model.
- Accurate color conversions on gpu (input matches output and no banding).
- Only modern Nvidia gpu is supported.
- TensorRT support.
- CLI only

Consider this release as very alpha.

TODO:
- improve blending
- improve performance (this version is very simple
- proper VR support
- TVAI and SeedVR
- Proper stream that can be played in Stash (and maybe others?)


### Usage
Go to releases page and download last package. Built for windows on cuda 13.0.
Make sure that ```ffmpeg``` and ```mkvmerge``` is in your path.
You can download mkvmerge [here](https://mkvtoolnix.download/downloads.html).

**First run might be slow because models will be compiled for your hardware (you can copy .engine files from models_directory to a new version!)**

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
