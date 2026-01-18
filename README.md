## Jasna

JAV model restoration tool inspired by [Lada](https://codeberg.org/ladaapp/lada).
Differences:
- GPU only processing (benchmarks TBD)
- Improved mosaic detection model.
- Accurate color conversions on gpu (input matches output and no banding).
- VR support (TDB).
- Only modern Nvidia gpu is supported.
- TensorRT support.

Consider this release as very alpha.


### Usage
Go to releases page and download last package. Built for windows on cuda 13.0.


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