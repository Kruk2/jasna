## jasna

Package name: `jasna`  
Version: `0.1`

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