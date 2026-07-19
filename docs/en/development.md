# Running from Source

This page is for developers. If you just want to use Jasna, download a
release package instead — it bundles everything, including its own
Python/Tk runtime, `ffmpeg`, and `ffprobe`.

Python requirement from `pyproject.toml`: **Python 3.12 or newer** (the
examples below use 3.13, which is what release builds ship).

On Linux, create the venv from a distribution-provided Python whose matching Tk package
uses Xft/fontconfig. Avoid a downloaded standalone Python that reports a `no-xft` Tk build;
it reduces all GUI text and CustomTkinter shapes to the legacy bitmap `fixed` font. For
example, when `/usr/bin/python3.13` is supplied by your distribution:

```bash
uv venv --python /usr/bin/python3.13 --no-managed-python --no-python-downloads .venv
source .venv/bin/activate
python -c "import tkinter; root = tkinter.Tk(); print(root.tk.call('info', 'patchlevel')); root.destroy()"
```

Ubuntu 22.04 does not provide Python 3.13 in its base repositories, so source development
there needs a separately installed or source-built Python 3.13 linked to the system `tk-dev`
and `libxft-dev`. This does not affect the prebuilt Linux release, which bundles its own
compatible Python/Tk runtime.

The public source checkout does not include the protection module. Running from source is fine for development and free models, but supporter-only models such as **unet-4x** and **SD 1.5 image restoration** will not be available from a plain source checkout.

Install runtime dependencies for the active vendor:

```bash
# NVIDIA (CUDA 13 wheels)
uv pip install ".[nvidia]" --extra-index-url https://download.pytorch.org/whl/cu130

# AMD Linux (inside a ROCm 7.2 environment)
uv pip install ".[amd]" \
  --find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/

# AMD Windows
uv pip install ".[amd]"
```

For Nvidia library builds, you also need:

- VS Build Tools 2022 with C++ support.
- CUDA 13.0 installed on the system.
- `cmake` and `ninja`:

```bash
uv pip install cmake ninja
```

Developer setup also requires:

- `ffmpeg` and `ffprobe` on `PATH`; `ffmpeg` major version must be **8**.
- Until PyAV 18.1.0 is published, a PyAV wheel built from upstream main commit `61e4aa8`.
  This contains the merged CUDA-current-context API used by Jasna; switch back to the PyPI
  wheel once 18.1.0 is released.

Then install Jasna in editable mode:

```bash
uv pip install -e ".[nvidia,dev]"  # or .[amd,dev]
```

## AMD release builds

These scripts live in the private protection submodule and are for the
maintainer's release environment — they are not available in the public
checkout:

```bash
jasna/protection/keytool/build_linux_amd.sh
jasna/protection/keytool/validate_amd_ssh.sh user@amd-host
python jasna/protection/keytool/build_windows_amd.py
```

The AMD build uses PyTorch/ROCm for BasicVSR++ and YOLO, ONNX Runtime for RF-DETR,
and AMF for H.264/HEVC/AV1 decode and encode. RF-DETR uses MIGraphX on Linux and
falls back to ONNX Runtime CPU inference on Windows. Decode falls back to FFmpeg
software decoding when AMF cannot handle the source. Secondary restoration and
segment smart rendering remain NVIDIA-only.

`--device cuda:N` selects the PyTorch GPU and, on Linux, the MIGraphX GPU.
FFmpeg 8's Linux AMF device context currently ignores its adapter
argument, so AMF decode/encode can use the default Vulkan adapter on a multi-GPU
AMD host. Isolate the target GPU at the container/host level when deterministic
AMF adapter selection matters.
