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

# AMD Linux (inside a ROCm 7.2 environment; torch/torchvision+rocm come from the
# rocm/pytorch base image, the find-links only backfills any missing rocm wheel)
uv pip install ".[amd]" \
  --find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/
```

**AMD Windows** (ROCm 7.2.1 — Python 3.12, Adrenalin ≥ 26.2.2). Install the ROCm
SDK + torch/torchvision ROCm wheels FIRST, with `--no-deps` so pip cannot silently
replace them with the CPU `torch==2.9.1` from PyPI when a later dependency
(torchvision, rfdetr, …) pulls torch — that swap is the usual cause of a
"non-ROCm torch/torchvision" env:

```powershell
$R = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1"
pip install --no-deps `
  "$R/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl" `
  "$R/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl" `
  "$R/torch-2.9.1+rocm7.2.1-cp312-cp312-win_amd64.whl" `
  "$R/torchvision-0.24.1+rocm7.2.1-cp312-cp312-win_amd64.whl"
# then the remaining AMD deps (torch/torchvision above already satisfy the pins)
uv pip install ".[amd]"
```

The `rocm_sdk_core` / `rocm_sdk_libraries_*` wheels are the ROCm runtime itself.
`import torch` reaches them through `rocm_sdk`, which resolves the package name at
call time, so Nuitka sees no import and bundles nothing — the release build copies
both wheel trees into the dist verbatim and then asserts that every library torch
preloads resolves there. Without that, the frozen app dies at startup with
`UnboundLocalError: cannot access local variable 'py_module'` (upstream
`rocm_sdk.find_libraries` reports an absent payload package that way).

Verify ROCm actually stuck on either OS (the Linux Docker build asserts the same):

```bash
python -c "import torch, torchvision; assert torch.version.hip and '+rocm' in torchvision.__version__; print('ROCm OK', torch.__version__, torchvision.__version__)"
```

`jasna[amd]` pins `torch==2.9.1` as a plain version on purpose: the ROCm wheels
(`2.9.1+rocm7.2.1`) satisfy it, and a `+rocm7.2.1` *local-version* pin can't be
shared across OSes. torchvision is pinned per OS (`0.24.0` on Linux, `0.24.1`
on Windows) because the ROCm channels ship different versions — the Linux
manylinux channel and the `rocm/pytorch` base image only have `0.24.0`, so a
shared `0.24.1` pin would make pip/uv silently install the PyPI CUDA build,
which breaks `torchvision.ops.nms` (and with it RF-DETR) at runtime. The
ROCm-build assertion above is the fail-loud guard on top.

For Nvidia library builds, you also need:

- VS Build Tools 2022 with C++ support.
- CUDA 13.0 installed on the system.
- `cmake` and `ninja`:

```bash
uv pip install cmake ninja
```

Developer setup also requires:

- `ffmpeg` and `ffprobe` on `PATH`; `ffmpeg` major version must be **8**.
- PyAV built from main branch. Currently released 18.0.0 doesn't ship some CUDA context improvements.
- Optional: a `python_vali` wheel built from <https://codeberg.org/Kruk2/vali>. Only that
  fork has `DecodeSingleSurfaceAsyncDetailed` and its corrupt-packet tolerance, which the
  VALI decode backend needs; with the stock PyPI wheel the reader falls back to PyAV for
  every video.

Then install Jasna in editable mode:

```bash
uv pip install -e ".[nvidia,dev]"  # or .[amd,dev]
```

## CUDA kernels

`jasna/media/*.cu` are compiled ahead of time into `.fatbin` files that are
committed alongside them, and loaded at run time through the CUDA driver API
(`jasna/media/cuda_kernel.py`). No CUDA toolkit is needed to *run* Jasna — only
to rebuild a kernel after editing its `.cu`:

```bash
scripts/build_fatbins.sh        # every kernel
scripts/build_fatbins.sh cas    # just jasna/media/cas.cu
```

That script runs, for each `.cu`:

```bash
GENCODE="-gencode arch=compute_75,code=[compute_75,sm_75]"
for arch in 80 86 87 88 89 90 100 103 110 120 121; do
    GENCODE="$GENCODE -gencode arch=compute_$arch,code=sm_$arch"
done
nvcc -ccbin g++-15 -std=c++17 -O3 -fatbin $GENCODE \
    -o jasna/media/cas.fatbin jasna/media/cas.cu
```

`-ccbin g++-15` is needed because CUDA 13 rejects newer host compilers (override
with `CCBIN=`). PTX is embedded for `compute_75` only, so future architectures
still load via JIT. The script prints each fatbin's size and architecture list;
`cuobjdump -lelf jasna/media/cas.fatbin` shows the detail. Add any new fatbin to
`CUDA_KERNEL_FATBINS` in `jasna/protection/keytool/build_nuitka.py` so frozen
builds bundle it.

Every kernel needs a Torch equivalent: ROCm has no fatbin path and falls back to
it, and the unit tests use it as the reference implementation.

## Benchmarks

Run by the maintainer only, on an otherwise idle GPU — anything else on the
card makes the numbers incomparable. The suite is deliberately small: **one
input per resolution plus the 8K VR clip**, all H.264 8-bit.

```bash
scripts/run_benchmarks.sh benchmarks/scratch            # ~8 min
scripts/run_benchmarks.sh benchmarks/scratch --codecs   # + HEVC 10-bit and AV1
scripts/run_benchmarks.sh benchmarks/scratch --scan     # + the GUI mosaic scan
```

It discards a warmup run, then reports the median of three per clip with RAM and
per-process VRAM sampled throughout (`scripts/bench_memory.py`). Fixed settings:
`--max-clip-size 180 --temporal-overlap 15 --secondary-restoration none`.

Why only H.264 by default: across five release steps a HEVC-10-bit or AV1
encoding of the same resolution never disagreed in sign with its H.264 sibling,
because the model sees identical 256² crops whatever the container held. H.264
also has the cheapest decode, so it hides the least of whatever changed
downstream. The other encodings do carry the decode-path signal — H.264 is
nearly flat across decode backends while AV1 and HEVC 10-bit spread ~19 % — so
pass `--codecs` when the change is in decode, encode or pixel-format code.

Write results to `benchmarks/<date>_<topic>.{csv,md}` and keep old CSVs intact —
the README tables are a summary that drops releases where nothing moved, so the
CSVs are the only full record. `scripts/benchmark_releases.py` compares frozen
release archives instead of the working tree, and
`scripts/benchmark_lada_flatpak.py` refreshes the Lada baseline column.

## AMD release builds

These scripts live in the private protection submodule and are for the
maintainer's release environment — they are not available in the public
checkout:

```bash
jasna/protection/keytool/build_linux_amd.sh
jasna/protection/keytool/validate_amd_ssh.sh user@amd-host
python jasna/protection/keytool/build_windows_amd.py
```

The AMD build uses PyTorch/ROCm for BasicVSR++, YOLO and RF-DETR, and AMF for
H.264/HEVC/AV1 decode and encode. RF-DETR runs the trained checkpoint through the
`rfdetr` torch model (`rfdetr==1.8.3` on `transformers==5.1.0`, bundled as
`rfdetr-v6.pt`) — no ONNX Runtime/MIGraphX, so no
per-model engine precompile step. NVIDIA builds keep the ONNX → TensorRT path
(`rfdetr-v6.onnx`). Decode falls back to FFmpeg software decoding when AMF cannot
handle the source. Secondary restoration and segment smart rendering remain
NVIDIA-only.

`--device cuda:N` selects the PyTorch GPU (ROCm reuses the CUDA device API).
FFmpeg 8's Linux AMF device context currently ignores its adapter
argument, so AMF decode/encode can use the default Vulkan adapter on a multi-GPU
AMD host. Isolate the target GPU at the container/host level when deterministic
AMF adapter selection matters.
