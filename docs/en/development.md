# Running from Source

This page is for developers. To just use Jasna, download a release package: it
bundles its own Python/Tk runtime, `ffmpeg` and `ffprobe`.

The public checkout runs the GUI, CLI and free **BasicVSR++** without the private
protection module. Activation and encrypted **unet-4x**, **SD 1.5** and **LTX**
models require the official release. LTX trial mode uses random weights and
does not restore videos. A missing private module disables activation; errors
inside an installed module still report a failure.

## Requirements

- Python **3.12+** (releases ship 3.13 on NVIDIA, 3.12 on AMD).
- `ffmpeg` and `ffprobe` **8** on `PATH`.
- libVLC 3 for GUI player audio (Ubuntu: `libvlc5 vlc-plugin-base`).
- Optional, NVIDIA: the CUDA 13 toolkit (`nvcc`), a C++ compiler `nvcc` accepts and
  `ninja` (in the `dev` extra) build the native LTX block on first use (see
  [Benchmarks](#benchmarks)); without them LTX runs its `torch.compile` path.
  On Windows start jasna from an x64 Developer prompt of VS 2022, so MSVC's
  `cl` is on `PATH` (CUDA 13.0 does not accept VS 2026's MSVC).
- Optional, NVIDIA: a `python_vali` wheel built from
  <https://codeberg.org/Kruk2/vali> (needs CUDA 13, `cmake`, `ninja`; on
  Windows also VS Build Tools 2022 with C++). Without that fork the decoder
  falls back to PyAV for every video.

## NVIDIA

Create the venv from your distribution's Python, whose Tk uses Xft. A
standalone Python with a `no-xft` Tk renders every GUI font as bitmap `fixed`.

```bash
uv venv --python /usr/bin/python3.13 --no-managed-python --no-python-downloads .venv
source .venv/bin/activate
uv pip install -e ".[nvidia,dev]" --extra-index-url https://download.pytorch.org/whl/cu130
```

Ubuntu 22.04 has no Python 3.13 package; build one against `tk-dev` and
`libxft-dev`.

On Windows (NVIDIA), run the same `uv pip install` line in a Python 3.13 venv.

Release builders must install the project's required `ultralytics>=8.4.96,<8.5` before
freezing. Older versions fail to load YOLO checkpoints under the release's
restricted loading policy, even when the same checkpoint works in development.
YOLO ONNX export needs neither onnxruntime nor onnxslim (releases bundle
neither).
The NVIDIA extras also install SciPy directly for the LTX worker; the release
bundler copies its Python modules, native extensions and bundled BLAS libraries.

## AMD (ROCm 10.0)

Linux: work inside the `rocm/pytorch` ROCm 10.0 image (torch comes from the
image), then `uv pip install -e ".[amd,dev]"`.

Windows (Python 3.12, Adrenalin 26.8.1 or newer): install torch from AMD's
index **first**, otherwise a later dependency pulls the CPU torch from PyPI:

```powershell
$I = "https://stable.repo.amd.com/rocm/whl-next/"
$D = "device-gfx1100,device-gfx1101,device-gfx1102,device-gfx1103,device-gfx1150,device-gfx1151,device-gfx1200,device-gfx1201"
pip install --index-url $I "torch[$D]==2.12.0+rocm10.0.0" "torchvision[$D]==0.27.0+rocm10.0.0"
uv pip install -e ".[amd,dev]"
```

Check that ROCm stuck (an unsatisfiable torch pin makes uv silently install the
CUDA build, which breaks RF-DETR):

```bash
python -c "import torch, torchvision; assert torch.version.hip and '+rocm' in torchvision.__version__; print('ROCm OK')"
```

A new GPU target needs its matching `device-gfx*` extra above.

AMD runs RF-DETR as the `rfdetr` torch model and decodes/encodes with AMF
(software decode when AMF can't). Secondary restoration is NVIDIA-only.
`--device cuda:N` picks the GPU, but FFmpeg 8's Linux AMF ignores the adapter,
so isolate the GPU at the container or host level on multi-GPU machines.

## Run

```bash
python -m jasna                                    # GUI
python -m jasna --input in.mp4 --output out.mp4    # CLI (see docs/en/cli.md)
```

`JASNA_DECODE_BACKEND=vali|pyav-hw|pyav-sw` forces a decoder; the default
`auto` prefers VALI on NVIDIA.

## Tests

```bash
python -m pytest
```

GUI tests need a screen; on a headless or locked desktop run them under
`xvfb-run -a -s "-screen 0 1920x1080x24"` (the default 640x480 screen cuts
off footer widgets). GPU tests need the matching GPU.

## CUDA kernels

`jasna/media/*.cu` are prebuilt into the committed `.fatbin` files next to
them, so running Jasna needs no CUDA toolkit. After editing a `.cu`, rebuild
with CUDA 13 and `g++-15` (override with `CCBIN=`):

```bash
scripts/build_fatbins.sh        # every kernel
scripts/build_fatbins.sh cas    # just jasna/media/cas.cu
```

Keep the release kernel inventory in sync when adding kernels. Every kernel needs
a Torch equivalent: ROCm uses it, and the tests use it as the reference.

## Benchmarks

Run by the maintainer only, on an otherwise idle GPU. Inputs live in the
gitignored `assets/benchmark/`. Write results to
`benchmarks/<date>_<topic>.{csv,md}` and never edit an old CSV.

The release suite is one H.264 input per resolution plus the 8K VR clip,
median of three runs, with `--max-clip-size 180 --temporal-overlap 15
--secondary-restoration none`. Add `--codecs` (HEVC 10-bit and AV1) when the
change touches decode, encode or pixel formats; H.264 hides decode differences.

```bash
scripts/run_benchmarks.sh benchmarks/scratch [--codecs] [--scan]       # ~8 min
P=~/.virtualenvs/jasna-linux/bin/python
$P scripts/benchmark_decode_backends.py --backends vali pyav-hw pyav-sw   # e2e per decoder
$P scripts/benchmark_scan_backends.py --backends vali                     # GUI mosaic scan
$P scripts/benchmark_releases.py --release v0.9.0=/path/to/jasna --source HEAD=. --workdir WORKDIR/releases --csv out.csv
$P scripts/benchmark_lada_flatpak.py --workdir WORKDIR/lada --csv out.csv   # Lada baseline
$P scripts/benchmark_ltx_denoise.py --model undistilled --csv out.csv    # LTX denoise, trial weights
$P scripts/benchmark_ltx_decode.py --variants shipped,kitchen,eager      # LTX VAE decode
$P scripts/benchmark_ltx_encode.py --variants fp32,eager,shipped        # LTX VAE encode, real weights
$P scripts/benchmark_na3d.py                                             # Triton NA kernel
```

`scripts/bench_memory.py` is the RAM/VRAM sampler the others share.

## LTX native block

- Source-mode LTX runs the public native block (`jasna/ltx/native/`): one
  C++/CUDA call per transformer block, built on first use into the compile
  cache (~1-2 min). Without the toolchain, or on a GPU missing from the kernel
  tables, it uses the `torch.compile`/Triton path.
- `JASNA_LTX_NATIVE=0` forces `torch.compile`, for an A/B in one tree.
- `jasna.ltx.native.prebuild(dir)` builds the extension and the cubins of every
  table architecture for a release; copy `dir` to `jasna/ltx/native/prebuilt/`.
- INT8 self-attention is comfy-kitchen 0.2.38's SageAttention, vendored in
  `jasna/ltx/native/sage_attention/` (bf16, unmasked subset; provenance and
  edits in its `NOTICE`). It compiles with comfy-kitchen's nvcc flags
  (`--use_fast_math` must not reach `glue.cu`) and must stay bit-identical with
  `comfy_kitchen.int8_attention` (`tests/test_ltx_native_gpu.py`). Re-vendor it
  when the torch path's comfy-kitchen pin changes.
- Release packaging keeps the vendored kernels' `NOTICE` in `licenses/`.
- A change to any path must be compared with identical weights, inputs, seeds
  and warmed caches.
- GPU-qualified on SM 12.0 (RTX 5090), SM 8.9 (RTX 4090), SM 8.6 (RTX 3090) and
  SM 7.5 (RTX 2080 Ti). Below SM 12.0 it is INT8 only: plain Triton GEMMs on
  SM 8.x, cuBLAS INT8 GEMMs with the Triton epilogue on SM 7.5. Cubins for
  another SM do not by themselves establish support.

## Release checklist

1. Same version in `pyproject.toml`, `jasna/__init__.py`, the git tag and
   `RELEASE_SOURCES.md`; that file must match the PyAV, VALI, FFmpeg, Python
   and other dependency revisions used by the build.
2. The unpacked package contains `LICENSE`, `LICENSING.md`, `NOTICE`,
   `RELEASE_SOURCES.md`, `assets/THIRD_PARTY_*.md` and `licenses/`.
3. Every bundled model hash matches `assets/THIRD_PARTY_MODELS.md`.
4. Upload every archive part **and** its `.sha256` to the same GitHub release.

Official release packaging is maintained separately.
