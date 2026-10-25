# Running from Source

This page is for developers. To just use Jasna, download a release package: it
bundles its own Python/Tk runtime, `ffmpeg` and `ffprobe`.

The public checkout runs the GUI, CLI and free **BasicVSR++** without the private
protection module. Activation and encrypted **unet-4x**, **SD 1.5** and **LTX**
models require the official release. LTX trial mode uses random weights and
does not restore videos. A missing private module disables activation; errors
inside an installed module still report a failure.

## LTX model versions

Each app release pins its model branch and expected file hashes in
`jasna/ltx/model_files.py`. `_MODEL_BRANCHES` lists published model versions;
the app selects the highest registered version no newer than its own version,
or `main` if none matches. Both Hugging Face and its mirror use that same branch.
Jasna 0.12.0 uses `0.12.0`; later app versions reuse it until a newer compatible
model branch is registered. Released 0.11.3 and older keep their original `main`
URLs. Branch selection is local; startup does not query the Hub for new branches.

When publishing new model files, create the version branch, verify its encrypted
checkpoints and public signatures, then register it with the matching sizes and
hashes before building the app. Retain earlier branch contents for their shipped
releases. The existing AMD beta archives still use `main/block-layout3`; keep
that compatibility path available. This release mapping is shared by NVIDIA and AMD.

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

## AMD (ROCm 10.1)

Linux: work inside `rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.14.0`.
This image installs Python packages system-wide. Inside the container, run:

```bash
python3 -m pip install --break-system-packages uv
uv pip install --system --break-system-packages -e ".[amd,dev]"
```

Windows (Python 3.12, Adrenalin 26.8.1 or newer): install the matching
PyTorch builds and GPU kernel packs from AMD's index:

```powershell
$I = "https://stable.repo.amd.com/rocm/whl-next/"
$D = "device-gfx1100,device-gfx1101,device-gfx1102,device-gfx1103,device-gfx1150,device-gfx1151,device-gfx1200,device-gfx1201"
pip install --index-url $I "torch[$D]==2.14.0+rocm10.1.0" "torchvision[$D]==0.29.0a0+rocm10.1.0"
uv pip install -e ".[amd,dev]"
```

AMD's matching torchvision 0.29 wheel is labeled `0.29.0a0` in its stable index.
Keep the complete ROCm version suffix so another vendor's wheel cannot satisfy the pin.
Check the installed builds:

```bash
python -c "import torch, torchvision; assert torch.version.hip and torch.__version__ == '2.14.0+rocm10.1.0'; assert torchvision.__version__ == '0.29.0a0+rocm10.1.0'; print('ROCm OK')"
```

To compile native LTX code on Windows, open an x64 Visual Studio developer
shell and initialize the matching development wheel. Its HIP headers require
the SDK's `clang-cl`; MSVC `cl` cannot compile them:

```powershell
pip install --no-deps --index-url $I rocm-sdk-devel==10.1.0
python -m rocm_sdk init --quiet
$R = (python -m rocm_sdk path --root).Trim()
$env:ROCM_HOME = $R
$env:HIP_PATH = $R
$env:HIP_DEVICE_LIB_PATH = "$R\lib\llvm\amdgcn\bitcode"
$env:CXX = "$R\lib\llvm\bin\clang-cl.exe"
```

A new GPU target needs its matching `device-gfx*` extra above.

AMD runs RF-DETR as the `rfdetr` torch model and decodes/encodes with AMF
(software decode when AMF can't). Secondary restoration is NVIDIA-only.
The model runs rfdetr's export forward with Jasna's own transformer and
deformable-attention forwards (`mosaic/rfdetr_sync_free.py`), which never wait
for the GPU; recheck them against rfdetr's code when bumping the `rfdetr` pin.
`--device cuda:N` picks the GPU, but FFmpeg 8's Linux AMF ignores the adapter,
so isolate the GPU at the container or host level on multi-GPU machines.

`accelerator.configure_rocm_process_env` sets MIOpen defaults at start-up; a
variable already in the environment wins. `MIOPEN_FIND_MODE=HYBRID` keeps the
conv benchmarks across runs, and `MIOPEN_DEBUG_CONV_GEMM=0` keeps the first
benchmark of a new shape from trying the im2col/GEMM solvers, whose workspace
(17 GiB for the LTX VAE encoder at 768) runs a 16 GB card out of memory.

AMF encodes nothing smaller than 96x38 (H.264), 384x128 (HEVC) or 320x128 (AV1) on an R9700;
smaller frames fail or hang, so `encoder_settings.require_encodable_size` rejects such
videos before the job. `av1_amf` pads the width to a multiple of 8 (854 → 856); that is left as is.

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

The suite passes on NVIDIA and AMD hosts and in a public checkout without the
protection submodule. A test of NVIDIA-only behavior that mocks the GPU takes the
`nvidia_vendor` fixture (`tests/conftest.py`); tests that need TensorRT, NVDEC,
NVFP4 or the protection source skip when they are missing.

On an idle AMD GPU, run the driver memory cleanup check explicitly:

```bash
JASNA_TEST_AMD_VRAM_CLEANUP=1 python -m pytest -q -s tests/test_amd_vram_cleanup_gpu.py
```

It fills most VRAM with a synthetic marker, kills that process, and scans new
HIP allocations in a fresh process. A pass applies only to the tested driver and GPU.

## CUDA kernels

`jasna/media/*.cu` are prebuilt into committed binaries next to them, so
running Jasna needs no GPU compiler: a `.fatbin` for NVIDIA and a `.hipfb`
offload bundle for AMD (same source, loaded through the HIP module API on Linux
and Windows). After editing a `.cu`, rebuild both. The fatbin needs CUDA 13 and
`g++-15` (override with `CCBIN=`); the bundle needs ROCm's `hipcc` (override with
`HIPCC=`), e.g. on the AMD test machine:

```bash
scripts/build_fatbins.sh        # every kernel
scripts/build_fatbins.sh cas    # just jasna/media/cas.cu
scripts/build_hipfbs.sh         # every kernel, all AMD targets
scripts/build_hipfbs.sh cas
```

The AMD target list in `build_hipfbs.sh` must match
`cuda_kernel.HIP_KERNEL_ARCHITECTURES` (`tests/test_cuda_kernel.py` checks the
committed bundles). Keep the release kernel inventory in sync when adding kernels.
Every
kernel needs a Torch equivalent: AMD GPUs outside the bundle and CPU use it, and
the tests use it as the reference. The GPU parity tests run on whichever vendor
the machine has.

Two AMD paths run Triton kernels compiled ahead of time, since AMD ships
Triton for Linux only: the BasicVSR++ propagation step
(`restorer/basicvsrpp_fused_body.py`) and the RF-DETR segmentation-head blocks
(`mosaic/rfdetr_fused_head.py`). `scripts/build_basicvsrpp_hipfbs.py` and
`scripts/build_rfdetr_hipfbs.py` hold their source; both build through
`scripts/triton_hipfb.py` on a Linux ROCm machine with the ROCm Torch + Triton
venv into `jasna/media/basicvsrpp_*.hipfb` / `rfdetr_dwblock_*.hipfb` (RDNA3,
RDNA3.5, RDNA4) plus the launch tables `jasna/restorer/basicvsrpp_hip_launch.py`
/ `jasna/mosaic/rfdetr_hip_launch.py`; commit all of them. Shapes and tiles
are fixed (RF-DETR: 144² and 192² mask features, 256 channels); `--tune` times
tile candidates on the local GPU (current tiles were picked on a gfx1201).
Other GPUs and fp32 run the torch modules, which are the reference for the
parity tests in `tests/test_basicvsrpp_fused_body.py` and
`tests/test_rfdetr_fused_head.py`. The bundles are AMD-only
(`AMD_GPU_KERNEL_MODULES` in `build_nuitka.py`).

```bash
python scripts/build_basicvsrpp_hipfbs.py          # ~1 min, all targets
python scripts/build_rfdetr_hipfbs.py              # ~30 s, all targets
python scripts/build_basicvsrpp_hipfbs.py --tune   # needs an AMD GPU (same for rfdetr)
```

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
- `jasna.ltx.native.prebuild(dir)` builds the extension and the kernels of every
  table architecture of the build's vendor for a release (cubins per SM; under
  ROCm `.hsaco` per gfx name, gfx1100/1101/1102/1200/1201, as gfx1103 has no
  tuned tiles); copy `dir` to `jasna/ltx/native/prebuilt/`. Shipped kernels load
  without Triton.
- Launch constants and tile tables live in `jasna/ltx/kernel_policy.py`, which
  imports no Triton; `kernels.py` and the native block both read it.
- INT8 self-attention is comfy-kitchen 0.2.38's SageAttention, vendored in
  `jasna/ltx/native/sage_attention/` (bf16, unmasked subset; provenance and
  edits in its `NOTICE`). It compiles with comfy-kitchen's nvcc flags
  (`--use_fast_math` must not reach `glue.cu`) and must stay bit-identical with
  `comfy_kitchen.int8_attention` (`tests/test_ltx_native_gpu.py`). Re-vendor it
  when the torch path's comfy-kitchen pin changes.
- AMD (Linux ROCm with Triton) builds the same sources: `gpu_runtime.h` maps
  the CUDA spellings, hipcc compiles `glue.cu` and comfy-kitchen 0.2.38's HIP
  INT8 attention (`sage_attention_hip/`, bit-identical with
  `comfy_kitchen.int8_attention` there) into objects, and the host compiler
  builds `block.cpp` (nothing is hipified). INT8 only, AMD tiles by gfx name,
  INT8 cross-attention and the gated-quantize epilogue as the AMD torch path.
  The q/k norm and RoPE run inside comfy-kitchen's Q/K quantization
  (`norm_rope_quant.hip`, compiled with its `quant_qk_int8.hip`); its codes must
  stay equal to `glue.cu`'s `norm_rope` followed by the vendored quantizer.
  `PYTORCH_ROCM_ARCH` picks the gfx targets (default: this GPU). Without
  Triton (Windows ROCm) the block needs shipped kernels; checked on Linux with
  Triton blocked. The Windows extension build (MSVC + HIP SDK flags) is
  untested.
- Release packaging keeps the vendored kernels' `NOTICE` in `licenses/`.
- A change to any path must be compared with identical weights, inputs, seeds
  and warmed caches.
- GPU-qualified on SM 12.0 (RTX 5090), SM 8.9 (RTX 4090), SM 8.6 (RTX 3090),
  SM 7.5 (RTX 2080 Ti) and gfx1201 (Radeon AI PRO R9700). Below SM 12.0 it is INT8 only: plain Triton GEMMs on
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
