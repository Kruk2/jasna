# Running from Source

For the Linux/Windows integration, start with the [English feature review index](feature_reviews.md)
or its [Chinese counterpart](../zh/feature_reviews.md). Each feature has bilingual usage,
dependency, and acceptance documentation; English is the default entry point.

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

# AMD Linux (inside the rocm/pytorch ROCm 10.0 image; torch/torchvision+rocm come
# from the image, see jasna/protection/keytool/Dockerfile.amd)
uv pip install ".[amd]"
```

**AMD Windows** (ROCm 10.0.0 — Python 3.12, Adrenalin ≥ 26.8.1). Install torch and
torchvision from AMD's ROCm index FIRST, with the GPU kernel packs the release
ships, so pip cannot silently replace them with the CPU `torch==2.12.0` from PyPI
when a later dependency (rfdetr, …) pulls torch — that swap is the usual cause of
a "non-ROCm torch/torchvision" env:

```powershell
$I = "https://stable.repo.amd.com/rocm/whl-next/"
$D = "device-gfx1100,device-gfx1101,device-gfx1102,device-gfx1103,device-gfx1150,device-gfx1151,device-gfx1200,device-gfx1201"
pip install --index-url $I "torch[$D]==2.12.0+rocm10.0.0" "torchvision[$D]==0.27.0+rocm10.0.0"
# then the remaining AMD deps (torch/torchvision above already satisfy the pins)
uv pip install ".[amd]"
```

Adding a GPU target means adding its `device-gfx*` extra here and keeping it in
the `Dockerfile.amd` keep-list, so both OSes ship the same targets.

The `rocm_sdk_core` / `rocm_sdk_libraries` / `rocm_sdk_device_gfx*` wheels are the
ROCm runtime itself, and torch's own GPU kernels sit in `torch/.kpack`.
`import torch` reaches the runtime through `rocm_sdk`, which resolves the package
name at call time, so Nuitka sees no import and bundles nothing — both AMD release
builds copy the wheel trees into the dist verbatim and then assert that every
library torch preloads resolves there. Without that, the frozen app dies at startup
with `UnboundLocalError: cannot access local variable 'py_module'` (upstream
`rocm_sdk.find_libraries` reports an absent payload package that way).

Verify ROCm actually stuck on either OS (the Linux Docker build asserts the same):

```bash
python -c "import torch, torchvision; assert torch.version.hip and '+rocm' in torchvision.__version__; print('ROCm OK', torch.__version__, torchvision.__version__)"
```

`jasna[amd]` pins `torch==2.12.0` and `torchvision==0.27.0` as plain versions, which
the ROCm wheels (`2.12.0+rocm10.0.0`, `0.27.0+rocm10.0.0`) satisfy on both OSes. A
pin no ROCm wheel satisfies makes pip/uv silently install the PyPI CUDA build,
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
- libVLC 3 installed for GUI player audio. The Python binding is installed
  from `pyproject.toml`; on Ubuntu install `libvlc5` and `vlc-plugin-base`.
- Optional: a `python_vali` wheel built from <https://codeberg.org/Kruk2/vali>. Only that
  fork has `DecodeSingleSurfaceAsyncDetailed` and its corrupt-packet tolerance, which the
  VALI decode backend needs; with the stock PyPI wheel the reader falls back to PyAV for
  every video.

Set `JASNA_DECODE_BACKEND` to `vali`, `pyav-hw`, or `pyav-sw` to force a
decoder backend. The default is `auto`, which prefers VALI on NVIDIA and falls
back to PyAV.

Then install Jasna in editable mode:

```bash
uv pip install -e ".[nvidia,dev]"  # or .[amd,dev]
```

## CUDA and Linux AMD HIP kernels

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

The two colour-conversion sources also have ahead-of-time AMD code objects for
the validated `gfx1100` target. Linux artifacts are rebuilt with:

```bash
scripts/build_hip_code_objects.sh                 # both colour kernels, gfx1100
scripts/build_hip_code_objects.sh gfx1100 yuv_to_rgb
```

The resulting `jasna/media/{yuv_to_rgb,rgb_to_yuv}.gfx1100.hsaco` files are
loaded through the HIP module API. A ROCm compiler is required only to rebuild
them. On Linux AMD `gfx1100`, Jasna automatically uses the pair when both files
are installed; `JASNA_AMD_HIP_COLOR_KERNELS=0` disables the route, while `=1`
requests it explicitly and fails closed on an unsupported or incomplete
installation. Other AMD architectures retain the Torch conversion path.

Windows artifacts must be rebuilt offline with the same HIP SDK used by the
installed ROCm PyTorch wheel:

```powershell
.\scripts\build_hip_code_objects_windows.ps1 `
  -Architecture gfx1100 `
  -Hipcc C:\path\to\matching\HIP\bin\hipcc.exe `
  -Python C:\path\to\rocm-python.exe
```

This writes two `.gfx1100.windows.co` files and
`hip_color_kernels.gfx1100.windows.json`. The manifest pins the source and
artifact hashes, architecture, parameter/code-object ABI, exact Torch HIP
version, HIP runtime API, and the SHA-256 of the HIP DLL already loaded by
PyTorch. Product startup never invokes `hipcc` or JIT compilation. Windows
selection is deliberately explicit-only with `JASNA_AMD_HIP_COLOR_KERNELS=1`;
unset/`auto` retains the Torch eager route and any mismatch fails closed.

Every kernel still needs a Torch equivalent. It is the fallback for targets
without an accepted AOT object and the reference implementation for parity
tests. Frozen Linux AMD builds must copy both `.hsaco` files next to the
executable. Frozen Windows builds must instead copy the two `.windows.co` files
and their Windows manifest next to the executable. Rebuild-only `.cu` sources
are not shipped in either frozen product. These locations match
`jasna.media.hip_kernel.code_object_path()`.

## Linux AMD BasicVSR++ B1 MIGraphX artifacts

Linux AMD `gfx1100` FP16 builds can automatically replace only the repeated
`i > 0` body of the four BasicVSR++ propagation directions with four static-B1
Torch-MIGraphX GraphModules. Model loading, optical flow, first-frame
propagation, reconstruction, and the rest of the pipeline stay on PyTorch.

Install the accepted artifact set next to the restoration checkpoint:

```text
model_weights/
├── lada_mosaic_restoration_model_generic_v1.2.pth
└── basicvsrpp-b1-migraphx-gfx1100/
    ├── B1_COLD_MANIFEST.json
    ├── B1_COLD_MANIFEST.sha256
    ├── b1_backward_1.torch
    ├── b1_forward_1.torch
    ├── b1_backward_2.torch
    ├── b1_forward_2.torch
    └── _torch_migraphx*.so
```

The loader validates the manifest and its digest, checkpoint and semantic-source
digests, Torch/ROCm/MIGraphX/Torch-MIGraphX versions, GPU identity, every static
tensor ABI, and the expected 28-node/three-partition graph topology. It rejects
artifacts that need loader-side compilation and immediately clones reusable
artifact outputs. When RF-DETR initialized Torch-MIGraphX first, the loader
reuses its already-loaded native extension if its SHA-256 matches the manifest.
Equal-content copies in the artifact directory and Torch extension cache may
have different paths; different content still fails closed.
`JASNA_BASICVSRPP_MIGRAPHX_B1=0` disables automatic selection;
`=1` explicitly requires the route and fails closed. Use
`JASNA_BASICVSRPP_MIGRAPHX_B1_DIR` only to point at an equivalent validated
artifact directory.

Compiled model artifacts and native Torch-MIGraphX extensions are release
assets, not Git source artifacts. Frozen Linux AMD packaging must include the
complete directory plus its Python/native runtime dependencies. The public
checkout does not contain the private `jasna/protection/keytool/build_nuitka.py`
asset list, so maintainers must update that private list before producing a
release. See `docs/LINUX_AMD_HIP_MIGRAPHX_OPTIMIZATION_CN.md` for the accepted
real-video measurements and reproduction boundaries.

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

## Release licensing checklist

Before publishing any platform archive:

1. Set the same version in `pyproject.toml`, `jasna/__init__.py`, the git tag,
   and `RELEASE_SOURCES.md`.
2. Confirm `RELEASE_SOURCES.md` matches the PyAV, VALI, FFmpeg, Python, and
   protection revisions used by the build.
3. Verify the unpacked package contains `LICENSE`, `LICENSING.md`, `NOTICE`,
   `RELEASE_SOURCES.md`, `assets/THIRD_PARTY_*.md`, and `licenses/`.
4. Recalculate every bundled model hash and compare it with
   `assets/THIRD_PARTY_MODELS.md`.
5. Upload every archive part **and** its generated `.sha256` file to the same
   GitHub release.

The release builder copies maintained license texts plus the license files
from all installed Python distributions. The public VALI fork is the source
for Jasna's modified `python_vali` wheel.

## AMD release builds

These scripts live in the private protection submodule and are for the
maintainer's release environment — they are not available in the public
checkout:

```bash
jasna/protection/keytool/build_linux_amd.sh
jasna/protection/keytool/validate_amd_ssh.sh user@amd-host
python jasna/protection/keytool/build_windows_amd.py
```

AMD builds use PyTorch/ROCm for their general model path and AMF for supported
H.264/HEVC/AV1 decode and encode. Windows AMD runs RF-DETR and BasicVSR++ in
PyTorch. Linux AMD `gfx1100` can instead auto-select the installed, strictly
versioned RF-DETR and BasicVSR++ B1 MIGraphX artifacts; when no accepted
artifact is installed, the corresponding PyTorch path remains available.
RF-DETR's source model is `rfdetr-v6.pt` (`rfdetr==1.8.3` on
`transformers==5.1.0`). NVIDIA builds keep the ONNX → TensorRT path
(`rfdetr-v6.onnx`). Decode falls back to FFmpeg software decoding when the
platform's product capability gate allows it and AMF cannot handle the source.
Secondary restoration remains NVIDIA-only; see the segment documentation for
current Smart Render platform/codec gates.

`--device cuda:N` selects the PyTorch GPU (ROCm reuses the CUDA device API).
FFmpeg 8's Linux AMF device context currently ignores its adapter
argument, so AMF decode/encode can use the default Vulkan adapter on a multi-GPU
AMD host. Isolate the target GPU at the container/host level when deterministic
AMF adapter selection matters.

## Latest-main local review stack (2026-10-03)

The local 26-feature review stack is based on upstream main
`81dc8b053fb317c063390daab1dab8289c2094df`, not the historical 0.10 tag.
The first 22 exact feature commits are retained; three Windows additions and a
refreshed documentation feature follow them. Each review must declare its exact
base and dependencies; later cumulative prefixes are not independent diffs
against upstream main. No branches or pull requests have been published.

Shared job/session/scan/render/output orchestration stays shared. Native media,
model and runtime backends retain vendor/platform capability gates. The new
Windows RF-DETR Math SDPA policy matches the verified Torch/HIP/DLL/gfx1100
identity before model construction and does not change Linux, CPU or NVIDIA
backends, precision selection, or explicit LTX attention contexts. A typed
Windows AMF host-transfer failure quarantines the queue and requires a new
process; this prevents reuse of an unsafe context, not a claim of fixing TDR.
The explicit synthetic precision probe is not a quality certificate.

The accepted Linux ROCm 10 collection was promoted to the desktop GUI with user
authorization. This review reorganization does not switch that launcher or
change deployed processing code. Windows SDK native asset rebuild, whole-card
telemetry and real-hardware A/B were waived by the user for this round:
`WAIVED_BY_USER_NOT_RUN`, never PASS. Existing binary identity gates and opt-in
defaults remain strict. New Windows/NVIDIA native paths and paid-model AMD
compatibility are not certified by CPU regression or historical evidence.
See `MAIN_INTEGRATION_ROCM10_20261001_CN.md` and
`WINDOWS_ROCM10_COMPATIBILITY_CN.md` for scope and evidence boundaries.
