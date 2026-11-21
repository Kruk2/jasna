# Windows on Arm (ARM64)

Jasna runs from source on Windows 11 on Arm. The GPU path is verified on
NVIDIA GB10 laptops (RTX Spark class, compute capability `sm_121`), using
NVIDIA's native `win_arm64` CUDA wheels — no emulation. Snapdragon X devices
have no CUDA path; they are not tested.

The official installers are x86_64 only. On ARM64, install from source.

## Prerequisites

- Windows 11 on ARM64.
- [uv](https://docs.astral.sh/uv/) (it has ARM64 Windows builds). python.org
  ships no ARM64 installers, so get Python through uv:

  ```powershell
  uv python install 3.13
  ```

  Use Python 3.13: the CUDA wheels pinned below are built for cp313.
- A native ARM64 build of FFmpeg 8 on `PATH`, for example the `winarm64`
  builds from [BtbN's FFmpeg builds](https://github.com/BtbN/FFmpeg-Builds/releases)
  (verified with `master-latest-winarm64-gpl`). If FFmpeg is missing, Jasna
  downloads an x86_64 build that runs under emulation; native is faster.

## Install

```powershell
git clone https://github.com/Kruk2/jasna.git
cd jasna
uv venv --python 3.13 .venv
.venv\Scripts\activate
```

Do **not** use `.[nvidia]` on ARM64: `torch` for `win_arm64` is not on PyPI
(it is on NVIDIA's wheel index, see below) and `tensorrt`, `torch-tensorrt`
and `python_vali` have no `win_arm64` wheels at all. Install like this:

```powershell
uv pip install torch==2.14.0+cu134 torchvision==0.29.0+cu134 --extra-index-url https://pypi.nvidia.com/nvtorch_oot
"opencv-python==5.1.0.95" | Out-File -Encoding ascii override.txt
uv pip install -e . --extra-index-url https://pypi.nvidia.com/nvtorch_oot --override override.txt
uv pip install triton-windows==3.8.0.post29
```

Two pins need the overrides, both unrelated to ARM64 behavior itself:

- `opencv-python<5`: the first release with a `win_arm64` wheel is 5.1.0.95,
  so `uv pip install -e .` fails to resolve on ARM64 without the override.
- `triton-windows==3.7.1.post27` ships `win_amd64` only; 3.8.0.post29 has
  `win_arm64` wheels.

`torch-tensorrt`, `python_vali` and `tensorrt` stay uninstalled — nothing in
the default pipeline needs them on this platform (see the table; FFmpeg/PyAV
is the decode fallback).

## Check

```powershell
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

On an RTX Spark laptop this prints something like
`NVIDIA RTX Spark N1X (6144-core Blackwell RTX GPU) (12, 1)` — native ARM64
Python talking to the GPU directly (`sm_121`).

## What works on GB10 today

| Area | Status | Notes |
| --- | --- | --- |
| PyTorch CUDA | works | native `win_arm64` wheels from [NVIDIA's index](https://pypi.nvidia.com/nvtorch_oot/), `sm_121` |
| Restoration inference (torch path, e.g. BasicVSR++ checkpoints) | works | model forward passes verified on GB10 hardware |
| Mosaic detection (RF-DETR / YOLO) | does not work yet | the NVIDIA path builds TensorRT engines; `tensorrt==10.16.1.11` has no `win_arm64` wheel, and engine builds with the 11.x ARM64 wheels fail |
| Output encoding | does not work yet | PyAV `win_arm64` wheels ship without NVENC (`hevc_nvenc` missing in `av==18.1.0`) and there is no CPU encoder fallback |
| LTX | not verified | `comfy-kitchen==0.2.38` does ship a `win_arm64` wheel; the Triton kernels have not been exercised on `sm_121` |

The detection and encoder gaps are being worked on as separate small PRs,
tracked in [#641](https://github.com/Kruk2/jasna/issues/641).
