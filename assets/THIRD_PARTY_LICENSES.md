# Third-party runtime notices

Official packages include the full texts named below in `licenses/`.
License files supplied by installed Python distributions are copied into
`licenses/python-packages/` during the release build. Exact custom source
revisions and build configurations are in `RELEASE_SOURCES.md`.

## Bundled native components

| Component | License | Source |
| --- | --- | --- |
| FFmpeg libraries in the PyAV and VALI wheels, including vendored x264 | GPLv3 for the combined payload; FFmpeg reports LGPLv3-or-later | [FFmpeg 8.1.2](https://github.com/FFmpeg/FFmpeg/tree/n8.1.2), built by [pyav-ffmpeg 8.1.2-1](https://github.com/PyAV-Org/pyav-ffmpeg/releases/tag/8.1.2-1) |
| Bundled `ffmpeg` and `ffprobe` tools | GPLv3 | [FFmpeg commit `9b6c8969e0`](https://github.com/FFmpeg/FFmpeg/commit/9b6c8969e05b4f0b29f0f85cd501be6b3e582e6b), built by [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds) |
| VALI / python_vali | Apache-2.0 | [Kruk2/vali](https://codeberg.org/Kruk2/vali) |
| libVLC 3 and plugins | LGPL-2.1-or-later | [VideoLAN VLC](https://www.videolan.org/vlc/) |
| CPython 3.13 | Python-2.0 | [CPython](https://github.com/python/cpython) |

The command-line build explicitly enables GPLv3. The shared wheel payload
vendors GPL-licensed x264 alongside FFmpeg libraries that report
LGPLv3-or-later, so Jasna applies GPLv3 to the combined payload. Exact source
commits, dependency versions, archive hashes, and build flags are in
`RELEASE_SOURCES.md` and the linked pyav-ffmpeg tag.

## Key Python components

| Component | License | Source |
| --- | --- | --- |
| PyAV | BSD-3-Clause | [PyAV](https://github.com/PyAV-Org/PyAV) |
| python-vlc | LGPL-2.1-or-later | [python-vlc](https://github.com/oaubert/python-vlc) |
| PyTorch and torchvision | BSD-3-Clause | [PyTorch](https://github.com/pytorch/pytorch), [torchvision](https://github.com/pytorch/vision) |
| RF-DETR | Apache-2.0 | [roboflow/rf-detr](https://github.com/roboflow/rf-detr) |
| MMagic and MMEngine | Apache-2.0 | [OpenMMLab MMagic](https://github.com/open-mmlab/mmagic), [MMEngine](https://github.com/open-mmlab/mmengine) |
| diffusers, accelerate, transformers, huggingface-hub | Apache-2.0 | [Hugging Face](https://github.com/huggingface) |
| OpenCV | Apache-2.0 | [OpenCV](https://github.com/opencv/opencv) |
| Ultralytics | AGPL-3.0 | [ultralytics](https://github.com/ultralytics/ultralytics) |

This table highlights the components most directly used or patched by Jasna;
the package-specific license directory in each binary release is the complete
notice set generated from that release environment.

## Vendored source

Jasna's `jasna/models/basicvsrpp/mmagic/` directory is an inference-only
subset derived from OpenMMLab MMagic. Its existing copyright headers and
Apache-2.0 terms are retained.

Jasna also contains code derived from
[Lada](https://codeberg.org/ladaapp/lada), licensed under AGPL-3.0. Jasna's
root `LICENSE` contains the complete AGPL-3.0 text.

## Proprietary vendor components

NVIDIA TensorRT, CUDA runtime libraries, RTX Video Effects, and AMD ROCm
runtime libraries retain their vendor terms. Their wheel/package license files
are copied into `licenses/python-packages/` when present. These components
are not relicensed by Jasna.
