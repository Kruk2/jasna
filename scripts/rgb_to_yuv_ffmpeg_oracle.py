"""Conformance oracle: validate RGB->NV12/P010 against FFmpeg's swscale.

Pipes a deterministic RGB field through real ``rgb24 -> nv12/p010le`` scaling at
every colour space and range we support, and compares it to both jasna
implementations: the Torch reference in ``rgb_to_nv12.py`` / ``rgb_to_p010.py``
and the fused CUDA kernel in ``rgb_to_yuv.cu``. This is the forward counterpart
to the swscale check on the reverse direction in ``tests/test_yuv_to_rgb.py``.

Three sources of disagreement are expected and reported separately:

* swscale's chroma downsample is not a plain 2x2 box average, so chroma differs
  wherever the image has detail. The field therefore holds flat blocks, where
  any filter agrees, and detailed blocks, which are reported but not asserted.
* swscale rounds luma with its own fixed-point pipeline, worth a code or two.
* The kernel sums in a different order than the Torch GEMM, worth one code at
  exact .5 ties.

Requires CUDA and an ffmpeg with rawvideo. Run:
  ~/.virtualenvs/jasna-linux/bin/python scripts/rgb_to_yuv_ffmpeg_oracle.py
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import torch

from jasna.media.rgb_to_yuv import _TORCH_CONVERTERS, RgbToYuvConverter

WIDTH = 128
HEIGHT = 192
FLAT_ROWS = 96
BAND_ROWS = 8  # 12 flat colours; 4 chroma rows each, of which 2 are interior

# swscale's luma rounding differs from ours by a code or so; a larger gap means
# the coefficients themselves are wrong. This caught a real error while the
# oracle was being written: naming out_primaries/out_transfer made swscale
# convert colour for real, which showed up here as luma 43 on bt2020 alone.
MAX_LUMA_DIFF = 2
# swscale derives chroma through a half-resolution fixed-point path that carries
# a small systematic offset against an exact box average. Roughly 4 codes at 8
# bits, and tests/test_rgb_to_nv12.py already allows 3.0 on the round trip.
MAX_CHROMA_DIFF = 4
# Kernel vs Torch is pure float reduction order, so at most one code.
MAX_KERNEL_DIFF = 1

# Only the matrix and range are set. Naming primaries or a transfer makes
# swscale perform an actual colour conversion, which changes the pixels and has
# nothing to do with the matrix under test.
SPACES = {"bt601": "smpte170m", "bt709": "bt709", "bt2020": "bt2020nc"}
RANGES = {"limited": "mpeg", "full": "jpeg"}


def build_field() -> np.ndarray:
    """Full-width flat bands on top, detail below.

    The bands are full width, so no 2x2 chroma quad straddles two colours
    horizontally. swscale's vertical chroma filter still reaches past the quad,
    so the comparison below scores only the chroma rows in a band's interior,
    where every downsample filter — swscale's or our box average — must agree.
    """
    rng = np.random.default_rng(1234)
    field = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    flat_colours = [
        (0, 0, 0), (255, 255, 255), (128, 128, 128),
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (16, 32, 48), (200, 120, 60), (60, 200, 120),
    ]
    assert len(flat_colours) * BAND_ROWS == FLAT_ROWS
    for index, colour in enumerate(flat_colours):
        field[index * BAND_ROWS : (index + 1) * BAND_ROWS] = colour

    field[FLAT_ROWS:] = rng.integers(0, 256, size=(HEIGHT - FLAT_ROWS, WIDTH, 3))
    return field


def interior_chroma_rows() -> np.ndarray:
    """Chroma rows that sit strictly inside one flat band."""
    per_band = BAND_ROWS // 2
    rows = np.arange(FLAT_ROWS // 2)
    return rows[(rows % per_band != 0) & (rows % per_band != per_band - 1)]


def run_ffmpeg(field: np.ndarray, pix_fmt: str, space: str, value_range: str) -> np.ndarray:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-i", "pipe:0",
        "-vf", f"scale=out_color_matrix={space}:out_range={value_range}",
        "-f", "rawvideo", "-pix_fmt", pix_fmt, "pipe:1",
    ]
    result = subprocess.run(command, input=field.tobytes(), capture_output=True, check=True)
    dtype = np.uint16 if pix_fmt == "p010le" else np.uint8
    return np.frombuffer(result.stdout, dtype=dtype).reshape(HEIGHT * 3 // 2, WIDTH)


def to_codes(packed: np.ndarray, ten_bit: bool) -> np.ndarray:
    codes = packed.astype(np.int32)
    return codes >> 6 if ten_bit else codes


def report(name: str, ours: np.ndarray, reference: np.ndarray, ten_bit: bool) -> int:
    ours_codes = to_codes(ours, ten_bit)
    reference_codes = to_codes(reference, ten_bit)
    difference = np.abs(ours_codes - reference_codes)

    chroma = difference[HEIGHT:]
    luma = difference[:HEIGHT].max()
    flat_chroma = chroma[interior_chroma_rows()].max()
    detailed_chroma = chroma[FLAT_ROWS // 2 :].max()

    scale = 4 if ten_bit else 1  # 10-bit codes span 4x the range of 8-bit ones
    within = luma <= MAX_LUMA_DIFF * scale and flat_chroma <= MAX_CHROMA_DIFF * scale
    status = "ok " if within else "FAIL"
    print(
        f"  {status} {name:24} luma {luma:3d}  flat chroma {flat_chroma:3d}  "
        f"detailed chroma {detailed_chroma:3d} (informational)"
    )
    return 0 if status == "ok " else 1


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is required to exercise the kernel", file=sys.stderr)
        return 2

    device = torch.device("cuda:0")
    field = build_field()
    frame = torch.from_numpy(field).permute(2, 0, 1).contiguous().to(device)
    failures = 0

    for pixel_format in ("nv12", "p010"):
        ten_bit = pixel_format == "p010"
        ffmpeg_format = "p010le" if ten_bit else "nv12"
        for space, matrix in SPACES.items():
            for value_range, ffmpeg_range in RANGES.items():
                variant = f"{pixel_format}_{space}_{value_range}"
                print(f"{variant}:")
                reference = run_ffmpeg(field, ffmpeg_format, matrix, ffmpeg_range)

                torch_packed = _TORCH_CONVERTERS[variant](frame).cpu().numpy()
                kernel_packed = (
                    RgbToYuvConverter(variant, device=device).convert(frame).cpu().numpy()
                )
                if ten_bit:
                    torch_packed = torch_packed.astype(np.uint16, copy=False)
                    kernel_packed = kernel_packed.astype(np.uint16, copy=False)

                failures += report("torch vs swscale", torch_packed, reference, ten_bit)
                failures += report("kernel vs swscale", kernel_packed, reference, ten_bit)

                kernel_gap = np.abs(
                    to_codes(kernel_packed, ten_bit) - to_codes(torch_packed, ten_bit)
                ).max()
                if kernel_gap > MAX_KERNEL_DIFF:
                    print(f"  FAIL kernel vs torch        max {kernel_gap}")
                    failures += 1
                else:
                    print(f"  ok  kernel vs torch         max {kernel_gap}")

    print()
    print("FAILED" if failures else "All variants agree within tolerance")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
