"""Conformance oracle: validate jasna.media.cas against FFmpeg's ``cas`` filter.

Pipes a deterministic luma field through real ``cas=strength=<s>:planes=1`` at
8 and 10 bits and compares it to our implementation. Also re-derives the two
constants that were recovered by measuring FFmpeg rather than read from its
source: the strength-to-weight curve ``-1 / lerp(16, 4, strength)`` and the
headroom ``1 << depth``. Run ``--fit-curve`` to print that derivation.

Residual disagreement is FFmpeg's approximate reciprocal/square-root; ours uses
exact ones, so a small number of pixels differ by one code.

CPU only. Requires ffmpeg with the cas filter. Run:
  ~/.virtualenvs/jasna-linux/bin/python scripts/cas_ffmpeg_oracle.py
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import torch

from jasna.media.cas import cas_weight_scale, sharpen_luma

WIDTH = 160
HEIGHT = 120
STRENGTHS = (0.1, 0.3, 0.5, 0.8, 1.0)
MIN_WITHIN_ONE = 99.0        # percent of pixels within one code, strength <= 0.8
MIN_WITHIN_ONE_AT_MAX = 94.0  # percent at strength 1.0, where the pole sits
MAX_MEAN_OFFSET = 0.25       # codes; a systematic shift would mean wrong rounding


def build_field(depth: int) -> np.ndarray:
    """Noise plus flat, step and ramp regions, so every amplitude regime is hit."""
    peak = (1 << depth) - 1
    rng = np.random.default_rng(1234)
    field = rng.integers(0, peak + 1, size=(HEIGHT, WIDTH))
    field[0:8, :] = 0                       # flat black: the zero-maxima guard
    field[8:16, :] = peak                   # flat white: the headroom term
    field[16:24, :] = peak // 2             # flat mid: the strength-1.0 pole
    field[24:32, :WIDTH // 2] = 0
    field[24:32, WIDTH // 2:] = peak        # hard step: maximum sharpening
    field[32:40, :] = np.linspace(0, peak, WIDTH)
    return field.astype(np.uint8 if depth == 8 else np.uint16)


def run_ffmpeg(field: np.ndarray, strength: float, pix_fmt: str) -> np.ndarray:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", f"{WIDTH}x{HEIGHT}", "-i", "pipe:0",
        "-vf", f"cas=strength={strength}:planes=1",
        "-f", "rawvideo", "-pix_fmt", pix_fmt, "pipe:1",
    ]
    result = subprocess.run(command, input=field.tobytes(), capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=field.dtype).reshape(HEIGHT, WIDTH)


def run_jasna(field: np.ndarray, strength: float, depth: int) -> np.ndarray:
    plane = torch.from_numpy(field.astype(np.float32))
    out = sharpen_luma(
        plane,
        weight_scale=cas_weight_scale(strength),
        headroom=float(2 << depth),
        peak=float((1 << depth) - 1),
    )
    return out.numpy()


def neighbourhood(field: np.ndarray):
    padded = np.pad(field.astype(np.float64), 1, mode="edge")
    return [
        padded[y:y + HEIGHT, x:x + WIDTH]
        for y in (0, 1, 2)
        for x in (0, 1, 2)
    ]


def fit_curve(depth: int) -> None:
    field = build_field(depth)
    pix_fmt = "gray" if depth == 8 else "gray10le"
    a, b, c, d, e, f, g, h, i = neighbourhood(field)
    minima = np.minimum.reduce([b, d, e, f, h]) + np.minimum.reduce([a, b, c, d, e, f, g, h, i])
    maxima = np.maximum.reduce([b, d, e, f, h]) + np.maximum.reduce([a, b, c, d, e, f, g, h, i])
    amp = np.sqrt(np.clip(np.minimum(minima, float(2 << depth) - maxima)
                          / np.maximum(maxima, 1.0), 0.0, 1.0))
    cross = b + d + f + h

    print(f"\n  strength -> measured divisor (expected 16 - 12*strength), depth {depth}")
    for strength in (0.0, 0.25, 0.5, 0.75, 1.0):
        out = run_ffmpeg(field, strength, pix_fmt).astype(np.float64)
        denominator = 4.0 * out - cross
        usable = (np.abs(denominator) > 1e-6) & (amp > 0.05)
        weight = (e - out)[usable] / denominator[usable]
        measured = -1.0 / np.median(weight / amp[usable])
        print(f"    {strength:<5} measured {measured:7.3f}   expected {16 - 12 * strength:7.3f}")


def main() -> int:
    if "--fit-curve" in sys.argv:
        for depth in (8, 10):
            fit_curve(depth)
        return 0

    failures = 0
    for depth, pix_fmt in ((8, "gray"), (10, "gray10le")):
        field = build_field(depth)
        print(f"\n{depth}-bit")
        for strength in STRENGTHS:
            reference = run_ffmpeg(field, strength, pix_fmt).astype(np.float64)
            ours = run_jasna(field, strength, depth).astype(np.float64)
            difference = ours - reference
            within_one = float(np.mean(np.abs(difference) <= 1) * 100)
            mean_offset = float(difference.mean())
            limit = MIN_WITHIN_ONE_AT_MAX if strength == 1.0 else MIN_WITHIN_ONE
            ok = (
                within_one >= limit
                and abs(mean_offset) <= MAX_MEAN_OFFSET
                and not np.isnan(ours).any()
            )
            failures += not ok
            print(
                f"  strength {strength:<4} exact {np.mean(difference == 0) * 100:6.2f}%  "
                f"within 1 code {within_one:6.2f}% (min {limit})  "
                f"max {np.abs(difference).max():4.0f}  mean {mean_offset:+.3f}  "
                f"{'ok' if ok else 'FAIL'}"
            )

    if failures:
        print(f"\n{failures} case(s) outside tolerance")
    else:
        print("\nall cases within tolerance")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
