"""Contrast Adaptive Sharpening of the luma plane on GPU.

Implements AMD's FidelityFX CAS (MIT):
https://github.com/GPUOpen-Effects/FidelityFX-CAS

The strength-to-weight curve and the headroom constant are matched to FFmpeg's
``cas`` filter, so a given strength reproduces ``cas=strength=<s>:planes=1``.
Both were recovered by measuring that filter's output rather than read from its
source; ``scripts/cas_ffmpeg_oracle.py`` re-derives them and checks parity.

NVIDIA runs the fused kernel in ``cas.cu``; ROCm and CPU use the Torch
implementation here, which produces identical results.
"""
from __future__ import annotations

import ctypes
import logging

import torch
import torch.nn.functional as F

from jasna.accelerator import is_nvidia_device
from jasna.media.cuda_kernel import check_cuda, cuda_driver, resolve_function

logger = logging.getLogger(__name__)

# FFmpeg maps strength onto the CAS weight as -1 / lerp(16, 4, strength).
_WEIGHT_DIVISOR_AT_ZERO = 16.0
_WEIGHT_DIVISOR_AT_ONE = 4.0

# CAS measures its headroom against 1 << depth rather than the peak code, and
# the "better diagonals" variant sums two min/max pairs, doubling the scale.
_HEADROOM_MULTIPLIER = 2

# P010 carries 10-bit codes in the high bits of each 16-bit sample.
_P010_SCALE = 64.0

_FATBIN = "cas.fatbin"
_BLOCK_WIDTH = 32
_BLOCK_HEIGHT = 8

# The Torch path holds several full-resolution intermediates, so it works in
# horizontal bands sized to keep one band near this many pixels: big enough to
# saturate the GPU, small enough that the working set stays resident. On an 8K
# plane that traded 7.4 ms / 1.4 GB whole-frame for 3.5 ms / 0.4 GB banded;
# 1080p and below still run as a single band.
_BAND_PIXELS = 4 << 20


def cas_weight_scale(strength: float) -> float:
    divisor = _WEIGHT_DIVISOR_AT_ZERO + (
        _WEIGHT_DIVISOR_AT_ONE - _WEIGHT_DIVISOR_AT_ZERO
    ) * strength
    return -1.0 / divisor


def _sharpen_band(
    strip: torch.Tensor,
    out: torch.Tensor,
    *,
    weight_scale: float,
    headroom: float,
    peak: float,
) -> None:
    """Sharpen one horizontal band.

    ``strip`` is ``out``'s rows grown by one replicated row above and below, so
    every output pixel sees the same neighbourhood it would in a whole-frame
    pass.
    """
    padded = F.pad(strip[None, None], (1, 1, 0, 0), mode="replicate")
    height, width = out.shape
    centre = padded[0, 0, 1:height + 1, 1:width + 1]
    above = padded[0, 0, 0:height, 1:width + 1]
    left = padded[0, 0, 1:height + 1, 0:width]
    right = padded[0, 0, 1:height + 1, 2:width + 2]
    below = padded[0, 0, 2:height + 2, 1:width + 1]
    delta = (above + left).add_(right).add_(below).sub_(centre * 4.0)

    # Half precision is exact here: codes reach 1023 and the folded sums 2046,
    # well inside fp16's exact-integer range. Both extremes come out of one
    # pooling pass by stacking the plane with its negation, since min(x) is
    # -max(-x), and the 3x3 window is separable into a row then a column pass.
    half = padded.half()
    signed = torch.cat([half, half.neg()], dim=0)
    del half
    rows = F.max_pool2d(signed, (1, 3), stride=1)
    cols = F.max_pool2d(signed, (3, 1), stride=1)
    del signed
    box = F.max_pool2d(rows, (3, 1), stride=1)
    cross = torch.maximum(rows[:, :, 1:-1, :], cols[:, :, :, 1:-1])
    del rows, cols
    folded = cross.add_(box).float()
    del box, cross

    maxima = folded[0, 0].clamp_min_(1.0)
    minima = folded[1, 0].neg_()
    amp = torch.minimum(minima, headroom - maxima).div_(maxima).clamp_(0.0, 1.0).sqrt_()
    weight = amp.mul_(weight_scale)
    denominator = weight.mul(4.0).add_(1.0)

    # Full strength on a flat neighbourhood drives the denominator to zero, but
    # delta is zero there too, so the centre pixel is the limit.
    sharpened = torch.where(
        denominator > 0.0,
        centre + delta * weight / denominator,
        centre,
    )
    # FFmpeg truncates rather than rounds on write; match it.
    out.copy_(sharpened.clamp_(0.0, peak).floor_())


def sharpen_luma(
    plane: torch.Tensor,
    *,
    weight_scale: float,
    headroom: float,
    peak: float,
) -> torch.Tensor:
    """Sharpen a 2-D float32 luma plane held in code units."""
    height, width = plane.shape
    band = max(1, min(height, _BAND_PIXELS // max(width, 1)))
    out = torch.empty_like(plane)
    for top in range(0, height, band):
        bottom = min(top + band, height)
        strip = plane[max(top - 1, 0):min(bottom + 1, height)]
        if top == 0:
            strip = torch.cat([strip[:1], strip])
        if bottom == height:
            strip = torch.cat([strip, strip[-1:]])
        _sharpen_band(
            strip,
            out[top:bottom],
            weight_scale=weight_scale,
            headroom=headroom,
            peak=peak,
        )
    return out


class _CasKernel:
    def __init__(self, function_name: str):
        self.function_name = function_name
        self._function: ctypes.c_void_p | None = None
        self._values = [
            ctypes.c_uint64(),  # source pointer
            ctypes.c_uint64(),  # destination pointer
            ctypes.c_int(),     # source row stride, in samples
            ctypes.c_int(),     # destination row stride, in samples
            ctypes.c_int(),     # height
            ctypes.c_int(),     # width
            ctypes.c_float(),   # weight scale
        ]
        self._params = (ctypes.c_void_p * len(self._values))(
            *(ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in self._values)
        )

    def launch(self, src: torch.Tensor, dst: torch.Tensor, weight_scale: float) -> None:
        if self._function is None:
            self._function = resolve_function(_FATBIN, self.function_name)
        height, width = src.shape
        values = self._values
        values[0].value = src.data_ptr()
        values[1].value = dst.data_ptr()
        values[2].value = src.stride(0)
        values[3].value = dst.stride(0)
        values[4].value = height
        values[5].value = width
        values[6].value = weight_scale
        check_cuda(
            cuda_driver().cuLaunchKernel(
                self._function,
                (width + _BLOCK_WIDTH - 1) // _BLOCK_WIDTH,
                (height + _BLOCK_HEIGHT - 1) // _BLOCK_HEIGHT,
                1,
                _BLOCK_WIDTH,
                _BLOCK_HEIGHT,
                1,
                0,
                ctypes.c_void_p(torch.cuda.current_stream(src.device).cuda_stream),
                self._params,
                None,
            ),
            f"cuLaunchKernel({self.function_name})",
        )


class GpuCasSharpener:
    """Sharpens the luma plane of a packed NV12/P010 frame in place."""

    def __init__(self, strength: float, *, ten_bit: bool, device: torch.device):
        if not 0.0 < strength <= 1.0:
            raise ValueError(f"Sharpening strength must be in (0, 1], got {strength}")
        self.weight_scale = cas_weight_scale(strength)
        self.ten_bit = ten_bit
        depth = 10 if ten_bit else 8
        self.headroom = float(_HEADROOM_MULTIPLIER << depth)
        self.peak = float((1 << depth) - 1)
        self._kernel = (
            _CasKernel("cas_luma10" if ten_bit else "cas_luma8")
            if is_nvidia_device(device)
            else None
        )

    def _disable_kernel(self, exc: Exception) -> None:
        # The Torch path is correct but roughly 30x slower, so a broken install
        # must not degrade to it quietly.
        logger.warning(
            "CUDA sharpening kernel unavailable (%s); falling back to the "
            "much slower Torch implementation",
            exc,
        )
        self._kernel = None

    def _sharpen_torch_(self, luma: torch.Tensor) -> None:
        plane = luma.to(torch.float32)
        if self.ten_bit:
            plane.div_(_P010_SCALE)
        sharpened = sharpen_luma(
            plane,
            weight_scale=self.weight_scale,
            headroom=self.headroom,
            peak=self.peak,
        )
        if self.ten_bit:
            sharpened.mul_(_P010_SCALE)
        luma.copy_(sharpened)

    def sharpen_into(self, source_luma: torch.Tensor, destination_luma: torch.Tensor) -> None:
        """Sharpen one luma plane into another.

        Saves the full-plane copy ``apply_luma_`` needs, because the caller can
        hand over a scratch plane the conversion wrote and the real destination.
        """
        if self.ten_bit:
            source_luma = source_luma.view(torch.uint16)
            destination_luma = destination_luma.view(torch.uint16)
        if self._kernel is not None:
            try:
                self._kernel.launch(source_luma, destination_luma, self.weight_scale)
            except RuntimeError as exc:
                self._disable_kernel(exc)
            else:
                return

        destination_luma.copy_(source_luma)
        self._sharpen_torch_(destination_luma)

    def apply_luma_(self, packed: torch.Tensor, height: int) -> None:
        luma = packed[:height]
        if self.ten_bit:
            luma = luma.view(torch.uint16)
        if self._kernel is not None:
            sharpened = torch.empty_like(luma)
            try:
                self._kernel.launch(luma, sharpened, self.weight_scale)
            except RuntimeError as exc:
                self._disable_kernel(exc)
            else:
                luma.copy_(sharpened)
                return

        self._sharpen_torch_(luma)
