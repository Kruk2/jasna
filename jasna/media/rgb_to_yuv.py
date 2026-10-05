"""Planar RGB to packed NV12/P010 conversion for the encoder.

NVIDIA and supported AMD devices launch the same fused colour-kernel source
through their CUDA/HIP backends. Other devices use the shared Torch reference.

The kernel takes separate luma and chroma destinations. That lets the caller
place the two planes in different buffers, which is what lets CAS sharpen
straight into the frame the encoder receives instead of copying a whole luma
plane back over the source.
"""
from __future__ import annotations

import ctypes
from typing import NamedTuple

import torch

from jasna.accelerator import is_nvidia_device
from jasna.media.cuda_kernel import Kernel, grid_size
from jasna.media.hip_kernel import Kernel as HipKernel, color_code_object_name, hip_color_kernels_enabled
from jasna.media.yuv_scratch import (
    YuvScratch,
    apply_matrix,
    average_quads,
    interleave_chroma,
)

_FATBIN = "rgb_to_yuv.fatbin"
_BLOCK_WIDTH = 16
_BLOCK_HEIGHT = 16

# Luma, U and V rows of the RGB->YUV matrix for each standard (BT.2020 is
# non-constant luminance).
_COEFFICIENTS = {
    "bt601": ((0.299, 0.587, 0.114), (-0.168736, -0.331264, 0.5), (0.5, -0.418688, -0.081312)),
    "bt709": ((0.2126, 0.7152, 0.0722), (-0.114572, -0.385428, 0.5), (0.5, -0.454153, -0.045847)),
    "bt2020": ((0.2627, 0.678, 0.0593), (-0.13963, -0.36037, 0.5), (0.5, -0.459786, -0.040214)),
}


class _CodeRange(NamedTuple):
    luma_scale: float
    chroma_scale: float
    peak: int
    luma_limits: tuple[int, int]
    chroma_limits: tuple[int, int]
    chroma_offset: float
    storage_scale: int  # P010 stores its 10-bit codes in the high bits of each sample


_CODE_RANGES = {
    "nv12": _CodeRange(219.0, 224.0, 255, (16, 235), (16, 240), 128.0, 1),
    "p010": _CodeRange(876.0, 896.0, 1023, (64, 940), (64, 960), 512.0, 64),
}


def _matrix_rows(standard: str, code_range: _CodeRange, full_range: bool) -> tuple[tuple[float, float, float], ...]:
    luma, u, v = _COEFFICIENTS[standard]
    matrix = torch.tensor([
        [code_range.luma_scale * c for c in luma],
        [code_range.chroma_scale * c for c in u],
        [code_range.chroma_scale * c for c in v],
    ], dtype=torch.float32)
    if full_range:
        matrix[0].mul_(code_range.peak / code_range.luma_scale)
        matrix[1:3].mul_(code_range.peak / code_range.chroma_scale)
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _rgb_to_yuv_into(
    frame: torch.Tensor,
    luma: torch.Tensor,
    chroma: torch.Tensor,
    scratch: YuvScratch,
    rows: tuple[tuple[float, float, float], ...],
    code_range: _CodeRange,
    *,
    full_range: bool,
) -> None:
    luma_offset = 0.0 if full_range else float(code_range.luma_limits[0])
    offsets = (luma_offset, code_range.chroma_offset, code_range.chroma_offset)
    yuv = scratch.yuv
    apply_matrix(frame, rows, offsets, yuv)

    luma_limits = (0, code_range.peak) if full_range else code_range.luma_limits
    chroma_limits = (0, code_range.peak) if full_range else code_range.chroma_limits
    y = yuv[0].round_().clamp_(*luma_limits)
    subsampled = scratch.chroma
    average_quads(yuv[1:3], subsampled)
    subsampled.round_().clamp_(*chroma_limits)
    if code_range.storage_scale != 1:
        y.mul_(code_range.storage_scale)
        subsampled.mul_(code_range.storage_scale)
    luma.copy_(y)
    interleave_chroma(subsampled, chroma)


_RGB_TO_YUV_ARG_TYPES = (
    ctypes.c_uint64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_uint64, ctypes.c_int64,
    ctypes.c_uint64, ctypes.c_int64,
    ctypes.c_int, ctypes.c_int,
)


def _launch_rgb_to_yuv(kernel: Kernel, rgb: torch.Tensor, luma: torch.Tensor, chroma: torch.Tensor) -> None:
    height, width = luma.shape
    quads_x = (width + 1) // 2
    quads_y = (height + 1) // 2
    kernel.launch(
        (grid_size(quads_x, _BLOCK_WIDTH), grid_size(quads_y, _BLOCK_HEIGHT), 1),
        (_BLOCK_WIDTH, _BLOCK_HEIGHT, 1),
        (
            rgb.data_ptr(), rgb.stride(0), rgb.stride(1),
            luma.data_ptr(), luma.stride(0),
            chroma.data_ptr(), chroma.stride(0),
            height, width,
        ),
        torch.cuda.current_stream(rgb.device).cuda_stream,
    )



class RgbToYuvConverter:
    """Converts a ``(3, H, W)`` uint8 planar RGB frame into a packed NV12/P010 frame.

    ``variant`` is ``<nv12|p010>_<bt601|bt709|bt2020>_<limited|full>``, which is
    also the kernel function name. ``convert_into`` writes the two planes into
    caller-owned buffers; ``convert`` allocates a single packed frame around it.
    Neither allocates per frame on the eager path: its float32 working set is
    built once per frame size and reused.
    """

    def __init__(self, variant: str, *, device: torch.device):
        pixel_format, _, rest = variant.partition("_")
        standard, _, value_range = rest.partition("_")
        if (
            pixel_format not in _CODE_RANGES
            or standard not in _COEFFICIENTS
            or value_range not in ("limited", "full")
        ):
            raise ValueError(f"Unknown RGB to YUV variant: {variant}")
        self.variant = variant
        self.ten_bit = pixel_format == "p010"
        self.sample_dtype = torch.int16 if self.ten_bit else torch.uint8
        self._code_range = _CODE_RANGES[pixel_format]
        self._full_range = value_range == "full"
        self._rows = _matrix_rows(standard, self._code_range, self._full_range)
        self._kernel = None
        if is_nvidia_device(device):
            self._kernel = Kernel(_FATBIN, variant, _RGB_TO_YUV_ARG_TYPES)
        elif hip_color_kernels_enabled(device):
            self._kernel = HipKernel(color_code_object_name("rgb_to_yuv"), variant, _RGB_TO_YUV_ARG_TYPES)
        self._scratch: YuvScratch | None = None

    @property
    def uses_kernel(self) -> bool:
        return self._kernel is not None

    def convert(self, frame: torch.Tensor) -> torch.Tensor:
        _, height, width = frame.shape
        packed = torch.empty(
            (height + height // 2, width), dtype=self.sample_dtype, device=frame.device
        )
        self.convert_into(frame, packed[:height], packed[height:])
        return packed

    def convert_into(
        self, frame: torch.Tensor, luma: torch.Tensor, chroma: torch.Tensor
    ) -> None:
        _, height, width = frame.shape
        if height % 2 or width % 2:
            raise ValueError(f"4:2:0 conversion requires even dimensions, got {height}x{width}")
        if frame.dtype is not torch.uint8:
            raise ValueError(f"Expected a uint8 RGB frame, got {frame.dtype}")
        if frame.stride(2) != 1:
            raise ValueError("RGB frame rows must be contiguous")
        if self._kernel is None:
            _rgb_to_yuv_into(
                frame,
                luma,
                chroma,
                self._scratch_for(frame),
                self._rows,
                self._code_range,
                full_range=self._full_range,
            )
            return
        if self.ten_bit:
            luma = luma.view(torch.uint16)
            chroma = chroma.view(torch.uint16)
        _launch_rgb_to_yuv(self._kernel, frame, luma, chroma)

    def _scratch_for(self, frame: torch.Tensor) -> YuvScratch:
        _, height, width = frame.shape
        scratch = self._scratch
        if scratch is None or not scratch.matches(height, width, frame.device):
            scratch = YuvScratch(height, width, frame.device)
            self._scratch = scratch
        return scratch
