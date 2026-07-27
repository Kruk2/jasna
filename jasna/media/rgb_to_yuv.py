"""Planar RGB to packed NV12/P010 conversion for the encoder.

NVIDIA runs the fused kernel in ``rgb_to_yuv.cu``; ROCm and CPU use the Torch
implementations in ``rgb_to_nv12.py`` / ``rgb_to_p010.py``, which are also the
reference the kernel is tested against.

The kernel takes separate luma and chroma destinations. That lets the caller
place the two planes in different buffers, which is what lets CAS sharpen
straight into the frame the encoder receives instead of copying a whole luma
plane back over the source.
"""
from __future__ import annotations

import ctypes

import torch

from jasna.accelerator import is_nvidia_device
from jasna.media.cuda_kernel import check_cuda, cuda_driver, resolve_function
from jasna.media.rgb_to_nv12 import (
    NV12_VARIANTS,
    _chw_rgb_to_nv12_into,
    chw_rgb_to_nv12_bt601_full,
    chw_rgb_to_nv12_bt601_limited,
    chw_rgb_to_nv12_bt709_full,
    chw_rgb_to_nv12_bt709_limited,
    chw_rgb_to_nv12_bt2020_full,
    chw_rgb_to_nv12_bt2020_limited,
)
from jasna.media.rgb_to_p010 import (
    P010_VARIANTS,
    _chw_rgb_to_p010_into,
    chw_rgb_to_p010_bt601_full,
    chw_rgb_to_p010_bt601_limited,
    chw_rgb_to_p010_bt709_full,
    chw_rgb_to_p010_bt709_limited,
    chw_rgb_to_p010_bt2020_full,
    chw_rgb_to_p010_bt2020_limited,
)
from jasna.media.yuv_scratch import YuvScratch

_FATBIN = "rgb_to_yuv.fatbin"

# Threads per block; each thread converts a 2x2 quad, so one block covers 32x32
# pixels.
_BLOCK_WIDTH = 16
_BLOCK_HEIGHT = 16

_TORCH_CONVERTERS = {
    "nv12_bt601_limited": chw_rgb_to_nv12_bt601_limited,
    "nv12_bt601_full": chw_rgb_to_nv12_bt601_full,
    "nv12_bt709_limited": chw_rgb_to_nv12_bt709_limited,
    "nv12_bt709_full": chw_rgb_to_nv12_bt709_full,
    "nv12_bt2020_limited": chw_rgb_to_nv12_bt2020_limited,
    "nv12_bt2020_full": chw_rgb_to_nv12_bt2020_full,
    "p010_bt601_limited": chw_rgb_to_p010_bt601_limited,
    "p010_bt601_full": chw_rgb_to_p010_bt601_full,
    "p010_bt709_limited": chw_rgb_to_p010_bt709_limited,
    "p010_bt709_full": chw_rgb_to_p010_bt709_full,
    "p010_bt2020_limited": chw_rgb_to_p010_bt2020_limited,
    "p010_bt2020_full": chw_rgb_to_p010_bt2020_full,
}


class _RgbToYuvKernel:
    def __init__(self, function_name: str):
        self.function_name = function_name
        self._function: ctypes.c_void_p | None = None
        self._values = [
            ctypes.c_uint64(),  # RGB pointer
            ctypes.c_int64(),   # RGB channel stride, in samples
            ctypes.c_int64(),   # RGB row stride, in samples
            ctypes.c_uint64(),  # luma pointer
            ctypes.c_int64(),   # luma row stride, in samples
            ctypes.c_uint64(),  # chroma pointer
            ctypes.c_int64(),   # chroma row stride, in samples
            ctypes.c_int(),     # height
            ctypes.c_int(),     # width
        ]
        self._params = (ctypes.c_void_p * len(self._values))(
            *(ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in self._values)
        )

    def launch(self, rgb: torch.Tensor, luma: torch.Tensor, chroma: torch.Tensor) -> None:
        if self._function is None:
            self._function = resolve_function(_FATBIN, self.function_name)
        height, width = luma.shape
        values = self._values
        values[0].value = rgb.data_ptr()
        values[1].value = rgb.stride(0)
        values[2].value = rgb.stride(1)
        values[3].value = luma.data_ptr()
        values[4].value = luma.stride(0)
        values[5].value = chroma.data_ptr()
        values[6].value = chroma.stride(0)
        values[7].value = height
        values[8].value = width
        quads_x = (width + 1) // 2
        quads_y = (height + 1) // 2
        check_cuda(
            cuda_driver().cuLaunchKernel(
                self._function,
                (quads_x + _BLOCK_WIDTH - 1) // _BLOCK_WIDTH,
                (quads_y + _BLOCK_HEIGHT - 1) // _BLOCK_HEIGHT,
                1,
                _BLOCK_WIDTH,
                _BLOCK_HEIGHT,
                1,
                0,
                ctypes.c_void_p(torch.cuda.current_stream(rgb.device).cuda_stream),
                self._params,
                None,
            ),
            f"cuLaunchKernel({self.function_name})",
        )


class RgbToYuvConverter:
    """Converts a ``(3, H, W)`` planar RGB frame into a packed NV12/P010 frame.

    ``convert_into`` writes the two planes into caller-owned buffers; ``convert``
    allocates a single packed frame around it. Neither allocates per frame on the
    eager path: its float32 working set is built once per frame size and reused.
    """

    def __init__(self, variant: str, *, device: torch.device):
        if variant not in _TORCH_CONVERTERS:
            raise ValueError(f"Unknown RGB to YUV variant: {variant}")
        self.variant = variant
        self.ten_bit = variant.startswith("p010")
        self.sample_dtype = torch.int16 if self.ten_bit else torch.uint8
        self._torch_convert = _TORCH_CONVERTERS[variant]
        self._eager_rows, self._eager_full_range = (
            P010_VARIANTS[variant] if self.ten_bit else NV12_VARIANTS[variant]
        )
        self._kernel = _RgbToYuvKernel(variant) if is_nvidia_device(device) else None
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
            convert_into = _chw_rgb_to_p010_into if self.ten_bit else _chw_rgb_to_nv12_into
            convert_into(
                frame,
                luma,
                chroma,
                self._scratch_for(frame),
                self._eager_rows,
                full_range=self._eager_full_range,
            )
            return
        if self.ten_bit:
            luma = luma.view(torch.uint16)
            chroma = chroma.view(torch.uint16)
        self._kernel.launch(frame, luma, chroma)

    def _scratch_for(self, frame: torch.Tensor) -> YuvScratch:
        _, height, width = frame.shape
        scratch = self._scratch
        if scratch is None or not scratch.matches(height, width, frame.device):
            scratch = YuvScratch(height, width, frame.device)
            self._scratch = scratch
        return scratch
