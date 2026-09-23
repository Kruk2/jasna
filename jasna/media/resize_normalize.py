"""Fused detector preprocess: resize, letterbox and normalize in one pass.

Detectors cast the whole frame to their input dtype and divide by 255 at source
resolution, then downscale. At 8K VR that writes a 384 MiB intermediate to
produce a 5 MiB one. ``ResizeNormalizer`` reads the frame once instead.

NVIDIA runs ``resize_normalize.cu``; ROCm and CPU keep the Torch expression the
caller already had, which is also what the kernel is tested against.
"""
from __future__ import annotations

import ctypes

import torch

from jasna.accelerator import is_nvidia_device
from jasna.media.cuda_kernel import Kernel, grid_size

_FATBIN = "resize_normalize.fatbin"
_BLOCK_WIDTH = 16
_BLOCK_HEIGHT = 16

_FUNCTIONS = {torch.float16: "resize_normalize_fp16", torch.float32: "resize_normalize_fp32"}


_RESIZE_NORMALIZE_ARG_TYPES = (
    ctypes.c_uint64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_uint64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
)


def _launch_resize_normalize(
    kernel: Kernel,
    frames: torch.Tensor,
    out: torch.Tensor,
    content: tuple[int, int, int, int],
    mean: torch.Tensor,
    std: torch.Tensor,
    fill: torch.Tensor,
) -> None:
    batch, _, src_height, src_width = frames.shape
    out_height, out_width = out.shape[2], out.shape[3]
    left, top, content_width, content_height = content
    kernel.launch(
        (grid_size(out_width, _BLOCK_WIDTH), grid_size(out_height, _BLOCK_HEIGHT), batch),
        (_BLOCK_WIDTH, _BLOCK_HEIGHT, 1),
        (
            frames.data_ptr(), frames.stride(0), frames.stride(1), frames.stride(2),
            out.data_ptr(), out.stride(0), out.stride(1), out.stride(2),
            batch, src_height, src_width, out_height, out_width,
            left, top, content_width, content_height,
            mean.data_ptr(), std.data_ptr(), fill.data_ptr(),
        ),
        torch.cuda.current_stream(frames.device).cuda_stream,
    )


class ResizeNormalizer:
    """Resizes a uint8 ``(B, 3, H, W)`` batch into a normalized detector input.

    ``mean``/``std`` are applied after the divide by 255. ``fill`` is the value
    letterbox padding takes, already expressed in normalized units.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        fill: tuple[float, float, float],
    ):
        self.device = device
        self.dtype = dtype
        self._mean = torch.tensor(mean, dtype=torch.float32, device=device)
        self._std = torch.tensor(std, dtype=torch.float32, device=device)
        self._fill = torch.tensor(fill, dtype=torch.float32, device=device)
        function = _FUNCTIONS.get(dtype)
        self._kernel = (
            Kernel(_FATBIN, function, _RESIZE_NORMALIZE_ARG_TYPES)
            if function is not None and is_nvidia_device(device)
            else None
        )

    @property
    def available(self) -> bool:
        return self._kernel is not None

    def run(
        self,
        frames_uint8_bchw: torch.Tensor,
        *,
        out_hw: tuple[int, int],
        content: tuple[int, int, int, int],
    ) -> torch.Tensor:
        if self._kernel is None:
            raise RuntimeError("The fused preprocess requires the CUDA kernel")
        if frames_uint8_bchw.dtype is not torch.uint8:
            raise ValueError(f"Expected a uint8 batch, got {frames_uint8_bchw.dtype}")
        if frames_uint8_bchw.ndim != 4 or frames_uint8_bchw.shape[1] != 3:
            raise ValueError(f"Expected (B, 3, H, W), got {tuple(frames_uint8_bchw.shape)}")
        if frames_uint8_bchw.stride(3) != 1:
            raise ValueError("Source rows must be contiguous")

        frames = frames_uint8_bchw
        if frames.device != self.device:
            frames = frames.to(self.device, non_blocking=True)

        out = torch.empty(
            (frames.shape[0], 3, out_hw[0], out_hw[1]), dtype=self.dtype, device=self.device
        )
        _launch_resize_normalize(self._kernel, frames, out, content, self._mean, self._std, self._fill)
        return out
