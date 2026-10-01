"""Fused detector preprocess: resize, letterbox and normalize in one pass.

Detectors cast the whole frame to their input dtype and divide by 255 at source
resolution, then downscale. At 8K VR that writes a 384 MiB intermediate to
produce a 5 MiB one. ``ResizeNormalizer`` reads the frame once instead.

NVIDIA runs ``resize_normalize.cu``. Windows AMD can explicitly select its
validated precompiled HIP backend; other routes retain the caller's Torch
expression, which is also the reference used to test these kernels.
"""
from __future__ import annotations

import ctypes
import os
import sys
import threading

import torch

from jasna.accelerator import is_nvidia_device
from jasna.media.cuda_kernel import check_cuda, cuda_driver, resolve_function

_FATBIN = "resize_normalize.fatbin"
_BLOCK_WIDTH = 16
_BLOCK_HEIGHT = 16

_FUNCTIONS = {torch.float16: "resize_normalize_fp16", torch.float32: "resize_normalize_fp32"}


class _ResizeNormalizeKernel:
    def __init__(self, function_name: str):
        self.function_name = function_name
        self._function: ctypes.c_void_p | None = None
        self._values = [
            ctypes.c_uint64(),  # source pointer
            ctypes.c_int64(),   # source batch stride
            ctypes.c_int64(),   # source channel stride
            ctypes.c_int64(),   # source row stride
            ctypes.c_uint64(),  # destination pointer
            ctypes.c_int64(),   # destination batch stride
            ctypes.c_int64(),   # destination channel stride
            ctypes.c_int64(),   # destination row stride
            ctypes.c_int(),     # batch
            ctypes.c_int(),     # source height
            ctypes.c_int(),     # source width
            ctypes.c_int(),     # output height
            ctypes.c_int(),     # output width
            ctypes.c_int(),     # content left
            ctypes.c_int(),     # content top
            ctypes.c_int(),     # content width
            ctypes.c_int(),     # content height
            ctypes.c_uint64(),  # mean pointer
            ctypes.c_uint64(),  # standard deviation pointer
            ctypes.c_uint64(),  # fill pointer
        ]
        self._params = (ctypes.c_void_p * len(self._values))(
            *(ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in self._values)
        )

    def launch(
        self,
        frames: torch.Tensor,
        out: torch.Tensor,
        content: tuple[int, int, int, int],
        mean: torch.Tensor,
        std: torch.Tensor,
        fill: torch.Tensor,
    ) -> None:
        if self._function is None:
            self._function = resolve_function(_FATBIN, self.function_name)
        batch, _, src_height, src_width = frames.shape
        out_height, out_width = out.shape[2], out.shape[3]
        left, top, content_width, content_height = content
        values = self._values
        for index, value in enumerate((
            frames.data_ptr(), frames.stride(0), frames.stride(1), frames.stride(2),
            out.data_ptr(), out.stride(0), out.stride(1), out.stride(2),
            batch, src_height, src_width, out_height, out_width,
            left, top, content_width, content_height,
            mean.data_ptr(), std.data_ptr(), fill.data_ptr(),
        )):
            values[index].value = value
        check_cuda(
            cuda_driver().cuLaunchKernel(
                self._function,
                (out_width + _BLOCK_WIDTH - 1) // _BLOCK_WIDTH,
                (out_height + _BLOCK_HEIGHT - 1) // _BLOCK_HEIGHT,
                batch,
                _BLOCK_WIDTH,
                _BLOCK_HEIGHT,
                1,
                0,
                ctypes.c_void_p(torch.cuda.current_stream(frames.device).cuda_stream),
                self._params,
                None,
            ),
            f"cuLaunchKernel({self.function_name})",
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
            _ResizeNormalizeKernel(function)
            if function is not None and is_nvidia_device(device)
            else None
        )
        if (self._kernel is None and function is not None and sys.platform == "win32"
                and device.type == "cuda" and getattr(torch.version, "hip", None)):
            from jasna.media.windows_hip_resize_contract import requested
            if requested(os.environ):
                self._kernel = _HipResizeNormalizeKernel(function, device, dtype)

    @property
    def available(self) -> bool:
        return self._kernel is not None

    def supports(self, frames: torch.Tensor, *, out_hw: tuple[int, int],
                 content: tuple[int, int, int, int]) -> bool:
        """Allow bounded backends to defer unsupported shapes to shared Torch."""
        if isinstance(self._kernel, _HipResizeNormalizeKernel):
            from jasna.media.windows_hip_resize_contract import supported_geometry
            return supported_geometry(tuple(frames.shape), tuple(frames.stride()), out_hw, content)
        return self.available

    def run(
        self,
        frames_uint8_bchw: torch.Tensor,
        *,
        out_hw: tuple[int, int],
        content: tuple[int, int, int, int],
    ) -> torch.Tensor:
        if self._kernel is None:
            raise RuntimeError("The fused preprocess requires an available GPU kernel")
        if frames_uint8_bchw.dtype is not torch.uint8:
            raise ValueError(f"Expected a uint8 batch, got {frames_uint8_bchw.dtype}")
        if frames_uint8_bchw.ndim != 4 or frames_uint8_bchw.shape[1] != 3:
            raise ValueError(f"Expected (B, 3, H, W), got {tuple(frames_uint8_bchw.shape)}")
        if frames_uint8_bchw.stride(3) != 1:
            raise ValueError("Source rows must be contiguous")
        if not self.supports(frames_uint8_bchw, out_hw=out_hw, content=content):
            raise ValueError("Input is outside the fused resize backend's supported geometry")

        frames = frames_uint8_bchw
        if frames.device != self.device:
            frames = frames.to(self.device, non_blocking=True)

        out = torch.empty(
            (frames.shape[0], 3, out_hw[0], out_hw[1]), dtype=self.dtype, device=self.device
        )
        self._kernel.launch(frames, out, content, self._mean, self._std, self._fill)
        return out


class _HipResizeNormalizeKernel(_ResizeNormalizeKernel):
    """Same parameter ABI with an explicitly admitted, precompiled HIP module."""

    def __init__(self, function_name: str, device: torch.device, dtype: torch.dtype):
        super().__init__(function_name)
        from jasna.media import hip_kernel as hip
        from jasna.media.windows_hip_resize_contract import CODE_OBJECT, MANIFEST, validate_bundle
        if device.type != "cuda" or device.index != 0:
            raise RuntimeError("The validated Windows HIP resize backend requires cuda:0")
        architecture = torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
        validate_bundle(hip.code_object_path(MANIFEST).parent, hip.hip_runtime_identity(),
                        str(torch.version.hip), architecture)
        self._hip = hip
        self._code_object = CODE_OBJECT
        self._device = device
        self._dtype = dtype
        self._launch_lock = threading.Lock()

    def launch(self, frames, out, content, mean, std, fill):
        from jasna.media.windows_hip_resize_contract import supported_geometry
        if (len(frames.shape) != 4 or len(out.shape) != 4
                or frames.device != self._device or out.device != self._device
                or frames.dtype != torch.uint8 or out.dtype != self._dtype
                or out.shape != (frames.shape[0], 3, out.shape[2], out.shape[3])
                or not supported_geometry(tuple(frames.shape), tuple(frames.stride()),
                                          tuple(out.shape[2:]), content)
                or not supported_geometry(tuple(out.shape), tuple(out.stride()),
                                          tuple(out.shape[2:]), (0, 0, out.shape[3], out.shape[2]))
                or out.stride(2) < out.shape[3]
                or out.stride(1) < out.stride(2) * out.shape[2]
                or out.stride(0) < out.stride(1) * out.shape[1]):
            raise ValueError("HIP resize input/output geometry, device or dtype mismatch")
        if any(t.device != self._device or t.dtype != torch.float32 or tuple(t.shape) != (3,)
               or not t.is_contiguous() for t in (mean, std, fill)):
            raise ValueError("HIP resize constants must be contiguous device float32 triples")
        batch, _, height, width = frames.shape
        oh, ow = out.shape[2:]
        left, top, cw, ch = content
        arguments = (frames.data_ptr(), *frames.stride()[:3], out.data_ptr(), *out.stride()[:3],
                     batch, height, width, oh, ow, left, top, cw, ch,
                     mean.data_ptr(), std.data_ptr(), fill.data_ptr())
        if len(arguments) != len(self._values):
            raise RuntimeError("HIP resize shared parameter ABI mismatch")
        # Parameter cells are mutable; keep packing and submission indivisible.
        # HIP modules use the existing product context-lifetime cache, not JIT.
        with self._launch_lock, torch.cuda.device(self._device):
            if self._function is None:
                self._function = self._hip.resolve_function(self._code_object, self.function_name)
            for cell, value in zip(self._values, arguments, strict=True):
                cell.value = value
            self._hip.launch_kernel(self._function, grid=((ow + 15) // 16, (oh + 15) // 16, batch),
                block=(16, 16, 1), stream=torch.cuda.current_stream(self._device).cuda_stream,
                params=self._params, operation=f"hipResizeNormalize({self.function_name})")
