import ctypes
import os
import sys
import threading
from pathlib import Path

import torch

from av.video.reformatter import Colorspace as AvColorspace

from jasna._frozen import is_frozen
from jasna.accelerator import is_nvidia_device

# YUV->RGB from standard luma coefficients (Kr, Kb):
#   R = Y' + 2(1-Kr) * V'
#   G = Y' - 2Kb(1-Kb)/Kg * U' - 2Kr(1-Kr)/Kg * V'
#   B = Y' + 2(1-Kb) * U'
_KR_KB = {
    "bt709": (0.2126, 0.0722),
    "bt601": (0.299, 0.114),
    "bt2020": (0.2627, 0.0593),
}

_BAYER8 = [
    [0, 48, 12, 60, 3, 51, 15, 63],
    [32, 16, 44, 28, 35, 19, 47, 31],
    [8, 56, 4, 52, 11, 59, 7, 55],
    [40, 24, 36, 20, 43, 27, 39, 23],
    [2, 50, 14, 62, 1, 49, 13, 61],
    [34, 18, 46, 30, 33, 17, 45, 29],
    [10, 58, 6, 54, 9, 57, 5, 53],
    [42, 26, 38, 22, 41, 25, 37, 21],
]


def _rgb_from_yuv_coeffs(name: str) -> tuple[float, float, float, float]:
    kr, kb = _KR_KB[name]
    kg = 1.0 - kr - kb
    return (
        2.0 * (1.0 - kr),
        2.0 * kb * (1.0 - kb) / kg,
        2.0 * kr * (1.0 - kr) / kg,
        2.0 * (1.0 - kb),
    )


_driver: ctypes.CDLL | None = None
_driver_lock = threading.Lock()
_modules: dict[int, ctypes.c_void_p] = {}
_CUDA_CONVERSION_BATCH = 8


def _fatbin_path() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent / "yuv_to_rgb.fatbin"
    return Path(__file__).resolve().with_name("yuv_to_rgb.fatbin")


def _cuda_driver() -> ctypes.CDLL:
    global _driver
    if _driver is not None:
        return _driver

    loader = ctypes.WinDLL if os.name == "nt" else ctypes.CDLL
    lib = loader("nvcuda.dll" if os.name == "nt" else "libcuda.so.1")
    lib.cuInit.argtypes = [ctypes.c_uint]
    lib.cuInit.restype = ctypes.c_int
    lib.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.cuCtxGetCurrent.restype = ctypes.c_int
    lib.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    lib.cuModuleLoadData.restype = ctypes.c_int
    lib.cuModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib.cuModuleGetFunction.restype = ctypes.c_int
    lib.cuLaunchKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.cuLaunchKernel.restype = ctypes.c_int
    lib.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    lib.cuGetErrorName.restype = ctypes.c_int
    lib.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    lib.cuGetErrorString.restype = ctypes.c_int
    _driver = lib
    return lib


def _check_cuda(result: int, operation: str) -> None:
    if result == 0:
        return
    lib = _cuda_driver()
    name = ctypes.c_char_p()
    message = ctypes.c_char_p()
    lib.cuGetErrorName(result, ctypes.byref(name))
    lib.cuGetErrorString(result, ctypes.byref(message))
    name_text = name.value.decode(errors="replace") if name.value else f"CUDA error {result}"
    message_text = message.value.decode(errors="replace") if message.value else "unknown error"
    raise RuntimeError(f"{operation} failed: {name_text}: {message_text}")


def _load_cuda_module() -> ctypes.c_void_p:
    lib = _cuda_driver()
    _check_cuda(lib.cuInit(0), "cuInit")

    context = ctypes.c_void_p()
    _check_cuda(lib.cuCtxGetCurrent(ctypes.byref(context)), "cuCtxGetCurrent")
    if not context.value:
        raise RuntimeError("No current CUDA context while loading YUV conversion kernel")

    with _driver_lock:
        cached = _modules.get(context.value)
        if cached is not None:
            return cached

        path = _fatbin_path()
        try:
            image = ctypes.create_string_buffer(path.read_bytes())
        except OSError as exc:
            raise RuntimeError(f"Missing precompiled YUV conversion kernel: {path}") from exc
        module = ctypes.c_void_p()
        _check_cuda(lib.cuModuleLoadData(ctypes.byref(module), image), "cuModuleLoadData")
        _modules[context.value] = module
        return module


class _CudaYuvKernel:
    def __init__(self, function_name: str):
        self.function_name = function_name
        self._function: ctypes.c_void_p | None = None
        self._values = [
            *(ctypes.c_uint64() for _ in range(2 * _CUDA_CONVERSION_BATCH)),
            ctypes.c_int(),       # y stride
            ctypes.c_int(),       # uv stride
            ctypes.c_uint64(),    # output pointer
            ctypes.c_int64(),     # output batch stride
            ctypes.c_int64(),     # output channel stride
            ctypes.c_int64(),     # output row stride
            ctypes.c_int(),       # batch size
            ctypes.c_int(),       # height
            ctypes.c_int(),       # width
        ]
        self._params = (ctypes.c_void_p * len(self._values))(
            *(ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in self._values)
        )

    def _resolve(self) -> ctypes.c_void_p:
        if self._function is not None:
            return self._function
        lib = _cuda_driver()
        context = ctypes.c_void_p()
        _check_cuda(lib.cuCtxGetCurrent(ctypes.byref(context)), "cuCtxGetCurrent")
        if not context.value:
            raise RuntimeError("No current CUDA context while launching YUV conversion kernel")
        module = _load_cuda_module()
        function = ctypes.c_void_p()
        _check_cuda(
            lib.cuModuleGetFunction(
                ctypes.byref(function), module, self.function_name.encode("ascii")
            ),
            f"cuModuleGetFunction({self.function_name})",
        )
        self._function = function
        return function

    def launch(self, y: torch.Tensor, uv: torch.Tensor, out: torch.Tensor) -> None:
        self.launch_ptrs(
            [y.data_ptr()],
            y.stride(0),
            [uv.data_ptr()],
            uv.stride(0),
            out,
        )

    def launch_ptr(
        self,
        y_ptr: int,
        y_stride: int,
        uv_ptr: int,
        uv_stride: int,
        out: torch.Tensor,
        stream: int | None = None,
    ) -> None:
        self.launch_ptrs([y_ptr], y_stride, [uv_ptr], uv_stride, out, stream)

    def launch_ptrs(
        self,
        y_ptrs: list[int],
        y_stride: int,
        uv_ptrs: list[int],
        uv_stride: int,
        out: torch.Tensor,
        stream: int | None = None,
    ) -> None:
        batch_size = len(y_ptrs)
        if batch_size != len(uv_ptrs) or not 1 <= batch_size <= _CUDA_CONVERSION_BATCH:
            raise ValueError(f"CUDA YUV conversion batch must contain 1-{_CUDA_CONVERSION_BATCH} frames")
        function = self._resolve()
        values = self._values
        for i in range(_CUDA_CONVERSION_BATCH):
            values[i].value = y_ptrs[i] if i < batch_size else 0
            values[_CUDA_CONVERSION_BATCH + i].value = uv_ptrs[i] if i < batch_size else 0
        base = 2 * _CUDA_CONVERSION_BATCH
        values[base].value = y_stride
        values[base + 1].value = uv_stride
        values[base + 2].value = out.data_ptr()
        values[base + 3].value = out.stride(0) if out.ndim == 4 else 0
        values[base + 4].value = out.stride(-3)
        values[base + 5].value = out.stride(-2)
        values[base + 6].value = batch_size
        values[base + 7].value = out.shape[-2]
        values[base + 8].value = out.shape[-1]
        threads = 256
        pixels = batch_size * out.shape[-2] * out.shape[-1]
        blocks = (pixels + threads - 1) // threads
        if stream is None:
            stream = torch.cuda.current_stream(out.device).cuda_stream
        _check_cuda(
            _cuda_driver().cuLaunchKernel(
                function,
                blocks,
                1,
                1,
                threads,
                1,
                1,
                0,
                ctypes.c_void_p(stream),
                self._params,
                None,
            ),
            f"cuLaunchKernel({self.function_name})",
        )


class YuvToRgbConverter:
    """NV12/P010 planes -> planar RGB uint8 (3, H, W) on GPU.

    CUDA conversion is one ahead-of-time-compiled kernel that writes directly
    into the destination tensor. The CPU implementation is the unit-test/reference
    path. 10-bit output uses the same 8x8 Bayer ordered dither as the VALI decoder.
    """

    def __init__(
        self,
        height: int,
        width: int,
        color_space: AvColorspace,
        full_range: bool,
        is_10bit: bool,
        device: torch.device,
    ):
        self.height = height
        self.width = width
        self.is_10bit = is_10bit
        color_names = {
            AvColorspace.ITU601: "bt601",
            AvColorspace.ITU709: "bt709",
            AvColorspace.BT2020: "bt2020",
        }
        try:
            name = color_names[color_space]
        except KeyError as exc:
            raise ValueError(f"Unsupported YUV color space: {color_space}") from exc

        self._cuda_kernel = None
        if is_nvidia_device(device):
            bits = 10 if is_10bit else 8
            value_range = "full" if full_range else "limited"
            self._cuda_kernel = _CudaYuvKernel(f"yuv{bits}_{name}_{value_range}")
            return

        a, b, c, d = _rgb_from_yuv_coeffs(name)

        out_max = 1023.0 if is_10bit else 255.0
        if full_range:
            luma_scale = out_max / (1023.0 if is_10bit else 255.0)
            chroma_scale = luma_scale
            luma_offset = 0.0
            chroma_center = 512.0 if is_10bit else 128.0
        else:
            luma_scale = out_max / (876.0 if is_10bit else 219.0)
            chroma_scale = out_max / (896.0 if is_10bit else 224.0)
            luma_offset = 64.0 if is_10bit else 16.0
            chroma_center = 512.0 if is_10bit else 128.0

        # rgb = luma_scale*Y + C @ [U, V] + off, in code units -> 0..out_max.
        # P010 stores the 10-bit value << 6; folding that /64 into the scales
        # keeps the plane tensors untouched (no extra kernels).
        raw_div = 64.0 if is_10bit else 1.0
        self._luma_scale = luma_scale / raw_div
        chroma_matrix = [
            [0.0, a * chroma_scale],
            [-b * chroma_scale, -c * chroma_scale],
            [d * chroma_scale, 0.0],
        ]
        chroma_matrix = [[value / raw_div for value in row] for row in chroma_matrix]
        offset = [
            -luma_offset * luma_scale - a * chroma_center * chroma_scale,
            -luma_offset * luma_scale + (b + c) * chroma_center * chroma_scale,
            -luma_offset * luma_scale - d * chroma_center * chroma_scale,
        ]
        self._chroma_matrix = torch.tensor(chroma_matrix, dtype=torch.float32, device=device)
        self._offset = torch.tensor(offset, dtype=torch.float32, device=device).view(3, 1, 1)

        if is_10bit:
            bayer = torch.tensor(_BAYER8, device=device, dtype=torch.float32)
            bayer = (bayer + 0.5) / 64.0
            y_mod8 = torch.arange(height, device=device) & 7
            x_mod8 = torch.arange(width, device=device) & 7
            t = bayer[y_mod8][:, x_mod8].unsqueeze(0)
            self._dither2 = torch.floor(t * 4.0).to(torch.int32)

    def convert(self, y: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        out = torch.empty((3, self.height, self.width), device=y.device, dtype=torch.uint8)
        self.convert_into(y, uv, out)
        return out

    def convert_frame_into(
        self, frame, out: torch.Tensor, stream: int | None = None
    ) -> None:
        """Convert a PyAV CUDA frame without constructing per-plane Torch tensors."""
        if self._cuda_kernel is None:
            raise RuntimeError("CUDA frame conversion requires a CUDA converter")
        if len(frame.planes) != 2:
            raise ValueError(f"Expected a two-plane NV12/P010 frame, got {len(frame.planes)}")
        y_plane, uv_plane = frame.planes
        bytes_per_sample = 2 if self.is_10bit else 1
        if y_plane.line_size % bytes_per_sample or uv_plane.line_size % bytes_per_sample:
            raise ValueError("YUV plane pitch is not aligned to its sample size")
        if y_plane.line_size < self.width * bytes_per_sample:
            raise ValueError("Luma plane pitch is smaller than the visible width")
        if uv_plane.line_size < self.width * bytes_per_sample:
            raise ValueError("Chroma plane pitch is smaller than the visible width")
        if out.shape != (3, self.height, self.width) or out.dtype != torch.uint8:
            raise ValueError(f"Unexpected RGB destination: {tuple(out.shape)} {out.dtype}")
        if not out.is_cuda or out.stride(2) != 1:
            raise ValueError("RGB destination must be a CUDA tensor with contiguous pixels")
        self._cuda_kernel.launch_ptr(
            y_plane.buffer_ptr,
            y_plane.line_size // bytes_per_sample,
            uv_plane.buffer_ptr,
            uv_plane.line_size // bytes_per_sample,
            out,
            stream,
        )

    def convert_frames_into(
        self, frames: list, out: torch.Tensor, stream: int | None = None
    ) -> None:
        """Convert a batch of PyAV CUDA frames with at most one launch per 8 frames."""
        if self._cuda_kernel is None:
            raise RuntimeError("CUDA frame conversion requires a CUDA converter")
        if out.shape != (len(frames), 3, self.height, self.width) or out.dtype != torch.uint8:
            raise ValueError(f"Unexpected RGB batch destination: {tuple(out.shape)} {out.dtype}")
        if not out.is_cuda or out.stride(3) != 1:
            raise ValueError("RGB batch destination must be a CUDA tensor with contiguous pixels")

        bytes_per_sample = 2 if self.is_10bit else 1
        for start in range(0, len(frames), _CUDA_CONVERSION_BATCH):
            chunk = frames[start : start + _CUDA_CONVERSION_BATCH]
            y_ptrs: list[int] = []
            uv_ptrs: list[int] = []
            y_stride = uv_stride = None
            for frame in chunk:
                if len(frame.planes) != 2:
                    raise ValueError(
                        f"Expected a two-plane NV12/P010 frame, got {len(frame.planes)}"
                    )
                y_plane, uv_plane = frame.planes
                if y_plane.line_size % bytes_per_sample or uv_plane.line_size % bytes_per_sample:
                    raise ValueError("YUV plane pitch is not aligned to its sample size")
                frame_y_stride = y_plane.line_size // bytes_per_sample
                frame_uv_stride = uv_plane.line_size // bytes_per_sample
                if frame_y_stride < self.width or frame_uv_stride < self.width:
                    raise ValueError("YUV plane pitch is smaller than the visible width")
                if y_stride is None:
                    y_stride, uv_stride = frame_y_stride, frame_uv_stride
                elif (frame_y_stride, frame_uv_stride) != (y_stride, uv_stride):
                    raise ValueError("YUV frame pitches changed within a conversion batch")
                y_ptrs.append(y_plane.buffer_ptr)
                uv_ptrs.append(uv_plane.buffer_ptr)

            self._cuda_kernel.launch_ptrs(
                y_ptrs,
                y_stride,
                uv_ptrs,
                uv_stride,
                out[start : start + len(chunk)],
                stream,
            )

    def convert_into(self, y: torch.Tensor, uv: torch.Tensor, out: torch.Tensor) -> None:
        """y (H, W) uint8/uint16, uv (H/2, W/2, 2) uint8/uint16 -> out (3, H, W) uint8.

        P010 planes store the 10-bit value in the top bits (value << 6).
        """
        if y.is_cuda:
            if self._cuda_kernel is None:
                raise RuntimeError("CPU YUV converter cannot process CUDA planes")
            expected = torch.uint16 if self.is_10bit else torch.uint8
            if y.dtype != expected or uv.dtype != expected:
                raise TypeError(
                    f"Expected {expected} {'P010' if self.is_10bit else 'NV12'} planes, "
                    f"got {y.dtype} and {uv.dtype}"
                )
            if y.shape != (self.height, self.width):
                raise ValueError(f"Unexpected luma shape: {tuple(y.shape)}")
            if uv.shape != (self.height // 2, self.width // 2, 2):
                raise ValueError(f"Unexpected chroma shape: {tuple(uv.shape)}")
            if out.shape != (3, self.height, self.width) or out.dtype != torch.uint8:
                raise ValueError(f"Unexpected RGB destination: {tuple(out.shape)} {out.dtype}")
            if y.stride(1) != 1 or uv.stride(1) != 2 or uv.stride(2) != 1 or out.stride(2) != 1:
                raise ValueError("YUV/RGB tensors have unsupported pixel strides")
            self._cuda_kernel.launch(y, uv, out)
            return

        if self._cuda_kernel is not None:
            raise RuntimeError("CUDA YUV converter cannot process CPU planes")
        self._convert_eager(y, uv, out)

    def _convert_eager(self, y: torch.Tensor, uv: torch.Tensor, out: torch.Tensor) -> None:
        H, W = self.height, self.width
        yf = y.to(torch.float32)
        uvf = uv.to(torch.float32).reshape(-1, 2)

        chroma = self._chroma_matrix.mm(uvf.T).reshape(1, 3, H // 2, W // 2)
        chroma_up = torch.nn.functional.interpolate(chroma, scale_factor=2, mode="nearest")

        rgb = chroma_up.squeeze(0).add_(self._offset)
        rgb.add_(yf, alpha=self._luma_scale)

        if self.is_10bit:
            rgb10 = rgb.round_().clamp_(0, 1023).to(torch.int32)
            out.copy_(((rgb10 + self._dither2) >> 2).clamp_(0, 255))
        else:
            out.copy_(rgb.round_().clamp_(0, 255))
