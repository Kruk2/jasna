import ctypes

import torch

from av.video.reformatter import Colorspace as AvColorspace

from jasna.accelerator import is_nvidia_device
from jasna.media.cuda_kernel import check_cuda, cuda_driver, resolve_function
from jasna.media.hip_kernel import (
    color_code_object_name,
    hip_color_kernels_enabled,
    launch_kernel as launch_hip_kernel,
    resolve_function as resolve_hip_function,
)

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


_CUDA_CONVERSION_BATCH = 8
_FATBIN = "yuv_to_rgb.fatbin"

# The eager AMD path used to keep four full-resolution working tensors for
# every reader. That is reasonable through 4K, but one 8K P010 converter then
# owns roughly 1 GiB before its packed/RGB batch or any decoder surfaces are
# counted. Keep at most a 4K-sized row tile and reuse it for the whole frame.
# 8K therefore takes four tiles while 4K retains the established single-tile
# path and its launch count.
_EAGER_SCRATCH_MAX_PIXELS = 3840 * 2160


def _eager_scratch_height(height: int, width: int) -> int:
    """Return an even, Bayer-aligned reusable row-tile height."""

    if height * width <= _EAGER_SCRATCH_MAX_PIXELS:
        return height
    rows = max(8, _EAGER_SCRATCH_MAX_PIXELS // width)
    rows -= rows % 8
    return min(height, max(8, rows))


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
        if self._function is None:
            self._function = resolve_function(_FATBIN, self.function_name)
        return self._function

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
        check_cuda(
            cuda_driver().cuLaunchKernel(
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


class _HipYuvKernel:
    """Single-frame Linux AMD launcher for the AOT colour kernel."""

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
        if self._function is None:
            self._function = resolve_hip_function(
                color_code_object_name("yuv_to_rgb"), self.function_name
            )
        return self._function

    def launch(
        self,
        y: torch.Tensor,
        uv: torch.Tensor,
        out: torch.Tensor,
        stream: int | None = None,
    ) -> None:
        function = self._resolve()
        values = self._values
        values[0].value = y.data_ptr()
        values[_CUDA_CONVERSION_BATCH].value = uv.data_ptr()
        for index in range(1, _CUDA_CONVERSION_BATCH):
            values[index].value = 0
            values[_CUDA_CONVERSION_BATCH + index].value = 0
        base = 2 * _CUDA_CONVERSION_BATCH
        values[base].value = y.stride(0)
        values[base + 1].value = uv.stride(0)
        values[base + 2].value = out.data_ptr()
        values[base + 3].value = 0
        values[base + 4].value = out.stride(-3)
        values[base + 5].value = out.stride(-2)
        values[base + 6].value = 1
        values[base + 7].value = out.shape[-2]
        values[base + 8].value = out.shape[-1]
        threads = 256
        pixels = out.shape[-2] * out.shape[-1]
        if stream is None:
            stream = int(torch.cuda.current_stream(out.device).cuda_stream)
        launch_hip_kernel(
            function,
            grid=((pixels + threads - 1) // threads, 1, 1),
            block=(threads, 1, 1),
            stream=int(stream),
            params=self._params,
            operation=f"hipModuleLaunchKernel({self.function_name})",
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
        self._hip_kernel = None
        if is_nvidia_device(device):
            bits = 10 if is_10bit else 8
            value_range = "full" if full_range else "limited"
            self._cuda_kernel = _CudaYuvKernel(f"yuv{bits}_{name}_{value_range}")
            return
        if hip_color_kernels_enabled(device):
            bits = 10 if is_10bit else 8
            value_range = "full" if full_range else "limited"
            self._hip_kernel = _HipYuvKernel(f"yuv{bits}_{name}_{value_range}")
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
        self._chroma_matrix = [[value / raw_div for value in row] for row in chroma_matrix]
        self._offset = [
            -luma_offset * luma_scale - a * chroma_center * chroma_scale,
            -luma_offset * luma_scale + (b + c) * chroma_center * chroma_scale,
            -luma_offset * luma_scale - d * chroma_center * chroma_scale,
        ]
        # See jasna/media/yuv_scratch.py for why the eager path allocates its
        # working set once instead of per frame.
        self._scratch_height = _eager_scratch_height(height, width)
        scratch_height = self._scratch_height
        self._rgb = torch.empty(
            (3, scratch_height, width), dtype=torch.float32, device=device
        )
        self._chroma = torch.empty(
            (3, scratch_height // 2, width // 2),
            dtype=torch.float32,
            device=device,
        )
        self._codes = (
            torch.empty(
                (3, scratch_height, width), dtype=torch.int32, device=device
            )
            if is_10bit
            else None
        )

        if is_10bit:
            bayer = torch.tensor(_BAYER8, device=device, dtype=torch.float32)
            bayer = (bayer + 0.5) / 64.0
            y_mod8 = torch.arange(scratch_height, device=device) & 7
            x_mod8 = torch.arange(width, device=device) & 7
            t = bayer[y_mod8][:, x_mod8].unsqueeze(0)
            self._dither2 = torch.floor(t * 4.0).to(torch.int32)

    def convert(self, y: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        out = torch.empty((3, self.height, self.width), device=y.device, dtype=torch.uint8)
        self.convert_into(y, uv, out)
        return out

    @property
    def uses_kernel(self) -> bool:
        return self._cuda_kernel is not None or self._hip_kernel is not None

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

    def convert_surface_into(
        self,
        y_ptr: int,
        uv_ptr: int,
        pitch: int,
        out: torch.Tensor,
        stream: int | None = None,
    ) -> None:
        """Convert one NV12/P010 device surface given raw plane pointers.

        Both planes must share ``pitch`` (in bytes), as in a single contiguous
        NVDEC surface with the chroma plane below the luma plane.
        """
        if self._cuda_kernel is None:
            raise RuntimeError("CUDA surface conversion requires a CUDA converter")
        bytes_per_sample = 2 if self.is_10bit else 1
        if pitch % bytes_per_sample:
            raise ValueError("YUV plane pitch is not aligned to its sample size")
        if pitch < self.width * bytes_per_sample:
            raise ValueError("YUV plane pitch is smaller than the visible width")
        if out.shape != (3, self.height, self.width) or out.dtype != torch.uint8:
            raise ValueError(f"Unexpected RGB destination: {tuple(out.shape)} {out.dtype}")
        if not out.is_cuda or out.stride(2) != 1:
            raise ValueError("RGB destination must be a CUDA tensor with contiguous pixels")
        stride = pitch // bytes_per_sample
        self._cuda_kernel.launch_ptr(y_ptr, stride, uv_ptr, stride, out, stream)

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
            kernel = self._cuda_kernel or self._hip_kernel
            if kernel is None:
                # AMD/ROCm path: the coefficient tensors already live on the
                # device, so the eager math runs there directly.
                self._convert_eager(y, uv, out)
                return
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
            kernel.launch(y, uv, out)
            return

        if self._cuda_kernel is not None:
            raise RuntimeError("CUDA YUV converter cannot process CPU planes")
        self._convert_eager(y, uv, out)

    def _convert_eager(self, y: torch.Tensor, uv: torch.Tensor, out: torch.Tensor) -> None:
        H, W = self.height, self.width
        tile_rows = self._scratch_height
        for top in range(0, H, tile_rows):
            bottom = min(H, top + tile_rows)
            rows = bottom - top
            half_rows = rows // 2
            uv_tile = uv[top // 2 : bottom // 2]
            u, v = uv_tile[..., 0], uv_tile[..., 1]

            chroma = self._chroma[:, :half_rows]
            for plane, (cu, cv) in enumerate(self._chroma_matrix):
                torch.mul(u, cu, out=chroma[plane])
                chroma[plane].add_(v, alpha=cv)

            # Nearest 2x upsample: broadcasting a copy into the split view of
            # the reusable tile replaces interpolate() and its allocation.
            rgb = self._rgb[:, :rows]
            rgb.view(3, half_rows, 2, W // 2, 2).copy_(
                chroma.unsqueeze(2).unsqueeze(4)
            )
            for plane, offset in enumerate(self._offset):
                rgb[plane].add_(offset)
            rgb.add_(y[top:bottom], alpha=self._luma_scale)

            out_tile = out[:, top:bottom]
            if self.is_10bit:
                codes = self._codes[:, :rows]
                codes.copy_(rgb.round_().clamp_(0, 1023))
                # Tiled heights are multiples of the 8-row Bayer period, so
                # every tile starts at the same phase as the full-frame path.
                codes.add_(self._dither2[:, :rows]).bitwise_right_shift_(2).clamp_(
                    0, 255
                )
                out_tile.copy_(codes)
            else:
                out_tile.copy_(rgb.round_().clamp_(0, 255))
