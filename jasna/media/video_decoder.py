import ctypes
import logging
import sys
from typing import Iterator

import av
import torch
from av.codec.hwaccel import HWAccel
from av.video.reformatter import ColorRange as AvColorRange, VideoReformatter

from jasna.media import VideoMetadata
from jasna.media.yuv_to_rgb import YuvToRgbConverter

log = logging.getLogger(__name__)

CORRUPT_PACKET_TOLERANCE = 10
_libcuda: ctypes.CDLL | None = None


class VideoDecodeError(RuntimeError):
    pass


def _cuda_driver() -> ctypes.CDLL:
    global _libcuda
    if _libcuda is None:
        loader = ctypes.WinDLL if sys.platform == "win32" else ctypes.CDLL
        lib = loader("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
        lib.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        lib.cuStreamCreate.restype = ctypes.c_int
        lib.cuStreamDestroy.argtypes = [ctypes.c_void_p]
        lib.cuStreamDestroy.restype = ctypes.c_int
        _libcuda = lib
    return _libcuda


def _create_blocking_cuda_stream(device: torch.device) -> tuple[int, torch.cuda.ExternalStream]:
    handle = ctypes.c_void_p()
    result = _cuda_driver().cuStreamCreate(ctypes.byref(handle), 0)
    if result != 0 or handle.value is None:
        raise RuntimeError(f"cuStreamCreate failed (CUDA error {result})")
    return handle.value, torch.cuda.ExternalStream(handle.value, device=device)


class NvidiaVideoReader:
    def __init__(
        self,
        file: str,
        batch_size: int,
        device: torch.device,
        metadata: VideoMetadata,
        *,
        frame_stride: int = 1,
    ):
        frame_stride = int(frame_stride)
        if frame_stride <= 0:
            raise ValueError("frame_stride must be > 0")
        self.device = device
        self.file = file
        self.batch_size = batch_size
        self.metadata = metadata
        self.frame_stride = frame_stride

    def __enter__(self):
        # Make torch's primary CUDA context current on this worker thread before
        # asking FFmpeg to reuse it.
        torch.cuda.current_stream(self.device)
        hwaccel = HWAccel(
            "cuda",
            device=str(self.device.index or 0),
            allow_software_fallback=True,
            is_hw_owned=True,
        )
        # Reuse the CUDA context already made current by torch without asking
        # FFmpeg to change the active primary context's scheduling flags.
        hwaccel.options["primary_ctx"] = "0"
        hwaccel.options["current_ctx"] = "1"
        try:
            self.container = av.open(self.file, hwaccel=hwaccel)
            self.video_stream = self.container.streams.video[0]
        except av.FFmpegError as e:
            raise VideoDecodeError(f"Failed to open {self.file}: {e}") from e

        ctx = self.video_stream.codec_context
        if not ctx.is_hwaccel:
            # Definite software decode: let FFmpeg pick frame/slice threading.
            # CUDA contexts must keep their default threading configuration.
            ctx.thread_type = "AUTO"
        self.width = ctx.width
        self.height = ctx.height
        self._full_range = (
            ctx.color_range == int(AvColorRange.JPEG)
            or self.metadata.color_range == AvColorRange.JPEG
        )
        self._raw_stream: int | None = None
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.container.close()
        if self._raw_stream is None:
            return
        result = _cuda_driver().cuStreamDestroy(ctypes.c_void_p(self._raw_stream))
        if result != 0 and exc_type is None:
            raise RuntimeError(f"cuStreamDestroy failed (CUDA error {result})")

    def _decode_packet(self, packet, consecutive_errors: int) -> tuple[list, int]:
        try:
            frames = packet.decode()
        except av.error.InvalidDataError as e:
            consecutive_errors += 1
            if consecutive_errors > CORRUPT_PACKET_TOLERANCE:
                raise VideoDecodeError(
                    f"Failed to decode {self.file}: too many consecutive corrupt packets "
                    f"({consecutive_errors}): {e}"
                ) from e
            log.warning("Recovered video corruption in %s: %s", self.file, e)
            return [], consecutive_errors
        except av.FFmpegError as e:
            raise VideoDecodeError(f"Failed to decode {self.file}: {e}") from e
        if frames:
            consecutive_errors = 0
        return frames, consecutive_errors

    def _decoded_frames(self, seek_ts: float | None):
        target_pts = None
        if seek_ts is not None:
            start = self.video_stream.start_time or 0
            target_pts = start + round(seek_ts / self.video_stream.time_base)
            self.container.seek(target_pts, stream=self.video_stream, backward=True)

        consecutive_errors = 0
        for packet in self.container.demux(self.video_stream):
            frames, consecutive_errors = self._decode_packet(packet, consecutive_errors)
            for frame in frames:
                if target_pts is not None and frame.pts is not None and frame.pts < target_pts:
                    continue
                target_pts = None
                yield frame

    def _read_group(self, decoded) -> list:
        group = []
        while len(group) < self.batch_size:
            frame = next(decoded, None)
            if frame is None:
                break
            group.append(frame)
        return group

    def _selected_frames(self, decoded):
        if self.frame_stride == 1:
            yield from decoded
            return
        for frame_index, frame in enumerate(decoded):
            if frame_index % self.frame_stride == 0:
                yield frame

    def frames(
        self,
        seek_ts: float | None = None,
    ) -> Iterator[tuple[torch.Tensor, list[int]]]:
        if seek_ts is not None and self.frame_stride != 1:
            raise ValueError(
                "frame_stride > 1 is not supported with seek_ts because frame selection "
                "must stay anchored to the start of the file"
            )
        # The first decoded frame's format is the final backend decision: a codec
        # can advertise a CUDA config and still fall back to software when
        # hardware initialization rejects a profile or pixel format. Dispatch
        # once here so neither per-frame loop carries a backend branch.
        decoded = self._selected_frames(self._decoded_frames(seek_ts))
        group = self._read_group(decoded)
        if not group:
            return
        if group[0].format.name == "cuda":
            backend = self._frames_hardware(decoded, group)
        else:
            log.warning(
                "CUDA/NVDEC cannot decode %s (codec %s, %s); using FFmpeg software "
                "decoding and uploading frames to CUDA",
                self.file,
                self.metadata.codec_name,
                group[0].format.name,
            )
            backend = self._frames_software(decoded, group)

        # The backend generator now owns the first group. Drop this outer
        # reference before yielding: retaining four 4K P010 NVDEC surfaces here
        # for the reader's lifetime costs about 96 MiB of avoidable VRAM.
        del group
        yield from backend

    def _frames_hardware(self, decoded, group: list) -> Iterator[tuple[torch.Tensor, list[int]]]:
        # FFmpeg 8 maps NVDEC output on CUDA stream 0. Conversion runs in a
        # blocking stream in that same context, so legacy-default-stream ordering
        # makes the decoded writes visible before this kernel without a race.
        # Decode one group ahead while conversion runs; keep both groups' frame
        # references alive until conversion is synchronized so their mapped
        # surfaces cannot be recycled underneath queued work.
        converter = YuvToRgbConverter(
            self.height,
            self.width,
            self.metadata.color_space,
            self._full_range,
            self.metadata.is_10bit,
            self.device,
        )
        if self._raw_stream is None:
            self._raw_stream, self.stream = _create_blocking_cuda_stream(self.device)
        while group:
            batch = torch.empty(
                (len(group), 3, self.height, self.width), device=self.device, dtype=torch.uint8
            )
            pts = [frame.pts for frame in group]
            with torch.cuda.stream(self.stream):
                converter.convert_frames_into(group, batch, self.stream.cuda_stream)

            next_group = self._read_group(decoded)
            self.stream.synchronize()
            group = next_group
            yield batch, pts

    def _frames_software(self, decoded, group: list) -> Iterator[tuple[torch.Tensor, list[int]]]:
        # Normalize CPU frames to the two layouts the CUDA conversion kernel
        # accepts (NV12 for <=8-bit sources, P010 above), keeping the resolved
        # matrix/range identical on both reformat sides so swscale changes only
        # layout/subsampling/depth. The one authoritative YUV->RGB conversion
        # stays in the CUDA kernel.
        depth = max(
            (component.bits for component in group[0].format.components if component.bits),
            default=10 if self.metadata.is_10bit else 8,
        )
        ten_bit = depth > 8
        if depth > 10:
            log.warning(
                "Reducing %d-bit source %s to 10-bit P010 before CUDA upload", depth, self.file
            )
        target_format = "p010le" if ten_bit else "nv12"
        dtype = torch.uint16 if ten_bit else torch.uint8
        bytes_per_sample = 2 if ten_bit else 1

        converter = YuvToRgbConverter(
            self.height,
            self.width,
            self.metadata.color_space,
            self._full_range,
            ten_bit,
            self.device,
        )
        reformatter = VideoReformatter()
        color_range = AvColorRange.JPEG if self._full_range else AvColorRange.MPEG
        H, W = self.height, self.width

        # One packed pinned host batch and one packed device staging frame bound
        # the fallback's extra memory: H2D copies and conversion kernels are
        # ordered on the same stream, so the next H2D overwrite of the staging
        # frame starts only after the prior conversion kernel consumed it.
        pinned = torch.empty((self.batch_size, H + H // 2, W), dtype=dtype, pin_memory=True)
        staging = torch.empty((H + H // 2, W), dtype=dtype, device=self.device)
        stream = torch.cuda.Stream(self.device)

        while group:
            batch = torch.empty((len(group), 3, H, W), device=self.device, dtype=torch.uint8)
            pts = [frame.pts for frame in group]
            for i, frame in enumerate(group):
                try:
                    normalized = reformatter.reformat(
                        frame,
                        width=W,
                        height=H,
                        format=target_format,
                        src_colorspace=self.metadata.color_space,
                        dst_colorspace=self.metadata.color_space,
                        src_color_range=color_range,
                        dst_color_range=color_range,
                    )
                except av.FFmpegError as e:
                    raise VideoDecodeError(f"Failed to decode {self.file}: {e}") from e
                y_plane, uv_plane = normalized.planes
                y = torch.frombuffer(y_plane, dtype=dtype).reshape(
                    H, y_plane.line_size // bytes_per_sample
                )[:, :W]
                uv = torch.frombuffer(uv_plane, dtype=dtype).reshape(
                    H // 2, uv_plane.line_size // bytes_per_sample
                )[:, :W]
                pinned[i, :H].copy_(y)
                pinned[i, H:].copy_(uv)

            with torch.cuda.stream(stream):
                for i in range(len(group)):
                    staging.copy_(pinned[i], non_blocking=True)
                    converter.convert_into(
                        staging[:H], staging[H:].view(H // 2, W // 2, 2), batch[i]
                    )

            next_group = self._read_group(decoded)
            stream.synchronize()
            group = next_group
            yield batch, pts
