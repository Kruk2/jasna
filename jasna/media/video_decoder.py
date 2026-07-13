import ctypes
import logging
import sys
from typing import Iterator

import av
import torch
from av.codec.hwaccel import HWAccel
from av.video.reformatter import ColorRange as AvColorRange

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
    def __init__(self, file: str, batch_size: int, device: torch.device, metadata: VideoMetadata):
        self.device = device
        self.file = file
        self.batch_size = batch_size
        self.metadata = metadata

    def __enter__(self):
        # Make torch's primary CUDA context current on this worker thread before
        # asking FFmpeg to reuse it.
        torch.cuda.current_stream(self.device)
        hwaccel = HWAccel(
            "cuda",
            device=str(self.device.index or 0),
            allow_software_fallback=False,
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
        self.width = ctx.width
        self.height = ctx.height
        full_range = (
            ctx.color_range == int(AvColorRange.JPEG)
            or self.metadata.color_range == AvColorRange.JPEG
        )
        self._converter = YuvToRgbConverter(
            self.height,
            self.width,
            self.metadata.color_space,
            full_range,
            self.metadata.is_10bit,
            self.device,
        )
        # A blocking stream participates in legacy-default-stream ordering.
        # FFmpeg's CUDA/NVDEC device uses stream 0, so this makes decoded-frame
        # writes happen-before conversion without an explicit cross-library event.
        self._raw_stream, self.stream = _create_blocking_cuda_stream(self.device)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.container.close()
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

    def frames(
        self,
        seek_ts: float | None = None,
    ) -> Iterator[tuple[torch.Tensor, list[int]]]:
        # FFmpeg 8 maps NVDEC output on CUDA stream 0. Conversion runs in a
        # blocking stream in that same context, so legacy-default-stream ordering
        # makes the decoded writes visible before this kernel without a race.
        # Decode one group ahead while conversion runs; keep both groups' frame
        # references alive until conversion is synchronized so their mapped
        # surfaces cannot be recycled underneath queued work.
        decoded = self._decoded_frames(seek_ts)
        group = self._read_group(decoded)
        while group:
            batch = torch.empty(
                (len(group), 3, self.height, self.width), device=self.device, dtype=torch.uint8
            )
            pts = [frame.pts for frame in group]
            with torch.cuda.stream(self.stream):
                self._converter.convert_frames_into(group, batch, self.stream.cuda_stream)

            next_group = self._read_group(decoded)
            self.stream.synchronize()
            group = next_group
            yield batch, pts
