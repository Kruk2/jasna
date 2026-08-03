import ctypes
import logging
import os
import sys
from typing import Iterator

import av
import torch
from av.codec.hwaccel import HWAccel
from av.video.reformatter import ColorRange as AvColorRange, VideoReformatter

from jasna.accelerator import (
    AcceleratorVendor,
    current_stream,
    new_stream,
    stream_context,
    vendor_for_device,
)
from jasna.media import VideoMetadata, resolve_video_start_pts
from jasna.media.yuv_to_rgb import YuvToRgbConverter

log = logging.getLogger(__name__)

CORRUPT_PACKET_TOLERANCE = 10
_libcuda: ctypes.CDLL | None = None

# Decode backend selection (`JASNA_DECODE_BACKEND` overrides the default):
# - "auto":    NVIDIA tries VALI first and falls back to PyAV hwaccel, then PyAV
#              software, when VALI cannot open or decode the first frame. AMD
#              keeps its AMF -> software escalation.
# - "vali":    VALI only; any failure raises (NVIDIA only).
# - "pyav-hw": skip VALI, use the PyAV hwaccel path with its software fallback.
# - "pyav-sw": force FFmpeg software decoding with GPU upload on every vendor.
DECODE_BACKEND = "auto"
DECODE_BACKEND_ENV = "JASNA_DECODE_BACKEND"
_DECODE_BACKENDS = ("auto", "vali", "pyav-hw", "pyav-sw")

# PyAV's avcodec_find_decoder returns libdav1d for AV1, which carries no NVDEC
# hwaccel config, so av.open silently decodes AV1 in software. Force the native
# FFmpeg av1 decoder, which does carry the CUDA hwaccel config. Keyed by codec
# name whose default PyAV decoder lacks NVDEC (only AV1 today).
_NVDEC_DECODER_OVERRIDES = {"av1": "av1"}
_NVDEC_MIN_CODED_SIZE = {"av1": (128, 128)}


class VideoDecodeError(RuntimeError):
    pass


def _decode_backend() -> str:
    backend = os.environ.get(DECODE_BACKEND_ENV, DECODE_BACKEND)
    if backend not in _DECODE_BACKENDS:
        raise ValueError(
            f"Unknown decode backend {backend!r} from {DECODE_BACKEND_ENV}/DECODE_BACKEND, "
            f"expected {_DECODE_BACKENDS}"
        )
    return backend


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
    # cuStreamCreate needs a current CUDA context; threads other than the one
    # torch initialized on have none until torch binds them to the device.
    torch.cuda.set_device(device)
    handle = ctypes.c_void_p()
    result = _cuda_driver().cuStreamCreate(ctypes.byref(handle), 0)
    if result != 0 or handle.value is None:
        raise RuntimeError(f"cuStreamCreate failed (CUDA error {result})")
    return handle.value, torch.cuda.ExternalStream(handle.value, device=device)


class _ValiFrameSource:
    """NVDEC decoding through python_vali, converted by the shared CUDA kernel.

    Construction is the whole VALI viability check: it opens the decoder and
    decodes the first frame, so any container/codec VALI cannot handle raises
    here, before the caller commits to this backend. Corrupt packets after that
    are skipped inside the VALI fork (up to its internal consecutive-packet
    tolerance) and surface only as recovery messages; exceeding the tolerance
    raises VideoDecodeError like the PyAV path.
    """

    def __init__(
        self,
        file: str,
        batch_size: int,
        device: torch.device,
        metadata: VideoMetadata,
        frame_stride: int,
    ):
        import python_vali as vali

        if not hasattr(vali.PyDecoder, "DecodeSingleSurfaceAsyncDetailed"):
            raise VideoDecodeError(
                "python_vali build lacks DecodeSingleSurfaceAsyncDetailed; "
                "the corruption-tolerant fork is required"
            )
        self._vali = vali
        self.file = file
        self.batch_size = batch_size
        self.device = device
        self.metadata = metadata
        self.frame_stride = frame_stride
        self.decoder = None
        self.surface = None
        self._raw_stream, self.stream = _create_blocking_cuda_stream(device)
        try:
            self.decoder = vali.PyDecoder(
                file,
                {},
                gpu_id=device.index or 0,
                stream=self.stream.cuda_stream,
            )
            if not self.decoder.IsAccelerated:
                raise VideoDecodeError(f"VALI decoder for {file} is not hardware accelerated")
            fmt = self.decoder.Format
            if fmt == vali.PixelFormat.NV12:
                self.is_10bit = False
            elif fmt == vali.PixelFormat.P10:
                self.is_10bit = True
            else:
                raise VideoDecodeError(
                    f"VALI surface format {fmt.name} of {file} has no CUDA conversion"
                )
            self.width = self.decoder.Width
            self.height = self.decoder.Height
            self.surface = vali.Surface.Make(
                format=fmt, width=self.width, height=self.height, gpu_id=device.index or 0
            )
            plane = self.surface.Planes[0]
            self._pitch = plane.Pitch
            self._y_ptr = plane.GpuMem
            self._uv_ptr = plane.GpuMem + plane.Pitch * self.height
            self._pkt_data = vali.PacketData()
            self._first_pts = self._decode_next(None)
        except BaseException:
            self.close()
            raise

    def _decode_next(self, seek_ctx) -> int | None:
        success, details = self.decoder.DecodeSingleSurfaceAsyncDetailed(
            self.surface, self._pkt_data, seek_ctx
        )
        if success:
            if details.message:
                log.warning("Recovered video corruption in %s: %s", self.file, details.message)
            return self._pkt_data.pts
        if details.info == self._vali.TaskExecInfo.END_OF_STREAM:
            return None
        raise VideoDecodeError(
            f"Failed to decode {self.file} ({details.info.name}): "
            f"{details.message or details.info.name}"
        )

    def frames(self, seek_ts: float | None) -> Iterator[tuple[torch.Tensor, list[int]]]:
        converter = YuvToRgbConverter(
            self.height,
            self.width,
            self.metadata.color_space,
            self.metadata.color_range == AvColorRange.JPEG,
            self.is_10bit,
            self.device,
        )
        # VALI seeks in absolute stream seconds while the reader contract is
        # seconds past the stream start, so shift by the start offset. VALI's
        # own StartTime is unreliable (wrong scale), use the probed metadata.
        seek_ctx = None
        if seek_ts is not None:
            start_seconds = float(
                resolve_video_start_pts(None, self.metadata.start_pts) * self.metadata.time_base
            )
            seek_ctx = self._vali.SeekContext(seek_ts=seek_ts + start_seconds)
        pending_pts = self._first_pts if seek_ctx is None else None
        exhausted = seek_ctx is None and self._first_pts is None
        self._first_pts = None
        frame_index = 0
        while not exhausted:
            batch = torch.empty(
                (self.batch_size, 3, self.height, self.width),
                device=self.device,
                dtype=torch.uint8,
            )
            pts: list[int] = []
            while len(pts) < self.batch_size:
                if pending_pts is not None:
                    frame_pts = pending_pts
                    pending_pts = None
                else:
                    frame_pts = self._decode_next(seek_ctx)
                    seek_ctx = None
                    if frame_pts is None:
                        exhausted = True
                        break
                selected = frame_index % self.frame_stride == 0
                frame_index += 1
                if not selected:
                    continue
                converter.convert_surface_into(
                    self._y_ptr,
                    self._uv_ptr,
                    self._pitch,
                    batch[len(pts)],
                    self.stream.cuda_stream,
                )
                pts.append(frame_pts)
            if pts:
                self.stream.synchronize()
                yield batch[: len(pts)], pts

    def close(self) -> None:
        self.decoder = None
        self.surface = None
        if self._raw_stream is None:
            return
        raw_stream, self._raw_stream = self._raw_stream, None
        result = _cuda_driver().cuStreamDestroy(ctypes.c_void_p(raw_stream))
        if result != 0:
            raise RuntimeError(f"cuStreamDestroy failed (CUDA error {result})")


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
        self.vendor = vendor_for_device(device)
        self._decoder_ctx = None
        self._amd_hardware_decode = False
        self._vali_source: _ValiFrameSource | None = None
        self._software_only = False

    def __enter__(self):
        self._decoder_ctx = None
        self._amd_hardware_decode = False
        self._vali_source = None
        current_stream(self.device)
        backend = _decode_backend()
        if backend in ("auto", "vali"):
            if self.vendor is AcceleratorVendor.NVIDIA:
                try:
                    self._vali_source = _ValiFrameSource(
                        self.file,
                        self.batch_size,
                        self.device,
                        self.metadata,
                        self.frame_stride,
                    )
                except (ImportError, RuntimeError, ValueError, VideoDecodeError) as exc:
                    if backend == "vali":
                        raise VideoDecodeError(f"VALI cannot decode {self.file}: {exc}") from exc
                    log.warning(
                        "VALI cannot decode %s (codec %s): %s; falling back to PyAV",
                        self.file,
                        self.metadata.codec_name,
                        exc,
                    )
                else:
                    self.width = self._vali_source.width
                    self.height = self._vali_source.height
                    log.info("Using VALI NVDEC decoder for %s", self.file)
                    return self
            elif backend == "vali":
                raise VideoDecodeError("The VALI decode backend requires an NVIDIA device")
        software_only = backend == "pyav-sw"
        self._software_only = software_only
        try:
            if not software_only and self.vendor is AcceleratorVendor.NVIDIA:
                hwaccel = HWAccel(
                    "cuda",
                    device=str(self.device.index or 0),
                    allow_software_fallback=True,
                    is_hw_owned=True,
                )
                # Reuse torch's current primary context without changing its
                # scheduling flags.
                hwaccel.options["primary_ctx"] = "0"
                hwaccel.options["current_ctx"] = "1"
                self.container = av.open(self.file, hwaccel=hwaccel)
            else:
                self.container = av.open(self.file)
            self.video_stream = self.container.streams.video[0]
        except av.FFmpegError as e:
            raise VideoDecodeError(f"Failed to open {self.file}: {e}") from e

        ctx = self.video_stream.codec_context
        if software_only:
            ctx.thread_type = "AUTO"
        elif self.vendor is AcceleratorVendor.AMD:
            self._setup_amf_decoder(ctx)
        elif not ctx.is_hwaccel:
            if self.vendor is AcceleratorVendor.NVIDIA:
                self._setup_nvdec_decoder(ctx)
            else:
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

    @property
    def start_pts(self) -> int:
        if self._vali_source is not None:
            return resolve_video_start_pts(None, self.metadata.start_pts)
        return resolve_video_start_pts(
            self.video_stream.start_time,
            self.metadata.start_pts,
        )

    def _setup_amf_decoder(self, source_ctx) -> None:
        decoder_name = {
            "h264": "h264_amf",
            "hevc": "hevc_amf",
            "av1": "av1_amf",
        }.get(str(source_ctx.name).lower())
        if decoder_name is None:
            source_ctx.thread_type = "AUTO"
            return
        try:
            hwaccel = HWAccel(
                "amf",
                device=str(self.device.index or 0),
                allow_software_fallback=False,
                is_hw_owned=False,
            )
            decoder = av.CodecContext.create(
                decoder_name,
                "r",
                hwaccel=hwaccel,
            )
            decoder.extradata = source_ctx.extradata
            decoder.width = source_ctx.width
            decoder.height = source_ctx.height
            # PyAV 18 rejects assigning time_base on a decoder ("Cannot access
            # 'time_base' as a decoder"); decoders take timing from packets.
            decoder.framerate = source_ctx.framerate
            decoder.sample_aspect_ratio = source_ctx.sample_aspect_ratio
            decoder.open(strict=False)
            self._decoder_ctx = decoder
            self._amd_hardware_decode = True
            log.info("Using AMF hardware decoder %s for %s", decoder_name, self.file)
        except (ValueError, av.FFmpegError, RuntimeError) as exc:
            source_ctx.thread_type = "AUTO"
            log.warning(
                "AMF cannot decode %s (codec %s): %s; using FFmpeg software "
                "decoding and uploading frames to ROCm",
                self.file,
                self.metadata.codec_name,
                exc,
            )

    def _setup_nvdec_decoder(self, source_ctx) -> None:
        decoder_name = _NVDEC_DECODER_OVERRIDES.get(str(self.metadata.codec_name).lower())
        if decoder_name is None:
            source_ctx.thread_type = "AUTO"
            return
        min_width, min_height = _NVDEC_MIN_CODED_SIZE[decoder_name]
        if source_ctx.width < min_width or source_ctx.height < min_height:
            source_ctx.thread_type = "AUTO"
            log.info(
                "Skipping NVDEC decoder %s for %s: %dx%d is below its %dx%d minimum",
                decoder_name,
                self.file,
                source_ctx.width,
                source_ctx.height,
                min_width,
                min_height,
            )
            return
        hwaccel = HWAccel(
            "cuda",
            device=str(self.device.index or 0),
            allow_software_fallback=True,
            is_hw_owned=True,
        )
        hwaccel.options["primary_ctx"] = "0"
        hwaccel.options["current_ctx"] = "1"
        try:
            decoder = av.CodecContext.create(
                decoder_name,
                "r",
                hwaccel=hwaccel,
            )
            decoder.extradata = source_ctx.extradata
            decoder.width = source_ctx.width
            decoder.height = source_ctx.height
            if source_ctx.pix_fmt is not None:
                decoder.pix_fmt = source_ctx.pix_fmt
            decoder.profile = source_ctx.profile
            decoder.open(strict=False)
            self._decoder_ctx = decoder
            log.info("Using NVDEC decoder %s for %s", decoder_name, self.file)
        except (ValueError, av.FFmpegError, RuntimeError) as exc:
            source_ctx.thread_type = "AUTO"
            log.warning(
                "NVDEC decoder %s unavailable for %s (codec %s): %s; using FFmpeg "
                "software decoding and uploading frames to CUDA",
                decoder_name,
                self.file,
                self.metadata.codec_name,
                exc,
            )

    def __exit__(self, exc_type, exc_value, traceback):
        if self._vali_source is not None:
            source, self._vali_source = self._vali_source, None
            source.close()
            return
        self.container.close()
        self._decoder_ctx = None
        if self._raw_stream is None:
            return
        result = _cuda_driver().cuStreamDestroy(ctypes.c_void_p(self._raw_stream))
        if result != 0 and exc_type is None:
            raise RuntimeError(f"cuStreamDestroy failed (CUDA error {result})")

    def _decode_packet(self, packet, consecutive_errors: int) -> tuple[list, int]:
        try:
            frames = (
                self._decoder_ctx.decode(packet)
                if getattr(self, "_decoder_ctx", None) is not None
                else packet.decode()
            )
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
            start = resolve_video_start_pts(
                self.video_stream.start_time,
                self.metadata.start_pts,
            )
            target_pts = start + round(seek_ts / self.video_stream.time_base)
            self.container.seek(target_pts, stream=self.video_stream, backward=True)
            if self._decoder_ctx is not None:
                self._decoder_ctx.flush_buffers()

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
        # With seek_ts, strided selection re-anchors at the first decoded frame
        # after the seek instead of the start of the file: sample phase is only
        # stable relative to the seek target.
        if self._vali_source is not None:
            yield from self._vali_source.frames(seek_ts)
            return
        # The first decoded frame's format is the final backend decision: a codec
        # can advertise a CUDA config and still fall back to software when
        # hardware initialization rejects a profile or pixel format. Dispatch
        # once here so neither per-frame loop carries a backend branch.
        decoded = self._selected_frames(self._decoded_frames(seek_ts))
        group = self._read_group(decoded)
        if not group:
            return
        vendor = getattr(self, "vendor", AcceleratorVendor.NVIDIA)
        if (
            vendor is AcceleratorVendor.NVIDIA
            and group[0].format.name == "cuda"
        ):
            backend = self._frames_hardware(decoded, group)
        else:
            if vendor is AcceleratorVendor.NVIDIA and not self._software_only:
                log.warning(
                    "CUDA/NVDEC cannot decode %s (codec %s, %s); using FFmpeg "
                    "software decoding and uploading frames to CUDA",
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

        # Pinned host batch is shared. Device staging is gated on
        # AcceleratorVendor.AMD (not "if not NVIDIA") so Intel/CPU keep the
        # historical single-staging private-stream fallback unless AMD-specific.
        # NVIDIA: one staging frame on a private stream (H2D+convert ordered so
        # the next overwrite starts only after the prior kernel consumed it).
        # AMD (issue #252): batch device YUV on current_stream. Isolated D1 was
        # clean; residual glitches are pipeline-contended (Phase 0). Per-frame
        # staging reuse under multi-thread load can race; batch staging removes
        # overwrite races. Phase 4 on gfx1201 cleared full-pipeline static.
        # Extra VRAM ≈ (batch_size - 1) × (1.5 × H × W × bytes) — a few MiB at
        # batch 4 @ 1080p8.
        pinned = torch.empty((self.batch_size, H + H // 2, W), dtype=dtype, pin_memory=True)
        if self.vendor is AcceleratorVendor.AMD:
            device_yuv = torch.empty(
                (self.batch_size, H + H // 2, W), dtype=dtype, device=self.device
            )
            staging = None
            stream = current_stream(self.device)
        else:
            device_yuv = None
            staging = torch.empty((H + H // 2, W), dtype=dtype, device=self.device)
            stream = new_stream(self.device)

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

            with stream_context(stream):
                # Shared copy/convert loop: AMD uses a per-frame plane from the
                # batch device buffer; NVIDIA reuses the single staging frame
                # (H2D + convert ordered on the private stream so the next
                # overwrite starts only after the prior kernel consumed it).
                for i in range(len(group)):
                    plane = staging if staging is not None else device_yuv[i]
                    plane.copy_(pinned[i], non_blocking=True)
                    converter.convert_into(
                        plane[:H],
                        plane[H:].view(H // 2, W // 2, 2),
                        batch[i],
                    )

            next_group = self._read_group(decoded)
            # Sync before yield so other pipeline threads never see in-flight planes.
            stream.synchronize()
            group = next_group
            yield batch, pts
