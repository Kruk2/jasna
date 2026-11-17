from __future__ import annotations

import heapq
import logging
import queue
import threading
from collections import deque
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Mapping

import av
import torch
from av.codec.hwaccel import HWAccel
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.accelerator import (
    AcceleratorVendor,
    current_stream,
    device_name,
    new_event,
    new_stream,
    set_device,
    stream_context,
    vendor_for_device,
)
from jasna.media.cas import GpuCasSharpener
from jasna.media.container_utils import (
    MOV_SUFFIXES,
    output_creation_time,
    with_output_creation_time,
)
from jasna.media.encoder_options import resolve_encoder_options
from jasna.media.lut import CubeLut, GpuLutApplier
from jasna.media.probe import VideoMetadata
from jasna.media.rgb_to_yuv import RgbToYuvConverter
from jasna.media.source_streams import SourceStreamCopier
from jasna.settings_tag import settings_comment

logger = logging.getLogger(__name__)

# ITU-T H.273 matrix, primaries, and transfer-characteristic code points.
_COLOR_TAGS = {
    AvColorspace.ITU709: (1, 1, 1),
    AvColorspace.ITU601: (6, 6, 6),
    AvColorspace.BT2020: (9, 9, 14),  # bt2020nc, bt2020 primaries, bt2020-10 transfer
}
_COLOR_PRIMARIES = {
    "bt709": 1,
    "bt470bg": 5,
    "smpte170m": 6,
    "bt2020": 9,
}
_COLOR_TRANSFERS = {
    "bt709": 1,
    "smpte170m": 6,
    "bt2020-10": 14,
    "smpte2084": 16,
    "arib-std-b67": 18,
}
_COLOR_VARIANTS = {
    (AvColorspace.ITU709, AvColorRange.MPEG): "bt709_limited",
    (AvColorspace.ITU709, AvColorRange.JPEG): "bt709_full",
    (AvColorspace.ITU601, AvColorRange.MPEG): "bt601_limited",
    (AvColorspace.ITU601, AvColorRange.JPEG): "bt601_full",
    (AvColorspace.BT2020, AvColorRange.MPEG): "bt2020_limited",
    (AvColorspace.BT2020, AvColorRange.JPEG): "bt2020_full",
}

_NVENC_PITCH_ALIGNMENT = 16
ENCODE_BUFFER_SIZE = 8


def nvenc_yuv_frame(height: int, width: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """A packed NV12/P010 frame whose rows start on the pitch NVENC accepts."""
    pitch_bytes = -(-width * dtype.itemsize // _NVENC_PITCH_ALIGNMENT) * _NVENC_PITCH_ALIGNMENT
    rows = torch.empty((height + height // 2, pitch_bytes // dtype.itemsize), dtype=dtype, device=device)
    return rows[:, :width]


def _mov_container_options(suffix: str, *, fmp4: bool) -> dict[str, str]:
    if suffix.lower() not in MOV_SUFFIXES:
        return {}
    # A fragmented MP4 writes a sample-free moov up front and one moof+mdat per
    # keyframe, so the growing file stays playable; +faststart instead relocates
    # a single moov at close, leaving the file unreadable until then. The two
    # are mutually exclusive. +delay_moov holds the moov until the first
    # fragment, because the MP4 sample entry for copied AC-3/E-AC-3 needs
    # header fields parsed from the first audio packet.
    flags = "+frag_keyframe+empty_moov+delay_moov" if fmp4 else "+faststart"
    return {"movflags": flags}


class EncodeOrder:
    """Pairs frames, in arrival order, with the smallest PTS still waiting once more than
    ``hold`` frames wait. A repeated PTS moves up by one and handed-out PTS always increase."""

    def __init__(self, *, hold: int, pts_origin: int):
        self._hold = hold
        self._pts_origin = pts_origin
        self._frames: deque = deque()
        self._pts_heap: list[int] = []
        self._pts_waiting: set[int] = set()
        self._last_pts: int | None = None

    def __len__(self) -> int:
        return len(self._frames)

    def push(self, frame, pts: int) -> None:
        pts = int(pts) - self._pts_origin
        while pts in self._pts_waiting:
            pts += 1
        heapq.heappush(self._pts_heap, pts)
        self._frames.append(frame)
        self._pts_waiting.add(pts)

    def pop(self, *, flush: bool) -> tuple[object, int] | None:
        """The next frame and its PTS, or ``None`` while it should wait for more frames."""
        if len(self._frames) <= self._hold and not (flush and self._frames):
            return None
        pts = heapq.heappop(self._pts_heap)
        self._pts_waiting.remove(pts)
        if self._last_pts is not None and pts <= self._last_pts:
            pts = self._last_pts + 1
        self._last_pts = pts
        return self._frames.popleft(), pts


class VideoEncoder:
    def __init__(
        self,
        file: str,
        device: torch.device,
        metadata: VideoMetadata,
        *,
        codec: str,
        encoder_settings: dict[str, object],
        lut: CubeLut | None = None,
        sharpen_strength: float = 0.0,
        output_fps: Fraction | None = None,
        pts_origin: int = 0,
        smart_fragment: bool = False,
        fmp4: bool = False,
        settings: Mapping[str, object] | None = None,
        frame_generation: bool = False,
    ):
        """``settings`` (from ``settings_tag.job_settings``) become the output's comment tag.
        ``frame_generation`` doubles ``output_fps`` and takes PTS in half the source time base
        (``jasna.frame_generation.FrameGenerationWriter``)."""
        self.device = torch.device(device)
        self.vendor = vendor_for_device(self.device)
        if self.vendor not in {AcceleratorVendor.NVIDIA, AcceleratorVendor.AMD}:
            raise RuntimeError(
                f"GPU video encoding is not supported on {self.vendor.value}"
            )
        spec, self.encoder_options = resolve_encoder_options(
            self.vendor, codec, metadata, encoder_settings, smart_fragment=smart_fragment,
            frame_generation=frame_generation,
        )
        color_variant = _COLOR_VARIANTS.get((metadata.color_space, metadata.color_range))
        if color_variant is None:
            raise ValueError(f"Unsupported color space or color range: {metadata.color_space} {metadata.color_range}")
        pixel_format = "p010" if spec.ten_bit else "nv12"
        self.metadata = metadata
        self.output_path = Path(file)
        self.codec = codec
        self.spec = spec
        self.encoder_name = spec.encoder_name
        self.pts_origin = int(pts_origin)
        self.smart_fragment = bool(smart_fragment)
        self.fmp4 = bool(fmp4)
        self.settings = settings
        self.output_fps = Fraction(
            metadata.video_fps_exact if output_fps is None else output_fps
        ) * (2 if frame_generation else 1)
        self.time_base = metadata.time_base / 2 if frame_generation else metadata.time_base

        self._lut_applier = None if lut is None else GpuLutApplier(lut, device)

        self._cas: GpuCasSharpener | None = None
        if sharpen_strength > 0.0:
            self._cas = GpuCasSharpener(
                sharpen_strength, ten_bit=spec.ten_bit, device=self.device
            )

        self._converter = RgbToYuvConverter(f"{pixel_format}_{color_variant}", device=self.device)

        # Only AMD reuses one packed frame (set in __enter__); NVENC still holds
        # its input frame after encode() returns, so NVIDIA allocates per frame.
        self._packed: torch.Tensor | None = None
        self._cas_luma: torch.Tensor | None = None

    def __enter__(self):
        try:
            av.Codec(self.encoder_name, "w")
        except ValueError as exc:  # av.codec.codec.UnknownCodecError
            raise RuntimeError(
                f"Encoder {self.encoder_name} (codec {self.codec}) is not available in the "
                f"bundled FFmpeg libraries: {exc}"
            ) from exc
        self._src = av.open(self.metadata.video_file, metadata_errors="replace")
        in_v = self._src.streams.video[0]

        container_options = _mov_container_options(
            self.output_path.suffix, fmp4=self.fmp4
        )
        self.dst = av.open(str(self.output_path), "w", container_options=container_options)

        stream_kwargs = {
            "rate": self.output_fps,
            "options": dict(self.encoder_options),
        }
        if self.vendor is AcceleratorVendor.AMD:
            stream_kwargs["hwaccel"] = HWAccel(
                "amf",
                device=str(self.device.index or 0),
                allow_software_fallback=False,
                is_hw_owned=False,
            )
            pix_fmt = self.spec.frame_format
        else:
            pix_fmt = "cuda"
        out_v = self.dst.add_stream(self.encoder_name, **stream_kwargs)
        if self.codec == "hevc" and self.output_path.suffix.lower() in MOV_SUFFIXES:
            out_v.codec_tag = "hvc1"
        out_v.width = self.metadata.video_width
        out_v.height = self.metadata.video_height
        out_v.time_base = self.time_base
        ctx = out_v.codec_context
        ctx.time_base = self.time_base
        ctx.framerate = self.output_fps
        ctx.pix_fmt = pix_fmt
        if self.smart_fragment:
            from av.codec.context import Flags

            ctx.flags |= Flags.closed_gop
        if self.metadata.sample_aspect_ratio != 1:
            ctx.sample_aspect_ratio = self.metadata.sample_aspect_ratio
        matrix, primaries, transfer = _COLOR_TAGS[self.metadata.color_space]
        primaries = _COLOR_PRIMARIES.get(self.metadata.color_primaries.lower(), primaries)
        transfer = _COLOR_TRANSFERS.get(self.metadata.color_transfer.lower(), transfer)
        ctx.color_range = int(self.metadata.color_range)
        ctx.colorspace = matrix
        ctx.color_primaries = primaries
        ctx.color_trc = transfer
        creation_time = output_creation_time(datetime.now(timezone.utc))
        out_v.metadata.update(with_output_creation_time(in_v.metadata, creation_time))
        out_v.disposition = in_v.disposition
        self.out_stream = out_v

        comment = (
            None
            if self.settings is None
            else settings_comment(self.settings, encoder=self.encoder_name, options=self.encoder_options)
        )
        self._source_streams = SourceStreamCopier(
            self._src,
            self.dst,
            self.output_path,
            video_index=in_v.index,
            creation_time=creation_time,
            comment=comment,
            smart_fragment=self.smart_fragment,
        )

        # Wrap torch's already-current primary context.  FFmpeg's primary_ctx
        # mode tries to change its scheduling flags and fails once torch has
        # initialized it; current_ctx leaves the context and its flags alone.
        # Keeping conversion and NVENC in one context also avoids a ~500 MiB
        # secondary CUDA context and cross-context scheduling overhead.
        # Conversion runs on its own stream so it overlaps the rest of the
        # pipeline. The worker makes that stream wait for the producer's event
        # and records it on the input frame, and the AMD buffers below live for
        # the whole encode, so no block is recycled while a conversion still
        # reads it (issue #252).
        height = self.metadata.video_height
        width = self.metadata.video_width
        self._cuda_ctx = None
        self._host_yuv = None
        self.stream = new_stream(self.device)
        if self._cas is not None:
            with stream_context(self.stream):
                self._cas_luma = torch.empty((height, width), dtype=self._converter.sample_dtype, device=self.device)
        if self.vendor is AcceleratorVendor.NVIDIA:
            from av.video.frame import CudaContext

            self._cuda_ctx = CudaContext(
                device_id=self.device.index or 0,
                primary_ctx=False,
                current_ctx=True,
                cuda_stream=self.stream.cuda_stream,
            )
        else:
            with stream_context(self.stream):
                self._packed = torch.empty(
                    (height + height // 2, width),
                    dtype=self._converter.sample_dtype,
                    device=self.device,
                )
            self._host_yuv = torch.empty(
                (height + height // 2, width),
                dtype=torch.uint16 if self.spec.ten_bit else torch.uint8,
                pin_memory=True,
            )
        self._order = EncodeOrder(hold=ENCODE_BUFFER_SIZE // 2, pts_origin=self.pts_origin)
        self._video_started = False
        self._options_validated = False
        self._worker_error: Exception | None = None

        self._stop_sentinel = object()
        self._encode_queue: queue.Queue = queue.Queue(maxsize=ENCODE_BUFFER_SIZE)
        self._encode_thread = threading.Thread(target=self._encode_worker, name="VideoEncoderWorker", daemon=True)
        self._encode_thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._close(flush_video=exc_type is None, copy_rest_of_source=exc_type is None)

    def abort(self) -> None:
        """Close an unfinished output after the frames given so far, without the rest of the
        source's audio and subtitles, so its length shows how far the job got."""
        self._close(flush_video=True, copy_rest_of_source=False)

    def _close(self, *, flush_video: bool, copy_rest_of_source: bool) -> None:
        try:
            if flush_video:
                while self._order:
                    self._process_buffer(flush_all=True)
            self._encode_queue.join()
            self._encode_queue.put(self._stop_sentinel)
            self._encode_thread.join()

            if flush_video and self._worker_error is None and self.out_stream.codec_context.is_open:
                for packet in self.out_stream.encode(None):
                    self._mux_video(packet)
                if copy_rest_of_source:
                    self._source_streams.drain()
        finally:
            self.dst.close()
            self._src.close()
        if copy_rest_of_source and self._worker_error is not None:
            raise self._worker_error

    def _encode_worker(self):
        set_device(self.device)

        while True:
            item = self._encode_queue.get()
            try:
                if item is self._stop_sentinel:
                    return
                if self._worker_error is None:
                    frame, pts, ready_event = item
                    self.stream.wait_event(ready_event)
                    frame.record_stream(self.stream)
                    self._encode_frame(frame, pts)
            except Exception as exc:
                self._worker_error = exc
                logger.exception("[encoder-worker] crashed")
            finally:
                self._encode_queue.task_done()

    def _build_encode_item(
        self,
        frame: torch.Tensor,
        pts: int,
    ) -> tuple[torch.Tensor, int, object]:
        producer_stream = current_stream(self.device)
        ready_event = new_event(self.device)
        producer_stream.record_event(ready_event)
        return frame, pts, ready_event

    def _validate_encoder_options(self):
        leftover = dict(self.out_stream.codec_context.options)
        if leftover:
            raise ValueError(f"{self.encoder_name} did not accept encoder option(s): {sorted(leftover)}")
        self._options_validated = True

    def _mux_video(self, packet: av.Packet):
        threshold = (
            float(packet.dts * packet.time_base)
            if packet.dts is not None and packet.time_base is not None
            else None
        )
        try:
            self.dst.mux(packet)
        except av.FFmpegError as exc:
            raise RuntimeError(
                f"Failed to mux {self.codec} video into '{self.output_path.suffix}' output: {exc}"
            ) from exc
        self._video_started = True
        if not self._options_validated:
            self._validate_encoder_options()
        if threshold is not None:
            self._source_streams.pump(threshold)

    def _process_buffer(self, flush_all=False):
        ready = self._order.pop(flush=flush_all)
        if ready is not None:
            self._encode_queue.put(self._build_encode_item(*ready))

    def _encoder_open_error(self, exc: Exception) -> RuntimeError:
        message = (
            f"Failed to open {self.codec} encoder ({self.encoder_name}) for "
            f"'{self.output_path.suffix}' output on {device_name(self.device)}: {exc}"
        )
        if self.codec == "av1":
            backend = "AMF" if self.vendor is AcceleratorVendor.AMD else "NVENC"
            message += (
                f". AV1 {backend} encoding requires a GPU/driver generation "
                "that provides it."
            )
        return RuntimeError(message)

    def _packed_frame(self, height: int, width: int) -> torch.Tensor:
        # NVENC takes the device frame by pointer and still owns it after
        # encode() returns, so NVIDIA hands it a fresh one every time. AMF has
        # no zero-copy path: the AMD branch synchronizes then blocking-copies
        # this buffer into pinned host memory, so one buffer can serve every frame.
        if self._packed is not None:
            return self._packed
        return nvenc_yuv_frame(height, width, self._converter.sample_dtype, self.device)

    def _to_yuv(self, frame: torch.Tensor, height: int) -> torch.Tensor:
        packed = self._packed_frame(height, frame.shape[2])
        if self._cas is None:
            self._converter.convert_into(frame, packed[:height], packed[height:])
            return packed
        # CAS is a 3x3 stencil, so it cannot run in place: the conversion writes
        # luma to a scratch plane and sharpening reads from there into the frame
        # the encoder receives, instead of copying a whole plane back.
        self._converter.convert_into(frame, self._cas_luma, packed[height:])
        self._cas.sharpen_into(self._cas_luma, packed[:height])
        return packed

    def _encode_frame(self, frame: torch.Tensor, pts: int):
        height = self.metadata.video_height
        with stream_context(self.stream):
            if self._lut_applier is not None:
                frame = self._lut_applier.apply(frame)
            packed = self._to_yuv(frame, height)
            if self.vendor is AcceleratorVendor.NVIDIA:
                if self.spec.frame_format == "p010le":
                    planes = [
                        packed[:height].view(torch.uint16),
                        packed[height:].view(torch.uint16),
                    ]
                else:
                    planes = [packed[:height], packed[height:]]
                hw_frame = av.VideoFrame.from_dlpack(
                    planes,
                    format=self.spec.frame_format,
                    stream=self.stream.cuda_stream,
                    cuda_context=self._cuda_ctx,
                )

        if self.vendor is AcceleratorVendor.AMD:
            # AMF reads host planes, and reading them while a non-blocking copy
            # was still in flight corrupted frames (issue #252): finish the
            # conversion, then blocking-copy into pinned memory.
            self.stream.synchronize()
            self._host_yuv.copy_(
                packed.view(torch.uint16) if self.spec.ten_bit else packed,
                non_blocking=False,
            )
            planes = [self._host_yuv[:height], self._host_yuv[height:]]
            hw_frame = av.VideoFrame.from_dlpack(
                planes,
                format=self.spec.frame_format,
            )
        hw_frame.pts = pts
        hw_frame.time_base = self.time_base
        try:
            packets = self.out_stream.encode(hw_frame)
        except av.FFmpegError as exc:
            if not self._video_started:
                raise self._encoder_open_error(exc) from exc
            raise
        for packet in packets:
            self._mux_video(packet)

    def encode(self, frame: torch.Tensor, pts: int):
        if self._worker_error is not None:
            raise self._worker_error
        self._order.push(frame, pts)
        self._process_buffer()
