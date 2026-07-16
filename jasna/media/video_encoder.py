from __future__ import annotations

import heapq
import logging
import queue
import threading
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import av
import torch
from av.video.frame import CudaContext
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.media import SUPPORTED_ENCODER_SETTINGS_BY_CODEC, VideoMetadata, validate_encoder_settings
from jasna.media.audio_utils import needs_audio_reencode
from jasna.media.lut import GpuLutApplier, parse_cube_file
from jasna.media.rgb_to_nv12 import (
    chw_rgb_to_nv12_bt2020_full,
    chw_rgb_to_nv12_bt2020_limited,
    chw_rgb_to_nv12_bt601_full,
    chw_rgb_to_nv12_bt601_limited,
    chw_rgb_to_nv12_bt709_full,
    chw_rgb_to_nv12_bt709_limited,
)
from jasna.media.rgb_to_p010 import (
    chw_rgb_to_p010_bt2020_full,
    chw_rgb_to_p010_bt2020_limited,
    chw_rgb_to_p010_bt601_full,
    chw_rgb_to_p010_bt601_limited,
    chw_rgb_to_p010_bt709_full,
    chw_rgb_to_p010_bt709_limited,
)

av.logging.set_level(logging.ERROR)

logger = logging.getLogger(__name__)

DEFAULT_ENCODER_OPTIONS: dict[str, str] = {
    "preset": "p5",
    "tune": "hq",
    "profile": "main10",
    "rc": "vbr",
    "cq": "25",
    "qmin": "17",
    "qmax": "34",
    "nonref_p": "1",
    "g": "250",
    "temporal-aq": "1",
    "rc-lookahead": "32",
    "lookahead_level": "1",
    "spatial_aq": "1",
    "aq-strength": "8",
    "init_qpI": "17",
    "init_qpP": "17",
    "init_qpB": "17",
    "bf": "4",
    "b_ref_mode": "middle",
}

# lookahead_level breaks avcodec_open2 on h264_nvenc with this lookahead/AQ
# combination (ENOSYS on RTX 5090), so H.264 deliberately omits it.
DEFAULT_H264_ENCODER_OPTIONS: dict[str, str] = {
    "preset": "p5",
    "tune": "hq",
    "profile": "high",
    "rc": "vbr",
    # CQ 24 matched HEVC CQ 25 in representative VMAF comparisons.
    "cq": "24",
    "qmin": "17",
    "qmax": "34",
    "nonref_p": "1",
    "g": "250",
    "temporal-aq": "1",
    "rc-lookahead": "32",
    "spatial_aq": "1",
    "aq-strength": "8",
    "init_qpI": "17",
    "init_qpP": "17",
    "init_qpB": "17",
    "bf": "4",
    "b_ref_mode": "middle",
}

# AV1 target quality uses a 0..63 scale rather than H.264/HEVC's 0..51.
# CQ 32 matched HEVC CQ 25 in representative VMAF/SSIM comparisons. AV1 QP limits use a separate
# 0..255 scale, so the HEVC qmin/qmax/init_qp values must not be copied here.
# No profile: P010 input makes av1_nvenc emit AV1 Main 10-bit on its own.
# av1_nvenc only consumes the hyphenated spatial-aq spelling.
DEFAULT_AV1_ENCODER_OPTIONS: dict[str, str] = {
    "preset": "p5",
    "tune": "hq",
    "rc": "vbr",
    "cq": "32",
    "nonref_p": "1",
    "g": "250",
    "temporal-aq": "1",
    "rc-lookahead": "32",
    "lookahead_level": "1",
    "spatial-aq": "1",
    "aq-strength": "8",
    "bf": "4",
    "b_ref_mode": "middle",
}


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    encoder_name: str
    frame_format: str  # PyAV hardware-frame software format: "nv12" or "p010le"
    default_options: Mapping[str, str]
    ten_bit: bool
    supported_settings: frozenset[str] = field(default_factory=frozenset)


ENCODER_SPECS: dict[str, EncoderSpec] = {
    "hevc": EncoderSpec(
        name="hevc",
        encoder_name="hevc_nvenc",
        frame_format="p010le",
        default_options=MappingProxyType(DEFAULT_ENCODER_OPTIONS),
        ten_bit=True,
        supported_settings=SUPPORTED_ENCODER_SETTINGS_BY_CODEC["hevc"],
    ),
    "h264": EncoderSpec(
        name="h264",
        encoder_name="h264_nvenc",
        frame_format="nv12",
        default_options=MappingProxyType(DEFAULT_H264_ENCODER_OPTIONS),
        ten_bit=False,
        supported_settings=SUPPORTED_ENCODER_SETTINGS_BY_CODEC["h264"],
    ),
    "av1": EncoderSpec(
        name="av1",
        encoder_name="av1_nvenc",
        frame_format="p010le",
        default_options=MappingProxyType(DEFAULT_AV1_ENCODER_OPTIONS),
        ten_bit=True,
        supported_settings=SUPPORTED_ENCODER_SETTINGS_BY_CODEC["av1"],
    ),
}

_CODEC_MAP = {spec.name: spec.encoder_name for spec in ENCODER_SPECS.values()}

# ITU-T H.273 matrix, primaries, and transfer-characteristic code points.
_COLOR_TAGS = {
    AvColorspace.ITU709: (1, 1, 1),
    AvColorspace.ITU601: (6, 6, 6),
    AvColorspace.BT2020: (9, 9, 14),  # bt2020nc, bt2020 primaries, bt2020-10 transfer
}
_COLOR_CONVERTERS = {
    (AvColorspace.ITU709, AvColorRange.MPEG): chw_rgb_to_p010_bt709_limited,
    (AvColorspace.ITU709, AvColorRange.JPEG): chw_rgb_to_p010_bt709_full,
    (AvColorspace.ITU601, AvColorRange.MPEG): chw_rgb_to_p010_bt601_limited,
    (AvColorspace.ITU601, AvColorRange.JPEG): chw_rgb_to_p010_bt601_full,
    (AvColorspace.BT2020, AvColorRange.MPEG): chw_rgb_to_p010_bt2020_limited,
    (AvColorspace.BT2020, AvColorRange.JPEG): chw_rgb_to_p010_bt2020_full,
}
_COLOR_CONVERTERS_NV12 = {
    (AvColorspace.ITU709, AvColorRange.MPEG): chw_rgb_to_nv12_bt709_limited,
    (AvColorspace.ITU709, AvColorRange.JPEG): chw_rgb_to_nv12_bt709_full,
    (AvColorspace.ITU601, AvColorRange.MPEG): chw_rgb_to_nv12_bt601_limited,
    (AvColorspace.ITU601, AvColorRange.JPEG): chw_rgb_to_nv12_bt601_full,
    (AvColorspace.BT2020, AvColorRange.MPEG): chw_rgb_to_nv12_bt2020_limited,
    (AvColorspace.BT2020, AvColorRange.JPEG): chw_rgb_to_nv12_bt2020_full,
}


def _option_value(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


class NvidiaVideoEncoder:
    def __init__(
        self,
        file: str,
        device: torch.device,
        metadata: VideoMetadata,
        *,
        codec: str,
        encoder_settings: dict[str, object],
        lut_path: str | Path | None = None,
        output_fps: Fraction | None = None,
    ):
        if codec not in ENCODER_SPECS:
            raise ValueError(f"Unsupported codec: {codec}")
        spec = ENCODER_SPECS[codec]
        converter_map = _COLOR_CONVERTERS if spec.frame_format == "p010le" else _COLOR_CONVERTERS_NV12
        converter = converter_map.get((metadata.color_space, metadata.color_range))
        if converter is None:
            raise ValueError(f"Unsupported color space or color range: {metadata.color_space} {metadata.color_range}")
        if encoder_settings:
            validate_encoder_settings(encoder_settings, codec=codec)

        self.metadata = metadata
        self.device = device
        self.file = file
        self.output_path = Path(file)
        self.codec = codec
        self.spec = spec
        self.encoder_name = spec.encoder_name
        self.output_fps = Fraction(
            metadata.video_fps_exact if output_fps is None else output_fps
        )

        self._lut_applier: GpuLutApplier | None = None
        if lut_path:
            lut = parse_cube_file(lut_path)
            self._lut_applier = GpuLutApplier(lut, device)

        self._to_yuv = converter

        self.encoder_options = dict(spec.default_options)
        if encoder_settings:
            overrides = {k: _option_value(v) for k, v in encoder_settings.items()}
            # FFmpeg accepts both spellings for HEVC/H.264, but their defaults
            # use the underscore key. Normalize the alias so a user override
            # replaces that default instead of passing two conflicting options.
            if "spatial-aq" in overrides and "spatial_aq" in self.encoder_options:
                overrides["spatial_aq"] = overrides.pop("spatial-aq")
            self.encoder_options.update(overrides)

        self.BUFFER_MAX_SIZE = 8

    def __enter__(self):
        try:
            av.Codec(self.encoder_name, "w")
        except ValueError as exc:  # av.codec.codec.UnknownCodecError
            raise RuntimeError(
                f"Encoder {self.encoder_name} (codec {self.codec}) is not available in the "
                f"bundled FFmpeg libraries: {exc}"
            ) from exc
        self._src = av.open(self.metadata.video_file)

        container_options = {}
        if self.output_path.suffix.lower() in {".mp4", ".mov"}:
            container_options["movflags"] = "+faststart"
        self.dst = av.open(str(self.output_path), "w", container_options=container_options)
        self.dst.metadata.update(self._src.metadata)

        out_v = self.dst.add_stream(
            self.encoder_name,
            rate=self.output_fps,
            options=dict(self.encoder_options),
        )
        out_v.width = self.metadata.video_width
        out_v.height = self.metadata.video_height
        out_v.time_base = self.metadata.time_base
        ctx = out_v.codec_context
        ctx.time_base = self.metadata.time_base
        ctx.framerate = self.output_fps
        ctx.pix_fmt = "cuda"
        if self.metadata.sample_aspect_ratio != 1:
            ctx.sample_aspect_ratio = self.metadata.sample_aspect_ratio
        matrix, primaries, transfer = _COLOR_TAGS[self.metadata.color_space]
        ctx.color_range = int(self.metadata.color_range)
        ctx.colorspace = matrix
        ctx.color_primaries = primaries
        ctx.color_trc = transfer
        self.out_stream = out_v

        self._setup_audio()

        # Wrap torch's already-current primary context.  FFmpeg's primary_ctx
        # mode tries to change its scheduling flags and fails once torch has
        # initialized it; current_ctx leaves the context and its flags alone.
        # Keeping conversion and NVENC in one context also avoids a ~500 MiB
        # secondary CUDA context and cross-context scheduling overhead.
        self._cuda_ctx = CudaContext(
            device_id=self.device.index or 0,
            primary_ctx=False,
            current_ctx=True,
        )
        self.stream = torch.cuda.Stream(self.device)
        self.pts_heap: list[int] = []
        self.frame_buffer: deque = deque()
        self.pts_set: set[int] = set()
        self._video_started = False
        self._options_validated = False
        self._worker_error: Exception | None = None

        self._stop_sentinel = object()
        self._encode_queue: queue.Queue = queue.Queue(maxsize=self.BUFFER_MAX_SIZE)
        self._encode_thread = threading.Thread(target=self._encode_worker, name="NvidiaVideoEncoderWorker", daemon=True)
        self._encode_thread.start()
        return self

    def _setup_audio(self):
        self._audio_pipes: dict[int, tuple[str, object, object]] = {}
        self._audio_backlog: deque = deque()
        self._audio_iter = None
        audio_streams = list(self._src.streams.audio)
        if not audio_streams:
            return
        for in_a in audio_streams:
            if needs_audio_reencode(in_a.codec_context.name, self.output_path.suffix):
                logger.info("re-encoding audio %s -> aac for %s", in_a.codec_context.name, self.output_path.suffix)
                out_a = self.dst.add_stream("aac", rate=in_a.codec_context.sample_rate)
                out_a.codec_context.layout = in_a.codec_context.layout
                out_a.bit_rate = 256_000
                resampler = av.AudioResampler(
                    format="fltp",
                    layout=in_a.codec_context.layout,
                    rate=in_a.codec_context.sample_rate,
                )
                self._audio_pipes[in_a.index] = ("transcode", out_a, resampler)
            else:
                out_a = self.dst.add_stream_from_template(in_a)
                self._audio_pipes[in_a.index] = ("copy", out_a, None)
            # add_stream_from_template copies neither of these
            out_a.metadata.update(in_a.metadata)
            out_a.disposition = in_a.disposition
        self._audio_iter = self._src.demux(audio_streams)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is None:
                while self.frame_buffer:
                    self._process_buffer(flush_all=True)
            self._encode_queue.join()
            self._encode_queue.put(self._stop_sentinel)
            self._encode_thread.join()

            if exc_type is None and self._worker_error is None and self.out_stream.codec_context.is_open:
                for packet in self.out_stream.encode(None):
                    self._mux_video(packet)
                self._drain_audio()
        finally:
            self.dst.close()
            self._src.close()
        if exc_type is None and self._worker_error is not None:
            raise self._worker_error

    def _encode_worker(self):
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        while True:
            item = self._encode_queue.get()
            try:
                if item is self._stop_sentinel:
                    return
                if self._worker_error is None:
                    frame, pts, ready_event = item
                    self._handle_encode_item(frame, pts, ready_event)
            except Exception as exc:
                self._worker_error = exc
                logger.exception("[encoder-worker] crashed")
            finally:
                self._encode_queue.task_done()

    def _build_encode_item(self, frame: torch.Tensor, pts: int) -> tuple[torch.Tensor, int, torch.cuda.Event]:
        producer_stream = torch.cuda.current_stream(self.device)
        ready_event = torch.cuda.Event()
        producer_stream.record_event(ready_event)
        return frame, pts, ready_event

    def _handle_encode_item(self, frame: torch.Tensor, pts: int, ready_event: torch.cuda.Event) -> None:
        self.stream.wait_event(ready_event)
        frame.record_stream(self.stream)
        self._encode_frame(frame, pts)

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
        if not self._video_started:
            self._video_started = True
        if not self._options_validated:
            self._validate_encoder_options()
        if threshold is not None:
            self._pump_audio(threshold)

    def _produce_audio_packets(self, in_packet) -> list:
        kind, out_a, resampler = self._audio_pipes[in_packet.stream.index]
        if kind == "copy":
            if in_packet.dts is None and in_packet.pts is None:
                return []
            in_packet.stream = out_a
            return [in_packet]
        out_packets = []
        for aframe in in_packet.decode():
            for rframe in resampler.resample(aframe):
                out_packets.extend(out_a.encode(rframe))
        return out_packets

    def _pump_audio(self, upto_seconds: float | None):
        if self._audio_iter is None:
            return
        while True:
            if self._audio_backlog:
                packet = self._audio_backlog[0]
                ts = packet.dts if packet.dts is not None else packet.pts
                if (
                    upto_seconds is not None
                    and ts is not None
                    and float(ts * packet.time_base) > upto_seconds
                ):
                    return
                self._audio_backlog.popleft()
                self.dst.mux(packet)
                continue
            in_packet = next(self._audio_iter, None)
            if in_packet is None:
                self._audio_iter = None
                return
            self._audio_backlog.extend(self._produce_audio_packets(in_packet))

    def _drain_audio(self):
        self._pump_audio(None)
        for kind, out_a, resampler in self._audio_pipes.values():
            if kind != "transcode":
                continue
            packets = []
            for rframe in resampler.resample(None):
                packets.extend(out_a.encode(rframe))
            packets.extend(out_a.encode(None))
            for packet in packets:
                self.dst.mux(packet)

    def _process_buffer(self, flush_all=False):
        if len(self.frame_buffer) > (self.BUFFER_MAX_SIZE // 2) or (flush_all and self.frame_buffer):
            frame_to_encode = self.frame_buffer.popleft()
            pts_to_assign = heapq.heappop(self.pts_heap)
            self.pts_set.remove(pts_to_assign)
            self._encode_queue.put(self._build_encode_item(frame_to_encode, pts_to_assign))

    def _encoder_open_error(self, exc: Exception) -> RuntimeError:
        try:
            gpu = torch.cuda.get_device_name(self.device)
        except Exception:
            gpu = str(self.device)
        message = (
            f"Failed to open {self.codec} encoder ({self.encoder_name}) for "
            f"'{self.output_path.suffix}' output on {gpu}: {exc}"
        )
        if self.codec == "av1":
            message += ". AV1 NVENC encoding requires a GPU/driver generation that provides it."
        return RuntimeError(message)

    def _encode_frame(self, frame: torch.Tensor, pts: int):
        with torch.cuda.stream(self.stream):
            if self._lut_applier is not None:
                frame = self._lut_applier.apply(frame)
            packed = self._to_yuv(frame)

        height = self.metadata.video_height
        # NVENC consumes these pointers asynchronously, so finish the conversion
        # before constructing the hardware frame on the shared CUDA context.
        self.stream.synchronize()
        if self.spec.frame_format == "p010le":
            planes = [packed[:height].view(torch.uint16), packed[height:].view(torch.uint16)]
        else:
            planes = [packed[:height], packed[height:]]
        hw_frame = av.VideoFrame.from_dlpack(
            planes,
            format=self.spec.frame_format,
            primary_ctx=False,
            cuda_context=self._cuda_ctx,
            current_ctx=True,
        )
        hw_frame.pts = pts
        hw_frame.time_base = self.metadata.time_base
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
        while pts in self.pts_set:
            pts += 1
        heapq.heappush(self.pts_heap, pts)
        self.frame_buffer.append(frame)
        self.pts_set.add(pts)
        self._process_buffer()
