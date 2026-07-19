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
from jasna.media import (
    AMF_SUPPORTED_ENCODER_SETTINGS_BY_CODEC,
    SUPPORTED_ENCODER_SETTINGS_BY_CODEC,
    VideoMetadata,
    validate_encoder_settings,
)
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

DEFAULT_AMF_H264_ENCODER_OPTIONS: dict[str, str] = {
    "usage": "high_quality",
    "quality": "quality",
    "rc": "qvbr",
    "qvbr_quality_level": "24",
    "g": "250",
    "preanalysis": "1",
    "vbaq": "1",
    "profile": "high",
}

DEFAULT_AMF_HEVC_ENCODER_OPTIONS: dict[str, str] = {
    "usage": "high_quality",
    "quality": "quality",
    "rc": "qvbr",
    "qvbr_quality_level": "25",
    "g": "250",
    "preanalysis": "1",
    "vbaq": "1",
    "profile": "main10",
    "bitdepth": "10",
}

DEFAULT_AMF_AV1_ENCODER_OPTIONS: dict[str, str] = {
    "usage": "high_quality",
    "quality": "quality",
    "rc": "qvbr",
    "qvbr_quality_level": "32",
    "g": "250",
    "preanalysis": "1",
    "vbaq": "1",
    "profile": "main",
    "bitdepth": "10",
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

AMF_ENCODER_SPECS: dict[str, EncoderSpec] = {
    "hevc": EncoderSpec(
        name="hevc",
        encoder_name="hevc_amf",
        frame_format="p010le",
        default_options=MappingProxyType(DEFAULT_AMF_HEVC_ENCODER_OPTIONS),
        ten_bit=True,
        supported_settings=AMF_SUPPORTED_ENCODER_SETTINGS_BY_CODEC["hevc"],
    ),
    "h264": EncoderSpec(
        name="h264",
        encoder_name="h264_amf",
        frame_format="nv12",
        default_options=MappingProxyType(DEFAULT_AMF_H264_ENCODER_OPTIONS),
        ten_bit=False,
        supported_settings=AMF_SUPPORTED_ENCODER_SETTINGS_BY_CODEC["h264"],
    ),
    "av1": EncoderSpec(
        name="av1",
        encoder_name="av1_amf",
        frame_format="p010le",
        default_options=MappingProxyType(DEFAULT_AMF_AV1_ENCODER_OPTIONS),
        ten_bit=True,
        supported_settings=AMF_SUPPORTED_ENCODER_SETTINGS_BY_CODEC["av1"],
    ),
}

_CODEC_MAP = {spec.name: spec.encoder_name for spec in ENCODER_SPECS.values()}

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

_NVENC_PITCH_ALIGNMENT = 16


def _option_value(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _drop_unsupported_nvenc_overrides(
    codec: str, overrides: dict[str, str], defaults: Mapping[str, str]
) -> None:
    # NVENC rejects these combinations at avcodec_open2, so dropping them with
    # a warning beats failing the whole job.
    if codec == "h264" and "lookahead_level" in overrides:
        overrides.pop("lookahead_level")
        logger.warning("dropping lookahead_level: h264_nvenc fails to open with it")
    if overrides.get("weighted_pred", "0") != "0":
        if codec == "av1":
            overrides.pop("weighted_pred")
            logger.warning("dropping weighted_pred: av1_nvenc does not support it")
        elif overrides.get("bf", defaults.get("bf", "0")) != "0":
            overrides.pop("weighted_pred")
            logger.warning("dropping weighted_pred: NVENC supports it only with bf=0")


def _align_yuv_pitch(packed: torch.Tensor) -> torch.Tensor:
    item_size = packed.element_size()
    if packed.stride(0) * item_size % _NVENC_PITCH_ALIGNMENT == 0:
        return packed

    width = packed.shape[1]
    row_bytes = width * item_size
    aligned_row_bytes = (
        row_bytes + _NVENC_PITCH_ALIGNMENT - 1
    ) // _NVENC_PITCH_ALIGNMENT * _NVENC_PITCH_ALIGNMENT
    pitch_elements = aligned_row_bytes // item_size
    storage = packed.new_empty((packed.shape[0], pitch_elements))
    storage[:, :width].copy_(packed)
    storage[:, width:].zero_()
    return storage[:, :width]


def _amf_host_input(packed: torch.Tensor, *, ten_bit: bool) -> torch.Tensor:
    return packed.view(torch.uint16) if ten_bit else packed


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
        mux_audio: bool = True,
        pts_origin: int = 0,
        match_input_bit_depth: bool = False,
        smart_fragment: bool = False,
    ):
        self.device = torch.device(device)
        self.vendor = vendor_for_device(self.device)
        if self.vendor not in {AcceleratorVendor.NVIDIA, AcceleratorVendor.AMD}:
            raise RuntimeError(
                f"GPU video encoding is not supported on {self.vendor.value}"
            )
        specs = (
            AMF_ENCODER_SPECS
            if self.vendor is AcceleratorVendor.AMD
            else ENCODER_SPECS
        )
        if codec not in specs:
            raise ValueError(f"Unsupported codec: {codec}")
        spec = specs[codec]
        if match_input_bit_depth and codec in {"hevc", "av1"} and not metadata.is_10bit:
            options = dict(spec.default_options)
            if codec == "hevc":
                options["profile"] = "main"
            spec = EncoderSpec(
                name=spec.name,
                encoder_name=spec.encoder_name,
                frame_format="nv12",
                default_options=MappingProxyType(options),
                ten_bit=False,
                supported_settings=spec.supported_settings,
            )
        converter_map = _COLOR_CONVERTERS if spec.frame_format == "p010le" else _COLOR_CONVERTERS_NV12
        converter = converter_map.get((metadata.color_space, metadata.color_range))
        if converter is None:
            raise ValueError(f"Unsupported color space or color range: {metadata.color_space} {metadata.color_range}")
        if encoder_settings:
            validate_encoder_settings(
                encoder_settings,
                codec=codec,
                vendor=self.vendor,
            )
        if smart_fragment and self.vendor is AcceleratorVendor.AMD:
            raise ValueError("Smart rendering is currently supported only with NVENC")

        self.metadata = metadata
        self.file = file
        self.output_path = Path(file)
        self.codec = codec
        self.spec = spec
        self.encoder_name = spec.encoder_name
        self.mux_audio = bool(mux_audio)
        self.pts_origin = int(pts_origin)
        self.smart_fragment = bool(smart_fragment)
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
            if self.vendor is AcceleratorVendor.AMD and "cq" in overrides:
                if "qvbr_quality_level" in overrides:
                    raise ValueError(
                        "Conflicting encoder settings: cq and "
                        "qvbr_quality_level are aliases on AMD; use only one"
                    )
                overrides["qvbr_quality_level"] = overrides.pop("cq")
            if self.vendor is AcceleratorVendor.NVIDIA:
                _drop_unsupported_nvenc_overrides(codec, overrides, self.encoder_options)
            self.encoder_options.update(overrides)
        if self.smart_fragment:
            self.encoder_options["forced-idr"] = "1"

        self.BUFFER_MAX_SIZE = 8
        self._lut_flags: deque[bool] = deque()

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
        out_v = self.dst.add_stream(self.encoder_name, **stream_kwargs)
        out_v.width = self.metadata.video_width
        out_v.height = self.metadata.video_height
        out_v.time_base = self.metadata.time_base
        ctx = out_v.codec_context
        ctx.time_base = self.metadata.time_base
        ctx.framerate = self.output_fps
        ctx.pix_fmt = (
            self.spec.frame_format
            if self.vendor is AcceleratorVendor.AMD
            else "cuda"
        )
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
        self.out_stream = out_v

        self._setup_audio()

        # Wrap torch's already-current primary context.  FFmpeg's primary_ctx
        # mode tries to change its scheduling flags and fails once torch has
        # initialized it; current_ctx leaves the context and its flags alone.
        # Keeping conversion and NVENC in one context also avoids a ~500 MiB
        # secondary CUDA context and cross-context scheduling overhead.
        self._cuda_ctx = None
        if self.vendor is AcceleratorVendor.NVIDIA:
            from av.video.frame import CudaContext

            self._cuda_ctx = CudaContext(
                device_id=self.device.index or 0,
                primary_ctx=False,
                current_ctx=True,
            )
        self.stream = new_stream(self.device)
        self._host_yuv = None
        if self.vendor is AcceleratorVendor.AMD:
            dtype = torch.uint16 if self.spec.ten_bit else torch.uint8
            self._host_yuv = torch.empty(
                (
                    self.metadata.video_height
                    + self.metadata.video_height // 2,
                    self.metadata.video_width,
                ),
                dtype=dtype,
                pin_memory=True,
            )
        self.pts_heap: list[int] = []
        self.frame_buffer: deque = deque()
        self._lut_flags.clear()
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
        if not self.mux_audio:
            return
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
                if self.mux_audio:
                    self._drain_audio()
        finally:
            self.dst.close()
            self._src.close()
        if exc_type is None and self._worker_error is not None:
            raise self._worker_error

    def _encode_worker(self):
        set_device(self.device)

        while True:
            item = self._encode_queue.get()
            try:
                if item is self._stop_sentinel:
                    return
                if self._worker_error is None:
                    if len(item) == 3:
                        frame, pts, ready_event = item
                        apply_lut = True
                    else:
                        frame, pts, apply_lut, ready_event = item
                    self._handle_encode_item(frame, pts, apply_lut, ready_event)
            except Exception as exc:
                self._worker_error = exc
                logger.exception("[encoder-worker] crashed")
            finally:
                self._encode_queue.task_done()

    def _build_encode_item(
        self,
        frame: torch.Tensor,
        pts: int,
        apply_lut: bool = True,
    ) -> tuple[torch.Tensor, int, bool, object]:
        producer_stream = current_stream(self.device)
        ready_event = new_event(self.device)
        producer_stream.record_event(ready_event)
        return frame, pts, bool(apply_lut), ready_event

    def _handle_encode_item(
        self,
        frame: torch.Tensor,
        pts: int,
        apply_lut: bool,
        ready_event: object,
    ) -> None:
        self.stream.wait_event(ready_event)
        frame.record_stream(self.stream)
        self._encode_frame(frame, pts, apply_lut=apply_lut)

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
            apply_lut = self._lut_flags.popleft() if self._lut_flags else True
            if apply_lut:
                item = self._build_encode_item(frame_to_encode, pts_to_assign)
            else:
                item = self._build_encode_item(frame_to_encode, pts_to_assign, False)
            self._encode_queue.put(item)

    def _encoder_open_error(self, exc: Exception) -> RuntimeError:
        try:
            gpu = device_name(self.device)
        except Exception:
            gpu = str(self.device)
        message = (
            f"Failed to open {self.codec} encoder ({self.encoder_name}) for "
            f"'{self.output_path.suffix}' output on {gpu}: {exc}"
        )
        if self.codec == "av1":
            backend = "AMF" if self.vendor is AcceleratorVendor.AMD else "NVENC"
            message += (
                f". AV1 {backend} encoding requires a GPU/driver generation "
                "that provides it."
            )
        return RuntimeError(message)

    def _encode_frame(self, frame: torch.Tensor, pts: int, *, apply_lut: bool = True):
        with stream_context(self.stream):
            if apply_lut and self._lut_applier is not None:
                frame = self._lut_applier.apply(frame)
            packed = self._to_yuv(frame)
            if self.vendor is AcceleratorVendor.NVIDIA:
                packed = _align_yuv_pitch(packed)
            else:
                self._host_yuv.copy_(
                    _amf_host_input(packed, ten_bit=self.spec.ten_bit),
                    non_blocking=True,
                )

        height = self.metadata.video_height
        self.stream.synchronize()
        if self.vendor is AcceleratorVendor.AMD:
            planes = [self._host_yuv[:height], self._host_yuv[height:]]
            hw_frame = av.VideoFrame.from_dlpack(
                planes,
                format=self.spec.frame_format,
            )
        else:
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
                cuda_context=self._cuda_ctx,
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

    def encode(self, frame: torch.Tensor, pts: int, *, apply_lut: bool = True):
        if self._worker_error is not None:
            raise self._worker_error
        pts = int(pts) - self.pts_origin
        while pts in self.pts_set:
            pts += 1
        heapq.heappush(self.pts_heap, pts)
        self.frame_buffer.append(frame)
        self._lut_flags.append(bool(apply_lut))
        self.pts_set.add(pts)
        self._process_buffer()
