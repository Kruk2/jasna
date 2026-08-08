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
from jasna.media.cas import GpuCasSharpener
from jasna.media.container_utils import (
    is_mov_chapter_stream,
    subtitle_transcode_codec,
)
from jasna.media.lut import GpuLutApplier, parse_cube_file
from jasna.media.rgb_to_yuv import RgbToYuvConverter

av.logging.set_level(logging.ERROR)

logger = logging.getLogger(__name__)

DEFAULT_ENCODER_OPTIONS: dict[str, str] = {
    "preset": "p5",
    "tune": "hq",
    "profile": "main10",
    "rc": "vbr",
    "cq": "28",
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
    # CQ 25 kept representative capped and uncapped H.264 encodes above VMAF 95.
    "cq": "25",
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
# CQ 35 matches HEVC CQ 28 on the same seven-above-HEVC scale. AV1 QP limits use a separate
# 0..255 scale, so the HEVC qmin/qmax/init_qp values must not be copied here.
# No profile: P010 input makes av1_nvenc emit AV1 Main 10-bit on its own.
# av1_nvenc only consumes the hyphenated spatial-aq spelling.
DEFAULT_AV1_ENCODER_OPTIONS: dict[str, str] = {
    "preset": "p5",
    "tune": "hq",
    "rc": "vbr",
    "cq": "35",
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
    "rc": "cbr",
    "qvbr_quality_level": "25",
    "g": "250",
    "preanalysis": "0",
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

NVENC_SMART_FRAGMENT_OPTIONS = MappingProxyType({"forced-idr": "1"})
AMF_SMART_FRAGMENT_OPTIONS = MappingProxyType({"forced_idr": "1"})


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    encoder_name: str
    frame_format: str  # PyAV hardware-frame software format: "nv12" or "p010le"
    default_options: Mapping[str, str]
    ten_bit: bool
    smart_fragment_options: Mapping[str, str]
    supported_settings: frozenset[str] = field(default_factory=frozenset)


ENCODER_SPECS: dict[str, EncoderSpec] = {
    "hevc": EncoderSpec(
        name="hevc",
        encoder_name="hevc_nvenc",
        frame_format="p010le",
        default_options=MappingProxyType(DEFAULT_ENCODER_OPTIONS),
        ten_bit=True,
        smart_fragment_options=NVENC_SMART_FRAGMENT_OPTIONS,
        supported_settings=SUPPORTED_ENCODER_SETTINGS_BY_CODEC["hevc"],
    ),
    "h264": EncoderSpec(
        name="h264",
        encoder_name="h264_nvenc",
        frame_format="nv12",
        default_options=MappingProxyType(DEFAULT_H264_ENCODER_OPTIONS),
        ten_bit=False,
        smart_fragment_options=NVENC_SMART_FRAGMENT_OPTIONS,
        supported_settings=SUPPORTED_ENCODER_SETTINGS_BY_CODEC["h264"],
    ),
    "av1": EncoderSpec(
        name="av1",
        encoder_name="av1_nvenc",
        frame_format="p010le",
        default_options=MappingProxyType(DEFAULT_AV1_ENCODER_OPTIONS),
        ten_bit=True,
        smart_fragment_options=NVENC_SMART_FRAGMENT_OPTIONS,
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
        smart_fragment_options=AMF_SMART_FRAGMENT_OPTIONS,
        supported_settings=AMF_SUPPORTED_ENCODER_SETTINGS_BY_CODEC["hevc"],
    ),
    "h264": EncoderSpec(
        name="h264",
        encoder_name="h264_amf",
        frame_format="nv12",
        default_options=MappingProxyType(DEFAULT_AMF_H264_ENCODER_OPTIONS),
        ten_bit=False,
        smart_fragment_options=AMF_SMART_FRAGMENT_OPTIONS,
        supported_settings=AMF_SUPPORTED_ENCODER_SETTINGS_BY_CODEC["h264"],
    ),
    "av1": EncoderSpec(
        name="av1",
        encoder_name="av1_amf",
        frame_format="p010le",
        default_options=MappingProxyType(DEFAULT_AMF_AV1_ENCODER_OPTIONS),
        ten_bit=True,
        smart_fragment_options=AMF_SMART_FRAGMENT_OPTIONS,
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
_COLOR_VARIANTS = {
    (AvColorspace.ITU709, AvColorRange.MPEG): "bt709_limited",
    (AvColorspace.ITU709, AvColorRange.JPEG): "bt709_full",
    (AvColorspace.ITU601, AvColorRange.MPEG): "bt601_limited",
    (AvColorspace.ITU601, AvColorRange.JPEG): "bt601_full",
    (AvColorspace.BT2020, AvColorRange.MPEG): "bt2020_limited",
    (AvColorspace.BT2020, AvColorRange.JPEG): "bt2020_full",
}

_NVENC_PITCH_ALIGNMENT = 16

# `cq` alone targets a fixed quality and ignores how the source was stored, so a
# cheaply encoded source is re-encoded far above its own quality point and grows
# several times over (issues #235, #243). A ceiling tied to the source bitrate
# bounds that without measurably costing quality: across 27 clips (1080p to 8K,
# VR and flat) capped encodes landed on the uncapped quality-vs-bitrate curve to
# within 0.07 VMAF, and the ceiling stays inert on sources that were already
# generously encoded. HEVC sources get restoration headroom; NVIDIA H.264 output
# gets a larger ceiling because the old 1x limit flattened CQ values (issue #282).
SOURCE_BITRATE_CAP_FACTORS: dict[str, float] = {"hevc": 1.25}
DEFAULT_SOURCE_BITRATE_CAP_FACTOR = 1.0
NVENC_H264_SOURCE_BITRATE_CAP_FACTOR = 2.0
# Any VBV buffer of roughly a second or more never becomes the binding
# constraint; only sub-second buffers throttle, which is the #243 unit trap.
SOURCE_BITRATE_CAP_BUFFER_RATIO = 2


def source_bitrate_cap_options(
    metadata: VideoMetadata,
    *,
    output_codec: str,
    vendor: AcceleratorVendor,
) -> dict[str, str]:
    if metadata.video_bitrate <= 0:
        logger.warning(
            "No source bitrate for %s; encoding without a source-tied bitrate ceiling",
            metadata.video_file,
        )
        return {}
    if vendor is AcceleratorVendor.NVIDIA and output_codec == "h264":
        factor = NVENC_H264_SOURCE_BITRATE_CAP_FACTOR
    else:
        factor = SOURCE_BITRATE_CAP_FACTORS.get(
            metadata.codec_name.lower(), DEFAULT_SOURCE_BITRATE_CAP_FACTOR
        )
    maxrate = int(metadata.video_bitrate * factor)
    return {
        "maxrate": str(maxrate),
        "bufsize": str(maxrate * SOURCE_BITRATE_CAP_BUFFER_RATIO),
    }


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


def _mov_container_options(suffix: str, *, fmp4: bool) -> dict[str, str]:
    if suffix.lower() not in {".mp4", ".mov"}:
        return {}
    # A fragmented MP4 writes a sample-free moov up front and one moof+mdat per
    # keyframe, so the growing file stays playable; +faststart instead relocates
    # a single moov at close, leaving the file unreadable until then. The two
    # are mutually exclusive.
    flags = "+frag_keyframe+empty_moov" if fmp4 else "+faststart"
    return {"movflags": flags}


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
        sharpen_strength: float = 0.0,
        output_fps: Fraction | None = None,
        mux_audio: bool = True,
        pts_origin: int = 0,
        match_input_bit_depth: bool = False,
        smart_fragment: bool = False,
        fmp4: bool = False,
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
            # AMF pins output depth via "bitdepth"; dropping it lets FFmpeg
            # derive 8-bit from the nv12 input instead of conflicting with it.
            options.pop("bitdepth", None)
            spec = EncoderSpec(
                name=spec.name,
                encoder_name=spec.encoder_name,
                frame_format="nv12",
                default_options=MappingProxyType(options),
                ten_bit=False,
                smart_fragment_options=spec.smart_fragment_options,
                supported_settings=spec.supported_settings,
            )
        color_variant = _COLOR_VARIANTS.get((metadata.color_space, metadata.color_range))
        if color_variant is None:
            raise ValueError(f"Unsupported color space or color range: {metadata.color_space} {metadata.color_range}")
        pixel_format = "p010" if spec.frame_format == "p010le" else "nv12"
        converter_variant = f"{pixel_format}_{color_variant}"
        if encoder_settings:
            validate_encoder_settings(
                encoder_settings,
                codec=codec,
                vendor=self.vendor,
            )
        self.metadata = metadata
        self.file = file
        self.output_path = Path(file)
        self.codec = codec
        self.spec = spec
        self.encoder_name = spec.encoder_name
        self.mux_audio = bool(mux_audio)
        self.pts_origin = int(pts_origin)
        self.smart_fragment = bool(smart_fragment)
        self.fmp4 = bool(fmp4)
        self.output_fps = Fraction(
            metadata.video_fps_exact if output_fps is None else output_fps
        )

        self._lut_applier: GpuLutApplier | None = None
        if lut_path:
            lut = parse_cube_file(lut_path)
            self._lut_applier = GpuLutApplier(lut, device)

        self._cas: GpuCasSharpener | None = None
        if sharpen_strength > 0.0:
            self._cas = GpuCasSharpener(
                sharpen_strength, ten_bit=spec.ten_bit, device=self.device
            )

        self._converter = RgbToYuvConverter(converter_variant, device=self.device)

        self.encoder_options = dict(spec.default_options)
        if "maxrate" not in encoder_settings:
            self.encoder_options.update(
                source_bitrate_cap_options(
                    metadata,
                    output_codec=codec,
                    vendor=self.vendor,
                )
            )
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
            self.encoder_options.update(spec.smart_fragment_options)

        self.BUFFER_MAX_SIZE = 8
        self._lut_flags: deque[bool] = deque()
        # Set on AMD in __enter__, where the frame size is known; NVIDIA leaves
        # them None and allocates per frame (NVENC outlives encode()).
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
        self._src = av.open(self.metadata.video_file)
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

        self._copy_source_metadata(in_v, out_v)
        self._setup_source_streams(in_v)

        # Wrap torch's already-current primary context.  FFmpeg's primary_ctx
        # mode tries to change its scheduling flags and fails once torch has
        # initialized it; current_ctx leaves the context and its flags alone.
        # Keeping conversion and NVENC in one context also avoids a ~500 MiB
        # secondary CUDA context and cross-context scheduling overhead.
        # NVENC consumes device memory, so conversion runs on its own stream and
        # overlaps the rest of the pipeline. AMF consumes host memory and the
        # conversion is eager Torch math, so on AMD everything stays on the
        # current stream: a private stream there let ROCm recycle in-flight
        # conversion buffers into the restorer's allocations (issue #252).
        self.stream = (
            current_stream(self.device)
            if self.vendor is AcceleratorVendor.AMD
            else new_stream(self.device)
        )
        self._cuda_ctx = None
        if self.vendor is AcceleratorVendor.NVIDIA:
            from av.video.frame import CudaContext

            self._cuda_ctx = CudaContext(
                device_id=self.device.index or 0,
                primary_ctx=False,
                current_ctx=True,
                cuda_stream=self.stream.cuda_stream,
            )
        height = self.metadata.video_height
        width = self.metadata.video_width
        self._packed = None
        self._cas_luma = None
        self._host_yuv = None
        if self.vendor is AcceleratorVendor.AMD:
            self._packed = torch.empty(
                (height + height // 2, width),
                dtype=self._converter.sample_dtype,
                device=self.device,
            )
            if self._cas is not None:
                self._cas_luma = torch.empty_like(self._packed[:height])
            dtype = torch.uint16 if self.spec.ten_bit else torch.uint8
            self._host_yuv = torch.empty(
                (height + height // 2, width),
                dtype=dtype,
                pin_memory=True,
            )
        self.pts_heap: list[int] = []
        self.frame_buffer: deque = deque()
        self._lut_flags.clear()
        self.pts_set: set[int] = set()
        self._last_emitted_pts: int | None = None
        self._video_started = False
        self._options_validated = False
        self._worker_error: Exception | None = None

        self._stop_sentinel = object()
        self._encode_queue: queue.Queue = queue.Queue(maxsize=self.BUFFER_MAX_SIZE)
        self._encode_thread = threading.Thread(target=self._encode_worker, name="NvidiaVideoEncoderWorker", daemon=True)
        self._encode_thread.start()
        return self

    def _copy_source_metadata(self, in_v, out_v) -> None:
        self.dst.metadata.update(self._src.metadata)
        out_v.metadata.update(in_v.metadata)
        out_v.disposition = in_v.disposition
        if not self.smart_fragment:
            self._source_chapters = self._src.chapters()
            self.dst.set_chapters(self._source_chapters)

    def _setup_source_streams(self, in_v) -> None:
        self._source_pipes: dict[int, tuple[str, object, object]] = {}
        self._source_backlog: deque = deque()
        self._source_iter = None
        if self.smart_fragment:
            return

        packet_streams = []
        output_formats = set(self.dst.format.name.split(","))
        source_formats = set(self._src.format.name.split(","))
        source_chapters = getattr(self, "_source_chapters", ())
        for in_stream in self._src.streams:
            if in_stream.index == in_v.index:
                continue
            if in_stream.type == "audio" and not self.mux_audio:
                continue
            if is_mov_chapter_stream(
                in_stream,
                source_formats=source_formats,
                chapters=source_chapters,
            ):
                continue
            if in_stream.type == "attachment" and "matroska" not in output_formats:
                logger.warning(
                    "Skipping attachment stream %s: %s output does not support attachments",
                    in_stream.index,
                    self.output_path.suffix,
                )
                continue
            if in_stream.codec_context is None and in_stream.type != "attachment":
                if in_stream.type != "data" or not source_formats & output_formats:
                    logger.warning(
                        "Skipping %s stream %s: it has no copyable codec",
                        in_stream.type,
                        in_stream.index,
                    )
                    continue

            if in_stream.type == "audio" and needs_audio_reencode(
                in_stream.codec_context.name, self.output_path.suffix
            ):
                logger.info(
                    "re-encoding audio %s -> aac for %s",
                    in_stream.codec_context.name,
                    self.output_path.suffix,
                )
                out_stream = self.dst.add_stream(
                    "aac", rate=in_stream.codec_context.sample_rate
                )
                out_stream.codec_context.layout = in_stream.codec_context.layout
                out_stream.bit_rate = 256_000
                resampler = av.AudioResampler(
                    format="fltp",
                    layout=in_stream.codec_context.layout,
                    rate=in_stream.codec_context.sample_rate,
                )
                kind = "transcode"
            else:
                try:
                    out_stream = self.dst.add_stream_from_template(
                        in_stream, opaque=True
                    )
                except ValueError as exc:
                    transcode_codec = subtitle_transcode_codec(
                        in_stream.codec_context.name,
                        output_formats=output_formats,
                        supported_codecs=getattr(
                            self.dst, "supported_codecs", frozenset()
                        ),
                    )
                    if in_stream.type != "subtitle" or transcode_codec is None:
                        logger.warning(
                            "Skipping %s stream %s: %s",
                            in_stream.type,
                            in_stream.index,
                            exc,
                        )
                        continue
                    logger.info(
                        "re-encoding subtitle %s -> %s for %s",
                        in_stream.codec_context.name,
                        transcode_codec,
                        self.output_path.suffix,
                    )
                    out_stream = self.dst.add_stream(transcode_codec)
                    subtitle_time_base = (
                        getattr(in_stream, "time_base", None)
                        or Fraction(1, 1_000)
                    )
                    out_stream.time_base = subtitle_time_base
                    out_stream.codec_context.time_base = subtitle_time_base
                    out_stream.codec_context.subtitle_header = b""
                    resampler = None
                    kind = "subtitle_transcode"
                else:
                    resampler = None
                    kind = "copy"

            out_stream.metadata.update(in_stream.metadata)
            out_stream.disposition = in_stream.disposition
            if in_stream.type == "attachment":
                continue
            self._source_pipes[in_stream.index] = (kind, out_stream, resampler)
            packet_streams.append(in_stream)

        if packet_streams:
            self._source_iter = self._src.demux(packet_streams)

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
                self._drain_source_streams()
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
            self._pump_source_streams(threshold)

    def _produce_source_packets(self, in_packet) -> list:
        kind, out_stream, resampler = self._source_pipes[in_packet.stream.index]
        if kind == "copy":
            if in_packet.size == 0:
                return []
            in_packet.stream = out_stream
            return [in_packet]
        if kind == "subtitle_transcode":
            if in_packet.size == 0:
                return []
            subtitle = in_packet.stream.codec_context.decode2(in_packet)
            if subtitle is None:
                return []
            packet = out_stream.codec_context.encode_subtitle(subtitle)
            output_time_base = (
                out_stream.time_base
                or out_stream.codec_context.time_base
                or in_packet.time_base
                or Fraction(1, 1_000)
            )

            def rescale(value):
                if value is None or in_packet.time_base is None:
                    return value
                return round(value * in_packet.time_base / output_time_base)

            packet.stream = out_stream
            packet.pts = rescale(in_packet.pts)
            packet.dts = rescale(
                in_packet.dts if in_packet.dts is not None else in_packet.pts
            )
            packet.duration = rescale(in_packet.duration)
            packet.time_base = output_time_base
            return [packet]
        out_packets = []
        for aframe in in_packet.decode():
            for rframe in resampler.resample(aframe):
                out_packets.extend(out_stream.encode(rframe))
        return out_packets

    def _pump_source_streams(self, upto_seconds: float | None):
        if self._source_iter is None:
            return
        while True:
            if self._source_backlog:
                packet = self._source_backlog[0]
                ts = packet.dts if packet.dts is not None else packet.pts
                if (
                    upto_seconds is not None
                    and ts is not None
                    and packet.time_base is not None
                    and float(ts * packet.time_base) > upto_seconds
                ):
                    return
                self._source_backlog.popleft()
                self.dst.mux(packet)
                continue
            in_packet = next(self._source_iter, None)
            if in_packet is None:
                self._source_iter = None
                return
            self._source_backlog.extend(self._produce_source_packets(in_packet))

    def _drain_source_streams(self):
        self._pump_source_streams(None)
        for kind, out_stream, resampler in self._source_pipes.values():
            if kind != "transcode":
                continue
            packets = []
            for rframe in resampler.resample(None):
                packets.extend(out_stream.encode(rframe))
            packets.extend(out_stream.encode(None))
            for packet in packets:
                self.dst.mux(packet)

    def _clamp_pts_monotonic(self, pts: int) -> int:
        last = self._last_emitted_pts
        if last is not None and pts <= last:
            pts = last + 1
        self._last_emitted_pts = pts
        return pts

    def _process_buffer(self, flush_all=False):
        if len(self.frame_buffer) > (self.BUFFER_MAX_SIZE // 2) or (flush_all and self.frame_buffer):
            frame_to_encode = self.frame_buffer.popleft()
            pts_to_assign = heapq.heappop(self.pts_heap)
            self.pts_set.remove(pts_to_assign)
            pts_to_assign = self._clamp_pts_monotonic(pts_to_assign)
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

    def _packed_frame(self, height: int, width: int) -> torch.Tensor:
        # NVENC takes the device frame by pointer and still owns it after
        # encode() returns, so NVIDIA hands it a fresh one every time. AMF has
        # no zero-copy path: the AMD branch synchronizes then blocking-copies
        # this buffer into pinned host memory, so one buffer can serve every frame.
        if self._packed is not None:
            return self._packed
        return torch.empty(
            (height + height // 2, width),
            dtype=self._converter.sample_dtype,
            device=self.device,
        )

    def _to_yuv(self, frame: torch.Tensor, height: int) -> torch.Tensor:
        # Sharpening happens here rather than after, because it must see a
        # contiguous plane: pitch alignment can hand back a strided view into a
        # wider buffer, and the AMD path copies straight to host memory.
        packed = self._packed_frame(height, frame.shape[2])
        if self._cas is None:
            self._converter.convert_into(frame, packed[:height], packed[height:])
            return packed
        # CAS is a 3x3 stencil, so it cannot run in place: the conversion writes
        # luma to a scratch plane and sharpening reads from there into the frame
        # the encoder receives, instead of copying a whole plane back.
        luma = self._cas_luma
        if luma is None:
            luma = torch.empty_like(packed[:height])
        self._converter.convert_into(frame, luma, packed[height:])
        self._cas.sharpen_into(luma, packed[:height])
        return packed

    def _encode_frame(self, frame: torch.Tensor, pts: int, *, apply_lut: bool = True):
        height = self.metadata.video_height
        with stream_context(self.stream):
            if apply_lut and self._lut_applier is not None:
                frame = self._lut_applier.apply(frame)
            packed = self._to_yuv(frame, height)
            if self.vendor is AcceleratorVendor.NVIDIA:
                packed = _align_yuv_pitch(packed)
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
            # Issue #252: isolated E1/E2 were clean; residual glitches under full
            # pipeline load matched AMF reading host planes while a non-blocking
            # D2H was still in flight. Finish convert on the stream, then
            # blocking-copy into pinned host so from_dlpack sees complete planes.
            # Phase 4 (gfx1201): stream.synchronize() + blocking copy cleared P1;
            # do not escalate to full device.synchronize() unless field reports return.
            self.stream.synchronize()
            self._host_yuv.copy_(
                _amf_host_input(packed, ten_bit=self.spec.ten_bit),
                non_blocking=False,
            )
            planes = [self._host_yuv[:height], self._host_yuv[height:]]
            hw_frame = av.VideoFrame.from_dlpack(
                planes,
                format=self.spec.frame_format,
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
