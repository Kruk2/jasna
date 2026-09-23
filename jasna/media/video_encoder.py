from __future__ import annotations

import heapq
import logging
import math
import queue
import threading
from collections import deque
from dataclasses import dataclass, replace
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
from jasna.media.audio_utils import needs_audio_reencode
from jasna.media.cas import GpuCasSharpener
from jasna.media.container_utils import (
    is_mov_chapter_stream,
    subtitle_transcode_codec,
)
from jasna.media.encoder_settings import encoder_cq_spec, validate_encoder_settings
from jasna.media.lut import GpuLutApplier, parse_cube_file
from jasna.media.probe import VideoMetadata
from jasna.media.rgb_to_yuv import RgbToYuvConverter

logger = logging.getLogger(__name__)

DEFAULT_ENCODER_OPTIONS: dict[str, str] = {
    "preset": "p5",
    "tune": "hq",
    "profile": "main10",
    "rc": "vbr",
    "cq": str(encoder_cq_spec("hevc", AcceleratorVendor.NVIDIA).default),
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
    "cq": str(encoder_cq_spec("h264", AcceleratorVendor.NVIDIA).default),
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
    "cq": str(encoder_cq_spec("av1", AcceleratorVendor.NVIDIA).default),
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
    "qvbr_quality_level": str(
        encoder_cq_spec("h264", AcceleratorVendor.AMD).default
    ),
    "g": "250",
    "preanalysis": "1",
    "vbaq": "1",
    "profile": "high",
}

DEFAULT_AMF_HEVC_ENCODER_OPTIONS: dict[str, str] = {
    "usage": "high_quality",
    "quality": "quality",
    "rc": "cqp",
    "qp_i": str(encoder_cq_spec("hevc", AcceleratorVendor.AMD).default),
    "qp_p": str(encoder_cq_spec("hevc", AcceleratorVendor.AMD).default),
    "g": "250",
    "preanalysis": "0",
    "vbaq": "0",
    "profile": "main10",
    "bitdepth": "10",
}

DEFAULT_AMF_AV1_ENCODER_OPTIONS: dict[str, str] = {
    "usage": "high_quality",
    "quality": "quality",
    "rc": "cqp",
    "qp_i": str(encoder_cq_spec("av1", AcceleratorVendor.AMD).default),
    "qp_p": str(encoder_cq_spec("av1", AcceleratorVendor.AMD).default),
    "qp_b": str(encoder_cq_spec("av1", AcceleratorVendor.AMD).default),
    "g": "250",
    "preanalysis": "0",
    "aq_mode": "none",
    "profile": "main",
    "bitdepth": "10",
}

NVENC_SMART_FRAGMENT_OPTIONS = MappingProxyType({"forced-idr": "1"})
AMF_SMART_FRAGMENT_OPTIONS = MappingProxyType({"forced_idr": "1"})


@dataclass(frozen=True)
class EncoderSpec:
    encoder_name: str
    frame_format: str  # PyAV hardware-frame software format: "nv12" or "p010le"
    default_options: Mapping[str, str]

    @property
    def ten_bit(self) -> bool:
        return self.frame_format == "p010le"


ENCODER_SPECS: dict[str, EncoderSpec] = {
    "hevc": EncoderSpec(
        encoder_name="hevc_nvenc",
        frame_format="p010le",
        default_options=MappingProxyType(DEFAULT_ENCODER_OPTIONS),
    ),
    "h264": EncoderSpec(
        encoder_name="h264_nvenc",
        frame_format="nv12",
        default_options=MappingProxyType(DEFAULT_H264_ENCODER_OPTIONS),
    ),
    "av1": EncoderSpec(
        encoder_name="av1_nvenc",
        frame_format="p010le",
        default_options=MappingProxyType(DEFAULT_AV1_ENCODER_OPTIONS),
    ),
}

AMF_ENCODER_SPECS: dict[str, EncoderSpec] = {
    "hevc": EncoderSpec(
        encoder_name="hevc_amf",
        frame_format="p010le",
        default_options=MappingProxyType(DEFAULT_AMF_HEVC_ENCODER_OPTIONS),
    ),
    "h264": EncoderSpec(
        encoder_name="h264_amf",
        frame_format="nv12",
        default_options=MappingProxyType(DEFAULT_AMF_H264_ENCODER_OPTIONS),
    ),
    "av1": EncoderSpec(
        encoder_name="av1_amf",
        frame_format="p010le",
        default_options=MappingProxyType(DEFAULT_AMF_AV1_ENCODER_OPTIONS),
    ),
}

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


def _normalize_amf_cq(
    codec: str,
    overrides: dict[str, str],
    defaults: dict[str, str],
    *,
    ten_bit: bool,
) -> None:
    rc = overrides.get("rc", defaults["rc"])
    cqp_modes = {"cqp", "0"}
    qvbr_modes = {"qvbr", "hqvbr", "4", "5"}
    if codec in {"hevc", "av1"} and rc not in cqp_modes:
        defaults.pop("qp_i", None)
        defaults.pop("qp_p", None)
        defaults.pop("qp_b", None)
    if codec in {"hevc", "av1"} and ten_bit and rc in qvbr_modes:
        raise ValueError(
            f"AMD {codec.upper()} Main10 does not support QVBR or HQVBR; use the default CQP mode"
        )

    aliases = [key for key in ("cq", "qvbr_quality_level") if key in overrides]
    if len(aliases) > 1:
        raise ValueError(
            "Conflicting encoder settings: cq and qvbr_quality_level are aliases "
            "on AMD; use only one"
        )
    if not aliases:
        return

    value = overrides.pop(aliases[0])
    if codec in {"hevc", "av1"}:
        if rc in cqp_modes:
            overrides["qp_i"] = value
            overrides["qp_p"] = value
            if codec == "av1":
                overrides["qp_b"] = value
        elif not ten_bit and rc in qvbr_modes:
            overrides["qvbr_quality_level"] = value
        else:
            raise ValueError(f"AMD {codec.upper()} CQ requires rc=cqp")
    else:
        overrides["qvbr_quality_level"] = value


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


def _mov_container_options(suffix: str, *, fmp4: bool) -> dict[str, str]:
    if suffix.lower() not in {".mp4", ".mov"}:
        return {}
    # A fragmented MP4 writes a sample-free moov up front and one moof+mdat per
    # keyframe, so the growing file stays playable; +faststart instead relocates
    # a single moov at close, leaving the file unreadable until then. The two
    # are mutually exclusive.
    flags = "+frag_keyframe+empty_moov" if fmp4 else "+faststart"
    return {"movflags": flags}


def _normalized_audio_layout(layout: av.AudioLayout) -> av.AudioLayout:
    if layout.name == "2 channels":
        return av.AudioLayout("stereo")
    return layout


def resolve_encoder_options(
    vendor: AcceleratorVendor,
    codec: str,
    metadata: VideoMetadata,
    encoder_settings: dict[str, object],
    *,
    smart_fragment: bool,
) -> tuple[EncoderSpec, dict[str, str]]:
    """Pick the encoder spec and the final FFmpeg options for one output."""
    specs = AMF_ENCODER_SPECS if vendor is AcceleratorVendor.AMD else ENCODER_SPECS
    if codec not in specs:
        raise ValueError(f"Unsupported codec: {codec}")
    spec = specs[codec]
    # Smart-render fragments are spliced between copied source spans, so they
    # keep the source bit depth instead of the codec's 10-bit default.
    if smart_fragment and codec in {"hevc", "av1"} and not metadata.is_10bit:
        options = dict(spec.default_options)
        if codec == "hevc":
            options["profile"] = "main"
        # AMF pins output depth via "bitdepth"; dropping it lets FFmpeg
        # derive 8-bit from the nv12 input instead of conflicting with it.
        options.pop("bitdepth", None)
        spec = replace(spec, frame_format="nv12", default_options=MappingProxyType(options))
    if encoder_settings:
        validate_encoder_settings(encoder_settings, codec=codec, vendor=vendor)

    encoder_options = dict(spec.default_options)
    overrides: dict[str, str] = {}
    if encoder_settings:
        overrides = {k: _option_value(v) for k, v in encoder_settings.items()}
        # FFmpeg accepts both spellings for HEVC/H.264, but their defaults
        # use the underscore key. Normalize the alias so a user override
        # replaces that default instead of passing two conflicting options.
        if "spatial-aq" in overrides and "spatial_aq" in encoder_options:
            overrides["spatial_aq"] = overrides.pop("spatial-aq")
        if vendor is AcceleratorVendor.AMD:
            _normalize_amf_cq(codec, overrides, encoder_options, ten_bit=spec.ten_bit)
        else:
            _drop_unsupported_nvenc_overrides(codec, overrides, encoder_options)
    uses_amf_cqp = (
        vendor is AcceleratorVendor.AMD
        and codec in {"hevc", "av1"}
        and overrides.get("rc", encoder_options["rc"]) in {"cqp", "0"}
    )
    if "maxrate" not in overrides and not uses_amf_cqp:
        encoder_options.update(source_bitrate_cap_options(metadata, output_codec=codec, vendor=vendor))
    encoder_options.update(overrides)
    if smart_fragment:
        encoder_options.update(
            AMF_SMART_FRAGMENT_OPTIONS if vendor is AcceleratorVendor.AMD else NVENC_SMART_FRAGMENT_OPTIONS
        )
    return spec, encoder_options


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
        pts_origin: int = 0,
        smart_fragment: bool = False,
        fmp4: bool = False,
    ):
        self.device = torch.device(device)
        self.vendor = vendor_for_device(self.device)
        if self.vendor not in {AcceleratorVendor.NVIDIA, AcceleratorVendor.AMD}:
            raise RuntimeError(
                f"GPU video encoding is not supported on {self.vendor.value}"
            )
        spec, self.encoder_options = resolve_encoder_options(
            self.vendor, codec, metadata, encoder_settings, smart_fragment=smart_fragment
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
        self.output_fps = Fraction(
            metadata.video_fps_exact if output_fps is None else output_fps
        )
        self._wmv_cfr_rate = self._vc1_wmv_cfr_rate(metadata)
        self._wmv_first_pts: int | None = None
        self._wmv_frame_index = 0

        self._lut_applier: GpuLutApplier | None = None
        if lut_path:
            lut = parse_cube_file(lut_path)
            self._lut_applier = GpuLutApplier(lut, device)

        self._cas: GpuCasSharpener | None = None
        if sharpen_strength > 0.0:
            self._cas = GpuCasSharpener(
                sharpen_strength, ten_bit=spec.ten_bit, device=self.device
            )

        self._converter = RgbToYuvConverter(f"{pixel_format}_{color_variant}", device=self.device)

        self._lut_flags: deque[bool] = deque()
        # Only AMD reuses one packed frame (set in __enter__); NVENC still holds
        # its input frame after encode() returns, so NVIDIA allocates per frame.
        self._packed: torch.Tensor | None = None
        self._cas_luma: torch.Tensor | None = None
        self._source_chapters = ()

    @staticmethod
    def _vc1_wmv_cfr_rate(metadata: VideoMetadata) -> Fraction | None:
        if (
            metadata.codec_name != "vc1"
            or Path(metadata.video_file).suffix.lower() not in {".wmv", ".asf"}
        ):
            return None
        rate = Fraction(30_000, 1_001)
        if abs(metadata.video_fps - float(rate)) >= 0.04:
            return None
        duration_rate = metadata.num_frames / metadata.duration if metadata.duration > 0 else 0
        if (
            abs(metadata.average_fps - float(rate)) >= 0.01
            and abs(duration_rate - float(rate)) >= 0.01
        ):
            return None
        return rate

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
            pix_fmt = self.spec.frame_format
        else:
            pix_fmt = "cuda"
        out_v = self.dst.add_stream(self.encoder_name, **stream_kwargs)
        if self.codec == "hevc" and self.output_path.suffix.lower() in {".mp4", ".mov"}:
            out_v.codec_tag = "hvc1"
        out_v.width = self.metadata.video_width
        out_v.height = self.metadata.video_height
        out_v.time_base = self.metadata.time_base
        ctx = out_v.codec_context
        ctx.time_base = self.metadata.time_base
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
        height = self.metadata.video_height
        width = self.metadata.video_width
        self._cuda_ctx = None
        self._host_yuv = None
        if self.vendor is AcceleratorVendor.NVIDIA:
            from av.video.frame import CudaContext

            self.stream = new_stream(self.device)
            self._cuda_ctx = CudaContext(
                device_id=self.device.index or 0,
                primary_ctx=False,
                current_ctx=True,
                cuda_stream=self.stream.cuda_stream,
            )
        else:
            self.stream = current_stream(self.device)
            self._packed = torch.empty(
                (height + height // 2, width),
                dtype=self._converter.sample_dtype,
                device=self.device,
            )
            if self._cas is not None:
                self._cas_luma = torch.empty_like(self._packed[:height])
            self._host_yuv = torch.empty(
                (height + height // 2, width),
                dtype=torch.uint16 if self.spec.ten_bit else torch.uint8,
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
        self._encode_queue: queue.Queue = queue.Queue(maxsize=ENCODE_BUFFER_SIZE)
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
        self._last_source_dts: dict[int, tuple[int, Fraction]] = {}
        if self.smart_fragment:
            return

        packet_streams = []
        output_formats = set(self.dst.format.name.split(","))
        source_formats = set(self._src.format.name.split(","))
        source_chapters = self._source_chapters
        for in_stream in self._src.streams:
            if in_stream.index == in_v.index:
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
                logger.warning(
                    "Skipping %s stream %s: it has no copyable codec",
                    in_stream.type,
                    in_stream.index,
                )
                continue

            source_audio_layout = (
                in_stream.codec_context.layout
                if in_stream.type == "audio"
                else None
            )
            audio_layout = (
                _normalized_audio_layout(source_audio_layout)
                if source_audio_layout is not None
                else None
            )
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
                out_stream.codec_context.layout = audio_layout
                out_stream.bit_rate = 256_000
                processor = av.AudioResampler(
                    format="fltp",
                    layout=audio_layout,
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
                    processor = None
                    kind = "subtitle_transcode"
                else:
                    if (
                        audio_layout is not None
                        and audio_layout is not source_audio_layout
                    ):
                        out_stream.codec_context.layout = audio_layout
                    processor = self._aac_copy_filter(
                        in_stream, out_stream, output_formats
                    )
                    kind = "copy"

            out_stream.metadata.update(in_stream.metadata)
            out_stream.disposition = in_stream.disposition
            if in_stream.type == "attachment":
                continue
            self._source_pipes[in_stream.index] = (kind, out_stream, processor)
            packet_streams.append(in_stream)

        if packet_streams:
            self._source_iter = self._src.demux(packet_streams)

    @staticmethod
    def _aac_copy_filter(in_stream, out_stream, output_formats):
        if (
            in_stream.type == "audio"
            and in_stream.codec_context.name == "aac"
            and output_formats & {"mp4", "mov"}
            and out_stream.codec_context.extradata
        ):
            return av.bitstream.BitStreamFilterContext("aac_adtstoasc", in_stream)
        return None

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
                    frame, pts, apply_lut, ready_event = item
                    self.stream.wait_event(ready_event)
                    frame.record_stream(self.stream)
                    self._encode_frame(frame, pts, apply_lut=apply_lut)
            except Exception as exc:
                self._worker_error = exc
                logger.exception("[encoder-worker] crashed")
            finally:
                self._encode_queue.task_done()

    def _build_encode_item(
        self,
        frame: torch.Tensor,
        pts: int,
        apply_lut: bool,
    ) -> tuple[torch.Tensor, int, bool, object]:
        producer_stream = current_stream(self.device)
        ready_event = new_event(self.device)
        producer_stream.record_event(ready_event)
        return frame, pts, apply_lut, ready_event

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
            self._pump_source_streams(threshold)

    def _produce_source_packets(self, in_packet) -> list:
        kind, out_stream, processor = self._source_pipes[in_packet.stream.index]
        if kind == "copy":
            return self._copy_source_packets(in_packet, out_stream, processor)
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
            for rframe in processor.resample(aframe):
                out_packets.extend(out_stream.encode(rframe))
        return out_packets

    @staticmethod
    def _copy_source_packets(in_packet, out_stream, bitstream_filter):
        if in_packet is not None and in_packet.size == 0:
            return []
        if bitstream_filter is None:
            packets = [in_packet]
        else:
            packets = bitstream_filter.filter(in_packet)
        for packet in packets:
            packet.stream = out_stream
        return packets

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
                self._mux_source_packet(packet)
                continue
            in_packet = next(self._source_iter, None)
            if in_packet is None:
                self._source_iter = None
                return
            self._source_backlog.extend(self._produce_source_packets(in_packet))

    def _drain_source_streams(self):
        self._pump_source_streams(None)
        for kind, out_stream, processor in self._source_pipes.values():
            if kind == "copy":
                if processor is not None:
                    self.dst.mux(self._copy_source_packets(None, out_stream, processor))
                continue
            if kind != "transcode":
                continue
            packets = []
            for rframe in processor.resample(None):
                packets.extend(out_stream.encode(rframe))
            packets.extend(out_stream.encode(None))
            for packet in packets:
                self._mux_source_packet(packet)

    def _mux_source_packet(self, packet):
        if packet.dts is not None:
            last = self._last_source_dts.get(packet.stream.index)
            if last is not None:
                time_base = Fraction(packet.time_base or packet.stream.time_base)
                minimum_time = (last[0] + 1) * last[1]
                if packet.dts * time_base < minimum_time:
                    logger.warning(
                        "Source DTS %s at %s overlaps last muxed DTS %s at %s in output stream %s; nudging forward",
                        packet.dts,
                        time_base,
                        last[0],
                        last[1],
                        packet.stream.index,
                    )
                    packet.dts = math.ceil(minimum_time / time_base)
                    if packet.pts is not None and packet.pts < packet.dts:
                        packet.pts = packet.dts
        self.dst.mux(packet)
        if packet.dts is not None:
            self._last_source_dts[packet.stream.index] = (
                packet.dts,
                Fraction(packet.time_base or packet.stream.time_base),
            )

    def _clamp_pts_monotonic(self, pts: int) -> int:
        last = self._last_emitted_pts
        if last is not None and pts <= last:
            pts = last + 1
        self._last_emitted_pts = pts
        return pts

    def _process_buffer(self, flush_all=False):
        if len(self.frame_buffer) > (ENCODE_BUFFER_SIZE // 2) or (flush_all and self.frame_buffer):
            frame_to_encode = self.frame_buffer.popleft()
            pts_to_assign = heapq.heappop(self.pts_heap)
            self.pts_set.remove(pts_to_assign)
            pts_to_assign = self._clamp_pts_monotonic(pts_to_assign)
            self._encode_queue.put(
                self._build_encode_item(frame_to_encode, pts_to_assign, self._lut_flags.popleft())
            )

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
        hw_frame.time_base = self.metadata.time_base
        try:
            packets = self.out_stream.encode(hw_frame)
        except av.FFmpegError as exc:
            if not self._video_started:
                raise self._encoder_open_error(exc) from exc
            raise
        for packet in packets:
            self._mux_video(packet)

    def _output_pts(self, pts: int) -> int:
        pts = int(pts) - self.pts_origin
        if self._wmv_cfr_rate is None:
            return pts
        if self._wmv_first_pts is None:
            self._wmv_first_pts = pts
        output_pts = self._wmv_first_pts + round(
            self._wmv_frame_index / (self._wmv_cfr_rate * self.metadata.time_base)
        )
        self._wmv_frame_index += 1
        return output_pts

    def encode(self, frame: torch.Tensor, pts: int, *, apply_lut: bool = True):
        if self._worker_error is not None:
            raise self._worker_error
        pts = self._output_pts(pts)
        while pts in self.pts_set:
            pts += 1
        heapq.heappush(self.pts_heap, pts)
        self.frame_buffer.append(frame)
        self._lut_flags.append(bool(apply_lut))
        self.pts_set.add(pts)
        self._process_buffer()
