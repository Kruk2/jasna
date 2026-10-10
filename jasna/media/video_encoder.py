from __future__ import annotations

import heapq
import logging
import math
import os
import queue
import sys
import threading
from collections import deque
from dataclasses import dataclass, field, replace
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
    MOV_SUFFIXES,
    is_mov_chapter_stream,
    subtitle_transcode_codec,
)
from jasna.media.encoder_settings import AMF_SUPPORTED_ENCODER_SETTINGS_BY_CODEC, SUPPORTED_ENCODER_SETTINGS_BY_CODEC, encoder_cq_spec, validate_encoder_settings
from jasna.media.lut import GpuLutApplier, parse_cube_file
from jasna.media.probe import VideoMetadata, hevc_level_to_amf_option
from jasna.media.rgb_to_yuv import RgbToYuvConverter

logger = logging.getLogger(__name__)

AMF_HEVC_VBR_PEAK_ENV = "JASNA_AMF_HEVC_VBR_PEAK"
AMF_HOST_ZERO_COPY_ENV = "JASNA_AMF_HOST_ZERO_COPY"
AMF_HOST_ZERO_COPY_MAIN8_MIN_PIXELS = 5760 * 2880
AMF_HOST_ZERO_COPY_MAIN10_MIN_PIXELS = 3840 * 2160


def _amf_host_native_output_eligible(
    *,
    width: int,
    height: int,
    ten_bit: bool,
    frame_format: str,
) -> bool:
    """Return whether this HEVC output shape has passed real-video A/B tests."""

    expected_format = "p010le" if ten_bit else "nv12"
    minimum_pixels = (
        AMF_HOST_ZERO_COPY_MAIN10_MIN_PIXELS
        if ten_bit
        else AMF_HOST_ZERO_COPY_MAIN8_MIN_PIXELS
    )
    return (
        width > 0
        and height > 0
        and width % 2 == 0
        and height % 2 == 0
        and width * height >= minimum_pixels
        and str(frame_format).strip().lower() == expected_format
    )


def _amf_hevc_vbr_peak_override() -> bool | None:
    """Return the AMF HEVC source-rate override, or ``None`` for auto."""

    raw_value = os.environ.get(AMF_HEVC_VBR_PEAK_ENV)
    if raw_value is None:
        return None
    value = raw_value.strip().casefold()
    if value == "auto":
        return None
    if value in {"", "0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Invalid {AMF_HEVC_VBR_PEAK_ENV} value {value!r}; expected auto, 0, or 1"
    )


def _amf_host_zero_copy_override() -> bool | None:
    """Return the Linux AMD HEVC host-native override, or ``None`` for auto."""

    raw_value = os.environ.get(AMF_HOST_ZERO_COPY_ENV)
    if raw_value is None:
        return None
    value = raw_value.strip().casefold()
    if value == "auto":
        return None
    if value in {"", "0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Invalid {AMF_HOST_ZERO_COPY_ENV} value {value!r}; expected auto, 0, or 1"
    )

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


def _amf_av1_qindex(cq: int) -> int:
    # AMF AV1 frame QPs are AV1 qindex 0..255; CQ uses the H.264/HEVC QP scale 0..51.
    return cq * 255 // 51


_AMF_AV1_DEFAULT_QINDEX = str(_amf_av1_qindex(encoder_cq_spec("av1", AcceleratorVendor.AMD).default))

DEFAULT_AMF_AV1_ENCODER_OPTIONS: dict[str, str] = {
    "usage": "high_quality",
    "quality": "quality",
    "rc": "cqp",
    "qp_i": _AMF_AV1_DEFAULT_QINDEX,
    "qp_p": _AMF_AV1_DEFAULT_QINDEX,
    "qp_b": _AMF_AV1_DEFAULT_QINDEX,
    "g": "250",
    "preanalysis": "0",
    "aq_mode": "none",
    "profile": "main",
    "bitdepth": "10",
}

NVENC_SMART_FRAGMENT_OPTIONS = MappingProxyType({"forced-idr": "1"})
AMF_SMART_FRAGMENT_OPTIONS = MappingProxyType({"forced_idr": "1"})

# CQP 30 measured near the portable CQ 28 source-quality point for Linux AMF
# HEVC fragments. Keep the shared CQ scale with a fragment-only offset.
AMD_HEVC_CQP_OFFSET = 2


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
_COLOR_PRIMARIES_BY_CODE = {value: key for key, value in _COLOR_PRIMARIES.items()}
_COLOR_TRANSFERS_BY_CODE = {value: key for key, value in _COLOR_TRANSFERS.items()}
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


def resolve_hevc_smart_render_vui(
    metadata: VideoMetadata,
) -> tuple[VideoMetadata, Fraction]:
    """Return encoder-only metadata matching the source HEVC SPS VUI."""

    replacements: dict[str, object] = {}
    output_fps = metadata.video_fps_exact
    try:
        with av.open(metadata.video_file) as source:
            stream = source.streams.video[0]
            context_rate = stream.codec_context.framerate or stream.codec_context.rate
            if context_rate is not None and context_rate > 0:
                output_fps = Fraction(context_rate)
            frame = next(source.decode(stream), None)
            if frame is not None:
                try:
                    color_range = AvColorRange(int(frame.color_range))
                    colorspace = AvColorspace(int(frame.colorspace))
                except ValueError:
                    color_range = None
                    colorspace = None
                if (colorspace, color_range) in _COLOR_VARIANTS:
                    replacements["color_range"] = color_range
                    replacements["color_space"] = colorspace
                primaries = _COLOR_PRIMARIES_BY_CODE.get(int(frame.color_primaries))
                transfer = _COLOR_TRANSFERS_BY_CODE.get(int(frame.color_trc))
                if primaries is not None:
                    replacements["color_primaries"] = primaries
                if transfer is not None:
                    replacements["color_transfer"] = transfer
    except (av.FFmpegError, IndexError, TypeError, ValueError, OSError) as exc:
        logger.warning(
            "Could not read HEVC source VUI from %s: %s",
            metadata.video_file,
            exc,
        )
    replacements.update(
        video_fps=float(output_fps),
        average_fps=float(output_fps),
        video_fps_exact=output_fps,
    )
    return replace(metadata, **replacements), output_fps


def add_amd_hevc_smart_fragment_source_level(
    encoder_settings: Mapping[str, object],
    metadata: VideoMetadata,
    *,
    codec: str,
    vendor: AcceleratorVendor,
) -> dict[str, object]:
    """Add source HEVC level to Linux AMF fragments when not explicit."""

    effective = dict(encoder_settings)
    if (
        vendor is not AcceleratorVendor.AMD
        or sys.platform != "linux"
        or codec != "hevc"
        or "level" in effective
    ):
        return effective
    level = hevc_level_to_amf_option(metadata.hevc_level)
    if level is not None:
        effective["level"] = level
    return effective

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
FFMPEG_ENCODER_RATE_MAX = 2_147_483_647
WINDOWS_AMF_HEVC_VBR_PEAK_BUFFER_TARGET_RATIO = 2


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
    bufsize = maxrate * SOURCE_BITRATE_CAP_BUFFER_RATIO
    if maxrate > FFMPEG_ENCODER_RATE_MAX or bufsize > FFMPEG_ENCODER_RATE_MAX:
        logger.warning(
            "Source bitrate ceiling for %s exceeds the encoder option range; "
            "encoding without a source-tied bitrate ceiling",
            metadata.video_file,
        )
        return {}
    return {
        "maxrate": str(maxrate),
        "bufsize": str(bufsize),
    }


def _windows_amf_hevc_vbr_peak_options(metadata: VideoMetadata) -> dict[str, str]:
    """Build the Windows AMF HEVC diagnostic rate contract, or fail closed."""

    target = int(metadata.video_bitrate)
    if target <= 0:
        logger.warning(
            "No source bitrate for %s; cannot derive the Windows AMF HEVC VBR Peak contract",
            metadata.video_file,
        )
        return {}
    maxrate = int(target * SOURCE_BITRATE_CAP_FACTORS["hevc"])
    bufsize = target * WINDOWS_AMF_HEVC_VBR_PEAK_BUFFER_TARGET_RATIO
    if maxrate > FFMPEG_ENCODER_RATE_MAX or bufsize > FFMPEG_ENCODER_RATE_MAX:
        logger.warning(
            "Windows AMF HEVC VBR Peak contract for %s exceeds the encoder option range; "
            "refusing to derive a source-tied rate contract",
            metadata.video_file,
        )
        return {}
    return {
        "maxrate": str(maxrate),
        "bufsize": str(bufsize),
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
        if rc in cqp_modes and codec == "av1":
            qindex = str(_amf_av1_qindex(int(value)))
            overrides.update(qp_i=qindex, qp_p=qindex, qp_b=qindex)
        elif rc in cqp_modes:
            overrides["qp_i"] = value
            overrides["qp_p"] = value
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


def _amf_host_input(packed: torch.Tensor, *, ten_bit: bool) -> torch.Tensor:
    return packed.view(torch.uint16) if ten_bit else packed


def _mov_container_options(suffix: str, *, fmp4: bool) -> dict[str, str]:
    if suffix.lower() not in MOV_SUFFIXES:
        return {}
    # A fragmented MP4 writes a sample-free moov up front and one moof+mdat per
    # keyframe, so the growing file stays playable; +faststart instead relocates
    # a single moov at close, leaving the file unreadable until then. The two
    # are mutually exclusive.
    flags = "+frag_keyframe+empty_moov" if fmp4 else "+faststart"
    return {"movflags": flags}


def _normalized_audio_layout(layout: av.AudioLayout) -> av.AudioLayout:
    channels = layout.nb_channels
    if layout.name == f"{channels} channels" and channels <= 8:
        return av.AudioLayout(f"{channels}c")
    return layout


@dataclass(frozen=True)
class _ResolvedEncoderPolicy:
    spec: EncoderSpec
    options: dict[str, str]
    target_bit_rate: int | None
    host_zero_copy: bool


def resolve_encoder_options(
    vendor: AcceleratorVendor,
    codec: str,
    metadata: VideoMetadata,
    encoder_settings: dict[str, object],
    *,
    smart_fragment: bool,
    match_input_bit_depth: bool = False,
    auto_source_rate: bool = False,
) -> tuple[EncoderSpec, dict[str, str]]:
    """Resolve the shared output policy without creating a GPU/codec session."""
    policy = _resolve_encoder_policy(
        vendor, codec, metadata, encoder_settings,
        smart_fragment=smart_fragment,
        match_input_bit_depth=match_input_bit_depth,
        auto_source_rate=auto_source_rate,
    )
    return policy.spec, policy.options


def _resolve_encoder_policy(
    vendor: AcceleratorVendor,
    codec: str,
    metadata: VideoMetadata,
    encoder_settings: dict[str, object],
    *,
    smart_fragment: bool,
    match_input_bit_depth: bool = False,
    auto_source_rate: bool = False,
    prefer_amf_host_native: bool = False,
    resident_coordinator: object | None = None,
) -> _ResolvedEncoderPolicy:
    """Pick the encoder spec and the final FFmpeg options for one output."""
    specs = AMF_ENCODER_SPECS if vendor is AcceleratorVendor.AMD else ENCODER_SPECS
    if codec not in specs:
        raise ValueError(f"Unsupported codec: {codec}")
    spec = specs[codec]
    # Smart-render fragments are spliced between copied source spans, so they
    # keep the source bit depth instead of the codec's 10-bit default.
    if (smart_fragment or match_input_bit_depth) and codec in {"hevc", "av1"} and not metadata.is_10bit:
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
    target_bit_rate: int | None = None
    amd_av1_main10 = (
        vendor is AcceleratorVendor.AMD
        and codec == "av1"
        and spec.ten_bit
    )
    amf_hevc_vbr_peak_override = _amf_hevc_vbr_peak_override()
    linux_amd_hevc = (
        vendor is AcceleratorVendor.AMD
        and sys.platform == "linux"
        and codec == "hevc"
    )
    linux_amd_h264_smart = (
        vendor is AcceleratorVendor.AMD
        and sys.platform == "linux"
        and codec == "h264"
        and smart_fragment
    )
    windows_amd_hevc = (
        vendor is AcceleratorVendor.AMD
        and sys.platform == "win32"
        and codec == "hevc"
    )
    windows_amd_hevc_full_encode = windows_amd_hevc and not smart_fragment
    amf_host_zero_copy_override = _amf_host_zero_copy_override()
    linux_amd_hevc_validated_host_output = (
        linux_amd_hevc
        and _amf_host_native_output_eligible(
            width=int(metadata.video_width),
            height=int(metadata.video_height),
            ten_bit=bool(spec.ten_bit),
            frame_format=spec.frame_format,
        )
    )
    # Main10/P010 uses the accepted automatic host-native route from the
    # 3840x2160-equivalent pixel count.
    # Main/NV12 showed only a small single-session gain, so it is selected
    # automatically only as the required input contract for a requested
    # and separately validated dual-GOP writer.  The explicit override
    # remains the fail-closed research entry for other HEVC output sizes.
    if (
        amf_host_zero_copy_override is True
        and resident_coordinator is None
        and not linux_amd_hevc
    ):
        raise ValueError(
            f"{AMF_HOST_ZERO_COPY_ENV}=1 is supported only for Linux AMD "
            "HEVC encoding"
        )
    host_zero_copy = (
        amf_host_zero_copy_override is True
        or (
            amf_host_zero_copy_override is None
            and linux_amd_hevc_validated_host_output
            and (
                bool(spec.ten_bit)
                or bool(prefer_amf_host_native)
            )
        )
    )
    if resident_coordinator is not None:
        if amf_host_zero_copy_override is True:
            raise ValueError(
                f"{AMF_HOST_ZERO_COPY_ENV}=1 conflicts with the Windows "
                "D3D11/HIP resident encoder"
            )
        host_zero_copy = False
    if amf_hevc_vbr_peak_override is True:
        if windows_amd_hevc and smart_fragment:
            raise ValueError(
                f"{AMF_HEVC_VBR_PEAK_ENV}=1 is not supported for Windows AMD "
                "HEVC Smart Render: the pinned runtime cannot complete copy/render "
                "seam validation"
            )
        if not (linux_amd_hevc or windows_amd_hevc_full_encode):
            raise ValueError(
                f"{AMF_HEVC_VBR_PEAK_ENV}=1 is supported only for Linux AMD HEVC "
                "or Windows AMD HEVC full encoding"
            )
    explicit_rate_policy = {
        "rc",
        "maxrate",
        "bufsize",
        "qp_i",
        "qp_p",
        "qvbr_quality_level",
    } & set(encoder_settings)
    auto_amf_hevc_vbr_peak = (
        amf_hevc_vbr_peak_override is None
        and linux_amd_hevc
        and (smart_fragment or auto_source_rate)
        and not explicit_rate_policy
    )
    auto_amf_hevc_rate_options: dict[str, str] | None = None
    if auto_amf_hevc_vbr_peak:
        auto_amf_hevc_rate_options = source_bitrate_cap_options(
            metadata,
            output_codec=codec,
            vendor=vendor,
        )
        if set(auto_amf_hevc_rate_options) != {"maxrate", "bufsize"}:
            logger.warning(
                "Linux AMD HEVC source-rate mode could not derive a "
                "vbr_peak contract; retaining the existing CQP policy"
            )
            auto_amf_hevc_rate_options = None
    amf_hevc_vbr_peak = (
        amf_hevc_vbr_peak_override is True
        or auto_amf_hevc_rate_options is not None
    )
    if amd_av1_main10:
        # Supported Linux and Windows AMF runtimes cannot reliably open
        # P010 AV1 while PreAnalysis is enabled.
        encoder_options["preanalysis"] = "0"
    use_amd_hevc_smart_fragment_cqp = (
        vendor is AcceleratorVendor.AMD
        and sys.platform == "linux"
        and codec == "hevc"
        and smart_fragment
        and not amf_hevc_vbr_peak
        and "cq" in encoder_settings
        and "rc" not in encoder_settings
        and "qvbr_quality_level" not in encoder_settings
    )
    if encoder_settings:
        overrides = {k: _option_value(v) for k, v in encoder_settings.items()}
        # FFmpeg accepts both spellings for HEVC/H.264, but their defaults
        # use the underscore key. Normalize the alias so a user override
        # replaces that default instead of passing two conflicting options.
        if "spatial-aq" in overrides and "spatial_aq" in encoder_options:
            overrides["spatial_aq"] = overrides.pop("spatial-aq")
        if vendor is AcceleratorVendor.AMD:
            if use_amd_hevc_smart_fragment_cqp:
                portable_cq = int(overrides.pop("cq"))
                cqp = max(0, min(51, portable_cq + AMD_HEVC_CQP_OFFSET))
                encoder_options.pop("qvbr_quality_level", None)
                encoder_options.pop("vbaq", None)
                encoder_options.update(
                    {
                        "rc": "cqp",
                        "qp_i": str(cqp),
                        "qp_p": str(cqp),
                        "preanalysis": "0",
                    }
                )
            else:
                _normalize_amf_cq(
                    codec,
                    overrides,
                    encoder_options,
                    ten_bit=spec.ten_bit,
                )
        else:
            _drop_unsupported_nvenc_overrides(codec, overrides, encoder_options)
    uses_amf_cqp = (
        vendor is AcceleratorVendor.AMD
        and codec in {"hevc", "av1"}
        and overrides.get("rc", encoder_options["rc"]) in {"cqp", "0"}
    )
    if "maxrate" not in overrides and not uses_amf_cqp:
        encoder_options.update(
            source_bitrate_cap_options(
                metadata,
                output_codec=codec,
                vendor=vendor,
            )
        )
    encoder_options.update(overrides)
    if linux_amd_h264_smart:
        # This AMF runtime requires PreAnalysis for QVBR, but persistent
        # H.264 PA sessions can stop returning packets on long Smart Render
        # spans.  Peak VBR preserves a source-tied size contract without PA.
        if metadata.video_bitrate <= 0:
            raise ValueError(
                "Linux AMD H.264 Smart Render requires a positive source "
                "video bitrate for its stable vbr_peak contract"
            )
        rate_options = source_bitrate_cap_options(
            metadata,
            output_codec=codec,
            vendor=vendor,
        )
        if set(rate_options) != {"maxrate", "bufsize"}:
            raise ValueError(
                "Linux AMD H.264 Smart Render could not derive a safe "
                "source-rate peak/buffer contract"
            )
        encoder_options.update(rate_options)
        encoder_options.update(
            {
                "rc": "vbr_peak",
                "preanalysis": "0",
                "vbaq": "0",
            }
        )
        encoder_options.pop("qvbr_quality_level", None)
        target_bit_rate = int(metadata.video_bitrate)
        logger.info(
            "Linux AMD H.264 Smart Render vbr_peak: target=%d peak=%s "
            "buffer=%s preanalysis=0",
            target_bit_rate,
            encoder_options["maxrate"],
            encoder_options["bufsize"],
        )
    if resident_coordinator is not None:
        resident_encoder_contract = {
            "g": "60",
            "bf": "0",
            "preanalysis": "0",
            # FFmpeg's AMF encoder defaults async_depth to 16 and only
            # blocks in QueryOutput once that many hardware surfaces are
            # queued.  The resident bridge intentionally owns exactly
            # four output surfaces, so a larger depth can consume all four
            # before FFmpeg drives output retrieval.  Match the codec's
            # progress threshold to the audited resident pool bound.
            "async_depth": "4",
        }
        incompatible = {
            name: value
            for name, value in overrides.items()
            if name in resident_encoder_contract
            and value != resident_encoder_contract[name]
        }
        if incompatible:
            raise ValueError(
                "The Windows D3D11/HIP resident encoder requires "
                "g=60, bf=0, preanalysis=0, and async_depth=4; "
                "incompatible overrides: "
                f"{incompatible}"
            )
        encoder_options.update(resident_encoder_contract)
        logger.info(
            "Windows D3D11/HIP resident encoder contract: "
            "g=60 bf=0 preanalysis=0 async_depth=4"
        )
    if amf_hevc_vbr_peak:
        explicit_rate_options = sorted(
            {"maxrate", "bufsize"} & overrides.keys()
        )
        if explicit_rate_options:
            raise ValueError(
                f"{AMF_HEVC_VBR_PEAK_ENV}=1 derives target/peak/buffer from "
                "the source; remove custom " + ", ".join(explicit_rate_options)
            )
        explicit_rc = overrides.get("rc")
        if explicit_rc not in {None, "vbr_peak", "2"}:
            raise ValueError(
                f"{AMF_HEVC_VBR_PEAK_ENV}=1 conflicts with rc={explicit_rc!r}"
            )
        if metadata.video_bitrate <= 0:
            raise ValueError(
                f"{AMF_HEVC_VBR_PEAK_ENV}=1 requires a positive source video bitrate"
            )
        if windows_amd_hevc_full_encode:
            rate_options = _windows_amf_hevc_vbr_peak_options(metadata)
        else:
            rate_options = auto_amf_hevc_rate_options or source_bitrate_cap_options(
                metadata,
                output_codec=codec,
                vendor=vendor,
            )
        if set(rate_options) != {"maxrate", "bufsize"}:
            raise ValueError(
                f"{AMF_HEVC_VBR_PEAK_ENV}=1 could not derive a safe peak/buffer"
            )
        encoder_options.update(rate_options)
        encoder_options.update(
            {
                "rc": "vbr_peak",
                "preanalysis": "0",
                "vbaq": "0",
            }
        )
        encoder_options.pop("qp_i", None)
        encoder_options.pop("qp_p", None)
        encoder_options.pop("qvbr_quality_level", None)
        target_bit_rate = int(metadata.video_bitrate)
        logger.info(
            "%s AMF HEVC vbr_peak: target=%d peak=%s buffer=%s preanalysis=0",
            (
                "Forced"
                if amf_hevc_vbr_peak_override is True
                else (
                    "Automatic Smart Render"
                    if smart_fragment
                    else "Automatic full encode"
                )
            ),
            target_bit_rate,
            encoder_options["maxrate"],
            encoder_options["bufsize"],
        )
    if (
        smart_fragment
        and vendor is AcceleratorVendor.AMD
        and sys.platform == "linux"
        and codec == "hevc"
    ):
        # Repeated HEVC fragment sessions can abort natively with
        # PreAnalysis enabled; custom settings cannot re-enable it.
        encoder_options["preanalysis"] = "0"
    if smart_fragment:
        encoder_options.update(AMF_SMART_FRAGMENT_OPTIONS if vendor is AcceleratorVendor.AMD else NVENC_SMART_FRAGMENT_OPTIONS)
    if host_zero_copy:
        # AMF retains every wrapped pointer until the associated output is
        # queried. Four inputs bound the live 8K P010 host owners to about
        # 384 MiB while preserving the measured encoder overlap.
        encoder_options.update(
            {
                "async_depth": "4",
                "host_zero_copy": "1",
            }
        )
        logger.info(
            "%s Linux AMD HEVC %s AMF host-native input: async_depth=4",
            "Forced" if amf_host_zero_copy_override is True else "Automatic",
            "Main10/P010" if spec.ten_bit else "Main/NV12",
        )

    return _ResolvedEncoderPolicy(spec, encoder_options, target_bit_rate, host_zero_copy)


@dataclass(order=True, frozen=True)
class _BufferedEncodeItem:
    """Keep each buffered tensor paired with its own PTS and LUT decision."""

    pts: int
    sequence: int
    frame: torch.Tensor = field(compare=False)
    apply_lut: bool = field(compare=False)


class VideoEncoder:
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
        mux_audio: bool | None = None,
        match_input_bit_depth: bool = False,
        auto_source_rate: bool = False,
        prefer_amf_host_native: bool = False,
        fmp4: bool = False,
        resident_coordinator: object | None = None,
    ):
        self.device = torch.device(device)
        self.vendor = vendor_for_device(self.device)
        self._resident_coordinator = resident_coordinator
        self.resident_encode_telemetry: list[dict[str, object]] = []
        if self.vendor not in {AcceleratorVendor.NVIDIA, AcceleratorVendor.AMD}:
            raise RuntimeError(
                f"GPU video encoding is not supported on {self.vendor.value}"
            )
        if smart_fragment:
            encoder_settings = add_amd_hevc_smart_fragment_source_level(
                encoder_settings,
                metadata,
                codec=codec,
                vendor=self.vendor,
            )
        if self._resident_coordinator is not None:
            if self.vendor is not AcceleratorVendor.AMD:
                raise ValueError(
                    "The Windows D3D11/HIP resident encoder requires AMD"
                )
            if sys.platform != "win32" or codec != "hevc" or smart_fragment:
                raise ValueError(
                    "The Windows D3D11/HIP resident encoder supports only "
                    "Windows AMD full HEVC encoding"
                )
        policy = _resolve_encoder_policy(
            self.vendor, codec, metadata, encoder_settings,
            smart_fragment=smart_fragment,
            match_input_bit_depth=match_input_bit_depth,
            auto_source_rate=auto_source_rate,
            prefer_amf_host_native=prefer_amf_host_native,
            resident_coordinator=resident_coordinator,
        )
        spec, self.encoder_options = policy.spec, policy.options
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
        self.mux_audio = not smart_fragment if mux_audio is None else bool(mux_audio)
        self.auto_source_rate = bool(auto_source_rate)
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

        self._converter = RgbToYuvConverter(f"{pixel_format}_{color_variant}", device=self.device)

        self._target_bit_rate = policy.target_bit_rate
        self._amf_host_zero_copy = policy.host_zero_copy

        # AMF receives a blocking host copy, so it needs only a small producer
        # window. Retaining the NVENC-sized window can pin up to twelve cloned
        # full RGB frames across the reorder heap and worker queue; at 8K that
        # is more than 1 GiB. Four entries preserve overlap and PTS ordering
        # while halving that AMD-only working set.
        self.BUFFER_MAX_SIZE = (
            4 if self.vendor is AcceleratorVendor.AMD else 8
        )
        self.frame_buffer: list[_BufferedEncodeItem] = []
        self._next_buffer_sequence = 0
        # Set on AMD in __enter__, where the frame size is known; NVIDIA leaves
        # them None and allocates per frame (NVENC outlives encode()).
        self._packed: torch.Tensor | None = None
        self._cas_luma: torch.Tensor | None = None
        self._source_chapters = ()

    def _video_stream_kwargs(self) -> dict[str, object]:
        stream_kwargs: dict[str, object] = {
            "rate": self.output_fps,
            "options": dict(self.encoder_options),
        }
        if (
            self.vendor is AcceleratorVendor.AMD
            and not self._amf_host_zero_copy
            and self._resident_coordinator is None
        ):
            stream_kwargs["hwaccel"] = HWAccel(
                "amf",
                device=str(self.device.index or 0),
                allow_software_fallback=False,
                is_hw_owned=False,
            )
        # PyAV's encoding HWAccel uploads software frames into an AMF hardware
        # frame pool before avcodec_send_frame(). The validated host-native
        # route deliberately leaves only AVCodecContext.hw_device_ctx in place
        # so the contiguous P010 frame reaches amfenc's guarded wrap branch.
        return stream_kwargs

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

        stream_kwargs = self._video_stream_kwargs()
        pix_fmt = self.spec.frame_format if self.vendor is AcceleratorVendor.AMD else "cuda"
        out_v = self.dst.add_stream(self.encoder_name, **stream_kwargs)
        if self.codec == "hevc" and self.output_path.suffix.lower() in MOV_SUFFIXES:
            out_v.codec_tag = "hvc1"
        out_v.width = self.metadata.video_width
        out_v.height = self.metadata.video_height
        out_v.time_base = self.metadata.time_base
        ctx = out_v.codec_context
        if self._target_bit_rate is not None:
            ctx.bit_rate = self._target_bit_rate
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

        if self._resident_coordinator is not None:
            self._resident_coordinator.bind_encoder_context(ctx)

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
            dtype = torch.uint16 if self.spec.ten_bit else torch.uint8
            if (
                not self._amf_host_zero_copy
                and self._resident_coordinator is None
            ):
                self._host_yuv = torch.empty(
                    (height + height // 2, width),
                    dtype=dtype,
                    pin_memory=True,
                )
        self.frame_buffer = []
        self._next_buffer_sequence = 0
        self.pts_set: set[int] = set()
        self._last_emitted_pts: int | None = None
        self._video_started = False
        self._options_validated = False
        self._worker_error: Exception | None = None

        self._stop_sentinel = object()
        self._encode_queue: queue.Queue = queue.Queue(maxsize=ENCODE_BUFFER_SIZE)
        self._encode_thread = threading.Thread(target=self._encode_worker, name="VideoEncoderWorker", daemon=True)
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
        self._warned_source_dts: set[int] = set()
        if self.smart_fragment or not self.mux_audio:
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
                if in_stream.type == "audio":
                    raise RuntimeError(
                        "Source audio stream "
                        f"{in_stream.index} has no codec context; the unified "
                        "FFmpeg runtime must include its audio decoder instead "
                        "of silently producing a video without sound"
                    )
                if in_stream.type != "data" or not source_formats & output_formats:
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
                in_stream.codec_context.name,
                self.output_path.suffix,
                self.dst.supported_codecs,
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
                        supported_codecs=self.dst.supported_codecs,
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
        worker_error = None
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
            worker_error = self._worker_error
            try:
                self.dst.close()
            finally:
                try:
                    self._src.close()
                finally:
                    self._release_runtime_references()
        if exc_type is None and worker_error is not None:
            raise worker_error

    def _release_runtime_references(self) -> None:
        """Drop per-session Torch/PyAV owners before the next Smart span opens."""

        self.frame_buffer.clear()
        self.pts_set.clear()
        self._packed = None
        self._cas_luma = None
        self._host_yuv = None
        self._converter = None
        self._lut_applier = None
        self._cas = None
        self._source_iter = None
        self._source_backlog.clear()
        self._source_pipes.clear()
        self.out_stream = None
        self.dst = None
        self._src = None
        self._cuda_ctx = None
        self.stream = None
        self._encode_thread = None
        self._encode_queue = None

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
        sample_time_base = Fraction(1, out_stream.codec_context.sample_rate)
        for aframe in in_packet.decode():
            if aframe.pts is not None:
                aframe.pts = round(aframe.pts * aframe.time_base / sample_time_base)
            aframe.time_base = sample_time_base
            for rframe in processor.resample(aframe):
                out_packets.extend(out_stream.encode(rframe))
        return out_packets

    @staticmethod
    def _copy_source_packets(in_packet, out_stream, bitstream_filter):
        if in_packet.size == 0:
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
                    first_nudge = packet.stream.index not in self._warned_source_dts
                    self._warned_source_dts.add(packet.stream.index)
                    logger.log(
                        logging.WARNING if first_nudge else logging.DEBUG,
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
        if len(self.frame_buffer) > (self.BUFFER_MAX_SIZE // 2) or (flush_all and self.frame_buffer):
            buffered = heapq.heappop(self.frame_buffer)
            self.pts_set.remove(buffered.pts)
            pts_to_assign = self._clamp_pts_monotonic(buffered.pts)
            item = self._build_encode_item(
                buffered.frame,
                pts_to_assign,
                buffered.apply_lut,
            )
            self._encode_queue.put(item)

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
        # no direct HIP-to-AMF path: the AMD branch synchronizes then performs
        # a blocking D2H into pinned host memory. One device buffer can therefore
        # serve every frame even when AMF retains distinct host-native owners.
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
            elif self._resident_coordinator is not None:
                hw_frame, telemetry = self._resident_coordinator.acquire_encoder_frame(
                    packed.data_ptr(),
                    packed.numel() * packed.element_size(),
                    int(pts),
                    int(self.stream.cuda_stream),
                )
                frame_format = getattr(
                    getattr(hw_frame, "format", None), "name", None
                )
                if (
                    frame_format != "amf"
                    or int(getattr(hw_frame, "width", -1))
                    != int(self.metadata.video_width)
                    or int(getattr(hw_frame, "height", -1))
                    != int(self.metadata.video_height)
                    or int(telemetry.get("slot_count", -1)) != 4
                    or int(telemetry.get("width", -1))
                    != int(self.metadata.video_width)
                    or int(telemetry.get("height", -1))
                    != int(self.metadata.video_height)
                ):
                    raise RuntimeError(
                        "Windows D3D11/HIP resident bridge returned an invalid "
                        f"encoder frame: format={frame_format}, telemetry={telemetry}"
                    )
                self.resident_encode_telemetry.append(telemetry)

        if (
            self.vendor is AcceleratorVendor.AMD
            and self._resident_coordinator is None
        ):
            # Issue #252: isolated E1/E2 were clean; residual glitches under full
            # pipeline load matched AMF reading host planes while a non-blocking
            # D2H was still in flight. Finish convert on the stream, then
            # blocking-copy into pinned host so from_dlpack sees complete planes.
            # Phase 4 (gfx1201): stream.synchronize() + blocking copy cleared P1;
            # do not escalate to full device.synchronize() unless field reports return.
            self.stream.synchronize()
            host_yuv = self._host_yuv
            if self._amf_host_zero_copy:
                # AMF may retain a wrapped host surface after encode() returns.
                # Give every submission an independent pinned owner; PyAV's
                # DLPack AVBuffer refs keep it alive until FFmpeg queries output.
                dtype = torch.uint16 if self.spec.ten_bit else torch.uint8
                host_yuv = torch.empty(
                    (height + height // 2, self.metadata.video_width),
                    dtype=dtype,
                    pin_memory=True,
                )
            host_yuv.copy_(
                _amf_host_input(packed, ten_bit=self.spec.ten_bit),
                non_blocking=False,
            )
            planes = [host_yuv[:height], host_yuv[height:]]
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
        # AMD decode batches can expose storage that is reused after the next
        # native batch. Own it before the asynchronous encoder window retains
        # the tensor; NVIDIA keeps the established producer-owned fast path.
        owned_frame = (
            frame.clone()
            if self.vendor is AcceleratorVendor.AMD and isinstance(frame, torch.Tensor)
            else frame
        )
        heapq.heappush(
            self.frame_buffer,
            _BufferedEncodeItem(
                pts=pts,
                sequence=self._next_buffer_sequence,
                frame=owned_frame,
                apply_lut=bool(apply_lut),
            ),
        )
        self._next_buffer_sequence += 1
        self.pts_set.add(pts)
        self._process_buffer()
