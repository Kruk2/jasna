"""Which NVENC/AMF encoder and FFmpeg options an output gets: per-codec defaults, user
overrides, the source-tied bitrate ceiling and NVENC VRAM estimates."""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping

from jasna.accelerator import AcceleratorVendor
from jasna.media.encoder_settings import cq_controls, encoder_cq_spec, validate_encoder_settings
from jasna.media.probe import VideoMetadata

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
    **{key: value for key, value in DEFAULT_ENCODER_OPTIONS.items() if key != "lookahead_level"},
    "profile": "high",
    # CQ 25 kept representative capped and uncapped H.264 encodes above VMAF 95.
    "cq": str(encoder_cq_spec("h264", AcceleratorVendor.NVIDIA).default),
}

# AV1 target quality uses a 0..63 scale rather than H.264/HEVC's 0..51.
# CQ 35 matches HEVC CQ 28 on the same seven-above-HEVC scale. AV1 QP limits use a separate
# 0..255 scale, so the HEVC qmin/qmax/init_qp values must not be copied here.
# No profile: P010 input makes av1_nvenc emit AV1 Main 10-bit on its own.
# av1_nvenc only consumes the hyphenated spatial-aq spelling.
_NOT_FOR_AV1_NVENC = {"profile", "qmin", "qmax", "init_qpI", "init_qpP", "init_qpB"}
DEFAULT_AV1_ENCODER_OPTIONS: dict[str, str] = {
    **{
        ("spatial-aq" if key == "spatial_aq" else key): value
        for key, value in DEFAULT_ENCODER_OPTIONS.items()
        if key not in _NOT_FOR_AV1_NVENC
    },
    "cq": str(encoder_cq_spec("av1", AcceleratorVendor.NVIDIA).default),
}


def _amf_av1_qindex(cq: int) -> int:
    # AMF AV1 frame QPs are AV1 qindex 0..255; CQ uses the H.264/HEVC QP scale 0..51.
    return cq * 255 // 51


def _amf_constant_qp(codec: str, cq: int) -> dict[str, str]:
    if codec == "av1":
        qindex = str(_amf_av1_qindex(cq))
        return {"qp_i": qindex, "qp_p": qindex, "qp_b": qindex}
    if codec == "h264":
        return {"qp_i": str(cq), "qp_p": str(cq), "qp_b": str(cq)}
    return {"qp_i": str(cq), "qp_p": str(cq)}


def _amf_default_options(codec: str, **codec_options: str) -> dict[str, str]:
    return {
        "usage": "high_quality",
        "quality": "quality",
        "rc": "cqp",
        **_amf_constant_qp(codec, encoder_cq_spec(codec, AcceleratorVendor.AMD).default),
        "g": "250",
        "preanalysis": "0",
        **codec_options,
    }


NVENC_SMART_FRAGMENT_OPTIONS = MappingProxyType({"forced-idr": "1"})
AMF_SMART_FRAGMENT_OPTIONS = MappingProxyType({"forced_idr": "1"})

_FRAME_FORMATS = {"hevc": "p010le", "h264": "nv12", "av1": "p010le"}


@dataclass(frozen=True)
class EncoderSpec:
    encoder_name: str
    frame_format: str  # PyAV hardware-frame software format: "nv12" or "p010le"
    default_options: Mapping[str, str]

    @property
    def ten_bit(self) -> bool:
        return self.frame_format == "p010le"


def _encoder_specs(backend: str, default_options: dict[str, dict[str, str]]) -> dict[str, EncoderSpec]:
    return {
        codec: EncoderSpec(f"{codec}_{backend}", _FRAME_FORMATS[codec], MappingProxyType(options))
        for codec, options in default_options.items()
    }


ENCODER_SPECS: dict[str, EncoderSpec] = _encoder_specs("nvenc", {
    "hevc": DEFAULT_ENCODER_OPTIONS,
    "h264": DEFAULT_H264_ENCODER_OPTIONS,
    "av1": DEFAULT_AV1_ENCODER_OPTIONS,
})

AMF_ENCODER_SPECS: dict[str, EncoderSpec] = _encoder_specs("amf", {
    "hevc": _amf_default_options("hevc", vbaq="0", profile="main10", bitdepth="10"),
    "h264": _amf_default_options("h264", vbaq="0", profile="high"),
    "av1": _amf_default_options("av1", aq_mode="none", profile="main", bitdepth="10"),
})

NVENC_MAX_SURFACES = 64
LARGE_FRAME_PIXELS = 4096 * 2160
"""Frames above DCI 4K get a shorter default NVENC lookahead: at 8K, rc-lookahead 32 holds 22
more 96 MiB frames than the B-frames need, which left LTX too little VRAM on 16 GB cards (#790)."""
NVENC_WORK_FRAMES = 20
"""NVENC's own buffers (references, lookahead analysis) in output frames: 16-22 measured at 4K
and 8K, HEVC and AV1, on an RTX 5090 (2026-11-16)."""


def nvenc_held_frames(rc_lookahead: int, b_frames: int) -> int:
    """Input frames NVENC keeps referenced: FFmpeg's ``nvenc_recalc_surfaces`` with the default
    ``surfaces`` and ``delay``, whose output starts only once all but one surface are queued."""
    frame_interval_p = b_frames + 1
    surfaces = max(4, frame_interval_p * 4)
    if rc_lookahead > 0:
        surfaces = max(surfaces, rc_lookahead + frame_interval_p + 5)
    return min(surfaces, NVENC_MAX_SURFACES)


def lookahead_within_b_frame_surfaces(b_frames: int) -> int:
    """The longest lookahead NVENC runs on the surfaces its B-frames already hold."""
    return max(0, nvenc_held_frames(0, b_frames) - (b_frames + 1) - 5)


def nvenc_vram_bytes(spec: EncoderSpec, options: Mapping[str, str], *, width: int, height: int) -> int:
    """VRAM an NVENC encode holds: its queued input frames plus its own buffers."""
    frame_bytes = width * height * 3 // 2 * (2 if spec.ten_bit else 1)
    held = nvenc_held_frames(int(options.get("rc-lookahead", 0)), int(options.get("bf", 0)))
    return (held + NVENC_WORK_FRAMES) * frame_bytes


def nvenc_vram_beside_ltx(spec: EncoderSpec, options: Mapping[str, str], *, width: int, height: int) -> int:
    """VRAM the 768 px decision keeps free for NVENC: only frames above DCI 4K, since its
    threshold was measured with a smaller encode already on the card."""
    if width * height <= LARGE_FRAME_PIXELS:
        return 0
    return nvenc_vram_bytes(spec, options, width=width, height=height)


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
    frame_generation: bool,
) -> dict[str, str]:
    """With ``frame_generation`` the output has twice the frames, so the ceiling doubles to
    keep each frame's share of it."""
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
    maxrate = int(metadata.video_bitrate * factor * (2 if frame_generation else 1))
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
    if rc not in cqp_modes:
        defaults.pop("qp_i", None)
        defaults.pop("qp_p", None)
        defaults.pop("qp_b", None)
    if codec in {"hevc", "av1"} and ten_bit and rc in qvbr_modes:
        raise ValueError(
            f"AMD {codec.upper()} Main10 does not support QVBR or HQVBR; use the default CQP mode"
        )

    aliases = cq_controls(overrides, AcceleratorVendor.AMD)
    if len(aliases) > 1:
        raise ValueError(
            "Conflicting encoder settings: cq and qvbr_quality_level are aliases "
            "on AMD; use only one"
        )
    if not aliases:
        return

    value = overrides.pop(aliases[0])
    if rc in cqp_modes:
        overrides.update(_amf_constant_qp(codec, int(value)))
    elif codec == "h264" or (not ten_bit and rc in qvbr_modes):
        overrides["qvbr_quality_level"] = value
    else:
        raise ValueError(f"AMD {codec.upper()} CQ requires rc=cqp")




def _requests_eight_bit_output(codec: str, overrides: Mapping[str, str]) -> bool:
    # NVENC and AMF take their output depth from the input frames, so an 8-bit
    # profile or bit depth only works when the frames are 8-bit too.
    return (codec == "hevc" and overrides.get("profile") == "main") or overrides.get("bitdepth") == "8"


def _eight_bit_spec(spec: EncoderSpec, codec: str) -> EncoderSpec:
    options = dict(spec.default_options)
    if codec == "hevc":
        options["profile"] = "main"
    # AMF pins output depth via "bitdepth"; dropping it lets FFmpeg
    # derive 8-bit from the nv12 input instead of conflicting with it.
    options.pop("bitdepth", None)
    return replace(spec, frame_format="nv12", default_options=MappingProxyType(options))


def resolve_encoder_options(
    vendor: AcceleratorVendor,
    codec: str,
    metadata: VideoMetadata,
    encoder_settings: dict[str, object],
    *,
    smart_fragment: bool,
    frame_generation: bool = False,
) -> tuple[EncoderSpec, dict[str, str]]:
    """Pick the encoder spec and the final FFmpeg options for one output."""
    specs = AMF_ENCODER_SPECS if vendor is AcceleratorVendor.AMD else ENCODER_SPECS
    if codec not in specs:
        raise ValueError(f"Unsupported codec: {codec}")
    spec = specs[codec]
    if encoder_settings:
        validate_encoder_settings(encoder_settings, codec=codec, vendor=vendor)
    overrides = {k: _option_value(v) for k, v in encoder_settings.items()}
    # Smart-render fragments are spliced between copied source spans, so they
    # keep the source bit depth instead of the codec's 10-bit default.
    matches_eight_bit_source = smart_fragment and not metadata.is_10bit
    if codec in {"hevc", "av1"} and (
        matches_eight_bit_source or _requests_eight_bit_output(codec, overrides)
    ):
        spec = _eight_bit_spec(spec, codec)

    encoder_options = dict(spec.default_options)
    if overrides:
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
        and overrides.get("rc", encoder_options["rc"]) in {"cqp", "0"}
    )
    if "maxrate" not in overrides and not uses_amf_cqp:
        encoder_options.update(
            source_bitrate_cap_options(metadata, output_codec=codec, vendor=vendor, frame_generation=frame_generation)
        )
    encoder_options.update(overrides)
    if (
        vendor is AcceleratorVendor.NVIDIA
        and metadata.video_width * metadata.video_height > LARGE_FRAME_PIXELS
        and not {"rc-lookahead", "lookahead_level"} & overrides.keys()
    ):
        encoder_options["rc-lookahead"] = str(lookahead_within_b_frame_surfaces(int(encoder_options["bf"])))
    if smart_fragment:
        encoder_options.update(
            AMF_SMART_FRAGMENT_OPTIONS if vendor is AcceleratorVendor.AMD else NVENC_SMART_FRAGMENT_OPTIONS
        )
    return spec, encoder_options
