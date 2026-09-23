"""User encoder settings: supported keys per vendor/codec, parsing, validation, and CQ ranges."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from jasna.accelerator import AcceleratorVendor, vendor_for_device


@dataclass(frozen=True)
class EncoderCqSpec:
    default: int
    minimum: int
    maximum: int


_ENCODER_CQ_SPECS: dict[
    AcceleratorVendor,
    dict[str, EncoderCqSpec],
] = {
    AcceleratorVendor.NVIDIA: {
        "h264": EncoderCqSpec(default=25, minimum=1, maximum=51),
        "hevc": EncoderCqSpec(default=28, minimum=1, maximum=51),
        "av1": EncoderCqSpec(default=35, minimum=1, maximum=63),
    },
    AcceleratorVendor.AMD: {
        "h264": EncoderCqSpec(default=24, minimum=0, maximum=51),
        "hevc": EncoderCqSpec(default=25, minimum=0, maximum=51),
        "av1": EncoderCqSpec(default=32, minimum=1, maximum=51),
    },
}


def encoder_cq_spec(
    codec: str,
    vendor: AcceleratorVendor | str,
) -> EncoderCqSpec:
    resolved_vendor = AcceleratorVendor(str(vendor))
    try:
        by_codec = _ENCODER_CQ_SPECS[resolved_vendor]
    except KeyError as exc:
        raise ValueError(
            f"CQ controls are not supported on {resolved_vendor.value}"
        ) from exc
    try:
        return by_codec[codec]
    except KeyError as exc:
        raise ValueError(f"Unsupported codec: {codec}") from exc


def validate_encoder_cq(
    value: object,
    *,
    codec: str,
    vendor: AcceleratorVendor | str,
) -> int | float:
    spec = encoder_cq_spec(codec, vendor)
    resolved_vendor = AcceleratorVendor(str(vendor))
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(
            f"CQ must be a number for {codec} on {resolved_vendor.value.upper()}"
        )
    if not spec.minimum <= value <= spec.maximum:
        raise ValueError(
            f"CQ for {codec} on {resolved_vendor.value.upper()} must be in "
            f"{spec.minimum}..{spec.maximum} (got {value})"
        )
    return value


# ffmpeg *_nvenc option names shared by every codec
_COMMON_ENCODER_SETTINGS: frozenset[str] = frozenset(
    {
        "preset",
        "tune",
        "rc",
        "cq",
        "qmin",
        "qmax",
        "nonref_p",
        "g",
        "temporal-aq",
        "rc-lookahead",
        "lookahead_level",
        "aq-strength",
        "init_qpI",
        "init_qpP",
        "init_qpB",
        "bf",
        "b_ref_mode",
        "maxrate",
        "bufsize",
        "multipass",
        "b_adapt",
        "weighted_pred",
        "tf_level",
    }
)

# hevc_nvenc/h264_nvenc accept both AQ spellings; av1_nvenc only the hyphen one.
SUPPORTED_ENCODER_SETTINGS_BY_CODEC: dict[str, frozenset[str]] = {
    "hevc": _COMMON_ENCODER_SETTINGS | {"profile", "tier", "spatial_aq", "spatial-aq"},
    "h264": _COMMON_ENCODER_SETTINGS | {"profile", "coder", "spatial_aq", "spatial-aq"},
    "av1": _COMMON_ENCODER_SETTINGS | {"tier", "spatial-aq", "tile-rows", "tile-columns"},
}

SUPPORTED_ENCODER_SETTINGS: frozenset[str] = frozenset().union(
    *SUPPORTED_ENCODER_SETTINGS_BY_CODEC.values()
)

# User-facing AMF settings. ``cq`` is kept as a portable Jasna option; the
# encoder maps it to QVBR for H.264 and constant QP for HEVC/AV1.
_COMMON_AMF_ENCODER_SETTINGS: frozenset[str] = frozenset(
    {
        "preset",
        "usage",
        "quality",
        "rc",
        "cq",
        "qvbr_quality_level",
        "g",
        "bf",
        "preanalysis",
        "maxrate",
        "bufsize",
        "profile",
        "level",
    }
)

AMF_SUPPORTED_ENCODER_SETTINGS_BY_CODEC: dict[str, frozenset[str]] = {
    "hevc": _COMMON_AMF_ENCODER_SETTINGS | {"tier", "bitdepth", "vbaq"},
    "h264": _COMMON_AMF_ENCODER_SETTINGS
    | {"coder", "bf_ref", "pa_adaptive_mini_gop", "vbaq"},
    "av1": _COMMON_AMF_ENCODER_SETTINGS | {"bitdepth", "aq_mode"},
}

AMF_SUPPORTED_ENCODER_SETTINGS: frozenset[str] = frozenset().union(
    *AMF_SUPPORTED_ENCODER_SETTINGS_BY_CODEC.values()
)


def _parse_encoder_setting_scalar(value: str) -> object:
    v = value.strip()
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def parse_encoder_settings(value: str) -> dict[str, object]:
    value = (value or "").strip()
    if value == "":
        return {}

    if value.startswith("{"):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("--encoder-settings JSON must be an object")
        return parsed

    settings: dict[str, object] = {}
    for part in value.split(","):
        part = part.strip()
        if part == "":
            continue
        if "=" not in part:
            raise ValueError(f"Invalid --encoder-settings item: {part!r} (expected key=value)")
        k, v = part.split("=", 1)
        k = k.strip()
        if k == "":
            raise ValueError(f"Invalid --encoder-settings item: {part!r} (empty key)")
        settings[k] = _parse_encoder_setting_scalar(v)

    return settings


def validate_encoder_settings(
    settings: dict[str, object],
    codec: str | None = None,
    *,
    vendor: AcceleratorVendor | str | None = None,
) -> dict[str, object]:
    resolved_vendor = (
        vendor_for_device()
        if vendor is None
        else AcceleratorVendor(str(vendor))
    )
    by_codec = (
        AMF_SUPPORTED_ENCODER_SETTINGS_BY_CODEC
        if resolved_vendor is AcceleratorVendor.AMD
        else SUPPORTED_ENCODER_SETTINGS_BY_CODEC
    )
    supported_all = (
        AMF_SUPPORTED_ENCODER_SETTINGS
        if resolved_vendor is AcceleratorVendor.AMD
        else SUPPORTED_ENCODER_SETTINGS
    )
    if "spatial_aq" in settings and "spatial-aq" in settings:
        raise ValueError(
            "Conflicting encoder settings: spatial_aq and spatial-aq are aliases; use only one"
        )
    if codec is None:
        supported = supported_all
        scope = "Supported"
    else:
        if codec not in by_codec:
            raise ValueError(f"Unsupported codec: {codec}")
        supported = by_codec[codec]
        scope = f"Supported for {codec}"
    invalid = sorted(set(settings.keys()) - set(supported))
    if invalid:
        raise ValueError(
            f"Unsupported encoder setting(s){'' if codec is None else f' for codec {codec}'}: "
            + ", ".join(invalid)
            + f". {scope}: "
            + ", ".join(sorted(supported))
        )
    return settings
