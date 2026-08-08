"""Literal encoder quality defaults and native ranges."""

from __future__ import annotations

import math
from dataclasses import dataclass

from jasna.accelerator import AcceleratorVendor


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
        "av1": EncoderCqSpec(default=32, minimum=0, maximum=51),
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
