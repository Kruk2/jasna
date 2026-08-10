from __future__ import annotations

import pytest

from jasna.accelerator import AcceleratorVendor
from jasna.main import _resolve_cli_encoder_settings
from jasna.media.encoder_quality import encoder_cq_spec, validate_encoder_cq


@pytest.mark.parametrize(
    ("vendor", "codec", "default", "minimum", "maximum"),
    [
        (AcceleratorVendor.NVIDIA, "h264", 25, 1, 51),
        (AcceleratorVendor.NVIDIA, "hevc", 28, 1, 51),
        (AcceleratorVendor.NVIDIA, "av1", 35, 1, 63),
        (AcceleratorVendor.AMD, "h264", 24, 0, 51),
        (AcceleratorVendor.AMD, "hevc", 25, 0, 51),
        (AcceleratorVendor.AMD, "av1", 32, 0, 51),
    ],
)
def test_encoder_cq_specs_follow_native_vendor_scales(
    vendor: AcceleratorVendor,
    codec: str,
    default: int,
    minimum: int,
    maximum: int,
) -> None:
    spec = encoder_cq_spec(codec, vendor)

    assert (spec.default, spec.minimum, spec.maximum) == (
        default,
        minimum,
        maximum,
    )


@pytest.mark.parametrize("value", [1, 25, 51])
def test_validate_encoder_cq_returns_literal_value(value: int) -> None:
    assert validate_encoder_cq(
        value,
        codec="h264",
        vendor=AcceleratorVendor.NVIDIA,
    ) == value


@pytest.mark.parametrize("value", [0, 52])
def test_validate_encoder_cq_rejects_values_outside_native_range(value: int) -> None:
    with pytest.raises(ValueError, match=r"h264.*NVIDIA.*1\.\.51"):
        validate_encoder_cq(
            value,
            codec="h264",
            vendor=AcceleratorVendor.NVIDIA,
        )


def test_validate_encoder_cq_rejects_non_numeric_value() -> None:
    with pytest.raises(ValueError, match="CQ must be a number"):
        validate_encoder_cq(
            "high",
            codec="hevc",
            vendor=AcceleratorVendor.NVIDIA,
        )


def test_cli_cq_defaults_follow_amd_codec() -> None:
    assert _resolve_cli_encoder_settings(
        "",
        cq=None,
        codec="av1",
        vendor=AcceleratorVendor.AMD,
    ) == {"cq": 32}


def test_cli_legacy_amf_quality_alias_suppresses_default() -> None:
    assert _resolve_cli_encoder_settings(
        "qvbr_quality_level=30,g=120",
        cq=None,
        codec="hevc",
        vendor="amd",
    ) == {"g": 120, "cq": 30}


def test_cli_legacy_amf_quality_alias_uses_native_range() -> None:
    with pytest.raises(ValueError, match=r"hevc.*AMD.*0\.\.51"):
        _resolve_cli_encoder_settings(
            "qvbr_quality_level=52",
            cq=None,
            codec="hevc",
            vendor=AcceleratorVendor.AMD,
        )


def test_cli_direct_cq_conflicts_with_amf_alias() -> None:
    with pytest.raises(ValueError, match="--cq.*qvbr_quality_level"):
        _resolve_cli_encoder_settings(
            "qvbr_quality_level=30",
            cq=25,
            codec="hevc",
            vendor=AcceleratorVendor.AMD,
        )


def test_cli_rejects_both_legacy_amf_cq_aliases() -> None:
    with pytest.raises(ValueError, match="multiple CQ controls"):
        _resolve_cli_encoder_settings(
            "cq=25,qvbr_quality_level=30",
            cq=None,
            codec="hevc",
            vendor=AcceleratorVendor.AMD,
        )
