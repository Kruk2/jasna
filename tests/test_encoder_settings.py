from __future__ import annotations

from contextlib import nullcontext

import pytest

from jasna.accelerator import AcceleratorVendor
from jasna.media.encoder_settings import resolve_custom_encoder_settings, validate_encoder_cq


@pytest.mark.parametrize("value", [1, 25, 51])
def test_validate_encoder_cq_returns_literal_value(value: int) -> None:
    assert validate_encoder_cq(
        value,
        codec="h264",
        vendor=AcceleratorVendor.NVIDIA,
    ) == value


@pytest.mark.parametrize("value", [0, 52])
def test_validate_encoder_cq_rejects_values_outside_supported_range(value: int) -> None:
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


@pytest.mark.parametrize(
    ("settings", "cq", "vendor", "expected"),
    [
        ({}, None, AcceleratorVendor.NVIDIA, {"cq": 28}),
        ({"g": 120}, 31, AcceleratorVendor.NVIDIA, {"g": 120, "cq": 31}),
        ({"cq": 20, "g": 120}, None, AcceleratorVendor.NVIDIA, {"cq": 20, "g": 120}),
        ({"qvbr_quality_level": 30}, None, AcceleratorVendor.AMD, {"cq": 30}),
    ],
    ids=["codec-default", "given", "in-settings", "amd-qvbr-alias"],
)
def test_resolve_encoder_settings_takes_the_given_then_the_set_then_the_default_cq(settings, cq, vendor, expected) -> None:
    from jasna.media.encoder_settings import resolve_encoder_settings

    assert resolve_encoder_settings(settings, cq=cq, codec="hevc", vendor=vendor) == expected


def test_qvbr_quality_level_is_a_cq_control_only_on_amd() -> None:
    from jasna.media.encoder_settings import cq_controls

    settings = {"cq": 20, "qvbr_quality_level": 30}
    assert cq_controls(settings, AcceleratorVendor.AMD) == ["cq", "qvbr_quality_level"]
    assert cq_controls(settings, AcceleratorVendor.NVIDIA) == ["cq"]


def test_cli_legacy_amf_quality_alias_suppresses_default() -> None:
    assert resolve_custom_encoder_settings(
        "qvbr_quality_level=30,g=120",
        cq=None,
        codec="hevc",
        vendor="amd",
    ) == {"g": 120, "cq": 30}


def test_cli_legacy_amf_quality_alias_uses_supported_range() -> None:
    with pytest.raises(ValueError, match=r"hevc.*AMD.*0\.\.51"):
        resolve_custom_encoder_settings(
            "qvbr_quality_level=52",
            cq=None,
            codec="hevc",
            vendor=AcceleratorVendor.AMD,
        )


def test_cli_direct_cq_conflicts_with_amf_alias() -> None:
    with pytest.raises(ValueError, match="remove qvbr_quality_level"):
        resolve_custom_encoder_settings(
            "qvbr_quality_level=30",
            cq=25,
            codec="hevc",
            vendor=AcceleratorVendor.AMD,
        )


def test_cli_rejects_both_legacy_amf_cq_aliases() -> None:
    with pytest.raises(ValueError, match="multiple CQ controls"):
        resolve_custom_encoder_settings(
            "cq=25,qvbr_quality_level=30",
            cq=None,
            codec="hevc",
            vendor=AcceleratorVendor.AMD,
        )


def test_amd_rejects_a_video_smaller_than_its_encoder_takes_and_names_codecs_that_fit() -> None:
    from jasna.media.encoder_settings import AMF_MIN_SIZE, require_encodable_size
    from jasna.media.probe import UnsupportedVideoError

    min_width, min_height = AMF_MIN_SIZE["hevc"]
    with pytest.raises(UnsupportedVideoError) as exc:
        require_encodable_size(AcceleratorVendor.AMD, "hevc", min_width - 2, 240)
    assert str(exc.value) == (
        f"The AMD video encoder can't make HEVC videos smaller than {min_width}x{min_height} pixels, "
        f"and this video is {min_width - 2}x240. Choose H.264 or AV1 instead."
    )
    with pytest.raises(UnsupportedVideoError, match="smaller than"):
        require_encodable_size(AcceleratorVendor.AMD, "hevc", 1920, min_height - 2)


@pytest.mark.parametrize("codec", ["h264", "hevc", "av1"])
@pytest.mark.parametrize(("vendor", "shrink", "refused"), [
    (AcceleratorVendor.AMD, 0, False),
    (AcceleratorVendor.AMD, 2, True),
    (AcceleratorVendor.NVIDIA, 2, False),
])
def test_only_amd_refuses_a_video_below_its_encoders_minimum_size(codec, vendor, shrink, refused) -> None:
    from jasna.media.encoder_settings import AMF_MIN_SIZE, require_encodable_size
    from jasna.media.probe import UnsupportedVideoError

    width, height = AMF_MIN_SIZE[codec]
    with pytest.raises(UnsupportedVideoError, match="smaller than") if refused else nullcontext():
        require_encodable_size(vendor, codec, width - shrink, height)
