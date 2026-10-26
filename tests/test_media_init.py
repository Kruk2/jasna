import json
from fractions import Fraction
from unittest.mock import MagicMock, patch

import pytest
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.accelerator import AcceleratorVendor
from jasna.media.encoder_settings import (
    SUPPORTED_ENCODER_SETTINGS_BY_CODEC,
    _parse_encoder_setting_scalar,
    parse_encoder_settings,
    validate_encoder_settings,
)
from jasna.media.probe import (
    is_stream_10bit,
    get_video_meta_data,
    parse_sample_aspect_ratio,
    parse_video_bitrate,
    resolve_video_start_pts,
)


@pytest.mark.parametrize(
    ("stream_start", "metadata_start", "expected"),
    [
        (900, 300, 900),
        (0, 300, 0),
        (None, 300, 300),
    ],
)
def test_resolve_video_start_pts(stream_start, metadata_start, expected) -> None:
    assert resolve_video_start_pts(stream_start, metadata_start) == expected


@pytest.mark.parametrize(
    ("stream", "fmt", "expected"),
    [
        ({"bit_rate": "18913000"}, {}, 18913000),
        ({"bit_rate": "18913000"}, {"bit_rate": "19200000"}, 18913000),
        ({"tags": {"BPS": "13567000"}}, {}, 13567000),
        ({"tags": {"BPS-eng": "6505000"}}, {}, 6505000),
        ({}, {"bit_rate": "25224000"}, 25224000),
        ({"bit_rate": "N/A"}, {"bit_rate": "25224000"}, 25224000),
        ({"bit_rate": "0"}, {}, 0),
        ({}, {}, 0),
    ],
)
def test_parse_video_bitrate(stream, fmt, expected) -> None:
    assert parse_video_bitrate(stream, fmt) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("true", True), ("True", True), ("TRUE", True), (" true ", True), ("false", False), ("False", False),
        ("42", 42), ("-1", -1), ("0", 0), ("  42  ", 42), ("3.14", 3.14), ("-0.5", -0.5),
        ("hello", "hello"), ("P5", "P5"),
    ],
)
def test_encoder_setting_values_parse_to_their_type(text, expected) -> None:
    value = _parse_encoder_setting_scalar(text)
    assert value == expected and type(value) is type(expected)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", {}),
        ("  ", {}),
        (None, {}),
        ('{"cq": 22, "lookahead": 32}', {"cq": 22, "lookahead": 32}),
        ("cq=22,lookahead=32", {"cq": 22, "lookahead": 32}),
        ("cq = 22 , lookahead = 32", {"cq": 22, "lookahead": 32}),
        ("preset=P5,temporalaq=true", {"preset": "P5", "temporalaq": True}),
        ("cq=22,", {"cq": 22}),
    ],
)
def test_parse_encoder_settings(text, expected) -> None:
    assert parse_encoder_settings(text) == expected


@pytest.mark.parametrize(("text", "error"), [("[1, 2]", "expected key=value"), ("cq22", "expected key=value"), ("=22", "empty key")])
def test_parse_encoder_settings_rejects_malformed_text(text, error) -> None:
    with pytest.raises(ValueError, match=error):
        parse_encoder_settings(text)


def _validate(settings, codec="hevc"):
    return validate_encoder_settings(settings, codec=codec, vendor=AcceleratorVendor.NVIDIA)


@pytest.mark.parametrize("codec", ["hevc", "h264", "av1"])
def test_every_supported_key_and_the_common_settings_are_accepted(codec):
    # spatial_aq/spatial-aq are aliases and may not be combined.
    supported = SUPPORTED_ENCODER_SETTINGS_BY_CODEC[codec]
    for alias in ("spatial-aq", "spatial_aq"):
        settings = {k: 0 for k in supported if k != alias}
        assert _validate(settings, codec) == settings
    common = {"preset": "p5", "cq": 25, "rc-lookahead": 32, "bf": 4, "maxrate": "10M"}
    assert _validate(common, codec) == common


@pytest.mark.parametrize(
    ("settings", "codec"),
    [
        ({"profile": "x"}, "hevc"),
        ({"profile": "x"}, "h264"),
        ({"spatial_aq": 1}, "hevc"),
        ({"spatial_aq": 1}, "h264"),
        ({"spatial-aq": 1}, "av1"),
        ({"tile-rows": 2, "tile-columns": 2}, "av1"),
        ({"coder": "cabac"}, "h264"),
        ({}, "hevc"),
    ],
)
def test_codec_specific_settings_are_accepted_for_their_codec(settings, codec):
    assert _validate(settings, codec) == settings


@pytest.mark.parametrize(
    ("settings", "codec", "error"),
    [
        ({"cq": 22, "bad_key": 1}, "hevc", "Unsupported encoder setting"),
        ({"profile": "main"}, "av1", "for codec av1.*profile"),
        ({"spatial_aq": 1}, "av1", "for codec av1.*spatial_aq"),
        ({"tile-rows": 2}, "hevc", "for codec hevc"),
        ({"coder": "cabac"}, "hevc", "for codec hevc"),
        ({"tier": "high"}, "h264", r"for codec h264.*tier.*Supported for h264"),
        ({"spatial_aq": 1, "spatial-aq": 1}, "hevc", "Conflicting encoder settings.*spatial"),
        ({}, "vp9", "Unsupported codec: vp9"),
    ],
)
def test_settings_outside_the_codec_are_rejected_by_name(settings, codec, error):
    with pytest.raises(ValueError, match=error):
        _validate(settings, codec)


@pytest.mark.parametrize(
    ("stream", "expected"),
    [
        ({"bits_per_raw_sample": 10}, True),
        ({"bits_per_raw_sample": 8}, False),
        ({"bits_per_raw_sample": "10"}, True),
        ({"bits_per_raw_sample": "8"}, False),
        ({"bits_per_raw_sample": 10.0}, True),
        ({"pix_fmt": "yuv420p10le"}, True),
        ({"pix_fmt": "p010"}, True),
        ({"pix_fmt": "x2rgb10le"}, True),
        ({"pix_fmt": "yuv420p"}, False),
        ({}, False),
        ({"bits_per_raw_sample": None, "pix_fmt": "yuv420p"}, False),
        ({"bits_per_raw_sample": "abc", "pix_fmt": "yuv420p"}, False),
    ],
)
def test_is_stream_10bit(stream, expected):
    assert is_stream_10bit(stream) is expected


@pytest.mark.parametrize(
    ("stream", "expected"),
    [
        ({"sample_aspect_ratio": "8:9"}, Fraction(8, 9)),
        ({"sample_aspect_ratio": "1:1"}, Fraction(1, 1)),
        ({}, Fraction(1, 1)),
        ({"sample_aspect_ratio": "0:1"}, Fraction(1, 1)),
        ({"sample_aspect_ratio": "N/A"}, Fraction(1, 1)),
    ],
)
def test_parse_sample_aspect_ratio_defaults_to_square(stream, expected):
    assert parse_sample_aspect_ratio(stream) == expected


def _probe(*, returncode: int = 0, drop: tuple[str, ...] = (), **overrides):
    """``get_video_meta_data`` over an ffprobe that reports one 1080p 23.976 fps HEVC stream."""
    stream = {
        "width": 1920,
        "height": 1080,
        "r_frame_rate": "24000/1001",
        "avg_frame_rate": "24000/1001",
        "time_base": "1/24000",
        "codec_name": "hevc",
        "nb_frames": "100",
        "duration": "4.17",
        "start_pts": 0,
        "color_range": "tv",
        "color_space": "bt709",
        "bits_per_raw_sample": "8",
        "pix_fmt": "yuv420p",
    }
    stream.update(overrides)
    for key in drop:
        del stream[key]
    process = MagicMock(returncode=returncode)
    process.communicate.return_value = (json.dumps({"streams": [stream], "format": {"duration": "4.17"}}).encode(), b"error message")
    with (
        patch("jasna.media.probe.resolve_executable", return_value="ffprobe"),
        patch("jasna.media.probe.subprocess.Popen", return_value=process),
    ):
        return get_video_meta_data("test.mp4")


def test_probe_reads_the_video_stream():
    meta = _probe()

    assert (meta.video_width, meta.video_height, meta.codec_name, meta.num_frames) == (1920, 1080, "hevc", 100)
    assert meta.video_fps == pytest.approx(24000 / 1001)
    assert (meta.video_fps_exact, meta.time_base) == (Fraction(24000, 1001), Fraction(1, 24000))
    assert (meta.is_10bit, meta.color_range, meta.color_space) == (False, AvColorRange.MPEG, AvColorspace.ITU709)
    assert meta.sample_aspect_ratio == Fraction(1, 1)


@pytest.mark.parametrize(
    ("stream", "field", "expected"),
    [
        (dict(bits_per_raw_sample="10", pix_fmt="yuv420p10le"), "is_10bit", True),
        (dict(color_primaries="bt2020"), "color_primaries", "bt2020"),
        (dict(color_transfer="smpte2084"), "color_transfer", "smpte2084"),
        (dict(color_space="bt601"), "color_space", AvColorspace.ITU601),
        (dict(color_space="bt470bg"), "color_space", AvColorspace.ITU601),
        (dict(color_space="smpte170m"), "color_space", AvColorspace.ITU601),
        (dict(color_range="jpeg"), "color_range", AvColorRange.JPEG),
        (dict(drop=("color_range",)), "color_range", AvColorRange.MPEG),
        (dict(drop=("color_space",)), "color_space", AvColorspace.ITU709),
        (dict(sample_aspect_ratio="8:9"), "sample_aspect_ratio", Fraction(8, 9)),
        (dict(side_data_list=[{"side_data_type": "Stereo 3D", "type": "side by side"}]), "stereo_layout", "side by side"),
        (dict(side_data_list=[{"side_data_type": "Spherical Mapping", "projection": "equirectangular"}]),
         "spherical_projection", "equirectangular"),
        (dict(tags={"stereo_mode": "left_right"}), "stereo_layout", "left_right"),
        (dict(tags={"projection": "equirectangular"}), "spherical_projection", "equirectangular"),
    ],
)
def test_probe_reads_color_and_spatial_metadata(stream, field, expected):
    assert getattr(_probe(**stream), field) == expected


def test_missing_frame_count_falls_back_to_counting():
    with patch("jasna.media.probe._frame_count_from_container", return_value=50) as count:
        assert _probe(nb_frames="0").num_frames == 50
    count.assert_called_once_with("test.mp4")


def test_ffprobe_failure_raises():
    with pytest.raises(Exception, match="error running ffprobe"):
        _probe(returncode=1)
