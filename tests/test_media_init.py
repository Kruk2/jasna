import json
from fractions import Fraction
from unittest.mock import MagicMock, patch

import pytest
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.media import (
    SUPPORTED_ENCODER_SETTINGS,
    _parse_encoder_setting_scalar,
    parse_encoder_settings,
    validate_encoder_settings,
    is_stream_10bit,
    get_video_meta_data,
    VideoMetadata,
)


class TestParseEncoderSettingScalar:
    def test_true(self):
        assert _parse_encoder_setting_scalar("true") is True
        assert _parse_encoder_setting_scalar("True") is True
        assert _parse_encoder_setting_scalar("TRUE") is True

    def test_false(self):
        assert _parse_encoder_setting_scalar("false") is False
        assert _parse_encoder_setting_scalar("False") is False

    def test_int(self):
        assert _parse_encoder_setting_scalar("42") == 42
        assert _parse_encoder_setting_scalar("-1") == -1
        assert _parse_encoder_setting_scalar("0") == 0

    def test_float(self):
        assert _parse_encoder_setting_scalar("3.14") == pytest.approx(3.14)
        assert _parse_encoder_setting_scalar("-0.5") == pytest.approx(-0.5)

    def test_string_fallback(self):
        assert _parse_encoder_setting_scalar("hello") == "hello"
        assert _parse_encoder_setting_scalar("P5") == "P5"

    def test_whitespace_stripped(self):
        assert _parse_encoder_setting_scalar("  42  ") == 42
        assert _parse_encoder_setting_scalar(" true ") is True


class TestParseEncoderSettings:
    def test_empty_string(self):
        assert parse_encoder_settings("") == {}
        assert parse_encoder_settings("  ") == {}
        assert parse_encoder_settings(None) == {}

    def test_json_object(self):
        result = parse_encoder_settings('{"cq": 22, "lookahead": 32}')
        assert result == {"cq": 22, "lookahead": 32}

    def test_json_array_starting_with_brace_is_not_dict(self):
        # This path is hard to trigger since JSON arrays start with '['.
        # Just verify array input goes to kv parsing.
        with pytest.raises(ValueError, match="expected key=value"):
            parse_encoder_settings("[1, 2]")

    def test_key_value_pairs(self):
        result = parse_encoder_settings("cq=22,lookahead=32")
        assert result == {"cq": 22, "lookahead": 32}

    def test_key_value_with_spaces(self):
        result = parse_encoder_settings("cq = 22 , lookahead = 32")
        assert result == {"cq": 22, "lookahead": 32}

    def test_key_value_bool_and_string(self):
        result = parse_encoder_settings("preset=P5,temporalaq=true")
        assert result == {"preset": "P5", "temporalaq": True}

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError, match="expected key=value"):
            parse_encoder_settings("cq22")

    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="empty key"):
            parse_encoder_settings("=22")

    def test_trailing_comma_ok(self):
        result = parse_encoder_settings("cq=22,")
        assert result == {"cq": 22}


class TestValidateEncoderSettings:
    def test_valid_settings(self):
        settings = {"cq": 22, "lookahead": 32, "preset": "P5"}
        assert validate_encoder_settings(settings) == settings

    def test_empty_settings(self):
        assert validate_encoder_settings({}) == {}

    def test_invalid_key_raises(self):
        with pytest.raises(ValueError, match="Unsupported encoder setting"):
            validate_encoder_settings({"cq": 22, "bad_key": 1})

    def test_all_supported_keys_accepted(self):
        settings = {k: 0 for k in SUPPORTED_ENCODER_SETTINGS}
        assert validate_encoder_settings(settings) == settings


class TestIsStream10bit:
    def test_bits_per_raw_sample_int_10(self):
        assert is_stream_10bit({"bits_per_raw_sample": 10}) is True

    def test_bits_per_raw_sample_int_8(self):
        assert is_stream_10bit({"bits_per_raw_sample": 8}) is False

    def test_bits_per_raw_sample_string_10(self):
        assert is_stream_10bit({"bits_per_raw_sample": "10"}) is True

    def test_bits_per_raw_sample_string_8(self):
        assert is_stream_10bit({"bits_per_raw_sample": "8"}) is False

    def test_bits_per_raw_sample_float_10(self):
        assert is_stream_10bit({"bits_per_raw_sample": 10.0}) is True

    def test_pix_fmt_p010(self):
        assert is_stream_10bit({"pix_fmt": "yuv420p10le"}) is True

    def test_pix_fmt_p10(self):
        assert is_stream_10bit({"pix_fmt": "p010"}) is True

    def test_pix_fmt_8bit(self):
        assert is_stream_10bit({"pix_fmt": "yuv420p"}) is False

    def test_no_relevant_fields(self):
        assert is_stream_10bit({}) is False

    def test_bits_per_raw_sample_none(self):
        assert is_stream_10bit({"bits_per_raw_sample": None, "pix_fmt": "yuv420p"}) is False

    def test_pix_fmt_rgb10(self):
        assert is_stream_10bit({"pix_fmt": "x2rgb10le"}) is True

    def test_bits_per_raw_sample_invalid_string(self):
        assert is_stream_10bit({"bits_per_raw_sample": "abc", "pix_fmt": "yuv420p"}) is False


class TestGetVideoMetaData:
    def _make_ffprobe_output(self, **overrides):
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
        return json.dumps({"streams": [stream], "format": {"duration": "4.17"}}).encode()

    @patch("jasna.media.resolve_executable", return_value="ffprobe")
    @patch("jasna.media.subprocess.Popen")
    def test_basic_metadata_extraction(self, mock_popen, mock_resolve):
        proc = MagicMock()
        proc.communicate.return_value = (self._make_ffprobe_output(), b"")
        proc.returncode = 0
        mock_popen.return_value = proc

        meta = get_video_meta_data("test.mp4")

        assert isinstance(meta, VideoMetadata)
        assert meta.video_width == 1920
        assert meta.video_height == 1080
        assert meta.codec_name == "hevc"
        assert meta.num_frames == 100
        assert meta.is_10bit is False
        assert meta.color_range == AvColorRange.MPEG
        assert meta.color_space == AvColorspace.ITU709

    @patch("jasna.media.resolve_executable", return_value="ffprobe")
    @patch("jasna.media.subprocess.Popen")
    def test_10bit_stream(self, mock_popen, mock_resolve):
        proc = MagicMock()
        proc.communicate.return_value = (
            self._make_ffprobe_output(bits_per_raw_sample="10", pix_fmt="yuv420p10le"),
            b"",
        )
        proc.returncode = 0
        mock_popen.return_value = proc

        meta = get_video_meta_data("test.mp4")
        assert meta.is_10bit is True

    @patch("jasna.media.resolve_executable", return_value="ffprobe")
    @patch("jasna.media.subprocess.Popen")
    def test_ffprobe_failure_raises(self, mock_popen, mock_resolve):
        proc = MagicMock()
        proc.communicate.return_value = (b"", b"error message")
        proc.returncode = 1
        mock_popen.return_value = proc

        with pytest.raises(Exception, match="error running ffprobe"):
            get_video_meta_data("test.mp4")

    @patch("jasna.media.resolve_executable", return_value="ffprobe")
    @patch("jasna.media.subprocess.Popen")
    def test_fps_fraction(self, mock_popen, mock_resolve):
        proc = MagicMock()
        proc.communicate.return_value = (self._make_ffprobe_output(), b"")
        proc.returncode = 0
        mock_popen.return_value = proc

        meta = get_video_meta_data("test.mp4")
        assert meta.video_fps == pytest.approx(24000 / 1001)
        assert meta.video_fps_exact == Fraction(24000, 1001)
        assert meta.time_base == Fraction(1, 24000)

    @patch("jasna.media.resolve_executable", return_value="ffprobe")
    @patch("jasna.media.subprocess.Popen")
    def test_missing_nb_frames_falls_back_to_counting(self, mock_popen, mock_resolve):
        proc = MagicMock()
        proc.communicate.return_value = (
            self._make_ffprobe_output(nb_frames="0"),
            b"",
        )
        proc.returncode = 0
        mock_popen.return_value = proc

        with patch("jasna.media._get_frame_count_by_counting", return_value=50) as mock_count:
            meta = get_video_meta_data("test.mp4")
            mock_count.assert_called_once_with("test.mp4")
            assert meta.num_frames == 50

    @patch("jasna.media.resolve_executable", return_value="ffprobe")
    @patch("jasna.media.subprocess.Popen")
    def test_color_space_bt601(self, mock_popen, mock_resolve):
        proc = MagicMock()
        proc.communicate.return_value = (
            self._make_ffprobe_output(color_space="bt601"),
            b"",
        )
        proc.returncode = 0
        mock_popen.return_value = proc

        meta = get_video_meta_data("test.mp4")
        assert meta.color_space == AvColorspace.ITU601

    @patch("jasna.media.resolve_executable", return_value="ffprobe")
    @patch("jasna.media.subprocess.Popen")
    def test_color_space_bt470bg(self, mock_popen, mock_resolve):
        proc = MagicMock()
        proc.communicate.return_value = (
            self._make_ffprobe_output(color_space="bt470bg"),
            b"",
        )
        proc.returncode = 0
        mock_popen.return_value = proc

        meta = get_video_meta_data("test.mp4")
        assert meta.color_space == AvColorspace.ITU601

    @patch("jasna.media.resolve_executable", return_value="ffprobe")
    @patch("jasna.media.subprocess.Popen")
    def test_color_space_smpte170m(self, mock_popen, mock_resolve):
        proc = MagicMock()
        proc.communicate.return_value = (
            self._make_ffprobe_output(color_space="smpte170m"),
            b"",
        )
        proc.returncode = 0
        mock_popen.return_value = proc

        meta = get_video_meta_data("test.mp4")
        assert meta.color_space == AvColorspace.ITU601

    @patch("jasna.media.resolve_executable", return_value="ffprobe")
    @patch("jasna.media.subprocess.Popen")
    def test_color_range_jpeg(self, mock_popen, mock_resolve):
        proc = MagicMock()
        proc.communicate.return_value = (
            self._make_ffprobe_output(color_range="jpeg"),
            b"",
        )
        proc.returncode = 0
        mock_popen.return_value = proc

        meta = get_video_meta_data("test.mp4")
        assert meta.color_range == AvColorRange.JPEG

    @patch("jasna.media.resolve_executable", return_value="ffprobe")
    @patch("jasna.media.subprocess.Popen")
    def test_missing_color_fields_default_to_bt709_mpeg(self, mock_popen, mock_resolve):
        output = self._make_ffprobe_output()
        data = json.loads(output)
        del data["streams"][0]["color_range"]
        del data["streams"][0]["color_space"]
        output = json.dumps(data).encode()

        proc = MagicMock()
        proc.communicate.return_value = (output, b"")
        proc.returncode = 0
        mock_popen.return_value = proc

        meta = get_video_meta_data("test.mp4")
        assert meta.color_range == AvColorRange.MPEG
        assert meta.color_space == AvColorspace.ITU709
