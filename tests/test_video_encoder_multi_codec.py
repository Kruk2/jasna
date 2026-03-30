"""Tests for multi-codec support (H.264, HEVC, AV1) in NvidiaVideoEncoder."""
from __future__ import annotations

import pytest
from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch

from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.media import VideoMetadata
from jasna.main import build_parser


def _fake_metadata(**overrides) -> VideoMetadata:
    defaults = dict(
        video_file="fake_input.mkv",
        num_frames=100,
        video_fps=24.0,
        average_fps=24.0,
        video_fps_exact=Fraction(24, 1),
        codec_name="hevc",
        duration=100.0 / 24.0,
        video_width=1920,
        video_height=1080,
        time_base=Fraction(1, 24),
        start_pts=0,
        color_space=AvColorspace.ITU709,
        color_range=AvColorRange.MPEG,
        is_10bit=True,
    )
    defaults.update(overrides)
    return VideoMetadata(**defaults)


class _FakeThread:
    def __init__(self, *args, **kwargs):
        pass
    def start(self):
        pass
    def join(self, timeout=None):
        pass


class _FakeCudaStream:
    def __init__(self, *args, **kwargs):
        self.cuda_stream = 0
        self.wait_event = MagicMock()
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


class TestMultiCodecSupport:
    """Test that different codecs use correct profiles and formats."""

    def test_h264_uses_high_profile_nv12(self, tmp_path):
        # Import here to allow CLI tests to run without PyNvVideoCodec
        from jasna.media.video_encoder import NvidiaVideoEncoder
        """H.264 should use 'high' profile and NV12 format."""
        output_path = tmp_path / "output" / "result.mkv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mock_nvc_encoder = MagicMock()
        mock_nvc_encoder.EndEncode.return_value = []

        with (
            patch("jasna.media.video_encoder.nvc") as mock_nvc,
            patch("jasna.media.video_encoder.threading.Thread", _FakeThread),
            patch("jasna.media.video_encoder.torch.cuda.Stream", _FakeCudaStream),
        ):
            mock_nvc.CreateEncoder.return_value = mock_nvc_encoder
            import torch
            enc = NvidiaVideoEncoder(
                file=str(output_path),
                device=torch.device("cuda:0"),
                metadata=_fake_metadata(),
                codec="h264",
                encoder_settings={},
                stream_mode=False,
            )

            call_kwargs = mock_nvc.CreateEncoder.call_args[1]
            assert call_kwargs["codec"] == "h264"
            assert call_kwargs["profile"] == "high"
            assert call_kwargs["fmt"] == "NV12"
            assert enc._codec == "h264"
            enc.raw_hevc.close()

    def test_hevc_uses_main10_profile_yuv420_10bit(self, tmp_path):
        # Import here to allow CLI tests to run without PyNvVideoCodec
        from jasna.media.video_encoder import NvidiaVideoEncoder
        """HEVC should use 'main10' profile and YUV420_10BIT format for 10-bit."""
        output_path = tmp_path / "output" / "result.mkv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mock_nvc_encoder = MagicMock()
        mock_nvc_encoder.EndEncode.return_value = []

        with (
            patch("jasna.media.video_encoder.nvc") as mock_nvc,
            patch("jasna.media.video_encoder.threading.Thread", _FakeThread),
            patch("jasna.media.video_encoder.torch.cuda.Stream", _FakeCudaStream),
        ):
            mock_nvc.CreateEncoder.return_value = mock_nvc_encoder
            import torch
            enc = NvidiaVideoEncoder(
                file=str(output_path),
                device=torch.device("cuda:0"),
                metadata=_fake_metadata(),
                codec="hevc",
                encoder_settings={},
                stream_mode=False,
            )

            call_kwargs = mock_nvc.CreateEncoder.call_args[1]
            assert call_kwargs["codec"] == "hevc"
            assert call_kwargs["profile"] == "main10"
            assert call_kwargs["fmt"] == "YUV420_10BIT"
            assert enc._codec == "hevc"
            enc.raw_hevc.close()

    def test_av1_uses_main_profile_nv12(self, tmp_path):
        # Import here to allow CLI tests to run without PyNvVideoCodec
        from jasna.media.video_encoder import NvidiaVideoEncoder
        """AV1 should use 'main' profile and NV12 format."""
        output_path = tmp_path / "output" / "result.mkv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mock_nvc_encoder = MagicMock()
        mock_nvc_encoder.EndEncode.return_value = []

        with (
            patch("jasna.media.video_encoder.nvc") as mock_nvc,
            patch("jasna.media.video_encoder.threading.Thread", _FakeThread),
            patch("jasna.media.video_encoder.torch.cuda.Stream", _FakeCudaStream),
        ):
            mock_nvc.CreateEncoder.return_value = mock_nvc_encoder
            import torch
            enc = NvidiaVideoEncoder(
                file=str(output_path),
                device=torch.device("cuda:0"),
                metadata=_fake_metadata(),
                codec="av1",
                encoder_settings={},
                stream_mode=False,
            )

            call_kwargs = mock_nvc.CreateEncoder.call_args[1]
            assert call_kwargs["codec"] == "av1"
            assert call_kwargs["profile"] == "main"
            assert call_kwargs["fmt"] == "NV12"
            assert enc._codec == "av1"
            enc.raw_hevc.close()

    def test_h264_has_tuning_info_and_bf(self, tmp_path):
        # Import here to allow CLI tests to run without PyNvVideoCodec
        from jasna.media.video_encoder import NvidiaVideoEncoder
        """H.264 should have tuning_info and bf parameters."""
        output_path = tmp_path / "output" / "result.mkv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mock_nvc_encoder = MagicMock()
        mock_nvc_encoder.EndEncode.return_value = []

        with (
            patch("jasna.media.video_encoder.nvc") as mock_nvc,
            patch("jasna.media.video_encoder.threading.Thread", _FakeThread),
            patch("jasna.media.video_encoder.torch.cuda.Stream", _FakeCudaStream),
        ):
            mock_nvc.CreateEncoder.return_value = mock_nvc_encoder
            import torch
            enc = NvidiaVideoEncoder(
                file=str(output_path),
                device=torch.device("cuda:0"),
                metadata=_fake_metadata(),
                codec="h264",
                encoder_settings={},
                stream_mode=False,
            )

            call_kwargs = mock_nvc.CreateEncoder.call_args[1]
            assert call_kwargs["tuning_info"] == "high_quality"
            assert "bf" in call_kwargs
            enc.raw_hevc.close()

    def test_hevc_has_tuning_info_and_bf(self, tmp_path):
        # Import here to allow CLI tests to run without PyNvVideoCodec
        from jasna.media.video_encoder import NvidiaVideoEncoder
        """HEVC should have tuning_info and bf parameters."""
        output_path = tmp_path / "output" / "result.mkv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mock_nvc_encoder = MagicMock()
        mock_nvc_encoder.EndEncode.return_value = []

        with (
            patch("jasna.media.video_encoder.nvc") as mock_nvc,
            patch("jasna.media.video_encoder.threading.Thread", _FakeThread),
            patch("jasna.media.video_encoder.torch.cuda.Stream", _FakeCudaStream),
        ):
            mock_nvc.CreateEncoder.return_value = mock_nvc_encoder
            import torch
            enc = NvidiaVideoEncoder(
                file=str(output_path),
                device=torch.device("cuda:0"),
                metadata=_fake_metadata(),
                codec="hevc",
                encoder_settings={},
                stream_mode=False,
            )

            call_kwargs = mock_nvc.CreateEncoder.call_args[1]
            assert call_kwargs["tuning_info"] == "high_quality"
            assert "bf" in call_kwargs
            enc.raw_hevc.close()

    def test_av1_no_tuning_info_or_bf(self, tmp_path):
        # Import here to allow CLI tests to run without PyNvVideoCodec
        from jasna.media.video_encoder import NvidiaVideoEncoder
        """AV1 should NOT have tuning_info or bf parameters."""
        output_path = tmp_path / "output" / "result.mkv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mock_nvc_encoder = MagicMock()
        mock_nvc_encoder.EndEncode.return_value = []

        with (
            patch("jasna.media.video_encoder.nvc") as mock_nvc,
            patch("jasna.media.video_encoder.threading.Thread", _FakeThread),
            patch("jasna.media.video_encoder.torch.cuda.Stream", _FakeCudaStream),
        ):
            mock_nvc.CreateEncoder.return_value = mock_nvc_encoder
            import torch
            enc = NvidiaVideoEncoder(
                file=str(output_path),
                device=torch.device("cuda:0"),
                metadata=_fake_metadata(),
                codec="av1",
                encoder_settings={},
                stream_mode=False,
            )

            call_kwargs = mock_nvc.CreateEncoder.call_args[1]
            assert "tuning_info" not in call_kwargs
            assert "bf" not in call_kwargs
            enc.raw_hevc.close()

    def test_unknown_codec_defaults_to_main_nv12(self, tmp_path):
        # Import here to allow CLI tests to run without PyNvVideoCodec
        from jasna.media.video_encoder import NvidiaVideoEncoder
        """Unknown codec should default to 'main' profile and NV12 format."""
        output_path = tmp_path / "output" / "result.mkv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mock_nvc_encoder = MagicMock()
        mock_nvc_encoder.EndEncode.return_value = []

        with (
            patch("jasna.media.video_encoder.nvc") as mock_nvc,
            patch("jasna.media.video_encoder.threading.Thread", _FakeThread),
            patch("jasna.media.video_encoder.torch.cuda.Stream", _FakeCudaStream),
        ):
            mock_nvc.CreateEncoder.return_value = mock_nvc_encoder
            import torch
            enc = NvidiaVideoEncoder(
                file=str(output_path),
                device=torch.device("cuda:0"),
                metadata=_fake_metadata(),
                codec="unknown",
                encoder_settings={},
                stream_mode=False,
            )

            call_kwargs = mock_nvc.CreateEncoder.call_args[1]
            assert call_kwargs["profile"] == "main"
            assert call_kwargs["fmt"] == "NV12"
            enc.raw_hevc.close()


class TestCodecSpecificFrameConversion:
    """Test that _encode_frame uses correct conversion based on codec."""

    def test_hevc_uses_p010_conversion(self, tmp_path):
        # Import here to allow CLI tests to run without PyNvVideoCodec
        from jasna.media.video_encoder import NvidiaVideoEncoder
        """HEVC should use chw_rgb_to_p010_bt709_limited for 10-bit."""
        output_path = tmp_path / "output" / "result.mkv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mock_nvc_encoder = MagicMock()
        mock_nvc_encoder.EndEncode.return_value = []
        mock_nvc_encoder.Encode.return_value = b'\x00\x00\x01\x40'

        with (
            patch("jasna.media.video_encoder.nvc") as mock_nvc,
            patch("jasna.media.video_encoder.threading.Thread", _FakeThread),
            patch("jasna.media.video_encoder.torch.cuda.Stream", _FakeCudaStream),
            patch("jasna.media.video_encoder.chw_rgb_to_p010_bt709_limited") as mock_p010,
            patch("jasna.media.video_encoder.chw_rgb_to_nv12_bt709_limited") as mock_nv12,
        ):
            mock_nvc.CreateEncoder.return_value = mock_nvc_encoder
            mock_p010.return_value = MagicMock()
            mock_nv12.return_value = MagicMock()

            import torch
            enc = NvidiaVideoEncoder(
                file=str(output_path),
                device=torch.device("cuda:0"),
                metadata=_fake_metadata(),
                codec="hevc",
                encoder_settings={},
                stream_mode=False,
            )

            frame = torch.zeros(3, 8, 8)
            with patch("jasna.media.video_encoder.torch.cuda.stream"):
                enc._encode_frame(frame, pts=0)

            mock_p010.assert_called_once()
            mock_nv12.assert_not_called()
            enc.raw_hevc.close()

    def test_h264_uses_nv12_conversion(self, tmp_path):
        # Import here to allow CLI tests to run without PyNvVideoCodec
        from jasna.media.video_encoder import NvidiaVideoEncoder
        """H.264 should use chw_rgb_to_nv12_bt709_limited for 8-bit."""
        output_path = tmp_path / "output" / "result.mkv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mock_nvc_encoder = MagicMock()
        mock_nvc_encoder.EndEncode.return_value = []
        mock_nvc_encoder.Encode.return_value = b'\x00\x00\x01\x40'

        with (
            patch("jasna.media.video_encoder.nvc") as mock_nvc,
            patch("jasna.media.video_encoder.threading.Thread", _FakeThread),
            patch("jasna.media.video_encoder.torch.cuda.Stream", _FakeCudaStream),
            patch("jasna.media.video_encoder.chw_rgb_to_p010_bt709_limited") as mock_p010,
            patch("jasna.media.video_encoder.chw_rgb_to_nv12_bt709_limited") as mock_nv12,
        ):
            mock_nvc.CreateEncoder.return_value = mock_nvc_encoder
            mock_p010.return_value = MagicMock()
            mock_nv12.return_value = MagicMock()

            import torch
            enc = NvidiaVideoEncoder(
                file=str(output_path),
                device=torch.device("cuda:0"),
                metadata=_fake_metadata(),
                codec="h264",
                encoder_settings={},
                stream_mode=False,
            )

            frame = torch.zeros(3, 8, 8)
            with patch("jasna.media.video_encoder.torch.cuda.stream"):
                enc._encode_frame(frame, pts=0)

            mock_nv12.assert_called_once()
            mock_p010.assert_not_called()
            enc.raw_hevc.close()

    def test_av1_uses_nv12_conversion(self, tmp_path):
        # Import here to allow CLI tests to run without PyNvVideoCodec
        from jasna.media.video_encoder import NvidiaVideoEncoder
        """AV1 should use chw_rgb_to_nv12_bt709_limited for 8-bit."""
        output_path = tmp_path / "output" / "result.mkv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mock_nvc_encoder = MagicMock()
        mock_nvc_encoder.EndEncode.return_value = []
        mock_nvc_encoder.Encode.return_value = b'\x00\x00\x01\x40'

        with (
            patch("jasna.media.video_encoder.nvc") as mock_nvc,
            patch("jasna.media.video_encoder.threading.Thread", _FakeThread),
            patch("jasna.media.video_encoder.torch.cuda.Stream", _FakeCudaStream),
            patch("jasna.media.video_encoder.chw_rgb_to_p010_bt709_limited") as mock_p010,
            patch("jasna.media.video_encoder.chw_rgb_to_nv12_bt709_limited") as mock_nv12,
        ):
            mock_nvc.CreateEncoder.return_value = mock_nvc_encoder
            mock_p010.return_value = MagicMock()
            mock_nv12.return_value = MagicMock()

            import torch
            enc = NvidiaVideoEncoder(
                file=str(output_path),
                device=torch.device("cuda:0"),
                metadata=_fake_metadata(),
                codec="av1",
                encoder_settings={},
                stream_mode=False,
            )

            frame = torch.zeros(3, 8, 8)
            with patch("jasna.media.video_encoder.torch.cuda.stream"):
                enc._encode_frame(frame, pts=0)

            mock_nv12.assert_called_once()
            mock_p010.assert_not_called()
            enc.raw_hevc.close()


class TestCLICodecSupport:
    """Test CLI argument parsing for codec selection."""

    def test_cli_accepts_h264_codec(self):
        """CLI should accept --codec h264."""
        parser = build_parser()
        args = parser.parse_args(["--codec", "h264", "--input", "test.mp4", "--output", "out.mp4"])
        assert args.codec == "h264"

    def test_cli_accepts_hevc_codec(self):
        """CLI should accept --codec hevc."""
        parser = build_parser()
        args = parser.parse_args(["--codec", "hevc", "--input", "test.mp4", "--output", "out.mp4"])
        assert args.codec == "hevc"

    def test_cli_accepts_av1_codec(self):
        """CLI should accept --codec av1."""
        parser = build_parser()
        args = parser.parse_args(["--codec", "av1", "--input", "test.mp4", "--output", "out.mp4"])
        assert args.codec == "av1"

    def test_cli_rejects_invalid_codec(self):
        """CLI should reject invalid codec values."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--codec", "invalid", "--input", "test.mp4", "--output", "out.mp4"])

    def test_cli_default_codec_is_hevc(self):
        """CLI should default to hevc codec."""
        parser = build_parser()
        args = parser.parse_args(["--input", "test.mp4", "--output", "out.mp4"])
        assert args.codec == "hevc"

    def test_cli_codec_choices_in_help(self):
        """CLI help should list all codec choices."""
        parser = build_parser()
        help_text = parser.format_help()
        assert "h264" in help_text
        assert "hevc" in help_text
        assert "av1" in help_text