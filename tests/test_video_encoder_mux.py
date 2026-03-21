from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.media import VideoMetadata
from jasna.media.audio_utils import (
    _INCOMPATIBLE_AUDIO,
    audio_codec_args,
    needs_audio_reencode,
    probe_audio_codec,
)
from jasna.media.video_encoder import mux_hevc_to_mkv, remux_with_audio_and_metadata


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
        time_base=Fraction(1, 24000),
        start_pts=0,
        color_space=AvColorspace.ITU709,
        color_range=AvColorRange.MPEG,
        is_10bit=True,
    )
    defaults.update(overrides)
    return VideoMetadata(**defaults)


class TestMuxHevcToMkv:
    @patch("jasna.media.video_encoder.get_subprocess_startup_info", return_value=None)
    @patch("jasna.media.video_encoder.resolve_executable", return_value="mkvmerge")
    @patch("jasna.media.video_encoder.subprocess.run")
    def test_success_writes_timecodes_and_cleans_up(self, mock_run, mock_resolve, mock_si, tmp_path):
        hevc_path = tmp_path / "video.hevc"
        hevc_path.write_bytes(b"\x00")
        output_path = tmp_path / "video.mkv"

        mock_run.return_value = MagicMock(returncode=0)
        pts_list = [0, 1001, 2002]
        time_base = Fraction(1, 24000)

        mux_hevc_to_mkv(hevc_path, output_path, pts_list, time_base)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "mkvmerge" in cmd[0]
        assert str(output_path) in cmd
        timecodes_path = output_path.with_suffix('.txt')
        assert not timecodes_path.exists()

    @patch("jasna.media.video_encoder.get_subprocess_startup_info", return_value=None)
    @patch("jasna.media.video_encoder.resolve_executable", return_value="mkvmerge")
    @patch("jasna.media.video_encoder.subprocess.run")
    def test_failure_raises_runtime_error(self, mock_run, mock_resolve, mock_si, tmp_path):
        hevc_path = tmp_path / "video.hevc"
        hevc_path.write_bytes(b"\x00")
        output_path = tmp_path / "video.mkv"

        mock_run.return_value = MagicMock(returncode=2, stdout=b"", stderr=b"mux error")

        with pytest.raises(RuntimeError, match="mkvmerge failed"):
            mux_hevc_to_mkv(hevc_path, output_path, [0, 1001], Fraction(1, 24000))

    @patch("jasna.media.video_encoder.get_subprocess_startup_info", return_value=None)
    @patch("jasna.media.video_encoder.resolve_executable", return_value="mkvmerge")
    @patch("jasna.media.video_encoder.subprocess.run")
    def test_timecodes_file_content(self, mock_run, mock_resolve, mock_si, tmp_path):
        hevc_path = tmp_path / "video.hevc"
        hevc_path.write_bytes(b"\x00")
        output_path = tmp_path / "video.mkv"
        timecodes_path = output_path.with_suffix('.txt')

        written_content = []

        def capture_run(cmd, **kwargs):
            written_content.append(timecodes_path.read_text())
            return MagicMock(returncode=0)

        mock_run.side_effect = capture_run

        pts_list = [0, 1001, 2002]
        time_base = Fraction(1, 24000)
        mux_hevc_to_mkv(hevc_path, output_path, pts_list, time_base)

        content = written_content[0]
        assert "# timestamp format v4" in content
        lines = content.strip().split("\n")
        assert len(lines) == 4


class TestNeedsAudioReencode:
    @pytest.mark.parametrize("codec,suffix,expected", [
        ("wmav2", ".mp4", True),
        ("wmav1", ".mp4", True),
        ("wmapro", ".mp4", True),
        ("vorbis", ".mp4", True),
        ("aac", ".mp4", False),
        ("mp3", ".mp4", False),
        ("opus", ".mp4", False),
        ("wmav2", ".mov", True),
        ("opus", ".mov", True),
        ("aac", ".mov", False),
        ("opus", ".avi", True),
        ("vorbis", ".avi", True),
        ("flac", ".avi", True),
        ("mp3", ".avi", False),
        ("aac", ".webm", True),
        ("mp3", ".webm", True),
        ("wmav2", ".webm", True),
        ("opus", ".webm", False),
        ("vorbis", ".webm", False),
        ("aac", ".mkv", False),
        ("wmav2", ".mkv", False),
        ("opus", ".mkv", False),
        (None, ".mp4", False),
        (None, ".mkv", False),
        ("WMAv2", ".mp4", True),
    ])
    def test_table(self, codec, suffix, expected):
        assert needs_audio_reencode(codec, suffix) == expected


class TestProbeAudioCodec:
    @patch("jasna.media.audio_utils.get_subprocess_startup_info", return_value=None)
    @patch("jasna.media.audio_utils.resolve_executable", return_value="ffprobe")
    @patch("jasna.media.audio_utils.subprocess.run")
    def test_returns_codec_name(self, mock_run, _resolve, _si):
        import json
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"streams": [{"codec_name": "wmav2"}]}).encode(),
        )
        assert probe_audio_codec("input.wmv") == "wmav2"

    @patch("jasna.media.audio_utils.get_subprocess_startup_info", return_value=None)
    @patch("jasna.media.audio_utils.resolve_executable", return_value="ffprobe")
    @patch("jasna.media.audio_utils.subprocess.run")
    def test_no_audio_stream(self, mock_run, _resolve, _si):
        import json
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"streams": []}).encode(),
        )
        assert probe_audio_codec("input.mp4") is None

    @patch("jasna.media.audio_utils.get_subprocess_startup_info", return_value=None)
    @patch("jasna.media.audio_utils.resolve_executable", return_value="ffprobe")
    @patch("jasna.media.audio_utils.subprocess.run")
    def test_ffprobe_failure(self, mock_run, _resolve, _si):
        mock_run.return_value = MagicMock(returncode=1)
        assert probe_audio_codec("input.mp4") is None


class TestAudioCodecArgs:
    @patch("jasna.media.audio_utils.probe_audio_codec", return_value="wmav2")
    def test_reencode_for_incompatible(self, _probe):
        assert audio_codec_args("input.wmv", Path("out.mp4")) == ["aac", "-b:a", "256k"]

    @patch("jasna.media.audio_utils.probe_audio_codec", return_value="aac")
    def test_copy_for_compatible(self, _probe):
        assert audio_codec_args("input.mp4", Path("out.mp4")) == ["copy"]

    @patch("jasna.media.audio_utils.probe_audio_codec", return_value=None)
    def test_copy_when_no_audio(self, _probe):
        assert audio_codec_args("input.mp4", Path("out.mp4")) == ["copy"]


class TestRemuxWithAudioAndMetadata:
    @patch("jasna.media.video_encoder.audio_codec_args", return_value=["copy"])
    @patch("jasna.media.video_encoder.get_subprocess_startup_info", return_value=None)
    @patch("jasna.media.video_encoder.resolve_executable", return_value="ffmpeg")
    @patch("jasna.media.video_encoder.subprocess.run")
    def test_success_bt709_mpeg(self, mock_run, mock_resolve, mock_si, _codec_args, tmp_path):
        video_input = tmp_path / "temp.mkv"
        video_input.touch()
        output_path = tmp_path / "output.mkv"

        mock_run.return_value = MagicMock(returncode=0)
        meta = _fake_metadata()

        remux_with_audio_and_metadata(video_input, output_path, meta)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "bt709" in cmd
        assert "tv" in cmd
        assert "-movflags" not in cmd

    @patch("jasna.media.video_encoder.audio_codec_args", return_value=["copy"])
    @patch("jasna.media.video_encoder.get_subprocess_startup_info", return_value=None)
    @patch("jasna.media.video_encoder.resolve_executable", return_value="ffmpeg")
    @patch("jasna.media.video_encoder.subprocess.run")
    def test_success_bt601_jpeg(self, mock_run, mock_resolve, mock_si, _codec_args, tmp_path):
        video_input = tmp_path / "temp.mkv"
        video_input.touch()
        output_path = tmp_path / "output.mkv"

        mock_run.return_value = MagicMock(returncode=0)
        meta = _fake_metadata(color_space=AvColorspace.ITU601, color_range=AvColorRange.JPEG)

        remux_with_audio_and_metadata(video_input, output_path, meta)

        cmd = mock_run.call_args[0][0]
        assert "smpte170m" in cmd
        assert "pc" in cmd

    @patch("jasna.media.video_encoder.audio_codec_args", return_value=["copy"])
    @patch("jasna.media.video_encoder.get_subprocess_startup_info", return_value=None)
    @patch("jasna.media.video_encoder.resolve_executable", return_value="ffmpeg")
    @patch("jasna.media.video_encoder.subprocess.run")
    def test_mp4_output_adds_faststart(self, mock_run, mock_resolve, mock_si, _codec_args, tmp_path):
        video_input = tmp_path / "temp.mkv"
        video_input.touch()
        output_path = tmp_path / "output.mp4"

        mock_run.return_value = MagicMock(returncode=0)
        meta = _fake_metadata()

        remux_with_audio_and_metadata(video_input, output_path, meta)

        cmd = mock_run.call_args[0][0]
        assert "-movflags" in cmd
        assert "+faststart" in cmd

    @patch("jasna.media.video_encoder.audio_codec_args", return_value=["copy"])
    @patch("jasna.media.video_encoder.get_subprocess_startup_info", return_value=None)
    @patch("jasna.media.video_encoder.resolve_executable", return_value="ffmpeg")
    @patch("jasna.media.video_encoder.subprocess.run")
    def test_failure_raises_runtime_error(self, mock_run, mock_resolve, mock_si, _codec_args, tmp_path):
        video_input = tmp_path / "temp.mkv"
        video_input.touch()
        output_path = tmp_path / "output.mkv"

        mock_run.return_value = MagicMock(returncode=1, stdout=b"", stderr=b"ffmpeg error")
        meta = _fake_metadata()

        with pytest.raises(RuntimeError, match="ffmpeg failed"):
            remux_with_audio_and_metadata(video_input, output_path, meta)

    @patch("jasna.media.video_encoder.audio_codec_args", return_value=["aac", "-b:a", "256k"])
    @patch("jasna.media.video_encoder.get_subprocess_startup_info", return_value=None)
    @patch("jasna.media.video_encoder.resolve_executable", return_value="ffmpeg")
    @patch("jasna.media.video_encoder.subprocess.run")
    def test_reencode_args_in_ffmpeg_cmd(self, mock_run, mock_resolve, mock_si, _codec_args, tmp_path):
        video_input = tmp_path / "temp.mkv"
        video_input.touch()
        output_path = tmp_path / "output.mp4"

        mock_run.return_value = MagicMock(returncode=0)
        meta = _fake_metadata()

        remux_with_audio_and_metadata(video_input, output_path, meta)

        cmd = mock_run.call_args[0][0]
        ca_idx = cmd.index("-c:a")
        assert cmd[ca_idx + 1] == "aac"
        assert cmd[ca_idx + 2] == "-b:a"
        assert cmd[ca_idx + 3] == "256k"
