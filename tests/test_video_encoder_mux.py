from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.media import VideoMetadata
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


class TestRemuxWithAudioAndMetadata:
    @patch("jasna.media.video_encoder.get_subprocess_startup_info", return_value=None)
    @patch("jasna.media.video_encoder.resolve_executable", return_value="ffmpeg")
    @patch("jasna.media.video_encoder.subprocess.run")
    def test_success_bt709_mpeg(self, mock_run, mock_resolve, mock_si, tmp_path):
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

    @patch("jasna.media.video_encoder.get_subprocess_startup_info", return_value=None)
    @patch("jasna.media.video_encoder.resolve_executable", return_value="ffmpeg")
    @patch("jasna.media.video_encoder.subprocess.run")
    def test_success_bt601_jpeg(self, mock_run, mock_resolve, mock_si, tmp_path):
        video_input = tmp_path / "temp.mkv"
        video_input.touch()
        output_path = tmp_path / "output.mkv"

        mock_run.return_value = MagicMock(returncode=0)
        meta = _fake_metadata(color_space=AvColorspace.ITU601, color_range=AvColorRange.JPEG)

        remux_with_audio_and_metadata(video_input, output_path, meta)

        cmd = mock_run.call_args[0][0]
        assert "smpte170m" in cmd
        assert "pc" in cmd

    @patch("jasna.media.video_encoder.get_subprocess_startup_info", return_value=None)
    @patch("jasna.media.video_encoder.resolve_executable", return_value="ffmpeg")
    @patch("jasna.media.video_encoder.subprocess.run")
    def test_mp4_output_adds_faststart(self, mock_run, mock_resolve, mock_si, tmp_path):
        video_input = tmp_path / "temp.mkv"
        video_input.touch()
        output_path = tmp_path / "output.mp4"

        mock_run.return_value = MagicMock(returncode=0)
        meta = _fake_metadata()

        remux_with_audio_and_metadata(video_input, output_path, meta)

        cmd = mock_run.call_args[0][0]
        assert "-movflags" in cmd
        assert "+faststart" in cmd

    @patch("jasna.media.video_encoder.get_subprocess_startup_info", return_value=None)
    @patch("jasna.media.video_encoder.resolve_executable", return_value="ffmpeg")
    @patch("jasna.media.video_encoder.subprocess.run")
    def test_failure_raises_runtime_error(self, mock_run, mock_resolve, mock_si, tmp_path):
        video_input = tmp_path / "temp.mkv"
        video_input.touch()
        output_path = tmp_path / "output.mkv"

        mock_run.return_value = MagicMock(returncode=1, stdout=b"", stderr=b"ffmpeg error")
        meta = _fake_metadata()

        with pytest.raises(RuntimeError, match="ffmpeg failed"):
            remux_with_audio_and_metadata(video_input, output_path, meta)
