from __future__ import annotations

import math
import threading
import time
from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jasna.streaming import HlsStreamingServer, _generate_vod_playlist


def _make_metadata(duration: float = 60.0, fps: float = 30.0, num_frames: int = 1800):
    from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange
    m = MagicMock()
    m.duration = duration
    m.video_fps = fps
    m.video_fps_exact = Fraction(30, 1)
    m.video_height = 1080
    m.video_width = 1920
    m.num_frames = num_frames
    m.time_base = Fraction(1, 90000)
    m.codec_name = "hevc"
    m.color_range = AvColorRange.MPEG
    m.color_space = AvColorspace.ITU709
    m.average_fps = fps
    m.start_pts = 0
    m.is_10bit = False
    m.video_file = "test.mp4"
    return m


class TestGenerateVodPlaylist:
    def test_basic_playlist_structure(self):
        text, count = _generate_vod_playlist(total_duration=20.0, segment_duration=4.0)
        assert count == 5
        assert "#EXTM3U" in text
        assert "#EXT-X-PLAYLIST-TYPE:VOD" in text
        assert "#EXT-X-ENDLIST" in text
        assert "seg_00000.ts" in text
        assert "seg_00004.ts" in text

    def test_segment_count_rounds_up(self):
        _text, count = _generate_vod_playlist(total_duration=10.5, segment_duration=4.0)
        assert count == 3

    def test_single_segment(self):
        text, count = _generate_vod_playlist(total_duration=2.0, segment_duration=4.0)
        assert count == 1
        assert "seg_00000.ts" in text

    def test_exact_division(self):
        _text, count = _generate_vod_playlist(total_duration=12.0, segment_duration=4.0)
        assert count == 3

    def test_last_segment_duration(self):
        text, count = _generate_vod_playlist(total_duration=10.0, segment_duration=4.0)
        assert count == 3
        lines = text.strip().split("\n")
        extinf_lines = [l for l in lines if l.startswith("#EXTINF:")]
        assert len(extinf_lines) == 3
        assert extinf_lines[0] == "#EXTINF:4.000,"
        assert extinf_lines[1] == "#EXTINF:4.000,"
        assert extinf_lines[2] == "#EXTINF:2.000,"


class TestHlsStreamingServer:
    def test_init_no_segments_dir(self):
        server = HlsStreamingServer(segment_duration=4.0, port=0)
        assert server.segments_dir is None
        assert not server.is_loaded

    def test_load_video_creates_segments_dir(self):
        meta = _make_metadata(duration=20.0)
        server = HlsStreamingServer(segment_duration=4.0, port=0)
        server.load_video(meta)
        assert server.segments_dir.exists()
        assert server.is_loaded
        server._cleanup_segments()

    def test_segment_count(self):
        meta = _make_metadata(duration=20.0)
        server = HlsStreamingServer(segment_duration=4.0, port=0)
        server.load_video(meta)
        assert server.segment_count == 5
        server._cleanup_segments()

    def test_segment_start_frame(self):
        meta = _make_metadata(duration=20.0, fps=30.0)
        server = HlsStreamingServer(segment_duration=4.0, port=0)
        server.load_video(meta)
        assert server.segment_start_frame(0) == 0
        assert server.segment_start_frame(1) == 120
        assert server.segment_start_frame(2) == 240
        server._cleanup_segments()

    def test_frames_per_segment(self):
        meta = _make_metadata(duration=20.0, fps=30.0)
        server = HlsStreamingServer(segment_duration=4.0, port=0)
        server.load_video(meta)
        assert server.frames_per_segment() == 120
        server._cleanup_segments()

    def test_seek_request_and_consume(self):
        server = HlsStreamingServer(segment_duration=4.0, port=0)
        assert server.consume_seek() is None

        server.request_seek(3)
        assert server.seek_requested.is_set()

        server._last_seek_time = 0.0
        target = server.consume_seek()
        assert target == 3
        assert not server.seek_requested.is_set()

        assert server.consume_seek() is None

    def test_multiple_seeks_last_wins(self):
        server = HlsStreamingServer(segment_duration=4.0, port=0)
        server.request_seek(1)
        server.request_seek(3)
        server.request_seek(2)

        server._last_seek_time = 0.0
        target = server.consume_seek()
        assert target == 2

    def test_url_property(self):
        server = HlsStreamingServer(segment_duration=4.0, port=9999)
        assert server.url == "http://localhost:9999/stream.m3u8"

    def test_start_and_stop(self):
        server = HlsStreamingServer(segment_duration=4.0, port=0)
        server.start()
        assert server._thread is not None
        assert server._thread.is_alive()
        server.stop()
        assert server._thread is None

    def test_cleanup_removes_dir(self):
        meta = _make_metadata(duration=20.0)
        server = HlsStreamingServer(segment_duration=4.0, port=0)
        server.load_video(meta)
        seg_dir = server.segments_dir
        assert seg_dir.exists()
        server.stop()
        assert not seg_dir.exists()

    def test_select_and_wait_for_video(self):
        server = HlsStreamingServer(segment_duration=4.0, port=0)
        result = [None]
        def _waiter():
            result[0] = server.wait_for_video()
        t = threading.Thread(target=_waiter)
        t.start()
        time.sleep(0.1)
        server.select_video(Path("test.mp4"))
        t.join(timeout=2.0)
        assert result[0] == Path("test.mp4")
        assert not server.video_change.is_set()

    def test_select_video_while_loaded_triggers_change(self):
        meta = _make_metadata(duration=20.0)
        server = HlsStreamingServer(segment_duration=4.0, port=0)
        server.load_video(meta)
        server.select_video(Path("other.mp4"))
        assert server.video_change.is_set()
        assert not server.is_loaded
        server._cleanup_segments()

    def test_stop_current_clears_metadata(self):
        meta = _make_metadata(duration=20.0)
        server = HlsStreamingServer(segment_duration=4.0, port=0)
        server.load_video(meta)
        assert server.is_loaded
        server.stop_current()
        assert not server.is_loaded
        assert server.video_change.is_set()
        server._cleanup_segments()

    def test_unload_video_clears_state(self):
        meta = _make_metadata(duration=20.0)
        server = HlsStreamingServer(segment_duration=4.0, port=0)
        server.load_video(meta)
        assert server.is_loaded
        server.unload_video()
        assert not server.is_loaded
        assert server.segment_count == 0

    def test_load_video_resets_seek_state(self):
        meta = _make_metadata(duration=20.0)
        server = HlsStreamingServer(segment_duration=4.0, port=0)
        server.request_seek(5)
        server.video_change.set()
        server.load_video(meta)
        assert not server.seek_requested.is_set()
        assert not server.video_change.is_set()
        server._cleanup_segments()


class TestDemandFlowControl:
    def test_wait_for_demand_proceeds_when_within_buffer(self):
        server = HlsStreamingServer(segment_duration=4.0, port=0)
        cancel = threading.Event()
        server.notify_segment_requested(0)
        server.wait_for_demand(current_segment=2, max_ahead=3, cancel_event=cancel)

    def test_wait_for_demand_blocks_when_too_far_ahead(self):
        server = HlsStreamingServer(segment_duration=4.0, port=0)
        cancel = threading.Event()
        server.notify_segment_requested(0)
        blocked = threading.Event()
        unblocked = threading.Event()

        def _waiter():
            blocked.set()
            server.wait_for_demand(current_segment=5, max_ahead=3, cancel_event=cancel)
            unblocked.set()

        t = threading.Thread(target=_waiter)
        t.start()
        blocked.wait(timeout=2.0)
        time.sleep(0.2)
        assert not unblocked.is_set()

        server.notify_segment_requested(3)
        unblocked.wait(timeout=2.0)
        assert unblocked.is_set()
        t.join(timeout=2.0)

    def test_wait_for_demand_unblocks_on_cancel(self):
        server = HlsStreamingServer(segment_duration=4.0, port=0)
        cancel = threading.Event()
        server.notify_segment_requested(0)
        done = threading.Event()

        def _waiter():
            server.wait_for_demand(current_segment=10, max_ahead=3, cancel_event=cancel)
            done.set()

        t = threading.Thread(target=_waiter)
        t.start()
        time.sleep(0.2)
        assert not done.is_set()
        cancel.set()
        done.wait(timeout=2.0)
        assert done.is_set()
        t.join(timeout=2.0)

    def test_initial_buffer_without_player(self):
        """With no player requests (highest=-1), pipeline runs freely."""
        server = HlsStreamingServer(segment_duration=4.0, port=0)
        cancel = threading.Event()
        server.wait_for_demand(current_segment=0, max_ahead=3, cancel_event=cancel)
        server.wait_for_demand(current_segment=10, max_ahead=3, cancel_event=cancel)
        server.wait_for_demand(current_segment=50, max_ahead=3, cancel_event=cancel)

    def test_reset_demand_default(self):
        server = HlsStreamingServer(segment_duration=4.0, port=0)
        server.notify_segment_requested(5)
        server.reset_demand()
        assert server._highest_requested_segment == -1

    def test_reset_demand_with_start_segment(self):
        server = HlsStreamingServer(segment_duration=4.0, port=0)
        cancel = threading.Event()
        server.reset_demand(start_segment=100)
        assert server._highest_requested_segment == 99
        server.wait_for_demand(current_segment=100, max_ahead=3, cancel_event=cancel)
        server.wait_for_demand(current_segment=102, max_ahead=3, cancel_event=cancel)

    def test_forward_seek_triggered_when_far_ahead(self):
        server = HlsStreamingServer(segment_duration=4.0, port=0)
        server.reset_demand(start_segment=0)
        server.update_production(3)
        assert not server.needs_seek(5)
        assert server.needs_seek(50)

    def test_no_forward_seek_when_production_close(self):
        server = HlsStreamingServer(segment_duration=4.0, port=0)
        server.reset_demand(start_segment=0)
        server.update_production(10)
        assert not server.needs_seek(14)

    def test_backward_seek_always_triggers(self):
        server = HlsStreamingServer(segment_duration=4.0, port=0)
        server.reset_demand(start_segment=50)
        server.update_production(55)
        assert server.needs_seek(10)

    def test_notify_tracks_highest(self):
        server = HlsStreamingServer(segment_duration=4.0, port=0)
        server.notify_segment_requested(2)
        server.notify_segment_requested(5)
        server.notify_segment_requested(3)
        assert server._highest_requested_segment == 5


def _make_rgb_frame(width=64, height=64):
    """Create a random RGB torch tensor in CHW format."""
    import torch
    return torch.randint(0, 255, (3, height, width), dtype=torch.uint8)


def _mock_ffmpeg_process():
    import io
    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.stdin.closed = False
    proc.stderr = io.BytesIO(b"")
    proc.returncode = 0
    proc.wait.return_value = 0
    return proc


class TestStreamingEncoder:
    def _make_encoder(self, tmp_path):
        from jasna.streaming_encoder import StreamingEncoder
        meta = _make_metadata(duration=20.0, fps=30.0)
        return StreamingEncoder(
            segments_dir=tmp_path,
            segment_duration=4.0,
            metadata=meta,
            source_video="nonexistent.mp4",
        )

    @patch('jasna.streaming_encoder.subprocess.Popen')
    def test_start_and_stop(self, mock_popen, tmp_path):
        mock_popen.return_value = _mock_ffmpeg_process()
        enc = self._make_encoder(tmp_path)
        enc.start(start_number=0)
        assert enc._started
        assert enc._process is not None
        enc.stop()
        assert not enc._started

    @patch('jasna.streaming_encoder.subprocess.Popen')
    def test_ffmpeg_cmd_contains_nvenc(self, mock_popen, tmp_path):
        mock_popen.return_value = _mock_ffmpeg_process()
        enc = self._make_encoder(tmp_path)
        enc.start(start_number=0)
        cmd = mock_popen.call_args[0][0]
        assert 'h264_nvenc' in cmd
        enc.stop()

    @patch('jasna.streaming_encoder.subprocess.Popen')
    def test_write_frame_sends_bytes(self, mock_popen, tmp_path):
        proc = _mock_ffmpeg_process()
        mock_popen.return_value = proc
        enc = self._make_encoder(tmp_path)
        enc.start(start_number=0)
        enc.write_frame(_make_rgb_frame(), pts=0)
        time.sleep(0.3)
        enc.stop()
        assert proc.stdin.write.called

    @patch('jasna.streaming_encoder.subprocess.Popen')
    def test_start_number_in_cmd(self, mock_popen, tmp_path):
        mock_popen.return_value = _mock_ffmpeg_process()
        enc = self._make_encoder(tmp_path)
        enc.start(start_number=5)
        cmd = mock_popen.call_args[0][0]
        idx = cmd.index('-start_number')
        assert cmd[idx + 1] == '5'
        enc.stop()

    @patch('jasna.streaming_encoder.subprocess.Popen')
    def test_flush_and_restart(self, mock_popen, tmp_path):
        mock_popen.return_value = _mock_ffmpeg_process()
        enc = self._make_encoder(tmp_path)
        enc.start(start_number=0)
        enc.flush_and_restart(start_number=5)
        assert enc._started
        assert mock_popen.call_count == 2
        cmd = mock_popen.call_args[0][0]
        idx = cmd.index('-start_number')
        assert cmd[idx + 1] == '5'
        enc.stop()

    @patch('jasna.streaming_encoder.subprocess.Popen')
    def test_no_audio_when_source_missing(self, mock_popen, tmp_path):
        mock_popen.return_value = _mock_ffmpeg_process()
        enc = self._make_encoder(tmp_path)
        enc.start(start_number=0)
        cmd = mock_popen.call_args[0][0]
        assert '-c:a' not in cmd
        enc.stop()

    def test_write_before_start_is_noop(self, tmp_path):
        enc = self._make_encoder(tmp_path)
        enc.write_frame(_make_rgb_frame(), pts=0)
        assert enc._process is None

    @patch('jasna.streaming_encoder.subprocess.Popen')
    @patch('jasna.streaming_encoder.os.path.isfile', return_value=True)
    def test_audio_copy_when_source_exists(self, mock_isfile, mock_popen, tmp_path):
        mock_popen.return_value = _mock_ffmpeg_process()
        enc = self._make_encoder(tmp_path)
        enc.source_video = "existing.mp4"
        enc.start(start_number=0)
        cmd = mock_popen.call_args[0][0]
        assert '-c:a' in cmd
        assert 'copy' in cmd
        assert 'existing.mp4' in cmd
        enc.stop()

    @patch('jasna.streaming_encoder.subprocess.Popen')
    @patch('jasna.streaming_encoder.os.path.isfile', return_value=True)
    def test_audio_seek_on_restart(self, mock_isfile, mock_popen, tmp_path):
        mock_popen.return_value = _mock_ffmpeg_process()
        enc = self._make_encoder(tmp_path)
        enc.source_video = "existing.mp4"
        enc.start(start_number=3)
        cmd = mock_popen.call_args[0][0]
        ss_idx = cmd.index('-ss')
        assert float(cmd[ss_idx + 1]) == pytest.approx(12.0)
        enc.stop()

    @patch('jasna.streaming_encoder.subprocess.Popen')
    def test_output_ts_offset_on_seek(self, mock_popen, tmp_path):
        mock_popen.return_value = _mock_ffmpeg_process()
        enc = self._make_encoder(tmp_path)
        enc.start(start_number=5)
        cmd = mock_popen.call_args[0][0]
        idx = cmd.index('-output_ts_offset')
        assert float(cmd[idx + 1]) == pytest.approx(20.0)
        enc.stop()

    @patch('jasna.streaming_encoder.subprocess.Popen')
    def test_no_ts_offset_at_start(self, mock_popen, tmp_path):
        mock_popen.return_value = _mock_ffmpeg_process()
        enc = self._make_encoder(tmp_path)
        enc.start(start_number=0)
        cmd = mock_popen.call_args[0][0]
        assert '-output_ts_offset' not in cmd
        enc.stop()


class TestMainParserStreamingArgs:
    def test_stream_flag_present(self):
        from jasna.main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--input", "test.mp4", "--stream"])
        assert args.stream is True
        assert args.stream_port == 8765
        assert args.stream_segment_duration == 4.0

    def test_stream_custom_port(self):
        from jasna.main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--input", "test.mp4", "--stream", "--stream-port", "9999"])
        assert args.stream_port == 9999

    def test_stream_custom_segment_duration(self):
        from jasna.main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--input", "test.mp4", "--stream", "--stream-segment-duration", "2.0"])
        assert args.stream_segment_duration == 2.0

    def test_no_output_required_with_stream(self):
        from jasna.main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--input", "test.mp4", "--stream"])
        assert args.output is None
        assert args.stream is True

    def test_stream_without_input(self):
        from jasna.main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--stream"])
        assert args.stream is True
        assert args.input is None
