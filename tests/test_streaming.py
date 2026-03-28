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
    def test_init_creates_segments_dir(self):
        meta = _make_metadata(duration=20.0)
        server = HlsStreamingServer(metadata=meta, segment_duration=4.0, port=0)
        assert server.segments_dir.exists()
        assert (server.segments_dir / "stream.m3u8").exists()
        server._cleanup_segments()

    def test_segment_count(self):
        meta = _make_metadata(duration=20.0)
        server = HlsStreamingServer(metadata=meta, segment_duration=4.0, port=0)
        assert server.segment_count == 5
        server._cleanup_segments()

    def test_segment_start_frame(self):
        meta = _make_metadata(duration=20.0, fps=30.0)
        server = HlsStreamingServer(metadata=meta, segment_duration=4.0, port=0)
        assert server.segment_start_frame(0) == 0
        assert server.segment_start_frame(1) == 120
        assert server.segment_start_frame(2) == 240
        server._cleanup_segments()

    def test_frames_per_segment(self):
        meta = _make_metadata(duration=20.0, fps=30.0)
        server = HlsStreamingServer(metadata=meta, segment_duration=4.0, port=0)
        assert server.frames_per_segment() == 120
        server._cleanup_segments()

    def test_seek_request_and_consume(self):
        meta = _make_metadata(duration=20.0)
        server = HlsStreamingServer(metadata=meta, segment_duration=4.0, port=0)

        assert server.consume_seek() is None

        server.request_seek(3)
        assert server.seek_requested.is_set()

        server._last_seek_time = 0.0
        target = server.consume_seek()
        assert target == 3
        assert not server.seek_requested.is_set()

        assert server.consume_seek() is None
        server._cleanup_segments()

    def test_multiple_seeks_last_wins(self):
        meta = _make_metadata(duration=20.0)
        server = HlsStreamingServer(metadata=meta, segment_duration=4.0, port=0)

        server.request_seek(1)
        server.request_seek(3)
        server.request_seek(2)

        server._last_seek_time = 0.0
        target = server.consume_seek()
        assert target == 2
        server._cleanup_segments()

    def test_url_property(self):
        meta = _make_metadata(duration=20.0)
        server = HlsStreamingServer(metadata=meta, segment_duration=4.0, port=9999)
        assert server.url == "http://localhost:9999/stream.m3u8"
        server._cleanup_segments()

    def test_start_and_stop(self):
        meta = _make_metadata(duration=20.0)
        server = HlsStreamingServer(metadata=meta, segment_duration=4.0, port=0)
        server.start()
        assert server._thread is not None
        assert server._thread.is_alive()
        server.stop()
        assert server._thread is None

    def test_cleanup_removes_dir(self):
        meta = _make_metadata(duration=20.0)
        server = HlsStreamingServer(metadata=meta, segment_duration=4.0, port=0)
        seg_dir = server.segments_dir
        assert seg_dir.exists()
        server.stop()
        assert not seg_dir.exists()


class TestDemandFlowControl:
    def test_wait_for_demand_proceeds_when_within_buffer(self):
        meta = _make_metadata(duration=60.0, fps=30.0)
        server = HlsStreamingServer(metadata=meta, segment_duration=4.0, port=0)
        cancel = threading.Event()
        server.notify_segment_requested(0)
        server.wait_for_demand(current_segment=2, max_ahead=3, cancel_event=cancel)
        server._cleanup_segments()

    def test_wait_for_demand_blocks_when_too_far_ahead(self):
        meta = _make_metadata(duration=60.0, fps=30.0)
        server = HlsStreamingServer(metadata=meta, segment_duration=4.0, port=0)
        cancel = threading.Event()
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
        server._cleanup_segments()

    def test_wait_for_demand_unblocks_on_cancel(self):
        meta = _make_metadata(duration=60.0, fps=30.0)
        server = HlsStreamingServer(metadata=meta, segment_duration=4.0, port=0)
        cancel = threading.Event()
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
        server._cleanup_segments()

    def test_initial_buffer_without_player(self):
        """With no player requests (highest=-1), pipeline can produce max_ahead-1 segments."""
        meta = _make_metadata(duration=60.0, fps=30.0)
        server = HlsStreamingServer(metadata=meta, segment_duration=4.0, port=0)
        cancel = threading.Event()
        server.wait_for_demand(current_segment=0, max_ahead=3, cancel_event=cancel)
        server.wait_for_demand(current_segment=1, max_ahead=3, cancel_event=cancel)
        server.wait_for_demand(current_segment=2, max_ahead=3, cancel_event=cancel)

        blocked = threading.Event()
        unblocked = threading.Event()

        def _waiter():
            blocked.set()
            server.wait_for_demand(current_segment=3, max_ahead=3, cancel_event=cancel)
            unblocked.set()

        t = threading.Thread(target=_waiter)
        t.start()
        blocked.wait(timeout=2.0)
        time.sleep(0.2)
        assert not unblocked.is_set()
        cancel.set()
        t.join(timeout=2.0)
        server._cleanup_segments()

    def test_reset_demand_default(self):
        meta = _make_metadata(duration=60.0, fps=30.0)
        server = HlsStreamingServer(metadata=meta, segment_duration=4.0, port=0)
        server.notify_segment_requested(5)
        server.reset_demand()
        assert server._highest_requested_segment == -1
        server._cleanup_segments()

    def test_reset_demand_with_start_segment(self):
        meta = _make_metadata(duration=60.0, fps=30.0)
        server = HlsStreamingServer(metadata=meta, segment_duration=4.0, port=0)
        cancel = threading.Event()
        server.reset_demand(start_segment=100)
        assert server._highest_requested_segment == 99
        server.wait_for_demand(current_segment=100, max_ahead=3, cancel_event=cancel)
        server.wait_for_demand(current_segment=102, max_ahead=3, cancel_event=cancel)
        server._cleanup_segments()

    def test_forward_seek_triggered_when_far_ahead(self):
        meta = _make_metadata(duration=60.0, fps=30.0)
        server = HlsStreamingServer(metadata=meta, segment_duration=4.0, port=0)
        server.reset_demand(start_segment=0)
        server.update_production(3)
        assert not server.needs_seek(5)
        assert server.needs_seek(50)
        server._cleanup_segments()

    def test_no_forward_seek_when_production_close(self):
        meta = _make_metadata(duration=60.0, fps=30.0)
        server = HlsStreamingServer(metadata=meta, segment_duration=4.0, port=0)
        server.reset_demand(start_segment=0)
        server.update_production(10)
        assert not server.needs_seek(14)
        server._cleanup_segments()

    def test_backward_seek_always_triggers(self):
        meta = _make_metadata(duration=60.0, fps=30.0)
        server = HlsStreamingServer(metadata=meta, segment_duration=4.0, port=0)
        server.reset_demand(start_segment=50)
        server.update_production(55)
        assert server.needs_seek(10)
        server._cleanup_segments()

    def test_notify_tracks_highest(self):
        meta = _make_metadata(duration=60.0, fps=30.0)
        server = HlsStreamingServer(metadata=meta, segment_duration=4.0, port=0)
        server.notify_segment_requested(2)
        server.notify_segment_requested(5)
        server.notify_segment_requested(3)
        assert server._highest_requested_segment == 5
        server._cleanup_segments()


class TestHlsEncoder:
    def test_start_and_stop(self, tmp_path):
        from jasna.streaming_encoder import HlsEncoder
        encoder = HlsEncoder(segments_dir=tmp_path, segment_duration=4.0, fps=29.97)
        encoder.start(start_number=0)
        assert encoder._proc is not None
        encoder.stop()

    def test_write_bytes(self, tmp_path):
        from jasna.streaming_encoder import HlsEncoder
        encoder = HlsEncoder(segments_dir=tmp_path, segment_duration=4.0, fps=29.97)
        encoder.start(start_number=0)
        encoder.write(b"\x00" * 100)
        encoder.stop()

    def test_flush_and_restart(self, tmp_path):
        from jasna.streaming_encoder import HlsEncoder
        encoder = HlsEncoder(segments_dir=tmp_path, segment_duration=4.0, fps=29.97)
        encoder.start(start_number=0)
        encoder.write(b"\x00" * 50)
        encoder.flush_and_restart(start_number=5)
        assert encoder._proc is not None
        encoder.stop()


class TestVideoEncoderBitstreamOnly:
    def test_bitstream_only_flag_accepted(self):
        """Verify bitstream_only parameter is accepted by NvidiaVideoEncoder constructor signature."""
        import inspect
        from jasna.media.video_encoder import NvidiaVideoEncoder
        sig = inspect.signature(NvidiaVideoEncoder.__init__)
        assert "bitstream_only" in sig.parameters
        assert "bitstream_sink" in sig.parameters


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
