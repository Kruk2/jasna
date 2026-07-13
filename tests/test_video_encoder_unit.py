"""Unit tests for NvidiaVideoEncoder internals (options, color guard, buffer, worker, audio pump)."""
from __future__ import annotations

import queue
import threading
from collections import deque
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.media import VideoMetadata
from jasna.media.rgb_to_p010 import (
    chw_rgb_to_p010_bt2020_full,
    chw_rgb_to_p010_bt2020_limited,
    chw_rgb_to_p010_bt601_full,
    chw_rgb_to_p010_bt601_limited,
    chw_rgb_to_p010_bt709_full,
    chw_rgb_to_p010_bt709_limited,
)
from jasna.media.video_encoder import DEFAULT_ENCODER_OPTIONS, NvidiaVideoEncoder


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


def _make_encoder(tmp_path, encoder_settings=None, **meta_overrides) -> NvidiaVideoEncoder:
    return NvidiaVideoEncoder(
        file=str(tmp_path / "result.mkv"),
        device=torch.device("cuda:0"),
        metadata=_fake_metadata(**meta_overrides),
        codec="hevc",
        encoder_settings=encoder_settings or {},
    )


class TestEncoderOptions:
    def test_defaults_used_when_no_settings(self, tmp_path):
        enc = _make_encoder(tmp_path)
        assert enc.encoder_options == DEFAULT_ENCODER_OPTIONS

    def test_settings_override_and_stringify(self, tmp_path):
        enc = _make_encoder(tmp_path, encoder_settings={"cq": 22, "temporal-aq": False, "maxrate": "10M"})
        assert enc.encoder_options["cq"] == "22"
        assert enc.encoder_options["temporal-aq"] == "0"
        assert enc.encoder_options["maxrate"] == "10M"
        assert enc.encoder_options["preset"] == "p5"

    def test_unsupported_codec_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported codec"):
            NvidiaVideoEncoder(
                file=str(tmp_path / "o.mkv"),
                device=torch.device("cuda:0"),
                metadata=_fake_metadata(),
                codec="av1",
                encoder_settings={},
            )

    def test_leftover_options_raise(self, tmp_path):
        enc = _make_encoder(tmp_path)
        enc.out_stream = MagicMock()
        enc.out_stream.codec_context.options = {"bogus": "1"}
        with pytest.raises(ValueError, match="did not accept encoder option.*bogus"):
            enc._validate_encoder_options()

    def test_no_leftover_options_pass(self, tmp_path):
        enc = _make_encoder(tmp_path)
        enc.out_stream = MagicMock()
        enc.out_stream.codec_context.options = {}
        enc._options_validated = False
        enc._validate_encoder_options()
        assert enc._options_validated


class TestColorHandling:
    def test_unsupported_color_range_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported color space or color range"):
            _make_encoder(tmp_path, color_range=AvColorRange.UNSPECIFIED)

    @pytest.mark.parametrize(
        ("color_space", "color_range", "expected"),
        [
            (AvColorspace.ITU709, AvColorRange.MPEG, chw_rgb_to_p010_bt709_limited),
            (AvColorspace.ITU709, AvColorRange.JPEG, chw_rgb_to_p010_bt709_full),
            (AvColorspace.ITU601, AvColorRange.MPEG, chw_rgb_to_p010_bt601_limited),
            (AvColorspace.ITU601, AvColorRange.JPEG, chw_rgb_to_p010_bt601_full),
            (AvColorspace.BT2020, AvColorRange.MPEG, chw_rgb_to_p010_bt2020_limited),
            (AvColorspace.BT2020, AvColorRange.JPEG, chw_rgb_to_p010_bt2020_full),
        ],
    )
    def test_selects_matrix_and_range_converter(self, tmp_path, color_space, color_range, expected):
        enc = _make_encoder(tmp_path, color_space=color_space, color_range=color_range)
        assert enc._to_p010 is expected


def _buffered_encoder(tmp_path) -> NvidiaVideoEncoder:
    enc = _make_encoder(tmp_path)
    enc.pts_heap = []
    enc.frame_buffer = deque()
    enc.pts_set = set()
    enc._worker_error = None
    enc._encode_queue = MagicMock()
    enc._build_encode_item = MagicMock(side_effect=lambda frame, pts: (frame, pts, None))
    return enc


class TestEncodeBuffer:
    def test_encode_pushes_to_buffer_and_heap(self, tmp_path):
        enc = _buffered_encoder(tmp_path)
        enc.encode("frame0", 10)
        assert list(enc.frame_buffer) == ["frame0"]
        assert enc.pts_heap == [10]
        assert enc.pts_set == {10}

    def test_encode_dedup_pts(self, tmp_path):
        enc = _buffered_encoder(tmp_path)
        enc.encode("a", 5)
        enc.encode("b", 5)
        assert sorted(enc.pts_set) == [5, 6]

    def test_flush_starts_above_half_buffer(self, tmp_path):
        enc = _buffered_encoder(tmp_path)
        for i in range(enc.BUFFER_MAX_SIZE // 2):
            enc.encode(f"f{i}", i)
        enc._encode_queue.put.assert_not_called()
        enc.encode("one-more", 99)
        enc._encode_queue.put.assert_called_once_with(("f0", 0, None))

    def test_smallest_pts_pairs_with_oldest_frame(self, tmp_path):
        enc = _buffered_encoder(tmp_path)
        for i, pts in enumerate([30, 10, 20, 40]):
            enc.encode(f"f{i}", pts)
        enc._process_buffer(flush_all=True)
        enc._encode_queue.put.assert_called_once_with(("f0", 10, None))

    def test_encode_raises_pending_worker_error(self, tmp_path):
        enc = _buffered_encoder(tmp_path)
        enc._worker_error = RuntimeError("nvenc exploded")
        with pytest.raises(RuntimeError, match="nvenc exploded"):
            enc.encode("frame", 0)


class TestWorkerErrorChannel:
    def test_worker_records_error_and_keeps_consuming(self, tmp_path):
        enc = _make_encoder(tmp_path)
        enc.device = torch.device("cpu")
        enc._worker_error = None
        enc._encode_queue = queue.Queue()
        enc._stop_sentinel = object()
        enc._handle_encode_item = MagicMock(side_effect=RuntimeError("mux failed"))

        worker = threading.Thread(target=enc._encode_worker, daemon=True)
        worker.start()
        enc._encode_queue.put(("frame", 0, None))
        enc._encode_queue.put(("frame", 1, None))
        enc._encode_queue.join()
        enc._encode_queue.put(enc._stop_sentinel)
        worker.join(timeout=5)

        assert not worker.is_alive()
        assert isinstance(enc._worker_error, RuntimeError)
        assert enc._handle_encode_item.call_count == 1


def _packet(stream_index, dts, time_base=Fraction(1, 1000)):
    return SimpleNamespace(
        stream=SimpleNamespace(index=stream_index),
        dts=dts,
        pts=dts,
        time_base=time_base,
    )


class TestAudioPump:
    def _audio_encoder(self, tmp_path, packets):
        enc = _make_encoder(tmp_path)
        out_a = MagicMock()
        enc._audio_pipes = {1: ("copy", out_a, None)}
        enc._audio_backlog = deque()
        enc._audio_iter = iter(packets)
        enc.dst = MagicMock()
        return enc, out_a

    def test_copy_reassigns_stream_and_skips_flush_packet(self, tmp_path):
        enc, out_a = self._audio_encoder(tmp_path, [])
        pkt = _packet(1, dts=0)
        assert enc._produce_audio_packets(pkt) == [pkt]
        assert pkt.stream is out_a

        flush = SimpleNamespace(stream=SimpleNamespace(index=1), dts=None, pts=None, time_base=None)
        assert enc._produce_audio_packets(flush) == []

    def test_pump_respects_threshold(self, tmp_path):
        packets = [_packet(1, dts=0), _packet(1, dts=500), _packet(1, dts=1500)]
        enc, _ = self._audio_encoder(tmp_path, packets)

        enc._pump_audio(1.0)
        assert enc.dst.mux.call_count == 2
        assert len(enc._audio_backlog) == 1  # dts=1500 held back

        enc._pump_audio(None)
        assert enc.dst.mux.call_count == 3

    def test_pump_without_audio_is_noop(self, tmp_path):
        enc = _make_encoder(tmp_path)
        enc._audio_iter = None
        enc._pump_audio(1.0)
