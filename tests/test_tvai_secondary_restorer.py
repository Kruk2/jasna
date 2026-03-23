"""Tests for jasna.restorer.tvai_secondary_restorer — persistent worker design."""
from __future__ import annotations

import threading
from collections import deque
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
import torch

from jasna.restorer.tvai_secondary_restorer import (
    TVAI_PIPELINE_DELAY,
    TvaiSecondaryRestorer,
    _ClipSegment,
    _FillerSegment,
    _TvaiWorker,
    _parse_tvai_args_kv,
)


class TestParseTvaiArgsKv:
    def test_empty_string(self):
        assert _parse_tvai_args_kv("") == {}

    def test_none_string(self):
        assert _parse_tvai_args_kv(None) == {}

    def test_whitespace_only(self):
        assert _parse_tvai_args_kv("   ") == {}

    def test_single_kv(self):
        assert _parse_tvai_args_kv("model=iris-2") == {"model": "iris-2"}

    def test_multiple_kv(self):
        result = _parse_tvai_args_kv("model=iris-2:scale=2:noise=0")
        assert result == {"model": "iris-2", "scale": "2", "noise": "0"}

    def test_trailing_colon(self):
        result = _parse_tvai_args_kv("model=iris-2:")
        assert result == {"model": "iris-2"}

    def test_leading_colon(self):
        result = _parse_tvai_args_kv(":model=iris-2")
        assert result == {"model": "iris-2"}

    def test_double_colon(self):
        result = _parse_tvai_args_kv("model=iris-2::scale=2")
        assert result == {"model": "iris-2", "scale": "2"}

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError, match="expected key=value"):
            _parse_tvai_args_kv("model")

    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="empty key"):
            _parse_tvai_args_kv("=value")


class TestTvaiInit:
    def test_valid_scales(self):
        for s in (1, 2, 4):
            r = TvaiSecondaryRestorer(ffmpeg_path="ffmpeg.exe", tvai_args="model=iris-2", scale=s, num_workers=1)
            assert r.scale == s

    def test_invalid_scale_raises(self):
        with pytest.raises(ValueError, match="Invalid tvai scale"):
            TvaiSecondaryRestorer(ffmpeg_path="ffmpeg.exe", tvai_args="model=iris-2", scale=3, num_workers=1)

    def test_filter_args_built_correctly(self):
        r = TvaiSecondaryRestorer(
            ffmpeg_path="ffmpeg.exe",
            tvai_args="model=iris-2:scale=4:w=256:h=256:noise=0",
            scale=2,
            num_workers=2,
        )
        assert r.tvai_filter_args == "model=iris-2:scale=2:noise=0"

    def test_num_workers_stored(self):
        r = TvaiSecondaryRestorer(ffmpeg_path="ffmpeg.exe", tvai_args="model=iris-2", scale=1, num_workers=3)
        assert r.num_workers == 3

    def test_out_size_calculated(self):
        r = TvaiSecondaryRestorer(ffmpeg_path="ffmpeg.exe", tvai_args="model=iris-2", scale=4, num_workers=1)
        assert r._out_size == 1024

    def test_not_started_on_init(self):
        r = TvaiSecondaryRestorer(ffmpeg_path="ffmpeg.exe", tvai_args="model=iris-2", scale=1, num_workers=1)
        assert not r._started
        assert r._workers == []


class TestTvaiBuildFfmpegCmd:
    def test_basic_cmd_structure(self):
        r = TvaiSecondaryRestorer(ffmpeg_path="ffmpeg.exe", tvai_args="model=iris-2", scale=1, num_workers=1)
        cmd = r.build_ffmpeg_cmd()
        assert cmd[0] == "ffmpeg.exe"
        assert "-f" in cmd
        assert "rawvideo" in cmd
        assert "pipe:0" in cmd
        assert "pipe:1" in cmd

    def test_filter_in_cmd(self):
        r = TvaiSecondaryRestorer(ffmpeg_path="ffmpeg.exe", tvai_args="model=iris-2", scale=2, num_workers=1)
        cmd = r.build_ffmpeg_cmd()
        assert "tvai_up=model=iris-2:scale=2" in cmd


class TestTvaiValidateEnvironment:
    def test_missing_data_dir(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TVAI_MODEL_DATA_DIR", raising=False)
        monkeypatch.setenv("TVAI_MODEL_DIR", str(tmp_path))
        ffmpeg = tmp_path / "ffmpeg.exe"
        ffmpeg.write_bytes(b"")
        r = TvaiSecondaryRestorer.__new__(TvaiSecondaryRestorer)
        r.ffmpeg_path = str(ffmpeg)
        with pytest.raises(RuntimeError, match="TVAI_MODEL_DATA_DIR"):
            r._validate_environment()

    def test_missing_model_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TVAI_MODEL_DATA_DIR", str(tmp_path))
        monkeypatch.delenv("TVAI_MODEL_DIR", raising=False)
        ffmpeg = tmp_path / "ffmpeg.exe"
        ffmpeg.write_bytes(b"")
        r = TvaiSecondaryRestorer.__new__(TvaiSecondaryRestorer)
        r.ffmpeg_path = str(ffmpeg)
        with pytest.raises(RuntimeError, match="TVAI_MODEL_DIR"):
            r._validate_environment()

    def test_data_dir_not_a_directory(self, monkeypatch, tmp_path):
        fake = tmp_path / "not_a_dir"
        fake.write_bytes(b"")
        monkeypatch.setenv("TVAI_MODEL_DATA_DIR", str(fake))
        monkeypatch.setenv("TVAI_MODEL_DIR", str(tmp_path))
        ffmpeg = tmp_path / "ffmpeg.exe"
        ffmpeg.write_bytes(b"")
        r = TvaiSecondaryRestorer.__new__(TvaiSecondaryRestorer)
        r.ffmpeg_path = str(ffmpeg)
        with pytest.raises(RuntimeError, match="not a directory"):
            r._validate_environment()

    def test_ffmpeg_not_found(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TVAI_MODEL_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("TVAI_MODEL_DIR", str(tmp_path))
        r = TvaiSecondaryRestorer.__new__(TvaiSecondaryRestorer)
        r.ffmpeg_path = str(tmp_path / "missing_ffmpeg.exe")
        with pytest.raises(FileNotFoundError, match="not found"):
            r._validate_environment()


class TestTvaiToNumpyHwc:
    def test_conversion(self):
        frames = np.random.rand(2, 3, 256, 256).astype(np.float32)
        result = TvaiSecondaryRestorer._to_numpy_hwc(frames)
        assert result.shape == (2, 256, 256, 3)
        assert result.dtype == np.uint8


class TestTvaiToTensors:
    def test_conversion(self):
        frames = [np.zeros((256, 256, 3), dtype=np.uint8), np.ones((256, 256, 3), dtype=np.uint8)]
        result = TvaiSecondaryRestorer._to_tensors(frames)
        assert result.shape == (2, 3, 256, 256)
        assert result.dtype == torch.uint8


def _make_restorer(scale=1, num_workers=1):
    r = TvaiSecondaryRestorer(ffmpeg_path="ffmpeg.exe", tvai_args="model=iris-2", scale=scale, num_workers=num_workers)
    r._validated = True
    return r


def _make_frame(out_size=256):
    return np.zeros((out_size, out_size, 3), dtype=np.uint8)


def _setup_mock_workers(r, num_workers=None):
    n = num_workers or r.num_workers
    r._workers = [MagicMock(spec=_TvaiWorker) for _ in range(n)]
    for w in r._workers:
        w.drain_available.return_value = []
        w.close_stdin_and_drain.return_value = []
    r._worker_segments = [deque() for _ in range(n)]
    r._worker_locks = [threading.Lock() for _ in range(n)]
    r._started = True
    return r._workers


class TestPushClip:
    def test_empty_range_returns_immediately(self):
        r = _make_restorer()
        seq = r.push_clip(torch.rand((5, 3, 256, 256)), keep_start=3, keep_end=3)
        assert seq == 0
        assert r._completed[0] == []
        assert not r._started

    def test_assigns_segment_and_pushes_to_worker(self):
        r = _make_restorer(num_workers=2)
        workers = _setup_mock_workers(r)
        seq = r.push_clip(torch.rand((5, 3, 256, 256)), keep_start=0, keep_end=5)
        assert seq == 0
        workers[0].push_frames.assert_called_once()
        segs = r._worker_segments[0]
        assert len(segs) == 1
        assert isinstance(segs[0], _ClipSegment)
        assert segs[0].seq == 0
        assert segs[0].expected == 5

    def test_least_pending_frames_assignment(self):
        r = _make_restorer(num_workers=2)
        workers = _setup_mock_workers(r)
        # Push a large clip (170 frames) — goes to worker 0 (both at 0 pending)
        r.push_clip(torch.rand((170, 3, 256, 256)), keep_start=0, keep_end=170)
        assert workers[0].push_frames.call_count == 1
        assert workers[1].push_frames.call_count == 0

        # Push a small clip (1 frame) — goes to worker 1 (0 pending < 170 pending)
        r.push_clip(torch.rand((1, 3, 256, 256)), keep_start=0, keep_end=1)
        assert workers[0].push_frames.call_count == 1
        assert workers[1].push_frames.call_count == 1

        # Push another small clip — still worker 1 (1 pending < 170 pending)
        r.push_clip(torch.rand((1, 3, 256, 256)), keep_start=0, keep_end=1)
        assert workers[0].push_frames.call_count == 1
        assert workers[1].push_frames.call_count == 2

        segs_0 = [s for s in r._worker_segments[0] if isinstance(s, _ClipSegment)]
        segs_1 = [s for s in r._worker_segments[1] if isinstance(s, _ClipSegment)]
        assert sum(s.expected for s in segs_0) == 170
        assert sum(s.expected for s in segs_1) == 2

    def test_equal_pending_uses_lower_index(self):
        r = _make_restorer(num_workers=2)
        workers = _setup_mock_workers(r)
        r.push_clip(torch.rand((6, 3, 256, 256)), keep_start=0, keep_end=6)
        r.push_clip(torch.rand((6, 3, 256, 256)), keep_start=0, keep_end=6)
        # Both workers have 6 pending, next clip goes to worker 0 (min picks lowest index)
        r.push_clip(torch.rand((6, 3, 256, 256)), keep_start=0, keep_end=6)
        assert workers[0].push_frames.call_count == 2
        assert workers[1].push_frames.call_count == 1

    def test_seq_increments(self):
        r = _make_restorer()
        _setup_mock_workers(r)
        s0 = r.push_clip(torch.rand((3, 3, 256, 256)), keep_start=0, keep_end=3)
        s1 = r.push_clip(torch.rand((3, 3, 256, 256)), keep_start=0, keep_end=3)
        assert s0 == 0
        assert s1 == 1

    def test_keep_slicing(self):
        r = _make_restorer()
        workers = _setup_mock_workers(r)
        r.push_clip(torch.rand((10, 3, 256, 256)), keep_start=2, keep_end=5)
        seg = r._worker_segments[0][0]
        assert seg.expected == 3

    def test_short_clip_no_padding(self):
        r = _make_restorer()
        workers = _setup_mock_workers(r)
        r.push_clip(torch.rand((2, 3, 256, 256)), keep_start=0, keep_end=2)
        segs = list(r._worker_segments[0])
        assert len(segs) == 1
        assert isinstance(segs[0], _ClipSegment)
        assert segs[0].expected == 2
        pushed = workers[0].push_frames.call_args[0][0]
        assert pushed.shape[0] == 2


class TestDrainWorker:
    def test_clip_segment_collection(self):
        r = _make_restorer()
        workers = _setup_mock_workers(r)
        out = _make_frame()
        r._worker_segments[0].append(_ClipSegment(seq=0, expected=2))
        r._process_drained_frames(0, [out, out])
        assert 0 in r._completed
        assert len(r._completed[0]) == 2

    def test_filler_segment_skipped(self):
        r = _make_restorer()
        workers = _setup_mock_workers(r)
        out = _make_frame()
        r._worker_segments[0].append(_FillerSegment(remaining=2))
        r._worker_segments[0].append(_ClipSegment(seq=0, expected=1))
        r._process_drained_frames(0, [out, out, out])
        assert 0 in r._completed
        assert len(r._completed[0]) == 1
        assert len(r._worker_segments[0]) == 0

    def test_partial_clip_not_completed(self):
        r = _make_restorer()
        workers = _setup_mock_workers(r)
        out = _make_frame()
        r._worker_segments[0].append(_ClipSegment(seq=0, expected=5))
        r._process_drained_frames(0, [out, out])
        assert 0 not in r._completed
        assert r._worker_segments[0][0].collected == [out, out]

    def test_multiple_clips_drain_in_order(self):
        r = _make_restorer()
        workers = _setup_mock_workers(r)
        out = _make_frame()
        r._worker_segments[0].append(_ClipSegment(seq=0, expected=2))
        r._worker_segments[0].append(_ClipSegment(seq=1, expected=1))
        r._process_drained_frames(0, [out, out, out])
        assert 0 in r._completed
        assert 1 in r._completed
        assert len(r._completed[0]) == 2
        assert len(r._completed[1]) == 1


class TestPopCompleted:
    def test_returns_sorted_by_seq(self):
        r = _make_restorer(num_workers=2)
        _setup_mock_workers(r)
        r._completed[2] = [_make_frame()]
        r._completed[0] = [_make_frame()]
        r._completed[1] = [_make_frame()]
        result = r.pop_completed()
        assert [s for s, _ in result] == [0, 1, 2]

    def test_empties_completed_dict(self):
        r = _make_restorer()
        _setup_mock_workers(r)
        r._completed[0] = [_make_frame()]
        r.pop_completed()
        assert len(r._completed) == 0

    def test_drains_workers_before_returning(self):
        r = _make_restorer()
        workers = _setup_mock_workers(r)
        out = _make_frame()
        r._worker_segments[0].append(_ClipSegment(seq=0, expected=1))
        workers[0].drain_available.return_value = [out]
        result = r.pop_completed()
        assert len(result) == 1
        assert result[0][0] == 0


class TestHasPending:
    def test_false_when_no_segments(self):
        r = _make_restorer()
        _setup_mock_workers(r)
        assert not r.has_pending

    def test_true_with_clip_segment(self):
        r = _make_restorer()
        _setup_mock_workers(r)
        r._worker_segments[0].append(_ClipSegment(seq=0, expected=5))
        assert r.has_pending

    def test_false_with_only_filler_segments(self):
        r = _make_restorer()
        _setup_mock_workers(r)
        r._worker_segments[0].append(_FillerSegment(remaining=10))
        assert not r.has_pending


class TestFlushPending:
    def test_pushes_filler_to_workers_with_clip_segments(self):
        r = _make_restorer(num_workers=2)
        workers = _setup_mock_workers(r)
        r._worker_segments[0].append(_ClipSegment(seq=0, expected=5))
        r.flush_pending()
        workers[0].push_frames.assert_called_once()
        filler_bytes = workers[0].push_frames.call_args[0][0]
        assert filler_bytes.shape[0] == TVAI_PIPELINE_DELAY
        workers[1].push_frames.assert_not_called()
        assert isinstance(r._worker_segments[0][-1], _FillerSegment)

    def test_skips_workers_without_clips(self):
        r = _make_restorer(num_workers=2)
        workers = _setup_mock_workers(r)
        r._worker_segments[0].append(_FillerSegment(remaining=10))
        r.flush_pending()
        workers[0].push_frames.assert_not_called()
        workers[1].push_frames.assert_not_called()

    def test_skips_worker_already_flushed(self):
        r = _make_restorer(num_workers=2)
        workers = _setup_mock_workers(r)
        r._worker_segments[0].append(_ClipSegment(seq=0, expected=5))
        r.flush_pending()
        assert workers[0].push_frames.call_count == 1
        assert isinstance(r._worker_segments[0][-1], _FillerSegment)
        r.flush_pending()
        assert workers[0].push_frames.call_count == 1

    def test_reflush_after_filler_consumed(self):
        r = _make_restorer(num_workers=1)
        workers = _setup_mock_workers(r)
        r._worker_segments[0].append(_ClipSegment(seq=0, expected=5))
        r.flush_pending()
        assert workers[0].push_frames.call_count == 1
        r._worker_segments[0][-1].remaining = 0
        r._worker_segments[0].pop()
        r.flush_pending()
        assert workers[0].push_frames.call_count == 2

    def test_noop_when_not_started(self):
        r = _make_restorer()
        r.flush_pending()

    def test_target_seqs_flushes_only_matching_worker(self):
        r = _make_restorer(num_workers=3)
        workers = _setup_mock_workers(r)
        r._worker_segments[0].append(_ClipSegment(seq=0, expected=5))
        r._worker_segments[1].append(_ClipSegment(seq=1, expected=5))
        r._worker_segments[2].append(_ClipSegment(seq=2, expected=5))
        r.flush_pending(target_seqs={1})
        workers[0].push_frames.assert_not_called()
        workers[1].push_frames.assert_called_once()
        workers[2].push_frames.assert_not_called()

    def test_target_seqs_flushes_multiple_matching_workers(self):
        r = _make_restorer(num_workers=3)
        workers = _setup_mock_workers(r)
        r._worker_segments[0].append(_ClipSegment(seq=0, expected=5))
        r._worker_segments[1].append(_ClipSegment(seq=1, expected=5))
        r._worker_segments[2].append(_ClipSegment(seq=2, expected=5))
        r.flush_pending(target_seqs={0, 2})
        workers[0].push_frames.assert_called_once()
        workers[1].push_frames.assert_not_called()
        workers[2].push_frames.assert_called_once()

    def test_target_seqs_none_flushes_all(self):
        r = _make_restorer(num_workers=2)
        workers = _setup_mock_workers(r)
        r._worker_segments[0].append(_ClipSegment(seq=0, expected=5))
        r._worker_segments[1].append(_ClipSegment(seq=1, expected=5))
        r.flush_pending(target_seqs=None)
        workers[0].push_frames.assert_called_once()
        workers[1].push_frames.assert_called_once()


class TestFlushAll:
    def test_drains_and_restarts_workers(self):
        r = _make_restorer(num_workers=2)
        workers = _setup_mock_workers(r)
        out = _make_frame()
        r._worker_segments[0].append(_ClipSegment(seq=0, expected=2))
        workers[0].close_stdin_and_drain.return_value = [out, out]
        r.flush_all()
        assert 0 in r._completed
        assert len(r._completed[0]) == 2
        for w in workers:
            w.restart.assert_called_once()
        assert all(len(s) == 0 for s in r._worker_segments)

    def test_handles_incomplete_clips(self):
        r = _make_restorer()
        workers = _setup_mock_workers(r)
        out = _make_frame()
        r._worker_segments[0].append(_ClipSegment(seq=0, expected=5))
        workers[0].close_stdin_and_drain.return_value = [out, out]
        r.flush_all()
        assert 0 in r._completed
        assert len(r._completed[0]) == 2

    def test_noop_when_not_started(self):
        r = _make_restorer()
        r.flush_all()


class TestRestore:
    def test_empty_range(self):
        r = _make_restorer()
        result = r.restore(torch.rand((5, 3, 256, 256)), keep_start=3, keep_end=3)
        assert result == []

    def test_sync_push_flush_pop(self):
        r = _make_restorer()
        workers = _setup_mock_workers(r)
        out = _make_frame()
        workers[0].close_stdin_and_drain.return_value = [out] * 3
        result = r.restore(torch.rand((3, 3, 256, 256)), keep_start=0, keep_end=3)
        assert len(result) == 3
        assert result[0].shape == (3, 256, 256)
        assert result[0].dtype == torch.uint8

    def test_sync_large_clip(self):
        r = _make_restorer()
        workers = _setup_mock_workers(r)
        out = _make_frame()
        workers[0].close_stdin_and_drain.return_value = [out] * 6
        result = r.restore(torch.rand((6, 3, 256, 256)), keep_start=0, keep_end=6)
        assert len(result) == 6


class TestClose:
    def test_kills_all_workers(self):
        r = _make_restorer(num_workers=2)
        workers = _setup_mock_workers(r)
        r.close()
        for w in workers:
            w.kill.assert_called_once()
        assert r._workers == []
        assert not r._started

    def test_noop_when_not_started(self):
        r = _make_restorer()
        r.close()


class TestPushClipFlushDeadlock:
    def test_push_clip_does_not_block_flush_pending(self):
        r = _make_restorer(num_workers=2)
        workers = _setup_mock_workers(r)

        push_blocked = threading.Event()

        def blocking_push(frames):
            push_blocked.set()
            threading.Event().wait(timeout=10)

        workers[0].push_frames.side_effect = blocking_push
        workers[1].push_frames.side_effect = lambda f: None

        r._worker_segments[1].append(_ClipSegment(seq=99, expected=5))

        push_thread = threading.Thread(
            target=r.push_clip,
            args=(torch.rand((3, 3, 256, 256)),),
            kwargs={"keep_start": 0, "keep_end": 3},
            daemon=True,
        )
        push_thread.start()
        assert push_blocked.wait(timeout=5), "push_clip never called push_frames"

        flush_done = threading.Event()

        def try_flush():
            r.flush_pending(target_seqs={99})
            flush_done.set()

        flush_thread = threading.Thread(target=try_flush, daemon=True)
        flush_thread.start()

        assert flush_done.wait(timeout=3), "flush_pending deadlocked on _push_lock held by push_clip"
