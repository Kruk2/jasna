"""Tests for jasna.restorer.tvai_secondary_restorer covering parsing, validation, worker lifecycle, and close."""
from __future__ import annotations

import os
import threading
from collections import deque
from io import BytesIO
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from jasna.restorer.tvai_secondary_restorer import (
    _parse_tvai_args_kv,
    _TvaiWorker,
    TvaiSecondaryRestorer,
    TVAI_MIN_FRAMES,
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


class TestTvaiWorker:
    def test_push_frames_writes_in_batches(self):
        worker = _TvaiWorker.__new__(_TvaiWorker)
        worker._in_frame_bytes = 256 * 256 * 3
        worker._proc = MagicMock()

        frames = np.zeros((6, 256, 256, 3), dtype=np.uint8)
        worker.push_frames(frames)

        assert worker._proc.stdin.write.call_count == 2
        assert worker._proc.stdin.flush.call_count == 1

    def test_push_frames_single_batch(self):
        worker = _TvaiWorker.__new__(_TvaiWorker)
        worker._in_frame_bytes = 256 * 256 * 3
        worker._proc = MagicMock()

        frames = np.zeros((3, 256, 256, 3), dtype=np.uint8)
        worker.push_frames(frames)

        assert worker._proc.stdin.write.call_count == 1

    def test_flush_returns_remaining(self):
        worker = _TvaiWorker.__new__(_TvaiWorker)
        worker._proc = MagicMock()
        worker._reader = MagicMock()
        worker._reader.is_alive.return_value = False
        worker._output = Queue()
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        worker._output.put(frame)
        worker._output.put(frame)

        remaining = worker.flush()
        assert len(remaining) == 2
        worker._proc.stdin.close.assert_called_once()
        worker._reader.join.assert_called_once_with(timeout=0)

    def test_alive_true(self):
        worker = _TvaiWorker.__new__(_TvaiWorker)
        worker._proc = MagicMock()
        worker._proc.poll.return_value = None
        assert worker.alive is True

    def test_alive_false(self):
        worker = _TvaiWorker.__new__(_TvaiWorker)
        worker._proc = MagicMock()
        worker._proc.poll.return_value = 1
        assert worker.alive is False

    def test_kill_when_running(self):
        worker = _TvaiWorker.__new__(_TvaiWorker)
        worker._proc = MagicMock()
        worker._proc.poll.return_value = None
        worker.kill()
        worker._proc.kill.assert_called_once()

    def test_kill_when_already_dead(self):
        worker = _TvaiWorker.__new__(_TvaiWorker)
        worker._proc = MagicMock()
        worker._proc.poll.return_value = 0
        worker.kill()
        worker._proc.kill.assert_not_called()

    def test_reader_loop_puts_all_frames(self):
        worker = _TvaiWorker.__new__(_TvaiWorker)
        worker.out_w = 256
        worker.out_h = 256
        worker._out_frame_bytes = 256 * 256 * 3
        worker._output = Queue()

        frame_data = bytes(worker._out_frame_bytes)
        mock_proc = MagicMock()
        mock_proc.stdout = BytesIO(frame_data * 3)
        mock_proc.stderr.read.return_value = b""
        mock_proc.wait.return_value = 0
        worker._proc = mock_proc

        worker._reader_loop()

        assert worker._output.qsize() == 3

    def test_reader_loop_handles_crash(self):
        worker = _TvaiWorker.__new__(_TvaiWorker)
        worker.out_w = 256
        worker.out_h = 256
        worker._out_frame_bytes = 256 * 256 * 3
        worker._output = Queue()
        worker._intentional_kill = False

        mock_proc = MagicMock()
        mock_proc.stdout = BytesIO(b"")
        mock_proc.stderr.read.return_value = b"crash info"
        mock_proc.wait.return_value = -1
        worker._proc = mock_proc

        worker._reader_loop()
        assert worker._output.qsize() == 0


class TestTvaiValidateEnvironment:
    def test_missing_data_dir(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TVAI_MODEL_DATA_DIR", raising=False)
        monkeypatch.setenv("TVAI_MODEL_DIR", str(tmp_path))
        ffmpeg = tmp_path / "ffmpeg.exe"
        ffmpeg.write_bytes(b"")

        restorer = TvaiSecondaryRestorer.__new__(TvaiSecondaryRestorer)
        restorer.ffmpeg_path = str(ffmpeg)
        with pytest.raises(RuntimeError, match="TVAI_MODEL_DATA_DIR"):
            restorer._validate_environment()

    def test_missing_model_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TVAI_MODEL_DATA_DIR", str(tmp_path))
        monkeypatch.delenv("TVAI_MODEL_DIR", raising=False)
        ffmpeg = tmp_path / "ffmpeg.exe"
        ffmpeg.write_bytes(b"")

        restorer = TvaiSecondaryRestorer.__new__(TvaiSecondaryRestorer)
        restorer.ffmpeg_path = str(ffmpeg)
        with pytest.raises(RuntimeError, match="TVAI_MODEL_DIR"):
            restorer._validate_environment()

    def test_data_dir_not_a_directory(self, monkeypatch, tmp_path):
        fake_file = tmp_path / "not_a_dir"
        fake_file.write_bytes(b"")
        monkeypatch.setenv("TVAI_MODEL_DATA_DIR", str(fake_file))
        monkeypatch.setenv("TVAI_MODEL_DIR", str(tmp_path))
        ffmpeg = tmp_path / "ffmpeg.exe"
        ffmpeg.write_bytes(b"")

        restorer = TvaiSecondaryRestorer.__new__(TvaiSecondaryRestorer)
        restorer.ffmpeg_path = str(ffmpeg)
        with pytest.raises(RuntimeError, match="not a directory"):
            restorer._validate_environment()

    def test_model_dir_not_a_directory(self, monkeypatch, tmp_path):
        fake_file = tmp_path / "not_a_dir"
        fake_file.write_bytes(b"")
        monkeypatch.setenv("TVAI_MODEL_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("TVAI_MODEL_DIR", str(fake_file))
        ffmpeg = tmp_path / "ffmpeg.exe"
        ffmpeg.write_bytes(b"")

        restorer = TvaiSecondaryRestorer.__new__(TvaiSecondaryRestorer)
        restorer.ffmpeg_path = str(ffmpeg)
        with pytest.raises(RuntimeError, match="not a directory"):
            restorer._validate_environment()

    def test_ffmpeg_not_found(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TVAI_MODEL_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("TVAI_MODEL_DIR", str(tmp_path))

        restorer = TvaiSecondaryRestorer.__new__(TvaiSecondaryRestorer)
        restorer.ffmpeg_path = str(tmp_path / "missing_ffmpeg.exe")
        with pytest.raises(FileNotFoundError, match="not found"):
            restorer._validate_environment()


class TestTvaiSecondaryRestorerInit:
    def test_invalid_scale(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TVAI_MODEL_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("TVAI_MODEL_DIR", str(tmp_path))
        ffmpeg = tmp_path / "ffmpeg.exe"
        ffmpeg.write_bytes(b"")

        with pytest.raises(ValueError, match="Invalid tvai scale"):
            TvaiSecondaryRestorer(
                ffmpeg_path=str(ffmpeg),
                tvai_args="model=iris-2",
                scale=3,
                num_workers=1,
            )


def _make_restorer(num_workers=1):
    restorer = TvaiSecondaryRestorer.__new__(TvaiSecondaryRestorer)
    restorer.num_workers = num_workers
    restorer.scale = 1
    workers = []
    for _ in range(num_workers):
        w = MagicMock()
        w.alive = True
        w._proc.pid = 100
        workers.append(w)
    restorer._workers = workers
    restorer._worker_locks = [threading.Lock() for _ in range(num_workers)]
    restorer._worker_pending_frames = [0] * num_workers
    restorer._assign_lock = threading.Lock()
    restorer._next_worker_idx = 0
    restorer._next_seq = 0
    restorer._ready_empty_seqs = deque()
    restorer._worker_pending_clips = [deque() for _ in range(num_workers)]
    restorer._worker_output_buf = [[] for _ in range(num_workers)]
    restorer._worker_last_frame = [None] * num_workers
    return restorer


class TestTvaiAsyncApi:
    def test_push_clip_empty_range(self):
        restorer = _make_restorer()
        seq = restorer.push_clip(torch.rand((5, 3, 256, 256)), keep_start=3, keep_end=3)
        assert seq == 0
        assert list(restorer._ready_empty_seqs) == [0]
        restorer._workers[0].push_frames.assert_not_called()

    def test_push_clip_pushes_to_worker(self):
        restorer = _make_restorer()
        frames = torch.rand((5, 3, 256, 256))
        seq = restorer.push_clip(frames, keep_start=1, keep_end=4)
        assert seq == 0
        assert len(restorer._worker_pending_clips[0]) == 1
        clip = restorer._worker_pending_clips[0][0]
        assert clip.frame_count == 3
        restorer._workers[0].push_frames.assert_called_once()

    def test_pop_completed_returns_empty_when_no_output(self):
        restorer = _make_restorer()
        restorer._workers[0].read_available.return_value = []
        frames = torch.rand((5, 3, 256, 256))
        restorer.push_clip(frames, keep_start=0, keep_end=5)
        completed = restorer.pop_completed()
        assert completed == []

    def test_pop_completed_returns_clip_when_output_ready(self):
        restorer = _make_restorer()
        out_frames = [np.zeros((256, 256, 3), dtype=np.uint8)] * TVAI_MIN_FRAMES
        restorer._workers[0].read_available.return_value = out_frames
        frames = torch.rand((5, 3, 256, 256))
        restorer.push_clip(frames, keep_start=0, keep_end=5)
        completed = restorer.pop_completed()
        assert len(completed) == 1
        seq, result = completed[0]
        assert seq == 0
        assert len(result) == 5

    def test_pop_completed_handles_excess_buffer(self):
        restorer = _make_restorer()
        out_frames = [np.zeros((256, 256, 3), dtype=np.uint8)] * TVAI_MIN_FRAMES
        restorer._workers[0].read_available.return_value = out_frames
        frames = torch.rand((2, 3, 256, 256))
        restorer.push_clip(frames, keep_start=0, keep_end=2)
        completed = restorer.pop_completed()
        assert len(completed) == 1
        _, result = completed[0]
        assert len(result) == 2

    def test_pop_completed_empty_range_clip(self):
        restorer = _make_restorer()
        restorer.push_clip(torch.rand((5, 3, 256, 256)), keep_start=3, keep_end=3)
        restorer._workers[0].read_available.return_value = []
        completed = restorer.pop_completed()
        assert len(completed) == 1
        assert completed[0] == (0, [])

    def test_pop_completed_preserves_order(self):
        restorer = _make_restorer()
        restorer._workers[0].read_available.return_value = []
        frames = torch.rand((5, 3, 256, 256))
        restorer.push_clip(frames, keep_start=0, keep_end=5)
        restorer.push_clip(frames, keep_start=0, keep_end=5)
        completed = restorer.pop_completed()
        assert completed == []

        restorer._workers[0].read_available.return_value = [np.zeros((256, 256, 3), dtype=np.uint8)] * (TVAI_MIN_FRAMES * 2)
        completed = restorer.pop_completed()
        assert len(completed) == 2
        assert completed[0][0] == 0
        assert completed[1][0] == 1

    def test_pop_completed_returns_other_worker_completion(self):
        restorer = _make_restorer(num_workers=2)
        frames = torch.rand((5, 3, 256, 256))
        restorer.push_clip(frames, keep_start=0, keep_end=5)
        restorer.push_clip(frames, keep_start=0, keep_end=5)

        restorer._workers[0].read_available.return_value = []
        restorer._workers[1].read_available.return_value = [np.zeros((256, 256, 3), dtype=np.uint8)] * TVAI_MIN_FRAMES
        completed = restorer.pop_completed()
        assert len(completed) == 1
        assert completed[0][0] == 1

    def test_flush_all_workers_alive_after_return(self):
        restorer = _make_restorer()
        remaining = [np.zeros((256, 256, 3), dtype=np.uint8)] * TVAI_MIN_FRAMES
        w = restorer._workers[0]

        def flush_kills_worker():
            w.alive = False
            return remaining

        def restart_revives_worker():
            w.alive = True

        w.flush.side_effect = flush_kills_worker
        w.restart.side_effect = restart_revives_worker

        frames = torch.rand((5, 3, 256, 256))
        restorer.push_clip(frames, keep_start=0, keep_end=5)
        restorer.flush_all()

        assert w.alive is True
        assert len(restorer._worker_output_buf[0]) == TVAI_MIN_FRAMES
        assert restorer._worker_pending_frames[0] == 5

        w.restart.reset_mock()
        restorer.push_clip(frames, keep_start=0, keep_end=5)
        w.restart.assert_not_called()

    def test_flush_all_skips_idle_workers(self):
        restorer = _make_restorer(num_workers=2)
        remaining = [np.zeros((256, 256, 3), dtype=np.uint8)] * TVAI_MIN_FRAMES
        restorer._workers[0].flush.return_value = remaining
        frames = torch.rand((5, 3, 256, 256))
        restorer.push_clip(frames, keep_start=0, keep_end=5)
        restorer.flush_all()
        restorer._workers[0].flush.assert_called_once()
        restorer._workers[1].flush.assert_not_called()

    def test_flush_pads_at_flush_time(self):
        restorer = _make_restorer()
        w = restorer._workers[0]
        w.flush.return_value = [np.zeros((256, 256, 3), dtype=np.uint8)] * TVAI_MIN_FRAMES
        w.read_available.return_value = []

        frames = torch.rand((2, 3, 256, 256))
        restorer.push_clip(frames, keep_start=0, keep_end=2)
        assert restorer._worker_pending_frames[0] == 2

        restorer.flush_all()
        w.push_frames.assert_called()
        pad_call_args = w.push_frames.call_args_list[-1][0][0]
        assert pad_call_args.shape[0] == 3

        assert len(restorer._worker_output_buf[0]) == 2

        completed = restorer.pop_completed()
        assert len(completed) == 1
        assert len(completed[0][1]) == 2

    def test_flush_no_padding_when_enough_frames(self):
        restorer = _make_restorer()
        remaining = [np.zeros((256, 256, 3), dtype=np.uint8)] * TVAI_MIN_FRAMES
        restorer._workers[0].flush.return_value = remaining
        restorer._workers[0].read_available.return_value = []
        frames = torch.rand((5, 3, 256, 256))
        restorer.push_clip(frames, keep_start=0, keep_end=5)
        restorer.flush_all()
        assert restorer._workers[0].push_frames.call_count == 1
        completed = restorer.pop_completed()
        assert len(completed) == 1
        assert len(completed[0][1]) == 5

    def test_push_clip_round_robin_assignment(self):
        restorer = _make_restorer(num_workers=2)
        big = torch.rand((20, 3, 256, 256))
        small = torch.rand((5, 3, 256, 256))
        restorer.push_clip(big, keep_start=0, keep_end=20)
        restorer.push_clip(small, keep_start=0, keep_end=5)
        restorer.push_clip(small, keep_start=0, keep_end=5)
        assert restorer._worker_pending_clips[0][0].worker_idx == 0
        assert restorer._worker_pending_clips[1][0].worker_idx == 1
        assert restorer._worker_pending_clips[0][1].worker_idx == 0

    def test_push_clip_restarts_dead_worker(self):
        restorer = _make_restorer()
        restorer._workers[0].alive = False
        frames = torch.rand((5, 3, 256, 256))
        restorer.push_clip(frames, keep_start=0, keep_end=5)
        restorer._workers[0].restart.assert_called_once()

    def test_sequence_numbers_increment(self):
        restorer = _make_restorer()
        frames = torch.rand((5, 3, 256, 256))
        s0 = restorer.push_clip(frames, keep_start=0, keep_end=5)
        s1 = restorer.push_clip(frames, keep_start=0, keep_end=5)
        s2 = restorer.push_clip(frames, keep_start=3, keep_end=3)
        assert s0 == 0
        assert s1 == 1
        assert s2 == 2


class TestTvaiClose:
    def test_close_kills_all_workers(self):
        restorer = _make_restorer(num_workers=2)
        restorer.close()
        for w in restorer._workers:
            w.kill.assert_called_once()


class TestTvaiToNumpyHwc:
    def test_conversion(self):
        restorer = TvaiSecondaryRestorer.__new__(TvaiSecondaryRestorer)
        frames = torch.rand((2, 3, 256, 256), dtype=torch.float32)
        result = restorer._to_numpy_hwc(frames)
        assert result.shape == (2, 256, 256, 3)
        assert result.dtype == np.uint8


class TestTvaiToTensors:
    def test_conversion(self):
        restorer = TvaiSecondaryRestorer.__new__(TvaiSecondaryRestorer)
        frames = [np.zeros((256, 256, 3), dtype=np.uint8), np.ones((256, 256, 3), dtype=np.uint8)]
        result = restorer._to_tensors(frames)
        assert len(result) == 2
        assert result[0].shape == (3, 256, 256)
        assert result[0].dtype == torch.uint8


