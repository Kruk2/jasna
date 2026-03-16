"""Tests for jasna.restorer.tvai_secondary_restorer covering parsing, validation, worker errors, and close."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    def test_restore_frames_ffmpeg_crash(self):
        mock_proc = MagicMock()
        mock_proc.pid = 999
        mock_proc.communicate.return_value = (b"", b"some error")
        mock_proc.returncode = 1

        worker = _TvaiWorker.__new__(_TvaiWorker)
        worker._proc = mock_proc
        worker.out_w = 256
        worker.out_h = 256
        worker._in_frame_bytes = 256 * 256 * 3
        worker._out_frame_bytes = 256 * 256 * 3

        frames = np.zeros((2, 256, 256, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="crashed"):
            worker.restore_frames(frames)

    def test_restore_frames_not_enough_output(self):
        out_frame_bytes = 256 * 256 * 3
        mock_proc = MagicMock()
        mock_proc.pid = 999
        mock_proc.communicate.return_value = (bytes(out_frame_bytes), b"")
        mock_proc.returncode = 0

        worker = _TvaiWorker.__new__(_TvaiWorker)
        worker._proc = mock_proc
        worker.out_w = 256
        worker.out_h = 256
        worker._in_frame_bytes = 256 * 256 * 3
        worker._out_frame_bytes = out_frame_bytes

        frames = np.zeros((2, 256, 256, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="expected 3 output frames, got 1"):
            worker.restore_frames(frames)

    def test_restore_frames_stderr_logged(self):
        out_frame_bytes = 256 * 256 * 3
        n_input = 1
        stdout = bytes(out_frame_bytes * (1 + n_input))
        mock_proc = MagicMock()
        mock_proc.pid = 999
        mock_proc.communicate.return_value = (stdout, b"some warning")
        mock_proc.returncode = 0

        worker = _TvaiWorker.__new__(_TvaiWorker)
        worker._proc = mock_proc
        worker.out_w = 256
        worker.out_h = 256
        worker._in_frame_bytes = 256 * 256 * 3
        worker._out_frame_bytes = out_frame_bytes

        frames = np.zeros((n_input, 256, 256, 3), dtype=np.uint8)
        result = worker.restore_frames(frames)
        assert len(result) == 1
        assert result[0].shape == (256, 256, 3)

    def test_kill_when_running(self):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        worker = _TvaiWorker.__new__(_TvaiWorker)
        worker._proc = mock_proc
        worker.kill()
        mock_proc.kill.assert_called_once()

    def test_kill_when_already_dead(self):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0

        worker = _TvaiWorker.__new__(_TvaiWorker)
        worker._proc = mock_proc
        worker.kill()
        mock_proc.kill.assert_not_called()


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
            with patch.object(TvaiSecondaryRestorer, "_spawn_warm_worker"):
                TvaiSecondaryRestorer(
                    ffmpeg_path=str(ffmpeg),
                    tvai_args="model=iris-2",
                    scale=3,
                    num_workers=1,
                )


class TestTvaiPadToMinimum:
    def test_no_padding_needed(self):
        restorer = TvaiSecondaryRestorer.__new__(TvaiSecondaryRestorer)
        frames = np.zeros((TVAI_MIN_FRAMES, 256, 256, 3), dtype=np.uint8)
        result, pad = restorer._pad_to_minimum(frames)
        assert pad == 0
        assert result is frames

    def test_padding_added(self):
        restorer = TvaiSecondaryRestorer.__new__(TvaiSecondaryRestorer)
        frames = np.zeros((2, 256, 256, 3), dtype=np.uint8)
        result, pad = restorer._pad_to_minimum(frames)
        assert pad == TVAI_MIN_FRAMES - 2
        assert result.shape[0] == TVAI_MIN_FRAMES


class TestTvaiReplenishClosed:
    def test_replenish_when_closed_returns_early(self):
        restorer = TvaiSecondaryRestorer.__new__(TvaiSecondaryRestorer)
        restorer._closed = True
        restorer._pool = MagicMock()

        with patch.object(TvaiSecondaryRestorer, "_spawn_warm_worker") as mock_spawn:
            restorer._replenish_async()
            import time
            time.sleep(0.1)
            mock_spawn.assert_not_called()


class TestTvaiClose:
    def test_close_kills_workers(self):
        from queue import Queue

        restorer = TvaiSecondaryRestorer.__new__(TvaiSecondaryRestorer)
        restorer._closed = False
        restorer._pool = Queue(maxsize=2)

        w1 = MagicMock()
        w2 = MagicMock()
        restorer._pool.put(w1)
        restorer._pool.put(w2)

        restorer.close()

        assert restorer._closed is True
        w1.kill.assert_called_once()
        w2.kill.assert_called_once()

    def test_close_empty_pool(self):
        from queue import Queue

        restorer = TvaiSecondaryRestorer.__new__(TvaiSecondaryRestorer)
        restorer._closed = False
        restorer._pool = Queue(maxsize=1)

        restorer.close()
        assert restorer._closed is True


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


class TestTvaiRestore:
    def test_restore_empty_keep_range(self):
        restorer = TvaiSecondaryRestorer.__new__(TvaiSecondaryRestorer)
        restorer._restore_call_count = 0
        frames = torch.rand((5, 3, 256, 256))
        result = restorer.restore(frames, keep_start=3, keep_end=3)
        assert result == []

    def test_restore_full_flow(self):
        restorer = TvaiSecondaryRestorer.__new__(TvaiSecondaryRestorer)
        restorer._restore_call_count = 0
        restorer.scale = 1
        restorer.out_w = 256
        restorer.out_h = 256

        mock_worker = MagicMock()
        out_frames = [np.zeros((256, 256, 3), dtype=np.uint8)] * 3
        mock_worker.restore_frames.return_value = out_frames

        restorer._pool = MagicMock()
        restorer._pool.get.return_value = mock_worker
        restorer._closed = False

        with patch.object(TvaiSecondaryRestorer, "_spawn_warm_worker", return_value=MagicMock()):
            frames = torch.rand((5, 3, 256, 256))
            result = restorer.restore(frames, keep_start=1, keep_end=4)

        assert len(result) == 3
        assert result[0].shape == (3, 256, 256)
