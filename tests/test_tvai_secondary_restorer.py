"""Tests for jasna.restorer.tvai_secondary_restorer — parsing, validation, communicate, restore, close."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from jasna.restorer.tvai_secondary_restorer import (
    TVAI_MIN_FRAMES,
    TvaiSecondaryRestorer,
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

    def test_model_dir_not_a_directory(self, monkeypatch, tmp_path):
        fake = tmp_path / "not_a_dir"
        fake.write_bytes(b"")
        monkeypatch.setenv("TVAI_MODEL_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("TVAI_MODEL_DIR", str(fake))
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


def _make_restorer(scale=1, num_workers=1):
    r = TvaiSecondaryRestorer(ffmpeg_path="ffmpeg.exe", tvai_args="model=iris-2", scale=scale, num_workers=num_workers)
    r._validated = True
    return r


def _fake_stdout(n_frames: int, frame_bytes: int) -> bytes:
    return bytes(n_frames * frame_bytes)


class TestTvaiCommunicate:
    def test_empty_input_returns_empty(self):
        r = _make_restorer()
        assert r._communicate(np.zeros((0, 256, 256, 3), dtype=np.uint8)) == []

    def test_normal_5_frames(self):
        r = _make_restorer()
        n = 5
        stdout = _fake_stdout(n, r._out_frame_bytes)
        mock_result = MagicMock(returncode=0, stdout=stdout, stderr=b"")
        with patch("jasna.restorer.tvai_secondary_restorer.subprocess.run", return_value=mock_result):
            frames_np = np.zeros((n, 256, 256, 3), dtype=np.uint8)
            result = r._communicate(frames_np)
        assert len(result) == 5
        assert result[0].shape == (256, 256, 3)

    def test_padding_for_short_clip(self):
        r = _make_restorer()
        n = 2
        stdout = _fake_stdout(TVAI_MIN_FRAMES, r._out_frame_bytes)
        mock_result = MagicMock(returncode=0, stdout=stdout, stderr=b"")
        with patch("jasna.restorer.tvai_secondary_restorer.subprocess.run", return_value=mock_result) as mock_run:
            frames_np = np.zeros((n, 256, 256, 3), dtype=np.uint8)
            result = r._communicate(frames_np)
        assert len(result) == 2
        call_args = mock_run.call_args
        stdin_bytes = call_args.kwargs["input"]
        assert len(stdin_bytes) == TVAI_MIN_FRAMES * r._in_frame_bytes

    def test_ffmpeg_error_raises(self):
        r = _make_restorer()
        mock_result = MagicMock(returncode=1, stdout=b"", stderr=b"some error")
        with patch("jasna.restorer.tvai_secondary_restorer.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="TVAI ffmpeg failed"):
                r._communicate(np.zeros((5, 256, 256, 3), dtype=np.uint8))

    def test_output_too_short_raises(self):
        r = _make_restorer()
        mock_result = MagicMock(returncode=0, stdout=b"short", stderr=b"")
        with patch("jasna.restorer.tvai_secondary_restorer.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="output too short"):
                r._communicate(np.zeros((5, 256, 256, 3), dtype=np.uint8))

    def test_scale_2_output_size(self):
        r = _make_restorer(scale=2)
        n = 5
        stdout = _fake_stdout(n, r._out_frame_bytes)
        mock_result = MagicMock(returncode=0, stdout=stdout, stderr=b"")
        with patch("jasna.restorer.tvai_secondary_restorer.subprocess.run", return_value=mock_result):
            result = r._communicate(np.zeros((n, 256, 256, 3), dtype=np.uint8))
        assert len(result) == 5
        assert result[0].shape == (512, 512, 3)


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
        assert len(result) == 2
        assert result[0].shape == (3, 256, 256)
        assert result[0].dtype == torch.uint8


class TestTvaiRestore:
    def test_empty_range_returns_empty(self):
        r = _make_restorer()
        frames = torch.rand((5, 3, 256, 256))
        assert r.restore(frames, keep_start=3, keep_end=3) == []

    def test_keep_start_end_slicing(self):
        r = _make_restorer()
        n_kept = 3
        # 3 frames < TVAI_MIN_FRAMES so _communicate pads to 5 internally
        stdout = _fake_stdout(TVAI_MIN_FRAMES, r._out_frame_bytes)
        mock_result = MagicMock(returncode=0, stdout=stdout, stderr=b"")
        with patch("jasna.restorer.tvai_secondary_restorer.subprocess.run", return_value=mock_result):
            result = r.restore(torch.rand((10, 3, 256, 256)), keep_start=2, keep_end=5)
        assert len(result) == 3
        assert result[0].shape == (3, 256, 256)
        assert result[0].dtype == torch.uint8

    def test_full_range(self):
        r = _make_restorer()
        n = 6
        stdout = _fake_stdout(n, r._out_frame_bytes)
        mock_result = MagicMock(returncode=0, stdout=stdout, stderr=b"")
        with patch("jasna.restorer.tvai_secondary_restorer.subprocess.run", return_value=mock_result):
            result = r.restore(torch.rand((n, 3, 256, 256)), keep_start=0, keep_end=n)
        assert len(result) == n

    def test_creates_pool_lazily(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TVAI_MODEL_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("TVAI_MODEL_DIR", str(tmp_path))
        ffmpeg = tmp_path / "ffmpeg.exe"
        ffmpeg.write_bytes(b"")
        r = TvaiSecondaryRestorer(ffmpeg_path=str(ffmpeg), tvai_args="model=iris-2", scale=1, num_workers=2)
        assert r._pool is None
        stdout = _fake_stdout(5, r._out_frame_bytes)
        mock_result = MagicMock(returncode=0, stdout=stdout, stderr=b"")
        with patch("jasna.restorer.tvai_secondary_restorer.subprocess.run", return_value=mock_result):
            r.restore(torch.rand((5, 3, 256, 256)), keep_start=0, keep_end=5)
        assert r._pool is not None
        r.close()


class TestTvaiClose:
    def test_close_shuts_down_pool(self):
        r = _make_restorer()
        mock_pool = MagicMock()
        r._pool = mock_pool
        r.close()
        mock_pool.shutdown.assert_called_once_with(wait=False)
        assert r._pool is None

    def test_close_noop_when_no_pool(self):
        r = _make_restorer()
        assert r._pool is None
        r.close()
