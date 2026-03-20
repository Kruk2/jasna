"""Tests for secondary restorer helpers."""
from __future__ import annotations

import torch

from jasna.restorer.restoration_pipeline import _IdentitySecondaryRestorer
from jasna.restorer.tvai_secondary_restorer import TvaiSecondaryRestorer


class TestIdentitySecondaryRestorer:
    def test_restore_returns_uint8_kept_frames(self):
        restorer = _IdentitySecondaryRestorer()
        frames = torch.rand((5, 3, 256, 256))
        result = restorer.restore(frames, keep_start=1, keep_end=4)
        assert len(result) == 3
        assert all(f.dtype == torch.uint8 for f in result)
        assert all(f.shape == (3, 256, 256) for f in result)

    def test_restore_empty_range(self):
        restorer = _IdentitySecondaryRestorer()
        frames = torch.rand((5, 3, 256, 256))
        assert restorer.restore(frames, keep_start=3, keep_end=3) == []

    def test_name_and_workers(self):
        restorer = _IdentitySecondaryRestorer()
        assert restorer.name == "identity"
        assert restorer.num_workers == 1


class TestTvaiSecondaryRestorerConfig:
    def test_constructor_keeps_config(self):
        restorer = TvaiSecondaryRestorer(
            ffmpeg_path="ffmpeg.exe",
            tvai_args="model=iris-2:scale=1",
            scale=1,
            num_workers=2,
        )
        assert restorer.name == "tvai"
        assert restorer.ffmpeg_path == "ffmpeg.exe"
        assert restorer.tvai_args == "model=iris-2:scale=1"
        assert restorer.scale == 1
        assert restorer.num_workers == 2

    def test_build_ffmpeg_cmd(self):
        restorer = TvaiSecondaryRestorer(
            ffmpeg_path="ffmpeg.exe",
            tvai_args="model=iris-2:scale=4:w=256:h=256:noise=0",
            scale=2,
            num_workers=2,
        )
        assert restorer.tvai_filter_args == "model=iris-2:scale=2:noise=0"
        cmd = restorer.build_ffmpeg_cmd()
        assert cmd[0] == "ffmpeg.exe"
        assert "tvai_up=model=iris-2:scale=2:noise=0" in cmd
