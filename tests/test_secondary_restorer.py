"""Tests for secondary restorer helpers."""
from __future__ import annotations

from jasna.restorer.secondary_restorer import AsyncSecondaryRestorer
from jasna.restorer.tvai_secondary_restorer import TvaiSecondaryRestorer


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

    def test_implements_async_protocol(self):
        restorer = TvaiSecondaryRestorer(
            ffmpeg_path="ffmpeg.exe",
            tvai_args="model=iris-2:scale=1",
            scale=1,
            num_workers=1,
        )
        assert isinstance(restorer, AsyncSecondaryRestorer)
