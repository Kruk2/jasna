"""E2E integration tests for TvaiSecondaryRestorer — require real Topaz Video ffmpeg."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

TVAI_FFMPEG_PATH = os.environ.get("TVAI_FFMPEG_PATH", r"C:\Program Files\Topaz Labs LLC\Topaz Video\ffmpeg.exe")

_skip_reason = None
if not os.environ.get("TVAI_MODEL_DATA_DIR"):
    _skip_reason = "TVAI_MODEL_DATA_DIR not set"
elif not os.environ.get("TVAI_MODEL_DIR"):
    _skip_reason = "TVAI_MODEL_DIR not set"
elif not Path(TVAI_FFMPEG_PATH).is_file():
    _skip_reason = f"TVAI ffmpeg not found at {TVAI_FFMPEG_PATH}"

pytestmark = pytest.mark.skipif(_skip_reason is not None, reason=_skip_reason or "")

TVAI_ARGS = "model=iris-2:scale=1:preblur=0:noise=0:details=0:halo=0:blur=0:compression=0:estimate=8:blend=0.2:device=-2:vram=1:instances=1"


def _make_restorer(num_workers: int = 1):
    from jasna.restorer.tvai_secondary_restorer import TvaiSecondaryRestorer
    return TvaiSecondaryRestorer(
        ffmpeg_path=TVAI_FFMPEG_PATH,
        tvai_args=TVAI_ARGS,
        scale=1,
        num_workers=num_workers,
    )


def _make_color_frames(n: int, color: tuple[float, float, float] = (0.5, 0.3, 0.7)) -> torch.Tensor:
    frames = torch.zeros((n, 3, 256, 256), dtype=torch.float32)
    for c in range(3):
        frames[:, c, :, :] = color[c]
    return frames


class TestTvaiSingleFrame:
    def test_single_frame_returns_one_frame(self) -> None:
        restorer = _make_restorer()
        frames = _make_color_frames(1)
        result = restorer.restore(frames, keep_start=0, keep_end=1)
        restorer.close()
        assert len(result) == 1

    def test_single_frame_shape(self) -> None:
        restorer = _make_restorer()
        frames = _make_color_frames(1)
        result = restorer.restore(frames, keep_start=0, keep_end=1)
        restorer.close()
        assert result[0].shape == (3, 256, 256)
        assert result[0].dtype == torch.uint8


class TestTvaiPaddingDiscard:
    def test_padding_stripped_for_2_frames(self) -> None:
        restorer = _make_restorer()
        frames = _make_color_frames(2)
        result = restorer.restore(frames, keep_start=0, keep_end=2)
        restorer.close()
        assert len(result) == 2

    def test_padding_stripped_for_4_frames(self) -> None:
        restorer = _make_restorer()
        frames = _make_color_frames(4)
        result = restorer.restore(frames, keep_start=0, keep_end=4)
        restorer.close()
        assert len(result) == 4

    def test_no_padding_for_5_frames(self) -> None:
        restorer = _make_restorer()
        frames = _make_color_frames(5)
        result = restorer.restore(frames, keep_start=0, keep_end=5)
        restorer.close()
        assert len(result) == 5


class TestTvaiOutputQuality:
    def test_not_black(self) -> None:
        restorer = _make_restorer()
        frames = _make_color_frames(5, color=(0.6, 0.4, 0.5))
        result = restorer.restore(frames, keep_start=0, keep_end=5)
        restorer.close()
        for i, frame in enumerate(result):
            mean_val = frame.float().mean().item()
            assert mean_val > 10, f"Frame {i} appears black (mean={mean_val:.1f})"

    def test_no_green_line_artifacts(self) -> None:
        restorer = _make_restorer()
        frames = _make_color_frames(5, color=(0.5, 0.5, 0.5))
        result = restorer.restore(frames, keep_start=0, keep_end=5)
        restorer.close()
        for i, frame in enumerate(result):
            r, g, b = frame[0].float(), frame[1].float(), frame[2].float()
            green_excess = (g - (r + b) / 2).mean().item()
            assert green_excess < 30, f"Frame {i} has green artifact (green_excess={green_excess:.1f})"

    def test_output_resembles_input(self) -> None:
        restorer = _make_restorer()
        color = (0.6, 0.3, 0.8)
        frames = _make_color_frames(5, color=color)
        result = restorer.restore(frames, keep_start=0, keep_end=5)
        restorer.close()
        for i, frame in enumerate(result):
            r_mean = frame[0].float().mean().item() / 255.0
            g_mean = frame[1].float().mean().item() / 255.0
            b_mean = frame[2].float().mean().item() / 255.0
            assert abs(r_mean - color[0]) < 0.25, f"Frame {i} R channel too far: {r_mean:.2f} vs {color[0]}"
            assert abs(g_mean - color[1]) < 0.25, f"Frame {i} G channel too far: {g_mean:.2f} vs {color[1]}"
            assert abs(b_mean - color[2]) < 0.25, f"Frame {i} B channel too far: {b_mean:.2f} vs {color[2]}"


class TestTvaiMultipleRestores:
    def test_sequential_restores(self) -> None:
        restorer = _make_restorer(num_workers=1)
        for call_idx in range(3):
            frames = _make_color_frames(5)
            result = restorer.restore(frames, keep_start=0, keep_end=5)
            assert len(result) == 5, f"Call {call_idx}: expected 5, got {len(result)}"
        restorer.close()

    def test_keep_start_keep_end_slicing(self) -> None:
        restorer = _make_restorer()
        frames = _make_color_frames(10)
        result = restorer.restore(frames, keep_start=2, keep_end=7)
        restorer.close()
        assert len(result) == 5


class TestTvaiEdgeCases:
    def test_empty_returns_empty(self) -> None:
        restorer = _make_restorer()
        frames = torch.zeros((0, 3, 256, 256), dtype=torch.float32)
        result = restorer.restore(frames, keep_start=0, keep_end=0)
        restorer.close()
        assert result == []

    def test_keep_range_empty_returns_empty(self) -> None:
        restorer = _make_restorer()
        frames = _make_color_frames(5)
        result = restorer.restore(frames, keep_start=3, keep_end=3)
        restorer.close()
        assert result == []
