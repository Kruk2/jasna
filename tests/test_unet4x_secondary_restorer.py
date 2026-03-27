from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

from jasna.restorer.unet4x_secondary_restorer import (
    UNET4X_INPUT_SIZE,
    UNET4X_OUTPUT_SIZE,
    UNET4X_HR_PREV_SIZE,
    Unet4xSecondaryRestorer,
    rgb_color_transfer,
    get_unet4x_engine_path,
)


def _make_fake_runner(device: torch.device, dtype: torch.dtype):
    runner = MagicMock()
    runner.outputs = {
        "hr_prev_out": torch.rand(1, UNET4X_HR_PREV_SIZE, UNET4X_HR_PREV_SIZE, 3, dtype=dtype, device=device),
        "hr_display": torch.rand(1, UNET4X_OUTPUT_SIZE, UNET4X_OUTPUT_SIZE, 3, dtype=dtype, device=device),
    }
    return runner


@pytest.fixture
def restorer(tmp_path: Path):
    device = torch.device("cpu")
    fake_engine = tmp_path / "fake_engine.trt"
    fake_engine.write_text("x")
    mock_stream = MagicMock()
    mock_stream.cuda_stream = 0
    with (
        patch("jasna.restorer.unet4x_secondary_restorer.get_unet4x_engine_path", return_value=fake_engine),
        patch("jasna.restorer.unet4x_secondary_restorer.TrtRunner", return_value=_make_fake_runner(device, torch.float32)),
        patch("torch.cuda.current_stream", return_value=mock_stream),
    ):
        r = Unet4xSecondaryRestorer(device=device, fp16=False)
        yield r


class TestUnet4xSecondaryRestorer:
    def test_protocol_attrs(self, restorer: Unet4xSecondaryRestorer):
        assert restorer.name == "unet-4x"
        assert restorer.num_workers == 1
        assert restorer.prefers_cpu_input is False

    def test_restore_basic(self, restorer: Unet4xSecondaryRestorer):
        T = 4
        frames = torch.rand(T, 3, UNET4X_INPUT_SIZE, UNET4X_INPUT_SIZE)
        result = restorer.restore(frames, keep_start=0, keep_end=T)
        assert len(result) == T
        for frame in result:
            assert frame.shape == (3, UNET4X_OUTPUT_SIZE, UNET4X_OUTPUT_SIZE)
            assert frame.dtype == torch.uint8

    def test_restore_keep_slice(self, restorer: Unet4xSecondaryRestorer):
        T = 8
        frames = torch.rand(T, 3, UNET4X_INPUT_SIZE, UNET4X_INPUT_SIZE)
        result = restorer.restore(frames, keep_start=2, keep_end=6)
        assert len(result) == 4

    def test_restore_empty(self, restorer: Unet4xSecondaryRestorer):
        frames = torch.rand(0, 3, UNET4X_INPUT_SIZE, UNET4X_INPUT_SIZE)
        result = restorer.restore(frames, keep_start=0, keep_end=0)
        assert result == []

    def test_restore_keep_out_of_range(self, restorer: Unet4xSecondaryRestorer):
        T = 4
        frames = torch.rand(T, 3, UNET4X_INPUT_SIZE, UNET4X_INPUT_SIZE)
        result = restorer.restore(frames, keep_start=5, keep_end=6)
        assert result == []

    def test_temporal_calls_per_frame(self, restorer: Unet4xSecondaryRestorer):
        T = 4
        frames = torch.rand(T, 3, UNET4X_INPUT_SIZE, UNET4X_INPUT_SIZE)
        restorer.restore(frames, keep_start=0, keep_end=T)
        # 1 bootstrap + T kept frames (no context frame when ks==0)
        assert restorer.runner.context.execute_async_v3.call_count == 1 + T

    def test_single_context_frame_for_overlap(self, restorer: Unet4xSecondaryRestorer):
        T = 8
        ks, ke = 3, 6
        kept = ke - ks
        frames = torch.rand(T, 3, UNET4X_INPUT_SIZE, UNET4X_INPUT_SIZE)
        restorer.restore(frames, keep_start=ks, keep_end=ke)
        # 1 bootstrap + 1 context frame + kept frames
        assert restorer.runner.context.execute_async_v3.call_count == 1 + 1 + kept

    def test_single_frame(self, restorer: Unet4xSecondaryRestorer):
        frames = torch.rand(1, 3, UNET4X_INPUT_SIZE, UNET4X_INPUT_SIZE)
        result = restorer.restore(frames, keep_start=0, keep_end=1)
        assert len(result) == 1
        assert result[0].shape == (3, UNET4X_OUTPUT_SIZE, UNET4X_OUTPUT_SIZE)
        assert result[0].dtype == torch.uint8


class TestRgbColorTransfer:
    def test_output_shape_and_range(self):
        ai = torch.rand(1, 64, 64, 3)
        ref = torch.rand(1, 64, 64, 3)
        result = rgb_color_transfer(ai, ref)
        assert result.shape == ai.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_identity_when_blend_zero(self):
        ai = torch.rand(1, 32, 32, 3).clamp(0.01, 0.99)
        ref = torch.rand(1, 32, 32, 3)
        result = rgb_color_transfer(ai, ref, blend=0.0)
        torch.testing.assert_close(result, ai, atol=1e-4, rtol=1e-4)

    def test_out_parameter(self):
        ai = torch.rand(1, 16, 16, 3)
        ref = torch.rand(1, 16, 16, 3)
        buf = torch.empty_like(ai)
        ret = rgb_color_transfer(ai, ref, out=buf)
        assert ret is buf
        assert buf.min() >= 0.0
        assert buf.max() <= 1.0


class TestGetUnet4xEnginePath:
    def test_returns_path(self):
        p = get_unet4x_engine_path(Path("model_weights/unet-4x.onnx"), fp16=True)
        assert isinstance(p, Path)
        assert "unet-4x" in str(p)

    def test_no_batch_size_in_path(self):
        p = get_unet4x_engine_path(Path("model_weights/unet-4x.onnx"), fp16=True)
        assert ".bs" not in str(p)
