from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

from jasna.restorer.unet4x_secondary_restorer import (
    UNET4X_INPUT_SIZE,
    UNET4X_OUTPUT_SIZE,
    Unet4xSecondaryRestorer,
)


def _make_fake_runner(device: torch.device, dtype: torch.dtype):
    runner = MagicMock()
    runner.outputs = {
        "out": torch.rand(1, UNET4X_OUTPUT_SIZE, UNET4X_OUTPUT_SIZE, 3, dtype=dtype, device=device),
    }
    runner.input_dtypes = {"lr_prev": dtype, "lr_curr": dtype, "hr_prev": dtype}
    return runner


@pytest.fixture
def plaintext_model(tmp_path, monkeypatch):
    model = tmp_path / "unet-4x.onnx"
    model.touch()
    monkeypatch.setattr("jasna.restorer.unet4x_secondary_restorer.UNET4X_ONNX_PATH", model)


@pytest.fixture
def restorer(tmp_path: Path, plaintext_model):
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


@pytest.mark.parametrize("frames, keep_start, keep_end, kept, runs", [
    (4, 0, 4, 4, 1 + 4),  # a bootstrap run, then one per kept frame
    (8, 3, 6, 3, 1 + 1 + 3),  # plus one run on the frame before the kept range
    (1, 0, 1, 1, 1 + 1),
    (4, 5, 6, 0, 0),
    (0, 0, 0, 0, 0),
])
def test_restore_returns_the_kept_frames_upscaled(restorer, frames, keep_start, keep_end, kept, runs):
    clip = torch.rand(frames, 3, UNET4X_INPUT_SIZE, UNET4X_INPUT_SIZE)
    result = restorer.restore(clip, keep_start=keep_start, keep_end=keep_end)
    assert [(frame.shape, frame.dtype) for frame in result] == [
        ((3, UNET4X_OUTPUT_SIZE, UNET4X_OUTPUT_SIZE), torch.uint8)] * kept
    assert restorer.runner.context.execute_async_v3.call_count == runs


class TestEngineDtypeMismatch:
    def test_buffers_match_half_engine_bindings_when_fp16_off(self, tmp_path: Path, plaintext_model):
        device = torch.device("cpu")
        fake_engine = tmp_path / "fake_engine.trt"
        fake_engine.write_text("x")
        mock_stream = MagicMock()
        mock_stream.cuda_stream = 0
        with (
            patch("jasna.restorer.unet4x_secondary_restorer.get_unet4x_engine_path", return_value=fake_engine),
            patch("jasna.restorer.unet4x_secondary_restorer.TrtRunner", return_value=_make_fake_runner(device, torch.float16)),
            patch("torch.cuda.current_stream", return_value=mock_stream),
        ):
            r = Unet4xSecondaryRestorer(device=device, fp16=False)

        assert r._g_lr_prev.dtype == torch.float16
        assert r._g_lr_curr.dtype == torch.float16
        assert r._g_hr_prev.dtype == torch.float16

        frames = torch.rand(2, 3, UNET4X_INPUT_SIZE, UNET4X_INPUT_SIZE)
        result = r.restore(frames, keep_start=0, keep_end=2)
        assert len(result) == 2
        assert all(f.dtype == torch.uint8 for f in result)
