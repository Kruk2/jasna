from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

from jasna.restorer.unet4x_secondary_restorer import (
    UNET4X_BATCH_SIZE,
    UNET4X_INPUT_SIZE,
    UNET4X_OUTPUT_SIZE,
    Unet4xSecondaryRestorer,
    get_unet4x_engine_path,
)


def _make_fake_runner(device: torch.device, dtype: torch.dtype):
    runner = MagicMock()
    runner.input_names = ["frames_stack", "hr_init", "lr_init"]
    runner.output_names = ["all_color_outputs", "hr_final"]
    runner.input_dtypes = {
        "frames_stack": dtype,
        "hr_init": dtype,
        "lr_init": dtype,
    }

    def fake_infer(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        bs = inputs["frames_stack"].shape[0]
        return {
            "all_color_outputs": torch.rand(bs, 1, UNET4X_OUTPUT_SIZE, UNET4X_OUTPUT_SIZE, 3, dtype=dtype, device=device),
            "hr_final": torch.rand(1, UNET4X_OUTPUT_SIZE, UNET4X_OUTPUT_SIZE, 3, dtype=dtype, device=device),
        }

    runner.infer = MagicMock(side_effect=fake_infer)
    return runner


@pytest.fixture
def restorer(tmp_path: Path):
    device = torch.device("cpu")
    fake_engine = tmp_path / "fake_engine.trt"
    fake_engine.write_text("x")
    with (
        patch("jasna.restorer.unet4x_secondary_restorer.get_unet4x_engine_path", return_value=fake_engine),
        patch("jasna.restorer.unet4x_secondary_restorer.TrtRunner", return_value=_make_fake_runner(device, torch.float32)),
    ):
        r = Unet4xSecondaryRestorer(device=device, fp16=False)
    return r


class TestUnet4xSecondaryRestorer:
    def test_protocol_attrs(self, restorer: Unet4xSecondaryRestorer):
        assert restorer.name == "unet-4x"
        assert restorer.num_workers == 1
        assert restorer.prefers_cpu_input is False

    def test_restore_exact_batch(self, restorer: Unet4xSecondaryRestorer):
        T = UNET4X_BATCH_SIZE
        frames = torch.rand(T, 3, UNET4X_INPUT_SIZE, UNET4X_INPUT_SIZE)
        result = restorer.restore(frames, keep_start=0, keep_end=T)
        assert len(result) == T
        for frame in result:
            assert frame.shape == (3, UNET4X_OUTPUT_SIZE, UNET4X_OUTPUT_SIZE)
            assert frame.dtype == torch.uint8

    def test_restore_non_multiple_batch(self, restorer: Unet4xSecondaryRestorer):
        T = 6
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

    def test_temporal_state_updated_across_batches(self, restorer: Unet4xSecondaryRestorer):
        T = 8
        frames = torch.rand(T, 3, UNET4X_INPUT_SIZE, UNET4X_INPUT_SIZE)
        restorer.restore(frames, keep_start=0, keep_end=T)
        assert restorer.runner.infer.call_count == 2
        second_call_inputs = restorer.runner.infer.call_args_list[1][0][0]
        assert "hr_init" in second_call_inputs
        assert "lr_init" in second_call_inputs

    def test_single_frame(self, restorer: Unet4xSecondaryRestorer):
        frames = torch.rand(1, 3, UNET4X_INPUT_SIZE, UNET4X_INPUT_SIZE)
        result = restorer.restore(frames, keep_start=0, keep_end=1)
        assert len(result) == 1
        assert result[0].shape == (3, UNET4X_OUTPUT_SIZE, UNET4X_OUTPUT_SIZE)
        assert result[0].dtype == torch.uint8


class TestGetUnet4xEnginePath:
    def test_returns_path(self):
        p = get_unet4x_engine_path(Path("model_weights/unet-4x.onnx"), fp16=True)
        assert isinstance(p, Path)
        assert "unet-4x" in str(p)
