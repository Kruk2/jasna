from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import tensorrt as trt
import torch

from jasna.trt.trt_runner import TrtRunner


def _build_runner(tmp_path, *, num_outputs=1):
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"fake")

    mock_engine = MagicMock()
    mock_engine.num_io_tensors = 1 + num_outputs
    names = ["input"] + [f"output_{i}" for i in range(num_outputs)]
    mock_engine.get_tensor_name = lambda i: names[i]
    mock_engine.get_tensor_mode = lambda name: (
        trt.TensorIOMode.INPUT if name == "input" else trt.TensorIOMode.OUTPUT
    )
    mock_engine.get_tensor_dtype = lambda name: trt.DataType.FLOAT

    mock_context = MagicMock()
    mock_context.get_tensor_shape = lambda name: (1, 3, 64, 64)
    mock_context.set_input_shape = MagicMock()
    mock_context.set_tensor_address = MagicMock()
    mock_context.execute_async_v3 = MagicMock()
    mock_engine.create_execution_context.return_value = mock_context

    mock_runtime = MagicMock()
    mock_runtime.deserialize_cuda_engine.return_value = mock_engine

    with (
        patch("jasna.trt.trt_runner.trt.Logger"),
        patch("jasna.trt.trt_runner.trt.Runtime", return_value=mock_runtime),
    ):
        runner = TrtRunner(
            engine_path=engine_path,
            input_shapes={"input": (1, 3, 64, 64)},
            device=torch.device("cuda:0"),
        )
    return runner, mock_context


class TestTrtRunnerInit:
    def test_basic_init(self, tmp_path):
        runner, ctx = _build_runner(tmp_path)
        assert runner.input_names == ["input"]
        assert len(runner.output_names) == 1
        assert "output_0" in runner.outputs
        assert runner.input_dtypes["input"] == torch.float32

    def test_multiple_outputs(self, tmp_path):
        runner, ctx = _build_runner(tmp_path, num_outputs=3)
        assert len(runner.output_names) == 3
        assert len(runner.outputs) == 3

    def test_deserialization_failure_raises(self, tmp_path):
        engine_path = tmp_path / "bad.engine"
        engine_path.write_bytes(b"bad")

        mock_runtime = MagicMock()
        mock_runtime.deserialize_cuda_engine.return_value = None

        with (
            patch("jasna.trt.trt_runner.trt.Logger"),
            patch("jasna.trt.trt_runner.trt.Runtime", return_value=mock_runtime),
        ):
            with pytest.raises(RuntimeError, match="Failed to deserialize"):
                TrtRunner(engine_path=engine_path, input_shapes={"input": (1, 3, 64, 64)}, device=torch.device("cuda:0"))

    def test_context_creation_failure_raises(self, tmp_path):
        engine_path = tmp_path / "bad.engine"
        engine_path.write_bytes(b"bad")

        mock_engine = MagicMock()
        mock_engine.create_execution_context.return_value = None

        mock_runtime = MagicMock()
        mock_runtime.deserialize_cuda_engine.return_value = mock_engine

        with (
            patch("jasna.trt.trt_runner.trt.Logger"),
            patch("jasna.trt.trt_runner.trt.Runtime", return_value=mock_runtime),
        ):
            with pytest.raises(RuntimeError, match="Failed to create TensorRT execution context"):
                TrtRunner(engine_path=engine_path, input_shapes={"input": (1, 3, 64, 64)}, device=torch.device("cuda:0"))


class TestTrtRunnerInfer:
    def test_infer_sets_address_and_executes(self, tmp_path):
        runner, ctx = _build_runner(tmp_path)
        x = torch.randn(1, 3, 64, 64, device="cuda:0")
        result = runner.infer({"input": x})
        ctx.set_tensor_address.assert_called()
        ctx.execute_async_v3.assert_called_once()
        assert result is runner.outputs

    def test_infer_returns_output_dict(self, tmp_path):
        runner, ctx = _build_runner(tmp_path, num_outputs=2)
        x = torch.randn(1, 3, 64, 64, device="cuda:0")
        result = runner.infer({"input": x})
        assert "output_0" in result
        assert "output_1" in result
