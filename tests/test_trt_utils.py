import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import tensorrt as trt
import torch

from jasna.trt import (
    _engine_io_names,
    _TRT_LOGGER,
    _trt_dtype_to_torch,
    get_onnx_tensorrt_engine_path,
    get_trt_logger,
    suppress_trt_logger_warnings,
    compile_onnx_to_tensorrt_engine,
)


class TestGetTrtLogger:
    def test_returns_fallback_when_torch_tensorrt_not_imported(self):
        with patch.dict("sys.modules", {"torch_tensorrt": None}):
            assert get_trt_logger() is _TRT_LOGGER

    def test_returns_torch_tensorrt_logger_when_available(self):
        sentinel = MagicMock()
        mock_torchtrt = MagicMock()
        mock_torchtrt.logging.TRT_LOGGER = sentinel
        with patch.dict("sys.modules", {"torch_tensorrt": mock_torchtrt}):
            assert get_trt_logger() is sentinel


class TestSuppressTrtLoggerWarnings:
    def _run_with_real_stderr(self, fn):
        """Run fn with real stderr so suppress_trt_logger_warnings can use fileno()."""
        real_stderr = sys.__stderr__
        old = sys.stderr
        sys.stderr = real_stderr
        try:
            return fn()
        finally:
            sys.stderr = old

    def test_filters_logger_mismatch_warning(self):
        leaked = []

        def go():
            saved_fd = os.dup(2)
            r, w = os.pipe()
            os.dup2(w, 2)
            os.close(w)
            try:
                with suppress_trt_logger_warnings():
                    os.write(2, b"WARNING The logger passed into createInferRuntime differs from one already registered\n")
            finally:
                os.dup2(saved_fd, 2)
                os.close(saved_fd)
                while True:
                    chunk = os.read(r, 4096)
                    if not chunk:
                        break
                    leaked.append(chunk)
                os.close(r)

        self._run_with_real_stderr(go)
        assert b"createInferRuntime" not in b"".join(leaked)

    def test_preserves_errors(self):
        leaked = []

        def go():
            saved_fd = os.dup(2)
            r, w = os.pipe()
            os.dup2(w, 2)
            os.close(w)
            try:
                with suppress_trt_logger_warnings():
                    os.write(2, b"WARNING The logger passed into createInferRuntime differs from one already registered\n")
                    os.write(2, b"REAL ERROR happened\n")
            finally:
                os.dup2(saved_fd, 2)
                os.close(saved_fd)
                while True:
                    chunk = os.read(r, 4096)
                    if not chunk:
                        break
                    leaked.append(chunk)
                os.close(r)

        self._run_with_real_stderr(go)
        output = b"".join(leaked)
        assert b"REAL ERROR happened" in output
        assert b"createInferRuntime" not in output

    def test_noop_when_fileno_unavailable(self):
        with suppress_trt_logger_warnings():
            pass


class TestTrtDtypeToTorch:
    def test_float(self):
        assert _trt_dtype_to_torch(trt.DataType.FLOAT) == torch.float32

    def test_half(self):
        assert _trt_dtype_to_torch(trt.DataType.HALF) == torch.float16

    def test_int8(self):
        assert _trt_dtype_to_torch(trt.DataType.INT8) == torch.int8

    def test_int32(self):
        assert _trt_dtype_to_torch(trt.DataType.INT32) == torch.int32

    def test_bool(self):
        assert _trt_dtype_to_torch(trt.DataType.BOOL) == torch.bool

    def test_unsupported_raises(self):
        with pytest.raises(ValueError, match="Unsupported TensorRT dtype"):
            _trt_dtype_to_torch(trt.DataType.INT64)


class TestGetOnnxTensorrtEnginePath:
    def test_basic_fp16_windows(self):
        with patch.object(os, "name", "nt"):
            result = get_onnx_tensorrt_engine_path("model.onnx", fp16=True)
            assert result == Path("model.fp16.win.engine")

    def test_basic_fp32_windows(self):
        with patch.object(os, "name", "nt"):
            result = get_onnx_tensorrt_engine_path("model.onnx", fp16=False)
            assert result == Path("model.win.engine")

    def test_with_batch_size(self):
        with patch.object(os, "name", "nt"):
            result = get_onnx_tensorrt_engine_path("model.onnx", batch_size=4, fp16=True)
            assert result == Path("model.bs4.fp16.win.engine")

    def test_batch_size_zero_raises(self):
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            get_onnx_tensorrt_engine_path("model.onnx", batch_size=0)

    def test_batch_size_negative_raises(self):
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            get_onnx_tensorrt_engine_path("model.onnx", batch_size=-1)

    def test_path_object_input(self):
        with patch.object(os, "name", "nt"):
            result = get_onnx_tensorrt_engine_path(Path("dir/model.onnx"), fp16=True)
            assert result == Path("dir/model.fp16.win.engine")


class TestEngineIoNames:
    def test_num_io_tensors_api(self):
        engine = MagicMock()
        engine.num_io_tensors = 3
        engine.get_tensor_name = lambda i: ["input", "output1", "output2"][i]
        engine.get_tensor_mode = lambda name: (
            trt.TensorIOMode.INPUT if name == "input" else trt.TensorIOMode.OUTPUT
        )

        inputs, outputs = _engine_io_names(engine)
        assert inputs == ["input"]
        assert outputs == ["output1", "output2"]

    def test_legacy_bindings_api(self):
        engine = MagicMock(spec=[])
        engine.num_bindings = 2
        engine.get_binding_name = lambda i: ["input", "output"][i]
        engine.binding_is_input = lambda i: i == 0

        inputs, outputs = _engine_io_names(engine)
        assert inputs == ["input"]
        assert outputs == ["output"]


def _setup_trt_mocks(tmp_path, *, num_inputs=1, num_outputs=1, dynamic_shape=False, parse_ok=True, build_result=b"engine_bytes"):
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake onnx")

    engine_path = get_onnx_tensorrt_engine_path(onnx_path, fp16=True)
    if engine_path.exists():
        engine_path.unlink()

    mock_input = MagicMock()
    mock_input.name = "input"
    mock_input.shape = (-1, 3, 64, 64) if dynamic_shape else (1, 3, 64, 64)
    mock_input.dtype = trt.DataType.FLOAT

    mock_output = MagicMock()
    mock_output.name = "output"
    mock_output.dtype = trt.DataType.FLOAT

    mock_network = MagicMock()
    mock_network.num_inputs = num_inputs
    mock_network.num_outputs = num_outputs
    mock_network.get_input = MagicMock(return_value=mock_input)
    mock_network.get_output = MagicMock(return_value=mock_output)

    mock_parser = MagicMock()
    mock_parser.parse.return_value = parse_ok
    mock_parser.num_errors = 0 if parse_ok else 1
    mock_parser.get_error = lambda i: "parse error"

    mock_config = MagicMock()
    mock_builder = MagicMock()
    mock_builder.create_network.return_value = mock_network
    mock_builder.create_builder_config.return_value = mock_config
    mock_builder.build_serialized_network.return_value = build_result

    return onnx_path, engine_path, mock_builder, mock_parser, mock_config, mock_network


class TestCompileOnnxToTensorrtEngine:
    def test_returns_existing_engine(self, tmp_path):
        onnx_path = tmp_path / "model.onnx"
        onnx_path.touch()
        engine_path = get_onnx_tensorrt_engine_path(onnx_path, fp16=True)
        engine_path.touch()

        result = compile_onnx_to_tensorrt_engine(onnx_path, torch.device("cuda:0"), fp16=True, workspace_gb=20)
        assert result == engine_path

    def test_missing_onnx_raises(self, tmp_path):
        onnx_path = tmp_path / "nonexistent.onnx"
        with pytest.raises(FileNotFoundError):
            compile_onnx_to_tensorrt_engine(onnx_path, torch.device("cuda:0"), fp16=True, workspace_gb=20)

    def test_successful_compilation(self, tmp_path):
        onnx_path, engine_path, mock_builder, mock_parser, mock_config, _ = _setup_trt_mocks(tmp_path)

        with (
            patch("jasna.trt.trt.Logger"),
            patch("jasna.trt.trt.Builder", return_value=mock_builder),
            patch("jasna.trt.trt.OnnxParser", return_value=mock_parser),
            patch("jasna.trt.torch.cuda.device", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))),
        ):
            result = compile_onnx_to_tensorrt_engine(onnx_path, torch.device("cuda:0"), fp16=True, workspace_gb=20)

        assert result == engine_path
        assert engine_path.exists()
        assert engine_path.read_bytes() == b"engine_bytes"
        mock_config.set_flag.assert_called_once_with(trt.BuilderFlag.FP16)

    def test_parse_failure_raises(self, tmp_path):
        onnx_path, _, mock_builder, mock_parser, _, _ = _setup_trt_mocks(tmp_path, parse_ok=False)

        with (
            patch("jasna.trt.trt.Logger"),
            patch("jasna.trt.trt.Builder", return_value=mock_builder),
            patch("jasna.trt.trt.OnnxParser", return_value=mock_parser),
        ):
            with pytest.raises(ValueError, match="ONNX parse failed"):
                compile_onnx_to_tensorrt_engine(onnx_path, torch.device("cuda:0"), fp16=True, workspace_gb=20)

    def test_dynamic_shape_raises(self, tmp_path):
        onnx_path, _, mock_builder, mock_parser, _, _ = _setup_trt_mocks(tmp_path, dynamic_shape=True)

        with (
            patch("jasna.trt.trt.Logger"),
            patch("jasna.trt.trt.Builder", return_value=mock_builder),
            patch("jasna.trt.trt.OnnxParser", return_value=mock_parser),
        ):
            with pytest.raises(ValueError, match="dynamic shape"):
                compile_onnx_to_tensorrt_engine(onnx_path, torch.device("cuda:0"), fp16=True, workspace_gb=20)

    def test_build_returns_none_raises(self, tmp_path):
        onnx_path, _, mock_builder, mock_parser, _, _ = _setup_trt_mocks(tmp_path, build_result=None)

        with (
            patch("jasna.trt.trt.Logger"),
            patch("jasna.trt.trt.Builder", return_value=mock_builder),
            patch("jasna.trt.trt.OnnxParser", return_value=mock_parser),
            patch("jasna.trt.torch.cuda.device", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))),
        ):
            with pytest.raises(ValueError, match="engine build returned None"):
                compile_onnx_to_tensorrt_engine(onnx_path, torch.device("cuda:0"), fp16=True, workspace_gb=20)

    def test_invalid_optimization_level_raises(self, tmp_path):
        onnx_path, _, mock_builder, mock_parser, _, _ = _setup_trt_mocks(tmp_path)

        with (
            patch("jasna.trt.trt.Logger"),
            patch("jasna.trt.trt.Builder", return_value=mock_builder),
            patch("jasna.trt.trt.OnnxParser", return_value=mock_parser),
        ):
            with pytest.raises(ValueError, match="optimization_level must be in"):
                compile_onnx_to_tensorrt_engine(
                    onnx_path, torch.device("cuda:0"), fp16=True, optimization_level=10, workspace_gb=20
                )

    def test_fp32_mode_no_fp16_flag(self, tmp_path):
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"fake")
        engine_path = get_onnx_tensorrt_engine_path(onnx_path, fp16=False)
        if engine_path.exists():
            engine_path.unlink()

        mock_network = MagicMock()
        mock_network.num_inputs = 0
        mock_network.num_outputs = 0
        mock_config = MagicMock()
        mock_parser = MagicMock()
        mock_parser.parse.return_value = True
        mock_builder = MagicMock()
        mock_builder.create_network.return_value = mock_network
        mock_builder.create_builder_config.return_value = mock_config
        mock_builder.build_serialized_network.return_value = b"fp32_engine"

        with (
            patch("jasna.trt.trt.Logger"),
            patch("jasna.trt.trt.Builder", return_value=mock_builder),
            patch("jasna.trt.trt.OnnxParser", return_value=mock_parser),
            patch("jasna.trt.torch.cuda.device", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))),
        ):
            result = compile_onnx_to_tensorrt_engine(onnx_path, torch.device("cuda:0"), fp16=False, workspace_gb=20)

        mock_config.set_flag.assert_not_called()
        assert result == engine_path
