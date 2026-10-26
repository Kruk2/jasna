import subprocess
import sys
from pathlib import Path

import onnx
import pytest
import torch
from onnx import TensorProto, helper

from jasna.trt import compile_onnx_to_tensorrt_engine, get_onnx_tensorrt_engine_path
from jasna.trt.trt_runner import TrtRunner

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _double_onnx(path: Path, shape: list) -> Path:
    """An ONNX model computing ``y = x + x`` for an input of ``shape`` (strings are dynamic axes)."""
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "x"], ["y"])], "double",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
    return path


def _compile(onnx_path: Path, **options) -> Path:
    return compile_onnx_to_tensorrt_engine(
        onnx_path, torch.device("cuda:0"), optimization_level=0, workspace_gb=1, **options)


def test_split_basicvsrpp_imports_without_tensorrt_installed():
    subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.modules['tensorrt'] = None; import jasna.restorer.basicvsrpp_sub_engines"],
        check=True,
    )


@pytest.mark.parametrize("options, name", [
    ({"fp16": True}, "model.fp16.win.engine"),
    ({"fp16": False}, "model.win.engine"),
    ({"batch_size": 4, "fp16": True}, "model.bs4.fp16.win.engine"),
    ({"batch_size": 4, "fp16": True, "dynamic_batch": True}, "model.bs1-4.fp16.win.engine"),
])
def test_engine_name_records_its_build_options(monkeypatch, options, name):
    monkeypatch.setattr("jasna.engine_paths.engine_system_suffix", lambda: ".win")
    assert get_onnx_tensorrt_engine_path(Path("dir/model.onnx"), **options) == Path("dir") / name


@pytest.mark.parametrize("batch_size", [0, -1])
def test_engine_name_needs_a_positive_batch(batch_size):
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        get_onnx_tensorrt_engine_path("model.onnx", batch_size=batch_size)


def test_an_existing_engine_is_reused_without_its_onnx(tmp_path):
    engine = get_onnx_tensorrt_engine_path(tmp_path / "model.onnx", fp16=True)
    engine.write_bytes(b"engine")
    assert compile_onnx_to_tensorrt_engine(
        tmp_path / "model.onnx", torch.device("cuda:0"), fp16=True, workspace_gb=1) == engine
    with pytest.raises(FileNotFoundError):
        compile_onnx_to_tensorrt_engine(tmp_path / "other.onnx", torch.device("cuda:0"), fp16=True, workspace_gb=1)


@needs_gpu
@pytest.mark.parametrize("shape, options, error", [
    (["batch", 3, 8, 8], {}, "provide batch_size"),
    ([4, 3, 8, 8], {"batch_size": 8}, "has fixed batch 4, requested batch_size=8"),
    ([4, 3, 8, 8], {"batch_size": 4, "dynamic_batch": True}, "requires an ONNX input with a dynamic batch axis"),
    (["batch", 3, "height", 8], {"batch_size": 4}, "non-batch dynamic shape"),
    ([1, 3, 8, 8], {"optimization_level": 10}, "optimization_level must be in"),
])
def test_compile_refuses_shapes_it_cannot_build(tmp_path, shape, options, error):
    onnx_path = _double_onnx(tmp_path / "model.onnx", shape)
    with pytest.raises(ValueError, match=error):
        compile_onnx_to_tensorrt_engine(onnx_path, torch.device("cuda:0"), workspace_gb=1, **options)


@needs_gpu
@pytest.mark.parametrize("fp16, dtype", [(True, torch.float16), (False, torch.float32)])
def test_dynamic_batch_engine_runs_every_batch_up_to_its_maximum(tmp_path, fp16, dtype):
    engine = _compile(_double_onnx(tmp_path / "model.onnx", ["batch", 3, 8, 8]),
                      batch_size=4, fp16=fp16, dynamic_batch=True)
    runner = TrtRunner(engine, {"x": (4, 3, 8, 8)}, torch.device("cuda:0"))
    assert runner.dynamic_batch and runner.input_dtypes == {"x": dtype}
    for batch in (4, 1, 3):
        x = torch.arange(batch * 3 * 64, device="cuda:0").reshape(batch, 3, 8, 8).to(dtype)
        assert torch.equal(runner.infer({"x": x})["y"], x * 2)
    torch.cuda.synchronize()


@needs_gpu
@pytest.mark.parametrize("batch_axis", [4, "batch"])
def test_fixed_batch_engine_pads_a_short_batch_and_refuses_a_longer_one(tmp_path, batch_axis):
    engine = _compile(_double_onnx(tmp_path / "model.onnx", [batch_axis, 3, 8, 8]), batch_size=4, fp16=False)
    runner = TrtRunner(engine, {"x": (4, 3, 8, 8)}, torch.device("cuda:0"))
    assert not runner.dynamic_batch
    x = torch.rand(1, 3, 8, 8, device="cuda:0")
    assert torch.equal(runner.infer({"x": x})["y"], x * 2)
    with pytest.raises(ValueError, match="exceeds fixed TensorRT batch 4"):
        runner.infer({"x": torch.rand(5, 3, 8, 8, device="cuda:0")})


@needs_gpu
def test_an_unreadable_engine_fails_to_load(tmp_path):
    engine = tmp_path / "broken.engine"
    engine.write_bytes(b"not an engine")
    with pytest.raises(RuntimeError, match="Failed to deserialize"):
        TrtRunner(engine, {"x": (1, 3, 8, 8)}, torch.device("cuda:0"))
