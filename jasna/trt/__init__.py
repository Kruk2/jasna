from __future__ import annotations

import logging
import os
from pathlib import Path

import tensorrt as trt
import torch

_TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


def _engine_io_names(engine: trt.ICudaEngine) -> tuple[list[str], list[str]]:
    input_names: list[str] = []
    output_names: list[str] = []

    if hasattr(engine, "num_io_tensors"):
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_names.append(name)
            else:
                output_names.append(name)
        return input_names, output_names

    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        if engine.binding_is_input(i):
            input_names.append(name)
        else:
            output_names.append(name)
    return input_names, output_names


def _trt_dtype_to_torch(trt_dtype: trt.DataType) -> torch.dtype:
    if trt_dtype == trt.DataType.FLOAT:
        return torch.float32
    if trt_dtype == trt.DataType.HALF:
        return torch.float16
    if trt_dtype == trt.DataType.INT8:
        return torch.int8
    if trt_dtype == trt.DataType.INT32:
        return torch.int32
    if trt_dtype == trt.DataType.BOOL:
        return torch.bool
    raise ValueError(f"Unsupported TensorRT dtype: {trt_dtype}")


from jasna.engine_paths import get_onnx_tensorrt_engine_path  # noqa: E402


def compile_onnx_to_tensorrt_engine(
    onnx_path: str | Path,
    device: torch.device,
    *,
    batch_size: int | None = None,
    fp16: bool = True,
    optimization_level: int = 5,
    workspace_gb: int,
) -> Path:
    onnx_path = Path(onnx_path)
    engine_path = get_onnx_tensorrt_engine_path(onnx_path, batch_size=batch_size, fp16=bool(fp16))

    if engine_path.exists():
        return engine_path

    if not onnx_path.exists():
        raise FileNotFoundError(str(onnx_path))
    log = logging.getLogger(__name__)
    msg = (
        f"Compiling TensorRT engine for {onnx_path} (this can take a few minutes). "
        f"Output: {engine_path}"
    )
    print(msg)
    log.info("%s", msg)

    builder = trt.Builder(_TRT_LOGGER)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, _TRT_LOGGER)

    onnx_bytes = onnx_path.read_bytes()
    if not parser.parse(onnx_bytes):
        errors = [parser.get_error(i) for i in range(parser.num_errors)]
        raise ValueError("TensorRT ONNX parse failed:\n" + "\n".join(str(e) for e in errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb) * 1024**3)
    if not (0 <= int(optimization_level) <= 5):
        raise ValueError(f"optimization_level must be in [0, 5], got {optimization_level}")
    config.builder_optimization_level = int(optimization_level)

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    needs_profile = False
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        t = network.get_input(i)
        shape = tuple(int(d) for d in t.shape)
        if any(d < 0 for d in shape):
            if batch_size is None:
                raise ValueError(
                    f"ONNX input '{t.name}' has dynamic shape {shape}; "
                    "provide batch_size to fix dynamic dimensions."
                )
            fixed = tuple(batch_size if d < 0 else d for d in shape)
            profile.set_shape(t.name, min=fixed, opt=fixed, max=fixed)
            needs_profile = True
        if fp16 and t.dtype == trt.DataType.FLOAT:
            t.dtype = trt.DataType.HALF
    if needs_profile:
        config.add_optimization_profile(profile)

    if fp16:
        for i in range(network.num_outputs):
            t = network.get_output(i)
            if t.dtype == trt.DataType.FLOAT:
                t.dtype = trt.DataType.HALF

    with torch.cuda.device(device):
        engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise ValueError("TensorRT engine build returned None")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(engine_bytes)
    return engine_path

