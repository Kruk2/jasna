from __future__ import annotations

import torch
from pathlib import Path
import tensorrt as trt
from jasna.trt import _engine_io_names, _trt_dtype_to_torch, get_trt_logger, suppress_trt_logger_warnings


class TrtRunner:
    def __init__(
        self,
        engine_path: Path,
        input_shapes: dict[str, tuple[int, ...]] | list[tuple[int, ...]],
        device: torch.device,
    ) -> None:
        self.engine_path = engine_path
        self.device = device

        with suppress_trt_logger_warnings():
            self.runtime = trt.Runtime(get_trt_logger())
            self.engine = self.runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")
        self.input_names, self.output_names = _engine_io_names(self.engine)

        if isinstance(input_shapes, list):
            input_shapes = dict(zip(self.input_names, input_shapes))

        self.input_dtypes: dict[str, torch.dtype] = {}
        for name in self.input_names:
            self.context.set_input_shape(name, input_shapes[name])
            self.input_dtypes[name] = _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))

        dev = torch.device(self.device)
        self.outputs: dict[str, torch.Tensor] = {}
        for name in self.output_names:
            shape = tuple(int(d) for d in self.context.get_tensor_shape(name))
            dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            t = torch.empty(size=shape, dtype=dtype, device=dev)
            self.outputs[name] = t
            self.context.set_tensor_address(name, int(t.data_ptr()))

        self._stream = torch.cuda.Stream(device=dev)

    def infer(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for name, tensor in inputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        current = torch.cuda.current_stream(self.device)
        self._stream.wait_stream(current)
        self.context.execute_async_v3(self._stream.cuda_stream)
        current.wait_stream(self._stream)
        return self.outputs

