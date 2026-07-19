from __future__ import annotations

from pathlib import Path

import tensorrt as trt
import torch

from jasna.trt import _engine_io_names, _trt_dtype_to_torch, get_trt_logger
from jasna.tensor_utils import pad_batch_with_last


class TrtRunner:
    def __init__(
        self,
        engine_path: Path,
        input_shapes: dict[str, tuple[int, ...]] | list[tuple[int, ...]],
        device: torch.device,
    ) -> None:
        self.engine_path = engine_path
        self._setup(engine_path.read_bytes(), input_shapes, device, str(engine_path))

    @classmethod
    def from_engine_bytes(
        cls,
        engine_bytes: bytes,
        input_shapes: dict[str, tuple[int, ...]] | list[tuple[int, ...]],
        device: torch.device,
        source: str = "<memory>",
    ) -> "TrtRunner":
        self = cls.__new__(cls)
        self.engine_path = None
        self._setup(engine_bytes, input_shapes, device, source)
        return self

    def _setup(
        self,
        engine_bytes: bytes,
        input_shapes: dict[str, tuple[int, ...]] | list[tuple[int, ...]],
        device: torch.device,
        source: str,
    ) -> None:
        self.device = device

        self.runtime = trt.Runtime(get_trt_logger())
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {source}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")
        self.input_names, self.output_names = _engine_io_names(self.engine)

        if isinstance(input_shapes, list):
            input_shapes = dict(zip(self.input_names, input_shapes))

        self.input_dtypes: dict[str, torch.dtype] = {
            name: _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            for name in self.input_names
        }
        input_name = self.input_names[0]
        engine_batch = int(self.engine.get_tensor_shape(input_name)[0])
        if engine_batch < 0:
            min_shape, _, max_shape = self.engine.get_tensor_profile_shape(
                input_name,
                0,
            )
            min_batch = int(min_shape[0])
            max_batch = int(max_shape[0])
            self.dynamic_batch = min_batch != max_batch
            self._engine_batch = max_batch
        else:
            self.dynamic_batch = False
            self._engine_batch = engine_batch
        self.outputs: dict[str, torch.Tensor] = {}
        self._cur_shapes: dict[str, tuple[int, ...]] = {}
        self._bind({name: tuple(int(d) for d in input_shapes[name]) for name in self.input_names})

    def _bind(self, input_shapes: dict[str, tuple[int, ...]]) -> None:
        for name in self.input_names:
            accepted = self.context.set_input_shape(name, input_shapes[name])
            if accepted is False:
                raise ValueError(
                    f"TensorRT engine rejected input shape for {name}: "
                    f"{input_shapes[name]}"
                )
        dev = torch.device(self.device)
        self.outputs = {}
        for name in self.output_names:
            shape = tuple(int(d) for d in self.context.get_tensor_shape(name))
            if any(d <= 0 for d in shape):
                raise RuntimeError(
                    f"TensorRT output shape for {name} is unresolved: {shape}"
                )
            dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            t = torch.empty(size=shape, dtype=dtype, device=dev)
            self.outputs[name] = t
            self.context.set_tensor_address(name, int(t.data_ptr()))
        self._cur_shapes = dict(input_shapes)

    def close(self) -> None:
        self.outputs.clear()
        self.context = None
        self.engine = None
        self.runtime = None

    def infer(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if set(inputs) != set(self.input_names):
            raise ValueError(
                f"TensorRT inputs must be {self.input_names}, got {list(inputs)}"
            )
        batch_sizes = {
            int(inputs[name].shape[0])
            for name in self.input_names
        }
        if len(batch_sizes) != 1:
            raise ValueError("TensorRT inputs must use the same batch size")
        trim = None
        n = batch_sizes.pop()
        if n <= 0:
            raise ValueError("TensorRT inference requires a non-empty batch")
        if not self.dynamic_batch:
            if n > self._engine_batch:
                raise ValueError(
                    f"input batch {n} exceeds fixed TensorRT batch "
                    f"{self._engine_batch}"
                )
            if n < self._engine_batch:
                inputs = {
                    name: pad_batch_with_last(
                        tensor,
                        batch_size=self._engine_batch,
                    )
                    for name, tensor in inputs.items()
                }
                trim = n
        shapes = {name: tuple(inputs[name].shape) for name in self.input_names}
        if shapes != self._cur_shapes:
            self._bind(shapes)
        for name, tensor in inputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        executed = self.context.execute_async_v3(
            torch.cuda.current_stream(self.device).cuda_stream
        )
        if executed is False:
            raise RuntimeError("TensorRT inference execution failed")
        if trim is not None:
            return {name: out[:trim] for name, out in self.outputs.items()}
        return self.outputs
