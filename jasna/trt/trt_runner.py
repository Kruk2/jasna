from __future__ import annotations

import torch
from pathlib import Path
import tensorrt as trt
from jasna.trt import _engine_io_names, _trt_dtype_to_torch, get_trt_logger


def _pad_batch(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    n = int(x.shape[0])
    if n >= batch_size:
        return x
    pad = x[-1:].expand(batch_size - n, *x.shape[1:])
    return torch.cat([x, pad], dim=0)


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
        # A fixed-batch engine only accepts its built batch; a partial batch is
        # padded to it (and outputs trimmed back) transparently in infer().
        engine_batch = int(self.engine.get_tensor_shape(self.input_names[0])[0])
        self.dynamic_batch = engine_batch < 0
        self._engine_batch = None if self.dynamic_batch else engine_batch
        self.outputs: dict[str, torch.Tensor] = {}
        self._cur_shapes: dict[str, tuple[int, ...]] = {}
        self._bind({name: tuple(int(d) for d in input_shapes[name]) for name in self.input_names})

    def _bind(self, input_shapes: dict[str, tuple[int, ...]]) -> None:
        """Set input shapes on the context and (re)allocate output tensors. For a
        dynamic-batch engine this runs whenever the fed batch changes."""
        for name in self.input_names:
            self.context.set_input_shape(name, input_shapes[name])
        dev = torch.device(self.device)
        self.outputs = {}
        for name in self.output_names:
            shape = tuple(int(d) for d in self.context.get_tensor_shape(name))
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
        trim = None
        if not self.dynamic_batch:
            n = int(inputs[self.input_names[0]].shape[0])
            if n < self._engine_batch:  # partial batch: pad up, trim outputs back
                inputs = {k: _pad_batch(v, self._engine_batch) for k, v in inputs.items()}
                trim = n
        shapes = {name: tuple(inputs[name].shape) for name in self.input_names}
        if shapes != self._cur_shapes:
            self._bind(shapes)
        for name, tensor in inputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        self.context.execute_async_v3(torch.cuda.current_stream(self.device).cuda_stream)
        if trim is not None:
            return {name: out[:trim] for name, out in self.outputs.items()}
        return self.outputs

