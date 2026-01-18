import torch
from pathlib import Path
import tensorrt as trt
from jasna.tensorrt import _engine_io_names, _trt_dtype_to_torch

class TrtRunner:
    def __init__(self, 
        engine_path: Path, 
        stream: torch.cuda.Stream, 
        input_shape: tuple[int, int, int, int], 
        device: torch.device) -> None:
        self.engine_path = engine_path

        self.logger = trt.Logger(trt.ILogger.Severity(trt.Logger.ERROR))
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")
        self.input_names, self.output_names = _engine_io_names(self.engine)
        self.input_name = self.input_names[0]
        self.context.set_input_shape(self.input_name, self.input_shape)

        dev = torch.device(self.device)
        self.outputs: dict[str, torch.Tensor] = {}
        for name in self.output_names:
            shape = tuple(int(d) for d in self.context.get_tensor_shape(name))
            dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            t = torch.empty(size=shape, dtype=dtype, device=dev)
            self.outputs[name] = t
            self.context.set_tensor_address(name, int(t.data_ptr()))

    def infer(
        self,
        x: torch.Tensor,
    ) -> dict[str, "torch.Tensor"]:
        self.context.set_tensor_address(self.input_name, int(x.data_ptr()))
        self.context.execute_async_v3(self.stream.cuda_stream)
        return self.outputs