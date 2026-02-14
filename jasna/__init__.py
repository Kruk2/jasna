__all__ = ["__version__"]

__version__ = "0.5.0-alpha3"

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"^To copy construct from a tensor, it is recommended to use sourceTensor\.detach\(\)\.clone\(\).*",
    category=UserWarning,
    module=r"^torch_tensorrt\.dynamo\.utils$",
)

import logging

logging.getLogger("torch_tensorrt.dynamo.conversion.converter_utils").setLevel(logging.ERROR)
logging.getLogger("torch_tensorrt.dynamo.conversion.aten_ops_converters").setLevel(logging.ERROR)
logging.getLogger("torch.export.pt2_archive._package").setLevel(logging.ERROR)

warnings.filterwarnings(
    "ignore",
    message=r"^Unable to import quantization op\..*",
)
warnings.filterwarnings(
    "ignore",
    message=r"^Unable to import quantize op\..*",
)
warnings.filterwarnings(
    "ignore",
    message=r"^TensorRT-LLM is not installed\..*",
)
warnings.filterwarnings(
    "ignore",
    message=r"^Unable to execute the generated python source code from the graph\..*",
    category=UserWarning,
    module=r"^torch\.export\.exported_program$",
)
warnings.filterwarnings(
    "ignore",
    message=r"^Attempted to insert a get_attr Node with no underlying reference in the owning GraphModule!.*",
    category=UserWarning,
    module=r"^torch_tensorrt\.dynamo\._exporter$",
)


class _SuppressTorchTensorRTNoises(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "Unable to import quantization op." in msg:
            return False
        if "Unable to import quantize op." in msg:
            return False
        if "TensorRT-LLM is not installed." in msg:
            return False
        if "Expect archive file to be a file ending in .pt2" in msg:
            return False
        return True


_trt_filter = _SuppressTorchTensorRTNoises()
logging.getLogger("torch_tensorrt").addFilter(_trt_filter)
logging.getLogger("torch_tensorrt.dynamo").addFilter(_trt_filter)
logging.getLogger("torch_tensorrt.dynamo.conversion.aten_ops_converters").addFilter(_trt_filter)
logging.getLogger("torch.export.pt2_archive._package").addFilter(_trt_filter)

