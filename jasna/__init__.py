__all__ = ["__version__"]

__version__ = "0.4.0-rc1"

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"^To copy construct from a tensor, it is recommended to use sourceTensor\.detach\(\)\.clone\(\).*",
    category=UserWarning,
    module=r"^torch_tensorrt\.dynamo\.utils$",
)

import logging

logging.getLogger("torch_tensorrt.dynamo.conversion.converter_utils").setLevel(logging.ERROR)

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


class _SuppressTorchTensorRTNoises(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "Unable to import quantization op." in msg:
            return False
        if "Unable to import quantize op." in msg:
            return False
        if "TensorRT-LLM is not installed." in msg:
            return False
        return True


_trt_filter = _SuppressTorchTensorRTNoises()
logging.getLogger("torch_tensorrt").addFilter(_trt_filter)
logging.getLogger("torch_tensorrt.dynamo").addFilter(_trt_filter)

