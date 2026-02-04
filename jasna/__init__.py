__all__ = ["__version__"]

__version__ = "0.3.0"

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"^To copy construct from a tensor, it is recommended to use sourceTensor\.detach\(\)\.clone\(\).*",
    category=UserWarning,
    module=r"^torch_tensorrt\.dynamo\.utils$",
)

import logging

logging.getLogger("torch_tensorrt.dynamo.conversion.converter_utils").setLevel(logging.ERROR)

