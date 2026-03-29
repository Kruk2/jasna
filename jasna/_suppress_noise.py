"""Suppress noisy warnings and log messages from torch / torch_tensorrt.

Call ``install()`` once before importing torch.  The function is idempotent.
"""
from __future__ import annotations

import logging
import warnings

_installed = False


def install() -> None:
    global _installed
    if _installed:
        return
    _installed = True

    warnings.filterwarnings(
        "ignore",
        message=r"^To copy construct from a tensor, it is recommended to use sourceTensor\.detach\(\)\.clone\(\).*",
        category=UserWarning,
        module=r"^torch_tensorrt\.dynamo\.utils$",
    )
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

    logging.getLogger("torch.export.pt2_archive._package").setLevel(logging.ERROR)
    for _name in (
        "torch_tensorrt",
        "torch_tensorrt._utils",
        "torch_tensorrt [TensorRT Conversion Context]",
    ):
        logging.getLogger(_name).setLevel(logging.ERROR)

    class _SuppressRedirectsWarning(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "Redirects are currently not supported" not in record.getMessage()

    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").addFilter(
        _SuppressRedirectsWarning()
    )
