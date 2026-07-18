"""Suppress noisy warnings and log messages from torch / torch_tensorrt.

Call ``install()`` once before importing torch.  The function is idempotent.
"""
from __future__ import annotations

from importlib.util import find_spec
import logging
import warnings

_installed = False


def _add_message_drop_filter(logger_name: str, *needles: str) -> None:
    """Drop any record on ``logger_name`` whose message contains one of ``needles``.
    Filtering on the source logger (not a handler) muting works regardless of which
    handler/formatter is upstream and survives torch._logging resetting logger levels."""

    class _Drop(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            message = record.getMessage()
            return not any(n in message for n in needles)

    logging.getLogger(logger_name).addFilter(_Drop())


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

    # torch._logging resets logger levels when torch is imported, so a level
    # change here would not stick; a filter on the source logger does.
    class _SuppressDeserializedSymbolWarning(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "did not appear in the graph that was deserialized" not in record.getMessage()

    logging.getLogger("torch._export.serde.serialize").addFilter(
        _SuppressDeserializedSymbolWarning()
    )

    # torch_tensorrt logs the CUDA/TRT-LLM-plugin note at ERROR level (so the level bump
    # above can't mute it), and torch's flop_counter logs "triton not found" through
    # torch's own glog handler. Both are harmless for jasna — drop them by message.
    _add_message_drop_filter(
        "torch_tensorrt._utils", "is not currently supported for TRT-LLM plugins"
    )
    _add_message_drop_filter("torch.utils.flop_counter", "triton not found")

    # tensorrt.plugin logs an "experimental" warning through TensorRT's C++
    # logger at import time (triggered by torch_tensorrt). It bypasses the
    # logging/warnings machinery, so the only way to mute it is to filter the
    # message in trt.Logger.log before the module is imported.
    if find_spec("tensorrt") is None:
        return

    import tensorrt as trt

    _original_trt_log = trt.Logger.log

    def _trt_log_without_plugin_experimental(self, severity, msg):
        if "tensorrt.plugin module is experimental" in str(msg):
            return
        return _original_trt_log(self, severity, msg)

    trt.Logger.log = _trt_log_without_plugin_experimental
