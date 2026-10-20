"""Suppress noisy warnings and log messages from torch / torch_tensorrt.

Call ``install()`` once before importing torch.  The function is idempotent.
"""
from __future__ import annotations

import logging
import sys
from types import ModuleType
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


def _tensorrt_proxy_getattr(self, name: str):
    return getattr(self._target_module, name)


class _QuietTensorRTProxyFinder:
    """torch_tensorrt replaces ``sys.modules["tensorrt"]`` with a proxy whose ``__getattr__``
    prints every missing attribute, and Inductor probes it while hashing compiled graphs.
    At the first import after the proxy appears, swap in a ``__getattr__`` that only raises.
    Filter TensorRT's logger before ``tensorrt.plugin`` imports, without loading TensorRT."""

    def __init__(self):
        self._logger_class = None
        self._proxy_patched = False

    def patch_loaded_logger(self):
        module = sys.modules.get("tensorrt")
        if module is None:
            return
        target = vars(module).get("_target_module", module)
        logger_class = vars(target).get("Logger")
        if logger_class is None or logger_class is self._logger_class:
            return
        original_log = logger_class.log

        def _trt_log_without_plugin_experimental(self, severity, msg):
            if "tensorrt.plugin module is experimental" in str(msg):
                return
            return original_log(self, severity, msg)

        logger_class.log = _trt_log_without_plugin_experimental
        self._logger_class = logger_class

    def find_spec(self, fullname, path, target=None):
        proxy_class = type(sys.modules.get("tensorrt"))
        if issubclass(proxy_class, ModuleType) and proxy_class is not ModuleType:
            proxy_class.__getattr__ = _tensorrt_proxy_getattr
            self._proxy_patched = True
        if fullname == "tensorrt.plugin":
            self.patch_loaded_logger()
        if self._proxy_patched and self._logger_class is not None:
            sys.meta_path = [finder for finder in sys.meta_path if finder is not self]
        return None


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
    warnings.filterwarnings("ignore", message=r".*isinstance\(treespec, LeafSpec\).*", category=FutureWarning)
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

    _add_message_drop_filter(
        "torch.distributed.elastic.multiprocessing.redirects", "Redirects are currently not supported"
    )
    _add_message_drop_filter(
        "torch._export.serde.serialize", "did not appear in the graph that was deserialized"
    )

    # torch_tensorrt logs the CUDA/TRT-LLM-plugin note at ERROR level (so the level bump
    # above can't mute it), and torch's flop_counter logs "triton not found" through
    # torch's own glog handler. Both are harmless for jasna — drop them by message.
    _add_message_drop_filter(
        "torch_tensorrt._utils", "is not currently supported for TRT-LLM plugins"
    )
    _add_message_drop_filter("torch.utils.flop_counter", "triton not found")
    # max-autotune tries Triton GEMM tiles sized for datacenter shared memory; consumer
    # GPUs reject them, and Inductor logs each skipped choice at ERROR.
    _add_message_drop_filter("torch._inductor.select_algorithm", "out of resource")

    # tensorrt.plugin logs an "experimental" warning through TensorRT's C++
    # logger at import time (triggered by torch_tensorrt). It bypasses the
    # logging/warnings machinery, so the only way to mute it is to filter the
    # message in trt.Logger.log before the module is imported.
    finder = _QuietTensorRTProxyFinder()
    finder.patch_loaded_logger()
    sys.meta_path.insert(0, finder)
