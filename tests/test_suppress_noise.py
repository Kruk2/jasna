from __future__ import annotations

import logging
import subprocess
import sys
import warnings
from importlib.util import find_spec
from pathlib import Path

import pytest

import jasna._suppress_noise as suppress_noise
from jasna._suppress_noise import install


_HIPBLAS_FALLBACK_WARNING = (
    "bgemm_internal_cublaslt error: HIPBLAS_STATUS_NOT_SUPPORTED when calling hipblasLtMatmul "
    "with transpose_mat1 t transpose_mat2 n m 461 n 1383 k 512 lda 512 ldb 512 ldc 461 "
    "abType 0 cType 0 computeType 6 scaleType 0. "
    "Will attempt to recover by calling cublas instead. "
    "(Triggered internally at /w/rockrel/pytorch/aten/src/ATen/hip/HIPBlas.cpp:575.)"
)


@pytest.mark.parametrize(("message", "muted"), [
    (_HIPBLAS_FALLBACK_WARNING, True),
    (_HIPBLAS_FALLBACK_WARNING.replace("HIPBLAS_STATUS_NOT_SUPPORTED", "HIPBLAS_STATUS_ALLOC_FAILED"), False),
    (_HIPBLAS_FALLBACK_WARNING.replace("HIPBLAS_STATUS", "CUBLAS_STATUS").replace("hipblasLtMatmul", "cublasLtMatmul"), False),
    (_HIPBLAS_FALLBACK_WARNING.split("Will attempt to recover")[0], False),
    ("an unrelated AMD warning", False),
], ids=["amd-fallback", "allocation-failure", "nvidia-fallback", "no-fallback", "unrelated"])
def test_only_unsupported_hipblas_fallback_warning_is_muted(monkeypatch, message, muted) -> None:
    monkeypatch.setattr(suppress_noise, "_installed", False)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        install()
        warnings.warn_explicit(message, UserWarning, "torch/functional.py", 373, module="torch.functional")
    assert len(caught) == (0 if muted else 1)


def test_deserialized_symbol_warning_is_filtered() -> None:
    install()
    logger = logging.getLogger("torch._export.serde.serialize")

    suppressed = logger.makeRecord(
        logger.name,
        logging.WARNING,
        "f",
        1,
        "Symbol %s did not appear in the graph that was deserialized",
        ("u0",),
        None,
    )
    kept = logger.makeRecord(
        logger.name, logging.WARNING, "f", 1, "a real warning", (), None
    )

    assert not logger.filter(suppressed)
    assert logger.filter(kept)


def test_torch_tensorrt_cuda_plugin_error_is_filtered() -> None:
    install()
    logger = logging.getLogger("torch_tensorrt._utils")

    suppressed = logger.makeRecord(
        logger.name,
        logging.ERROR,
        "f",
        1,
        "CUDA 13 is not currently supported for TRT-LLM plugins. "
        "Please install pytorch with CUDA 12.x support",
        (),
        None,
    )
    kept = logger.makeRecord(
        logger.name, logging.ERROR, "f", 1, "a genuine conversion failure", (), None
    )

    assert not logger.filter(suppressed)
    assert logger.filter(kept)


def test_flop_counter_triton_warning_is_filtered() -> None:
    install()
    logger = logging.getLogger("torch.utils.flop_counter")

    suppressed = logger.makeRecord(
        logger.name,
        logging.WARNING,
        "f",
        1,
        "triton not found; flop counting will not work for triton kernels",
        (),
        None,
    )
    kept = logger.makeRecord(
        logger.name, logging.WARNING, "f", 1, "some other flop message", (), None
    )

    assert not logger.filter(suppressed)
    assert logger.filter(kept)


def test_tensorrt_plugin_experimental_warning_is_muted(monkeypatch) -> None:
    trt = pytest.importorskip("tensorrt")
    # TensorRT's C++ logger writes to the original stderr handle and bypasses
    # Python capture, so assert the patch's behaviour directly: the experimental
    # message must not reach the wrapped logger, while other messages do.
    logged: list[str] = []

    def spy(self, severity, msg):
        logged.append(msg)

    monkeypatch.setattr(trt.Logger, "log", spy)
    monkeypatch.setattr(suppress_noise, "_installed", False)
    install()

    logger = trt.Logger()
    logger.log(
        trt.Logger.WARNING,
        "Functionality provided through tensorrt.plugin module is experimental.",
    )
    logger.log(trt.Logger.WARNING, "an-unrelated-trt-warning")

    assert logged == ["an-unrelated-trt-warning"]


def test_install_succeeds_without_tensorrt(monkeypatch) -> None:
    monkeypatch.setattr(suppress_noise, "_installed", False)
    monkeypatch.delitem(sys.modules, "tensorrt", raising=False)

    class RejectTensorRTFinder:
        def find_spec(self, fullname, path, target=None):
            if fullname == "tensorrt" or fullname.startswith("tensorrt."):
                raise AssertionError("install eagerly probed TensorRT")
            return None

    monkeypatch.setattr(sys, "meta_path", [RejectTensorRTFinder(), *sys.meta_path])

    install()
    assert "tensorrt" not in sys.modules


def test_duplicate_install_does_not_add_filters_or_finders(monkeypatch) -> None:
    monkeypatch.setattr(suppress_noise, "_installed", False)
    monkeypatch.delitem(sys.modules, "tensorrt", raising=False)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))

    install()
    finders = tuple(sys.meta_path)
    filters = tuple(logging.getLogger("torch._inductor.select_algorithm").filters)
    install()

    assert tuple(sys.meta_path) == finders
    assert tuple(logging.getLogger("torch._inductor.select_algorithm").filters) == filters
    assert "tensorrt" not in sys.modules


@pytest.mark.parametrize("proxy_before_plugin", [False, True])
def test_real_tensorrt_plugin_logger_is_filtered_lazily(proxy_before_plugin) -> None:
    if find_spec("tensorrt") is None:
        pytest.skip("TensorRT is not installed")
    script = (
        "import sys\n"
        "from importlib.util import find_spec\n"
        "from types import ModuleType\n"
        "from jasna._suppress_noise import install\n"
        "assert 'tensorrt' not in sys.modules\n"
        "install()\n"
        "assert 'tensorrt' not in sys.modules\n"
        "import tensorrt as trt\n"
        "assert 'tensorrt.plugin' not in sys.modules\n"
        "logged = []\n"
        "def spy(self, severity, msg):\n"
        "    logged.append(msg)\n"
        "trt.Logger.log = spy\n"
        "class Proxy(ModuleType):\n"
        "    def __init__(self):\n"
        "        super().__init__('tensorrt')\n"
        "        self._target_module = trt\n"
        "        self.__path__ = trt.__path__\n"
        "        self.__spec__ = trt.__spec__\n"
        "    def __getattr__(self, name):\n"
        "        print('NOISY_PROXY_ATTRIBUTE', name)\n"
        "        return getattr(self._target_module, name)\n"
        f"if {proxy_before_plugin!r}:\n"
        "    sys.modules['tensorrt'] = Proxy()\n"
        "    assert find_spec('__jasna_missing_import_for_proxy_filter__') is None\n"
        "    assert not hasattr(sys.modules['tensorrt'], 'missing_test_attribute')\n"
        "import tensorrt.plugin\n"
        "logger = trt.Logger()\n"
        "logger.log(trt.Logger.WARNING, 'Functionality provided through tensorrt.plugin module is experimental.')\n"
        "logger.log(trt.Logger.WARNING, 'an-unrelated-trt-warning')\n"
        "assert logged == ['an-unrelated-trt-warning'], logged\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=Path(__file__).parents[1], capture_output=True, text=True, check=True
    )
    assert "NOISY_PROXY_ATTRIBUTE" not in result.stdout


def test_autotune_choice_that_exceeds_shared_memory_is_filtered() -> None:
    install()
    logger = logging.getLogger("torch._inductor.select_algorithm")

    suppressed = logger.makeRecord(
        logger.name,
        logging.ERROR,
        "f",
        1,
        "Runtime error during autotuning: \n%s. \nIgnoring this choice.",
        ("No valid triton configs. OutOfMemoryError: out of resource: triton_mm Required: 131072 Hardware limit:101376",),
        None,
    )
    kept = logger.makeRecord(
        logger.name, logging.ERROR, "f", 1, "Runtime error during autotuning: \n%s. \nIgnoring this choice.", ("illegal memory access",), None
    )

    assert not logger.filter(suppressed)
    assert logger.filter(kept)


def test_tensorrt_proxy_misses_attributes_without_printing() -> None:
    if find_spec("torch_tensorrt") is None:
        pytest.skip("Torch-TensorRT is not installed")
    script = (
        "import sys\n"
        "from unittest.mock import patch\n"
        "import torch\n"
        "from jasna._suppress_noise import install\n"
        "install()\n"
        "with (\n"
        "    patch('torch.cuda.get_device_capability', return_value=(0, 0)),\n"
        "    patch('torch.cuda.current_device', return_value=0),\n"
        "):\n"
        "    import torch_tensorrt\n"
        "assert not hasattr(sys.modules['tensorrt'], 'torch')\n"
        "assert sys.modules['tensorrt'].Logger is not None\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=Path(__file__).parents[1], capture_output=True, text=True, check=True
    )
    assert "AttributeError" not in result.stdout


@pytest.mark.skipif(not Path("assets/test_clip1_1080p.mp4").exists(), reason="needs the test clip")
def test_threaded_decoders_stop_early_after_install() -> None:
    script = (
        "import av\n"
        "from jasna._suppress_noise import install\n"
        "install()\n"
        "for _ in range(5):\n"
        "    with av.open('assets/test_clip1_1080p.mp4') as container:\n"
        "        stream = container.streams.video[0]\n"
        "        stream.codec_context.thread_type = 'AUTO'\n"
        "        for index, _frame in enumerate(container.decode(stream)):\n"
        "            if index == 5:\n"
        "                break\n"
    )
    subprocess.run(
        [sys.executable, "-c", script], cwd=Path(__file__).parents[1], capture_output=True, check=True, timeout=60
    )
