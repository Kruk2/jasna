from __future__ import annotations

import logging
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

import jasna._suppress_noise as suppress_noise
from jasna._suppress_noise import install


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
    monkeypatch.setattr(suppress_noise, "find_spec", lambda _name: None)

    install()


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
