from __future__ import annotations

import logging

import tensorrt as trt

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
