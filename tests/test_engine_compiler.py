from __future__ import annotations

import logging
import subprocess
import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jasna.engine_compiler import (
    EngineCompilationRequest,
    _detection_engine_exists,
    _subprocess_compile,
    _unet4x_engine_exists,
    ensure_engines_compiled,
)


def _mock_proc(lines: list[str], returncode: int = 0) -> MagicMock:
    stdout = MagicMock()
    stdout.__iter__ = MagicMock(return_value=iter(lines))
    proc = MagicMock()
    proc.stdout = stdout
    proc.wait.return_value = returncode
    return proc


def test_request_json_roundtrip() -> None:
    req = EngineCompilationRequest(
        device="cuda:0", fp16=True, basicvsrpp=True,
        basicvsrpp_model_path="/path/to/model.pth",
        detection=True, detection_model_name="rfdetr-v5",
        detection_model_path="/path/to/det.onnx", detection_batch_size=8, unet4x=True,
    )
    assert EngineCompilationRequest.from_json(req.to_json()) == req


def test_request_defaults() -> None:
    req = EngineCompilationRequest(device="cuda:0", fp16=True)
    assert req.basicvsrpp is False
    assert req.detection is False
    assert req.unet4x is False


def test_ensure_no_subprocess_when_basicvsrpp_exists(monkeypatch) -> None:
    monkeypatch.setattr("jasna.engine_compiler.all_basicvsrpp_sub_engines_exist", lambda *_a, **_kw: True)
    req = EngineCompilationRequest(device="cuda:0", fp16=True, basicvsrpp=True, basicvsrpp_model_path="x")
    assert ensure_engines_compiled(req).use_basicvsrpp_tensorrt is True


def test_ensure_no_subprocess_when_not_requested() -> None:
    req = EngineCompilationRequest(device="cuda:0", fp16=True)
    result = ensure_engines_compiled(req)
    assert result.use_basicvsrpp_tensorrt is False


def test_ensure_all_exist_no_subprocess(monkeypatch) -> None:
    monkeypatch.setattr("jasna.engine_compiler.all_basicvsrpp_sub_engines_exist", lambda *_a, **_kw: True)
    monkeypatch.setattr("jasna.engine_compiler._detection_engine_exists", lambda *_a, **_kw: True)
    monkeypatch.setattr("jasna.engine_compiler._unet4x_engine_exists", lambda *_a, **_kw: True)
    req = EngineCompilationRequest(
        device="cuda:0", fp16=True, basicvsrpp=True, basicvsrpp_model_path="x",
        detection=True, detection_model_name="rfdetr-v5", detection_model_path="x", unet4x=True,
    )
    assert ensure_engines_compiled(req).use_basicvsrpp_tensorrt is True


def test_ensure_basicvsrpp_fp32_no_tensorrt() -> None:
    req = EngineCompilationRequest(device="cuda:0", fp16=False, basicvsrpp=True, basicvsrpp_model_path="x")
    assert ensure_engines_compiled(req).use_basicvsrpp_tensorrt is False


def test_ensure_spawns_subprocess_on_missing(monkeypatch) -> None:
    popen_calls = []
    proc = _mock_proc(["Compiling...\n", "Done.\n"])
    monkeypatch.setattr("jasna.engine_compiler.subprocess.Popen", lambda cmd, **kw: (popen_calls.append(cmd), proc)[1])

    call_count = [0]
    def engines_exist_after_compile(*_a, **_kw):
        call_count[0] += 1
        return call_count[0] > 1
    monkeypatch.setattr("jasna.engine_compiler.all_basicvsrpp_sub_engines_exist", engines_exist_after_compile)

    log_messages = []
    req = EngineCompilationRequest(device="cuda:0", fp16=True, basicvsrpp=True, basicvsrpp_model_path="model.pth")
    result = ensure_engines_compiled(req, log_callback=log_messages.append)

    assert len(popen_calls) == 1
    assert "-m" in popen_calls[0]
    assert "jasna.engine_compiler" in popen_calls[0]
    assert result.use_basicvsrpp_tensorrt is True
    assert any("Compiling" in m for m in log_messages)


def test_ensure_subprocess_failure_raises(monkeypatch) -> None:
    monkeypatch.setattr("jasna.engine_compiler.all_basicvsrpp_sub_engines_exist", lambda *_a, **_kw: False)
    monkeypatch.setattr("jasna.engine_compiler.subprocess.Popen", lambda *a, **kw: _mock_proc(["error\n"], returncode=1))

    req = EngineCompilationRequest(device="cuda:0", fp16=True, basicvsrpp=True, basicvsrpp_model_path="x")
    with pytest.raises(RuntimeError, match="exit code 1"):
        ensure_engines_compiled(req)


def test_ensure_frozen_exe_uses_compile_engines_flag(monkeypatch) -> None:
    monkeypatch.setattr("jasna.engine_compiler.all_basicvsrpp_sub_engines_exist", lambda *_a, **_kw: False)
    monkeypatch.setattr("jasna.engine_compiler.is_frozen", lambda: True)
    monkeypatch.setattr("jasna.engine_compiler.get_application_executable", lambda: "C:/app/jasna.exe")

    popen_calls = []
    proc = _mock_proc([])
    monkeypatch.setattr("jasna.engine_compiler.subprocess.Popen", lambda cmd, **kw: (popen_calls.append(cmd), proc)[1])

    req = EngineCompilationRequest(device="cuda:0", fp16=True, basicvsrpp=True, basicvsrpp_model_path="x")
    ensure_engines_compiled(req)

    assert len(popen_calls) == 1
    assert popen_calls[0][0] == "C:/app/jasna.exe"
    assert popen_calls[0][1] == "--compile-engines"
    assert "-m" not in popen_calls[0]


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_frozen_engine_compiler_removes_parent_tcl_paths_before_native_startup(monkeypatch, platform) -> None:
    import torch
    import jasna.engine_compiler as compiler

    pytest.importorskip("jasna.protection.runtime")
    monkeypatch.setattr(compiler, "is_frozen", lambda: True)
    monkeypatch.setattr(compiler.sys, "platform", platform)
    monkeypatch.setattr(compiler, "get_application_executable", lambda: "/app/jasna.bin")
    monkeypatch.setattr(compiler, "subprocess_no_window_kwargs", lambda: {})
    monkeypatch.setattr(compiler, "_missing_engines", lambda *_args, **_kwargs: compiler._MissingEngines(False, True, False))
    for name in ("TCL_LIBRARY", "TK_LIBRARY", "TCLLIBPATH"):
        monkeypatch.setenv(name, "/parent/runtime/tcl")
    monkeypatch.setenv("JASNA_ENGINE_ENV_SENTINEL", "preserved")
    calls = []
    monkeypatch.setattr(compiler.subprocess, "Popen", lambda command, **kwargs: (calls.append(kwargs), _mock_proc([]))[1])

    ensure_engines_compiled(EngineCompilationRequest(device="cpu", fp16=False, detection=True))

    environment = calls[0]["env"]
    assert all(name not in environment for name in ("TCL_LIBRARY", "TK_LIBRARY", "TCLLIBPATH"))
    assert environment["JASNA_ENGINE_ENV_SENTINEL"] == "preserved"


def test_frozen_linux_engine_compiler_uses_the_running_binary(monkeypatch) -> None:
    import torch  # noqa: F401  Initialize before mocking subprocess.
    import jasna.engine_compiler as compiler
    from jasna import os_utils

    monkeypatch.setattr(compiler, "is_frozen", lambda: True)
    monkeypatch.setattr(os_utils, "is_frozen", lambda: True)
    monkeypatch.setattr(os_utils.sys, "platform", "linux")
    monkeypatch.setattr(compiler.sys, "executable", "/relocated/app/python")
    monkeypatch.setattr(os_utils.os, "readlink", lambda path: "/relocated/app/jasna.bin")
    monkeypatch.setattr(compiler, "_missing_engines", lambda *_args, **_kwargs: compiler._MissingEngines(False, True, False))
    commands = []
    monkeypatch.setattr(compiler.subprocess, "Popen", lambda command, **kwargs: (commands.append(command), _mock_proc([]))[1])

    ensure_engines_compiled(EngineCompilationRequest(device="cpu", fp16=False, detection=True))

    assert commands[0][:2] == ["/relocated/app/jasna.bin", "--compile-engines"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows native module path API")
def test_frozen_windows_engine_compiler_uses_running_binary_when_python_placeholder_is_missing(monkeypatch) -> None:
    import torch  # noqa: F401  Initialize before mocking subprocess.
    import jasna.engine_compiler as compiler
    from jasna import os_utils

    expected = sys.executable
    monkeypatch.setattr(compiler, "is_frozen", lambda: True)
    monkeypatch.setattr(os_utils, "is_frozen", lambda: True)
    monkeypatch.setattr(compiler.sys, "executable", "C:/relocated/app/python.exe")
    monkeypatch.setattr(compiler, "_missing_engines", lambda *_args, **_kwargs: compiler._MissingEngines(False, True, False))
    commands = []
    monkeypatch.setattr(compiler.subprocess, "Popen", lambda command, **kwargs: (commands.append(command), _mock_proc([]))[1])

    ensure_engines_compiled(EngineCompilationRequest(device="cpu", fp16=False, detection=True))

    assert commands[0][0].casefold() == expected.casefold()
    assert commands[0][1] == "--compile-engines"


def test_ensure_create_no_window_on_windows(monkeypatch) -> None:
    monkeypatch.setattr("jasna.engine_compiler.all_basicvsrpp_sub_engines_exist", lambda *_a, **_kw: False)
    monkeypatch.setattr(
        "jasna.engine_compiler.subprocess_no_window_kwargs", lambda: {"creationflags": 0x08000000}
    )

    popen_kwargs = {}
    monkeypatch.setattr(
        "jasna.engine_compiler.subprocess.Popen",
        lambda cmd, **kw: (popen_kwargs.update(kw), _mock_proc([]))[1],
    )

    req = EngineCompilationRequest(device="cuda:0", fp16=True, basicvsrpp=True, basicvsrpp_model_path="x")
    ensure_engines_compiled(req)
    assert popen_kwargs.get("creationflags") == 0x08000000


def test_ensure_does_not_print_when_log_callback_given(monkeypatch) -> None:
    # The frozen GUI drops its console (FreeConsole), so stdout is invalid — an
    # unconditional print() would raise WinError 6. With a log_callback (GUI path), the
    # progress message must go to the callback and never to print().
    monkeypatch.setattr("jasna.engine_compiler.all_basicvsrpp_sub_engines_exist", lambda *_a, **_kw: False)
    monkeypatch.setattr("jasna.engine_compiler.subprocess.Popen", lambda *a, **kw: _mock_proc(["Done.\n"]))
    printed: list = []
    monkeypatch.setattr("jasna.engine_compiler.print", lambda *a, **kw: printed.append(a), raising=False)

    log_messages: list = []
    req = EngineCompilationRequest(device="cuda:0", fp16=True, basicvsrpp=True, basicvsrpp_model_path="x")
    ensure_engines_compiled(req, log_callback=log_messages.append)

    assert printed == []
    assert any("Compiling" in m for m in log_messages)


def test_ensure_popen_stdin_is_devnull(monkeypatch) -> None:
    # The detached GUI's stdin handle is invalid; the child must not inherit it.
    monkeypatch.setattr("jasna.engine_compiler.all_basicvsrpp_sub_engines_exist", lambda *_a, **_kw: False)
    popen_kwargs: dict = {}
    monkeypatch.setattr(
        "jasna.engine_compiler.subprocess.Popen",
        lambda cmd, **kw: (popen_kwargs.update(kw), _mock_proc([]))[1],
    )

    req = EngineCompilationRequest(device="cuda:0", fp16=True, basicvsrpp=True, basicvsrpp_model_path="x")
    ensure_engines_compiled(req)
    assert popen_kwargs.get("stdin") == subprocess.DEVNULL


def test_subprocess_compile_patches_frozen_torch(monkeypatch) -> None:
    # In the compiled binary the compile subprocess imports torch_tensorrt -> torch._inductor
    # directly; without patch_frozen_torch the source-introspection raises. An empty request
    # compiles nothing, so this only exercises the early import-torch + patch path.
    called = []
    monkeypatch.setattr("jasna._frozen.patch_frozen_torch", lambda: called.append(True))
    try:
        with warnings.catch_warnings():
            _subprocess_compile(EngineCompilationRequest(device="cpu", fp16=False))
    finally:
        logging.disable(logging.NOTSET)
    assert called, "patch_frozen_torch must run before any torch_tensorrt/_inductor import"


def test_detection_engine_exists_rfdetr(tmp_path: Path) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_text("x")
    assert _detection_engine_exists(
        "rfdetr-v5", str(onnx_path), 4, True, "cuda:0"
    ) is False

    from jasna.trt import get_onnx_tensorrt_engine_path
    engine = get_onnx_tensorrt_engine_path(
        onnx_path,
        batch_size=4,
        fp16=True,
        dynamic_batch=False,
    )
    engine.parent.mkdir(parents=True, exist_ok=True)
    engine.write_text("x")
    assert _detection_engine_exists(
        "rfdetr-v5", str(onnx_path), 4, True, "cuda:0"
    ) is True


def test_detection_engine_exists_rfdetr_v6_uses_dynamic_path(
    tmp_path: Path,
) -> None:
    onnx_path = tmp_path / "rfdetr-v6.onnx"
    onnx_path.write_text("x")

    from jasna.trt import get_onnx_tensorrt_engine_path

    engine = get_onnx_tensorrt_engine_path(
        onnx_path,
        batch_size=8,
        fp16=True,
        dynamic_batch=True,
    )
    engine.write_text("x")
    assert _detection_engine_exists(
        "rfdetr-v6",
        str(onnx_path),
        8,
        True,
        "cuda:0",
    )


def test_dynamic_batch_engine_path_is_distinct_from_fixed() -> None:
    from jasna.engine_paths import get_onnx_tensorrt_engine_path

    onnx = Path("model_weights/rfdetr-v6.onnx")
    fixed = get_onnx_tensorrt_engine_path(onnx, batch_size=4, fp16=True)
    dyn = get_onnx_tensorrt_engine_path(onnx, batch_size=4, fp16=True, dynamic_batch=True)
    assert ".bs4." in fixed.name
    assert ".bs1-4." in dyn.name
    assert fixed != dyn


def test_unet4x_engine_exists_plaintext(monkeypatch, tmp_path: Path) -> None:
    onnx_path = tmp_path / "unet-4x.onnx"
    onnx_path.write_bytes(b"onnx")
    monkeypatch.setattr("jasna.engine_paths.UNET4X_ONNX_PATH", onnx_path)
    assert _unet4x_engine_exists(fp16=True) is False

    from jasna.engine_paths import get_unet4x_engine_path
    engine = get_unet4x_engine_path(onnx_path, fp16=True)
    engine.parent.mkdir(parents=True, exist_ok=True)
    engine.write_text("x")
    assert _unet4x_engine_exists(fp16=True) is True


def test_unet4x_engine_exists_encrypted(monkeypatch, tmp_path: Path) -> None:
    onnx_path = tmp_path / "unet-4x.onnx"  # absent → encrypted branch
    monkeypatch.setattr("jasna.engine_paths.UNET4X_ONNX_PATH", onnx_path)
    enc_engine = tmp_path / "unet-4x.fp16.linux.engine.enc"
    monkeypatch.setattr("jasna.engine_paths.get_unet4x_encrypted_engine_path", lambda fp16=True: enc_engine)
    monkeypatch.setattr(
        "jasna.protection.protected_model.decrypt_engine_bytes",
        lambda model_id, data: b"decrypted-engine",
    )

    assert _unet4x_engine_exists(fp16=True) is False
    enc_engine.write_text("x")
    assert _unet4x_engine_exists(fp16=True) is True
