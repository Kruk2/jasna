from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jasna.engine_compiler import (
    EngineCompilationRequest,
    _detection_engine_exists,
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
        basicvsrpp_model_path="/path/to/model.pth", basicvsrpp_max_clip_size=90,
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
    monkeypatch.setattr("jasna.engine_compiler._basicvsrpp_engines_exist", lambda *_a, **_kw: True)
    req = EngineCompilationRequest(device="cuda:0", fp16=True, basicvsrpp=True, basicvsrpp_model_path="x")
    assert ensure_engines_compiled(req).use_basicvsrpp_tensorrt is True


def test_ensure_no_subprocess_when_not_requested() -> None:
    req = EngineCompilationRequest(device="cuda:0", fp16=True)
    result = ensure_engines_compiled(req)
    assert result.use_basicvsrpp_tensorrt is False


def test_ensure_all_exist_no_subprocess(monkeypatch) -> None:
    monkeypatch.setattr("jasna.engine_compiler._basicvsrpp_engines_exist", lambda *_a, **_kw: True)
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
    monkeypatch.setattr("jasna.engine_compiler._basicvsrpp_engines_exist", engines_exist_after_compile)

    log_messages = []
    req = EngineCompilationRequest(device="cuda:0", fp16=True, basicvsrpp=True, basicvsrpp_model_path="model.pth")
    result = ensure_engines_compiled(req, log_callback=log_messages.append)

    assert len(popen_calls) == 1
    assert "-m" in popen_calls[0]
    assert "jasna.engine_compiler" in popen_calls[0]
    assert result.use_basicvsrpp_tensorrt is True
    assert any("Compiling" in m for m in log_messages)


def test_ensure_subprocess_failure_raises(monkeypatch) -> None:
    monkeypatch.setattr("jasna.engine_compiler._basicvsrpp_engines_exist", lambda *_a, **_kw: False)
    monkeypatch.setattr("jasna.engine_compiler.subprocess.Popen", lambda *a, **kw: _mock_proc(["error\n"], returncode=1))

    req = EngineCompilationRequest(device="cuda:0", fp16=True, basicvsrpp=True, basicvsrpp_model_path="x")
    with pytest.raises(RuntimeError, match="exit code 1"):
        ensure_engines_compiled(req)


def test_ensure_frozen_exe_uses_compile_engines_flag(monkeypatch) -> None:
    monkeypatch.setattr("jasna.engine_compiler._basicvsrpp_engines_exist", lambda *_a, **_kw: False)
    fake_sys = type("FakeSys", (), {"executable": "C:/app/jasna.exe", "frozen": True})()
    monkeypatch.setattr("jasna.engine_compiler.sys", fake_sys)

    popen_calls = []
    proc = _mock_proc([])
    monkeypatch.setattr("jasna.engine_compiler.subprocess.Popen", lambda cmd, **kw: (popen_calls.append(cmd), proc)[1])

    req = EngineCompilationRequest(device="cuda:0", fp16=True, basicvsrpp=True, basicvsrpp_model_path="x")
    ensure_engines_compiled(req)

    assert len(popen_calls) == 1
    assert popen_calls[0][0] == "C:/app/jasna.exe"
    assert popen_calls[0][1] == "--compile-engines"
    assert "-m" not in popen_calls[0]


def test_ensure_create_no_window_on_windows(monkeypatch) -> None:
    monkeypatch.setattr("jasna.engine_compiler._basicvsrpp_engines_exist", lambda *_a, **_kw: False)
    monkeypatch.setattr("jasna.engine_compiler.os.name", "nt")

    popen_kwargs = {}
    monkeypatch.setattr(
        "jasna.engine_compiler.subprocess.Popen",
        lambda cmd, **kw: (popen_kwargs.update(kw), _mock_proc([]))[1],
    )

    req = EngineCompilationRequest(device="cuda:0", fp16=True, basicvsrpp=True, basicvsrpp_model_path="x")
    ensure_engines_compiled(req)
    assert popen_kwargs.get("creationflags") == subprocess.CREATE_NO_WINDOW


def test_detection_engine_exists_rfdetr(tmp_path: Path) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_text("x")
    assert _detection_engine_exists("rfdetr-v5", str(onnx_path), 4, True) is False

    from jasna.trt import get_onnx_tensorrt_engine_path
    engine = get_onnx_tensorrt_engine_path(onnx_path, batch_size=4, fp16=True)
    engine.parent.mkdir(parents=True, exist_ok=True)
    engine.write_text("x")
    assert _detection_engine_exists("rfdetr-v5", str(onnx_path), 4, True) is True


def test_unet4x_engine_exists(monkeypatch, tmp_path: Path) -> None:
    onnx_path = tmp_path / "unet-4x.onnx"
    monkeypatch.setattr("jasna.engine_paths.UNET4X_ONNX_PATH", onnx_path)
    assert _unet4x_engine_exists(fp16=True) is False

    from jasna.engine_paths import get_unet4x_engine_path
    engine = get_unet4x_engine_path(onnx_path, fp16=True)
    engine.parent.mkdir(parents=True, exist_ok=True)
    engine.write_text("x")
    assert _unet4x_engine_exists(fp16=True) is True
