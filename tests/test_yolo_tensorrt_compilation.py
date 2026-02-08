from __future__ import annotations

import sys
import types
from pathlib import Path


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def test_compile_yolo_to_tensorrt_engine_exports_when_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model_weights").mkdir(parents=True, exist_ok=True)

    pt = Path("model_weights") / "lada_mosaic_detection_model_v4_fast.pt"
    _touch(pt)

    calls = {"export": 0}

    class _FakeYOLO:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def export(self, **_kwargs):
            calls["export"] += 1
            onnx = pt.with_suffix(".onnx")
            _touch(onnx)
            return str(onnx)

    ultra = types.ModuleType("ultralytics")
    ultra.YOLO = _FakeYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", ultra)

    from jasna.mosaic.yolo_tensorrt_compilation import compile_yolo_to_tensorrt_engine, get_yolo_tensorrt_engine_path
    import jasna.trt as jt

    def _fake_compile(onnx_path, batch_size=None, fp16=True, **_kwargs):
        engine = jt.get_onnx_tensorrt_engine_path(onnx_path, batch_size=batch_size, fp16=fp16)
        _touch(engine)
        return engine

    monkeypatch.setattr(jt, "compile_onnx_to_tensorrt_engine", _fake_compile)

    expected = get_yolo_tensorrt_engine_path(pt, fp16=True)
    engine_path = compile_yolo_to_tensorrt_engine(pt, batch=8, fp16=True, imgsz=640)
    assert engine_path == expected
    assert engine_path.is_file()
    assert calls["export"] == 1


def test_compile_yolo_to_tensorrt_engine_skips_when_present(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model_weights").mkdir(parents=True, exist_ok=True)

    pt = Path("model_weights") / "lada_mosaic_detection_model_v4_fast.pt"
    _touch(pt)
    from jasna.mosaic.yolo_tensorrt_compilation import get_yolo_tensorrt_engine_path
    engine = get_yolo_tensorrt_engine_path(pt, fp16=True)
    _touch(engine)

    ultra = types.ModuleType("ultralytics")

    class _NeverCalled:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("YOLO constructor should not be called when engine exists")

    ultra.YOLO = _NeverCalled  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", ultra)

    from jasna.mosaic.yolo_tensorrt_compilation import compile_yolo_to_tensorrt_engine

    out = compile_yolo_to_tensorrt_engine(pt, batch=8, fp16=True, imgsz=640)
    assert out == engine

