from __future__ import annotations

import sys
import types
from pathlib import Path

import torch


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
    ultra_utils = types.ModuleType("ultralytics.utils")
    ultra_utils.nms = types.ModuleType("nms")
    ultra_utils.ops = types.ModuleType("ops")
    monkeypatch.setitem(sys.modules, "ultralytics", ultra)
    monkeypatch.setitem(sys.modules, "ultralytics.utils", ultra_utils)

    import jasna.trt as jt

    def _fake_compile(onnx_path, device, *, batch_size=None, fp16=True, **_kwargs):
        engine = jt.get_onnx_tensorrt_engine_path(onnx_path, batch_size=batch_size, fp16=fp16)
        _touch(engine)
        return engine

    monkeypatch.setattr(jt, "compile_onnx_to_tensorrt_engine", _fake_compile)

    from jasna.mosaic.yolo_tensorrt_compilation import compile_yolo_to_tensorrt_engine, get_yolo_tensorrt_engine_path

    expected = get_yolo_tensorrt_engine_path(pt, fp16=True)
    engine_path = compile_yolo_to_tensorrt_engine(pt, batch=8, fp16=True, imgsz=640, device=torch.device("cuda:0"))
    assert engine_path == expected
    assert engine_path.is_file()
    assert calls["export"] == 1


def test_compile_yolo_to_tensorrt_engine_skips_when_present(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model_weights").mkdir(parents=True, exist_ok=True)

    pt = Path("model_weights") / "lada_mosaic_detection_model_v4_fast.pt"
    _touch(pt)

    ultra = types.ModuleType("ultralytics")
    ultra_utils = types.ModuleType("ultralytics.utils")
    ultra_utils.nms = types.ModuleType("nms")
    ultra_utils.ops = types.ModuleType("ops")

    class _NeverCalled:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("YOLO constructor should not be called when engine exists")

    ultra.YOLO = _NeverCalled  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", ultra)
    monkeypatch.setitem(sys.modules, "ultralytics.utils", ultra_utils)

    from jasna.mosaic.yolo_tensorrt_compilation import get_yolo_tensorrt_engine_path, compile_yolo_to_tensorrt_engine
    engine = get_yolo_tensorrt_engine_path(pt, fp16=True)
    _touch(engine)


    out = compile_yolo_to_tensorrt_engine(pt, batch=8, fp16=True, imgsz=640, device=torch.device("cuda:0"))
    assert out == engine
