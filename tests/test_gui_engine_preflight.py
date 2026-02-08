from __future__ import annotations

import os
from pathlib import Path

from jasna.gui.engine_preflight import run_engine_preflight
from jasna.gui.models import AppSettings


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def test_preflight_detects_missing_engines(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model_weights").mkdir(parents=True, exist_ok=True)

    settings = AppSettings()
    res = run_engine_preflight(settings)

    keys = {r.key for r in res.requirements}
    assert "rfdetr" in keys
    assert "basicvsrpp" in keys
    assert res.should_warn_first_run_slow
    assert {r.key for r in res.missing} == {"rfdetr", "basicvsrpp"}


def test_preflight_no_warning_when_all_expected_engines_exist(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model_weights").mkdir(parents=True, exist_ok=True)

    settings = AppSettings()
    first = run_engine_preflight(settings)
    for req in first.requirements:
        for p in req.paths:
            _touch(p)

    res = run_engine_preflight(settings)
    assert not res.should_warn_first_run_slow
    assert res.missing == ()


def test_preflight_includes_swin2sr_only_when_selected_and_fp16(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model_weights").mkdir(parents=True, exist_ok=True)

    settings = AppSettings(secondary_restoration="swin2sr", swin2sr_tensorrt=True, fp16_mode=True)
    res = run_engine_preflight(settings)
    assert "swin2sr" in {r.key for r in res.requirements}

    settings_no_fp16 = AppSettings(secondary_restoration="swin2sr", swin2sr_tensorrt=True, fp16_mode=False)
    res2 = run_engine_preflight(settings_no_fp16)
    assert "swin2sr" not in {r.key for r in res2.requirements}


def test_preflight_basicvsrpp_risky_only_when_main_engine_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model_weights").mkdir(parents=True, exist_ok=True)

    import jasna.restorer.basicvrspp_tenorrt_compilation as br

    def fake_get_approx(_device):
        return 8.0, 30

    monkeypatch.setattr(br, "_get_approx_max_tensorrt_clip_length", fake_get_approx)

    settings = AppSettings(max_clip_size=60, compile_basicvsrpp=True, fp16_mode=True)
    res = run_engine_preflight(settings)
    assert res.basicvsrpp_risk.is_risky

    basic_req = next(r for r in res.requirements if r.key == "basicvsrpp")
    main_engine = basic_req.paths[0]
    _touch(main_engine)

    res2 = run_engine_preflight(settings)
    assert not res2.basicvsrpp_risk.is_risky


def test_get_onnx_tensorrt_engine_path_matches_compile_return_when_present(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model_weights").mkdir(parents=True, exist_ok=True)

    from jasna.trt import compile_onnx_to_tensorrt_engine, get_onnx_tensorrt_engine_path

    onnx = Path("model_weights") / "rfdetr-v3.onnx"
    _touch(onnx)
    engine = get_onnx_tensorrt_engine_path(onnx, batch_size=4, fp16=True)
    _touch(engine)

    out = compile_onnx_to_tensorrt_engine(onnx, batch_size=4, fp16=True)
    assert out == engine


def test_preflight_uses_yolo_engine_name_when_selected(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model_weights").mkdir(parents=True, exist_ok=True)

    settings = AppSettings(detection_model="lada-yolo-v4")
    res = run_engine_preflight(settings)

    keys = {r.key for r in res.requirements}
    assert "yolo" in keys
    assert "rfdetr" not in keys

    yolo_req = next(r for r in res.requirements if r.key == "yolo")
    suffix = ".fp16.win.engine" if os.name == "nt" else ".fp16.linux.engine"
    assert yolo_req.paths == (Path("model_weights") / f"lada_mosaic_detection_model_v4_fast{suffix}",)

