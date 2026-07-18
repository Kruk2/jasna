from __future__ import annotations

import os
from pathlib import Path

import torch

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


def test_preflight_basicvsrpp_missing_then_found(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model_weights").mkdir(parents=True, exist_ok=True)

    settings = AppSettings(max_clip_size=60, compile_basicvsrpp=True, fp16_mode=True)
    res = run_engine_preflight(settings)

    basic_req = next(r for r in res.requirements if r.key == "basicvsrpp")
    assert not basic_req.exists
    assert len(basic_req.missing_paths) == 6

    for p in basic_req.paths:
        _touch(p)

    res2 = run_engine_preflight(settings)
    basic_req2 = next(r for r in res2.requirements if r.key == "basicvsrpp")
    assert basic_req2.exists
    assert len(basic_req2.missing_paths) == 0


def test_get_onnx_tensorrt_engine_path_matches_compile_return_when_present(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model_weights").mkdir(parents=True, exist_ok=True)

    from jasna.trt import compile_onnx_to_tensorrt_engine, get_onnx_tensorrt_engine_path

    onnx = Path("model_weights") / "rfdetr-v3.onnx"
    _touch(onnx)
    engine = get_onnx_tensorrt_engine_path(onnx, batch_size=4, fp16=True)
    _touch(engine)

    out = compile_onnx_to_tensorrt_engine(onnx, torch.device("cuda:0"), batch_size=4, fp16=True, workspace_gb=20)
    assert out == engine


def test_preflight_unet4x_encrypted_engine_satisfies(monkeypatch, tmp_path: Path) -> None:
    """Frozen builds ship the encrypted unet-4x engine (.enc); preflight must accept it
    instead of always looking for the plaintext engine and warning about compilation."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model_weights").mkdir(parents=True, exist_ok=True)

    import jasna.engine_paths as ep
    monkeypatch.setattr(ep, "unet4x_plaintext_available", lambda: False)

    settings = AppSettings(secondary_restoration="unet-4x", fp16_mode=True)

    res = run_engine_preflight(settings)
    unet_req = next(r for r in res.requirements if r.key == "unet_4x")
    assert not unet_req.exists

    enc_engine = ep.get_unet4x_encrypted_engine_path(fp16=True)
    _touch(enc_engine)

    res2 = run_engine_preflight(settings)
    unet_req2 = next(r for r in res2.requirements if r.key == "unet_4x")
    assert unet_req2.exists
    assert unet_req2.paths == (enc_engine,)


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


def test_amd_preflight_checks_only_migraphx_cache(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model_weights").mkdir(parents=True, exist_ok=True)

    import jasna.gui.engine_preflight as module
    import jasna.mosaic.migraphx_runner as migraphx

    cache = tmp_path / "model_weights" / "rfdetr-v5.migraphx" / "test"
    monkeypatch.setattr(module, "is_amd_device", lambda _device: True)
    monkeypatch.setattr(migraphx, "migraphx_cache_dir", lambda *_args, **_kwargs: cache)
    monkeypatch.setattr(
        migraphx,
        "migraphx_cache_is_ready",
        lambda *_args, **_kwargs: False,
    )

    settings = AppSettings(compile_basicvsrpp=True)
    result = run_engine_preflight(settings)

    assert [requirement.key for requirement in result.requirements] == ["rfdetr"]
    assert result.missing[0].paths == (cache,)
