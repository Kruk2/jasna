from __future__ import annotations

from pathlib import Path

import torch


def test_compile_skips_when_all_sub_engines_exist(monkeypatch, tmp_path: Path) -> None:
    import jasna.restorer.basicvrspp_tenorrt_compilation as comp
    import jasna.restorer.basicvsrpp_sub_engines as sub

    monkeypatch.chdir(tmp_path)
    model_path = str(tmp_path / "model_weights" / "model.pth")
    (tmp_path / "model_weights").mkdir(parents=True, exist_ok=True)

    paths = sub.get_sub_engine_paths(model_path, fp16=True)
    for p in paths.values():
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_text("engine", encoding="utf-8")

    result = comp.compile_mosaic_restoration_model(
        mosaic_restoration_model_path=model_path,
        device=torch.device("cuda:0"),
        fp16=True,
    )
    assert result is True


def test_compile_skips_on_low_vram(monkeypatch, tmp_path: Path) -> None:
    import jasna.restorer.basicvrspp_tenorrt_compilation as comp

    monkeypatch.chdir(tmp_path)
    model_path = str(tmp_path / "model_weights" / "model.pth")
    (tmp_path / "model_weights").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(comp, "get_gpu_vram_gb", lambda _dev: 3.0)

    result = comp.compile_mosaic_restoration_model(
        mosaic_restoration_model_path=model_path,
        device=torch.device("cuda:0"),
        fp16=True,
    )
    assert result is False


def test_compile_skips_on_fp32(monkeypatch, tmp_path: Path) -> None:
    import jasna.restorer.basicvrspp_tenorrt_compilation as comp

    monkeypatch.chdir(tmp_path)
    model_path = str(tmp_path / "model_weights" / "model.pth")
    (tmp_path / "model_weights").mkdir(parents=True, exist_ok=True)

    result = comp.compile_mosaic_restoration_model(
        mosaic_restoration_model_path=model_path,
        device=torch.device("cuda:0"),
        fp16=False,
    )
    assert result is False


def test_startup_policy_false_when_compile_disabled(tmp_path: Path) -> None:
    import jasna.restorer.basicvrspp_tenorrt_compilation as comp

    result = comp.basicvsrpp_startup_policy(
        restoration_model_path=str(tmp_path / "model.pth"),
        device=torch.device("cuda:0"),
        fp16=True,
        compile_basicvsrpp=False,
    )
    assert result is False


def test_startup_policy_true_when_engines_exist(monkeypatch, tmp_path: Path) -> None:
    import jasna.restorer.basicvrspp_tenorrt_compilation as comp
    import jasna.restorer.basicvsrpp_sub_engines as sub

    model_path = str(tmp_path / "model.pth")
    paths = sub.get_sub_engine_paths(model_path, fp16=True)
    for p in paths.values():
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_text("engine", encoding="utf-8")

    result = comp.basicvsrpp_startup_policy(
        restoration_model_path=model_path,
        device=torch.device("cuda:0"),
        fp16=True,
        compile_basicvsrpp=True,
    )
    assert result is True

