from __future__ import annotations

from pathlib import Path

import torch

import jasna.restorer.basicvsrpp_sub_engines as sub
from jasna.engine_paths import get_basicvsrpp_sub_engine_paths


def _model_path(tmp_path: Path) -> str:
    (tmp_path / "model_weights").mkdir(parents=True, exist_ok=True)
    return str(tmp_path / "model_weights" / "model.pth")


def test_compile_skips_when_all_sub_engines_exist(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    model_path = _model_path(tmp_path)
    for p in get_basicvsrpp_sub_engine_paths(model_path, fp16=True).values():
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_text("engine", encoding="utf-8")

    assert sub.compile_basicvsrpp_engines(model_path, torch.device("cuda:0"), True) is True


def test_compile_skips_on_low_vram(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sub.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sub, "_gpu_vram_gb", lambda _dev: 3.0)

    assert sub.compile_basicvsrpp_engines(_model_path(tmp_path), torch.device("cuda:0"), True) is False


def test_compile_skips_on_fp32(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sub.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sub, "_gpu_vram_gb", lambda _dev: 32.0)

    assert sub.compile_basicvsrpp_engines(_model_path(tmp_path), torch.device("cuda:0"), False) is False


def test_compile_skips_off_cuda(tmp_path: Path) -> None:
    assert sub.compile_basicvsrpp_engines(_model_path(tmp_path), torch.device("cpu"), True) is False
