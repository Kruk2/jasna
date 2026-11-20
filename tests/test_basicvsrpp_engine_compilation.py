from __future__ import annotations

from pathlib import Path

import pytest
import torch

import jasna.restorer.basicvsrpp_sub_engines as sub
from jasna.engine_paths import get_basicvsrpp_sub_engine_paths


def _model_path(tmp_path: Path) -> str:
    (tmp_path / "model_weights").mkdir(parents=True, exist_ok=True)
    return str(tmp_path / "model_weights" / "model.pth")


def test_compile_skips_when_all_sub_engines_exist(monkeypatch, tmp_path: Path, cuda_capability) -> None:
    cuda_capability((8, 9))
    monkeypatch.setattr(sub.torch.cuda, "is_available", lambda: True)
    monkeypatch.chdir(tmp_path)
    model_path = _model_path(tmp_path)
    for p in get_basicvsrpp_sub_engine_paths(model_path, True, "sm89").values():
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_text("engine", encoding="utf-8")

    assert sub.compile_basicvsrpp_engines(model_path, torch.device("cuda:0"), True) is True


@pytest.mark.parametrize(
    ("device", "vram_gb", "fp16", "hip"),
    [
        ("cuda:0", 3.0, True, None),
        ("cuda:0", 32.0, False, None),
        ("cpu", 32.0, True, None),
        ("cuda:0", 32.0, True, "7.1"),
    ],
    ids=["low-vram", "fp32", "off-cuda", "amd"],
)
def test_compile_skips(monkeypatch, tmp_path: Path, cuda_capability, device, vram_gb, fp16, hip) -> None:
    cuda_capability((8, 9))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sub.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sub, "_gpu_vram_gb", lambda _dev: vram_gb)
    monkeypatch.setattr(torch.version, "hip", hip)

    assert sub.compile_basicvsrpp_engines(_model_path(tmp_path), torch.device(device), fp16) is False
