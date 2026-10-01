from __future__ import annotations

import sys
from pathlib import Path

import pytest

import jasna.engine_paths as engine_paths
from jasna.engine_paths import model_weights_dir
from jasna.mosaic import detection_registry


@pytest.fixture
def nvidia_vendor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ONNX weight expectations independent of the host GPU."""

    monkeypatch.setattr(
        detection_registry, "is_amd_device", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(
        detection_registry, "is_nvidia_device", lambda *_args, **_kwargs: True
    )


def test_model_weights_dir_in_dev_is_cwd_relative(monkeypatch) -> None:
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert model_weights_dir() == Path("model_weights")


def test_model_weights_dir_when_frozen_is_next_to_executable(monkeypatch, tmp_path: Path) -> None:
    fake_exe = tmp_path / "jasna-cli.exe"
    fake_exe.touch()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(fake_exe))

    assert model_weights_dir() == tmp_path / "model_weights"


def test_detection_model_weights_path_uses_helper(
    monkeypatch, tmp_path: Path, nvidia_vendor: None
) -> None:
    fake_exe = tmp_path / "jasna-cli.exe"
    fake_exe.touch()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(fake_exe))

    p = detection_registry.detection_model_weights_path("rfdetr-v5")
    assert p == tmp_path / "model_weights" / "rfdetr-v5.onnx"


def test_discover_available_detection_models_uses_helper(
    monkeypatch, tmp_path: Path, nvidia_vendor: None
) -> None:
    weights_dir = tmp_path / "model_weights"
    weights_dir.mkdir()
    (weights_dir / "rfdetr-v5.onnx").touch()
    (weights_dir / "lada_mosaic_detection_model_v4_fast.pt").touch()

    fake_exe = tmp_path / "jasna-cli.exe"
    fake_exe.touch()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(fake_exe))

    names = detection_registry.discover_available_detection_models()
    assert "rfdetr-v5" in names
    assert "lada-yolo-v4" in names


def test_discover_available_detection_models_explicit_dir_overrides(
    tmp_path: Path, nvidia_vendor: None
) -> None:
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    (weights_dir / "rfdetr-v3.onnx").touch()

    names = detection_registry.discover_available_detection_models(weights_dir)
    assert names == ["rfdetr-v3"]
