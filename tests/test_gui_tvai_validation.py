from __future__ import annotations

from pathlib import Path

from jasna.gui.models import AppSettings
from jasna.gui.validation import validate_gui_start


def test_validate_gui_start_non_tvai_returns_empty() -> None:
    settings = AppSettings(secondary_restoration="none")
    assert validate_gui_start(settings) == []


def test_validate_gui_start_tvai_missing_env_vars(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TVAI_MODEL_DATA_DIR", raising=False)
    monkeypatch.delenv("TVAI_MODEL_DIR", raising=False)

    ffmpeg = tmp_path / "ffmpeg.exe"
    ffmpeg.write_bytes(b"")
    settings = AppSettings(secondary_restoration="tvai", tvai_ffmpeg_path=str(ffmpeg))

    errors = validate_gui_start(settings)
    assert "TVAI_MODEL_DATA_DIR env var is not set" in errors
    assert "TVAI_MODEL_DIR env var is not set" in errors


def test_validate_gui_start_tvai_env_dirs_must_exist(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TVAI_MODEL_DATA_DIR", str(tmp_path / "missing_data"))
    monkeypatch.setenv("TVAI_MODEL_DIR", str(tmp_path / "missing_model"))

    ffmpeg = tmp_path / "ffmpeg.exe"
    ffmpeg.write_bytes(b"")
    settings = AppSettings(secondary_restoration="tvai", tvai_ffmpeg_path=str(ffmpeg))

    errors = validate_gui_start(settings)
    assert any(e.startswith("TVAI_MODEL_DATA_DIR does not point to an existing directory:") for e in errors)
    assert any(e.startswith("TVAI_MODEL_DIR does not point to an existing directory:") for e in errors)


def test_validate_gui_start_tvai_ffmpeg_path_must_exist(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"
    data_dir.mkdir()
    model_dir.mkdir()
    monkeypatch.setenv("TVAI_MODEL_DATA_DIR", str(data_dir))
    monkeypatch.setenv("TVAI_MODEL_DIR", str(model_dir))

    settings = AppSettings(secondary_restoration="tvai", tvai_ffmpeg_path=str(tmp_path / "missing_ffmpeg.exe"))
    errors = validate_gui_start(settings)
    assert any(e.startswith("TVAI ffmpeg not found:") for e in errors)


def test_validate_gui_start_tvai_ok(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"
    data_dir.mkdir()
    model_dir.mkdir()
    monkeypatch.setenv("TVAI_MODEL_DATA_DIR", str(data_dir))
    monkeypatch.setenv("TVAI_MODEL_DIR", str(model_dir))

    ffmpeg = tmp_path / "ffmpeg.exe"
    ffmpeg.write_bytes(b"")

    settings = AppSettings(secondary_restoration="tvai", tvai_ffmpeg_path=str(ffmpeg))
    assert validate_gui_start(settings) == []

