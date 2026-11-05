from __future__ import annotations

import json
from pathlib import Path

import pytest

from jasna import os_utils
from jasna.gui.models import AppSettings, PresetManager
from jasna.user_settings import get_settings_path


def test_get_user_config_dir_windows_uses_appdata(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(os_utils.sys, "platform", "win32", raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    assert os_utils.get_user_config_dir("jasna") == (tmp_path / "Roaming" / "jasna")


def test_get_user_config_dir_linux_uses_xdg_config_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(os_utils.sys, "platform", "linux", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert os_utils.get_user_config_dir("jasna") == (tmp_path / "xdg" / "jasna")


@pytest.fixture
def windows_settings(monkeypatch, tmp_path: Path) -> Path:
    """Settings live under a Windows %APPDATA% in ``tmp_path``; returns the settings file."""
    monkeypatch.setattr(os_utils.sys, "platform", "win32", raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    return tmp_path / "Roaming" / "jasna" / "settings.json"


def test_preset_manager_saves_to_user_config_dir(windows_settings) -> None:
    assert PresetManager().create_preset("MyPreset", AppSettings())

    assert get_settings_path() == windows_settings
    assert "MyPreset" in json.loads(windows_settings.read_text(encoding="utf-8"))["user_presets"]


def test_preset_manager_preserves_other_settings_keys(windows_settings) -> None:
    windows_settings.parent.mkdir(parents=True)
    windows_settings.write_text(json.dumps({"language": "zh"}), encoding="utf-8")

    assert PresetManager().create_preset("MyPreset", AppSettings())

    assert json.loads(windows_settings.read_text(encoding="utf-8"))["language"] == "zh"


@pytest.mark.parametrize(
    "settings",
    [
        AppSettings(lut_path=r"C:\luts\film.cube"),
        AppSettings(working_directory=r"D:\scratch"),
        AppSettings(retarget_high_fps=True),
        AppSettings(sharpen_strength=0.45),
        AppSettings(post_export_action="command", post_export_command="echo done", post_export_video_command="remux {output}"),
        AppSettings(restoration_model="ltx", ltx_seed=42, ltx_fast=True, ltx_large_canvas=True, ltx_overlap=25),
    ],
    ids=["lut", "working-directory", "frame-rate", "sharpen", "post-export", "ltx"],
)
def test_preset_manager_persists_user_preset(windows_settings, settings) -> None:
    assert PresetManager().create_preset("Mine", settings)

    assert PresetManager().get_preset("Mine") == settings


@pytest.mark.parametrize(
    ("getter", "setter", "value"),
    [
        ("get_last_output_folder", "set_last_output_folder", "/some/output"),
        ("get_last_output_pattern", "set_last_output_pattern", "{original}_done.mkv"),
        ("get_settings_view", "set_settings_view", "advanced"),
    ],
)
def test_preset_manager_remembers_the_last_output(windows_settings, getter, setter, value) -> None:
    manager = PresetManager()
    assert getattr(manager, getter)() != value

    getattr(manager, setter)(value)

    assert getattr(PresetManager(), getter)() == value


def test_preset_manager_resolve_falls_back_to_default(windows_settings) -> None:
    manager = PresetManager()
    assert manager.resolve("DoesNotExist") == ("Default", AppSettings())

    assert manager.create_preset("Mine", AppSettings(encoder_cq=30))
    name, preset = manager.resolve("Mine")
    assert (name, preset.encoder_cq) == ("Mine", 30)


def test_preset_manager_saves_do_not_overwrite_keys_changed_elsewhere(windows_settings) -> None:
    panel_mgr = PresetManager()
    app_mgr = PresetManager()
    assert panel_mgr.create_preset("MyPreset", AppSettings())
    app_mgr.set_system_check_passed_version("9.9.9")
    panel_mgr.set_last_selected("MyPreset")

    reloaded = PresetManager()
    assert reloaded.get_preset("MyPreset") is not None
    assert reloaded.get_system_check_passed_version() == "9.9.9"
    assert reloaded.get_last_selected() == "MyPreset"


@pytest.mark.parametrize("content", ["[1, 2]", "{\"user_presets\": ", "\"text\""])
def test_unreadable_settings_start_from_defaults_and_are_rewritten(windows_settings, content) -> None:
    windows_settings.parent.mkdir(parents=True)
    windows_settings.write_text(content, encoding="utf-8")

    manager = PresetManager()
    assert manager.get_last_selected() == "Default"
    assert manager.create_preset("Mine", AppSettings())

    assert list(json.loads(windows_settings.read_text(encoding="utf-8"))) == ["user_presets"]
    assert [path.name for path in windows_settings.parent.iterdir()] == ["settings.json"]


def test_settings_keep_non_ascii_preset_names(windows_settings) -> None:
    assert PresetManager().create_preset("Ostrość ゆ", AppSettings())

    assert PresetManager().get_preset("Ostrość ゆ") == AppSettings()
