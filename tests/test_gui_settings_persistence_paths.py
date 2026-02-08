from __future__ import annotations

import json
from pathlib import Path

from jasna import os_utils
from jasna.gui.models import AppSettings, PresetManager, get_settings_path


def test_get_user_config_dir_windows_uses_appdata(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(os_utils.sys, "platform", "win32", raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    assert os_utils.get_user_config_dir("jasna") == (tmp_path / "Roaming" / "jasna")


def test_get_user_config_dir_linux_uses_xdg_config_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(os_utils.sys, "platform", "linux", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert os_utils.get_user_config_dir("jasna") == (tmp_path / "xdg" / "jasna")


def test_preset_manager_saves_to_user_config_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(os_utils.sys, "platform", "win32", raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))

    mgr = PresetManager()
    assert mgr.create_preset("MyPreset", AppSettings())

    settings_path = get_settings_path()
    assert settings_path == tmp_path / "Roaming" / "jasna" / "settings.json"
    assert settings_path.exists()

    data = json.loads(settings_path.read_text(encoding="utf-8"))
    assert "user_presets" in data
    assert "MyPreset" in data["user_presets"]


def test_preset_manager_preserves_other_settings_keys(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(os_utils.sys, "platform", "win32", raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))

    settings_path = get_settings_path()
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps({"language": "zh"}, indent=2), encoding="utf-8")

    mgr = PresetManager()
    assert mgr.create_preset("MyPreset", AppSettings())

    data = json.loads(settings_path.read_text(encoding="utf-8"))
    assert data.get("language") == "zh"
