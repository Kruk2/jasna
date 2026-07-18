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


def test_preset_manager_saves_and_loads_last_output_folder(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(os_utils.sys, "platform", "win32", raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))

    mgr = PresetManager()
    assert mgr.get_last_output_folder() == ""
    mgr.set_last_output_folder("/some/output")
    assert mgr.get_last_output_folder() == "/some/output"

    mgr2 = PresetManager()
    assert mgr2.get_last_output_folder() == "/some/output"

    data = json.loads(get_settings_path().read_text(encoding="utf-8"))
    assert data.get("last_output_folder") == "/some/output"


def test_preset_manager_persists_lut_path_in_user_preset(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(os_utils.sys, "platform", "win32", raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))

    mgr = PresetManager()
    settings = AppSettings(lut_path=r"C:\luts\film.cube")
    assert mgr.create_preset("WithLut", settings)

    mgr2 = PresetManager()
    loaded = mgr2.get_preset("WithLut")
    assert loaded is not None
    assert loaded.lut_path == r"C:\luts\film.cube"


def test_preset_manager_persists_working_directory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(os_utils.sys, "platform", "win32", raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))

    mgr = PresetManager()
    settings = AppSettings(working_directory=r"D:\scratch")
    assert mgr.create_preset("WithWorkDir", settings)

    mgr2 = PresetManager()
    loaded = mgr2.get_preset("WithWorkDir")
    assert loaded is not None
    assert loaded.working_directory == r"D:\scratch"


def test_preset_manager_persists_frame_rate_retargeting(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(os_utils.sys, "platform", "win32", raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))

    mgr = PresetManager()
    assert mgr.create_preset("HalfRate", AppSettings(retarget_high_fps=True))

    loaded = PresetManager().get_preset("HalfRate")
    assert loaded is not None
    assert loaded.retarget_high_fps is True


def test_preset_manager_persists_post_export_action(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(os_utils.sys, "platform", "win32", raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))

    mgr = PresetManager()
    settings = AppSettings(post_export_action="command", post_export_command="echo done")
    assert mgr.create_preset("WithAction", settings)

    mgr2 = PresetManager()
    loaded = mgr2.get_preset("WithAction")
    assert loaded is not None
    assert loaded.post_export_action == "command"
    assert loaded.post_export_command == "echo done"


def test_preset_manager_saves_and_loads_last_output_pattern(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(os_utils.sys, "platform", "win32", raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))

    mgr = PresetManager()
    default = mgr.get_last_output_pattern()
    assert "{original}" in default
    mgr.set_last_output_pattern("{original}_done.mkv")
    assert mgr.get_last_output_pattern() == "{original}_done.mkv"

    mgr2 = PresetManager()
    assert mgr2.get_last_output_pattern() == "{original}_done.mkv"
