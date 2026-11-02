from __future__ import annotations


import customtkinter as ctk
import pytest

from jasna import os_utils

from jasna.gui.components import CollapsibleSection
from jasna.gui.models import PresetManager
from jasna.gui.settings_panel import SettingsPanel
from jasna.gui.theme import Colors
from factories import ltx_models


@pytest.fixture
def fresh_settings(monkeypatch, tmp_path):
    """A new user's settings.json, so the panel opens in the simple view."""
    monkeypatch.setattr(os_utils.sys, "platform", "linux", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def _settings_panel(root: ctk.CTk, tmp_path) -> SettingsPanel:
    panel = SettingsPanel(root, PresetManager(), ltx_models=ltx_models(root, tmp_path, installed=True))
    panel.pack(fill="both", expand=True)
    root.update()
    return panel


def _collapse_all_sections(panel: SettingsPanel) -> None:
    for section in panel._scroll.winfo_children():
        if isinstance(section, CollapsibleSection) and section._expanded:
            section._toggle()


def test_settings_scrollbar_thumb_contrasts_with_panel_without_hover(tk_root, tmp_path, fresh_settings) -> None:
    panel = _settings_panel(tk_root, tmp_path)

    assert panel._scroll.cget("scrollbar_button_color") != Colors.BG_PANEL


def test_settings_scrollbar_only_appears_when_sections_overflow(tk_root, tmp_path, fresh_settings) -> None:
    tk_root.geometry("420x480")
    panel = _settings_panel(tk_root, tmp_path)

    assert panel._scroll._scrollbar.winfo_ismapped()

    _collapse_all_sections(panel)
    tk_root.update()

    assert not panel._scroll._scrollbar.winfo_ismapped()
