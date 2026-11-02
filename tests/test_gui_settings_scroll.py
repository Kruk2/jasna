from __future__ import annotations


import customtkinter as ctk
import pytest

from jasna import os_utils

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


def test_settings_scrollbar_thumb_contrasts_with_panel_without_hover(tk_root, tmp_path, fresh_settings) -> None:
    panel = _settings_panel(tk_root, tmp_path)

    assert panel._scroll.cget("scrollbar_button_color") != Colors.BG_PANEL


def test_settings_scrollbar_only_appears_when_sections_overflow(tk_root, tmp_path, fresh_settings) -> None:
    tk_root.geometry("420x480")
    panel = _settings_panel(tk_root, tmp_path)
    panel.set_view("expert")
    panel.show_tab("output")
    tk_root.update()

    assert panel._scroll._scrollbar.winfo_ismapped()

    panel.show_tab("images")
    panel.set_view("simple")
    tk_root.update()

    assert not panel._scroll._scrollbar.winfo_ismapped()


@pytest.fixture
def opens_in_advanced(monkeypatch, tmp_path):
    """A user who left the settings in the Advanced view, written before the app starts."""
    monkeypatch.setattr(os_utils.sys, "platform", "linux", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    PresetManager().set_settings_view("advanced")


def test_no_scrollbar_after_the_gpu_check_relays_out_a_tab_that_fits(opens_in_advanced, jasna_app) -> None:
    from jasna.accelerator import AcceleratorVendor

    jasna_app.geometry("1500x800")
    jasna_app.update()
    jasna_app._settings_panel.set_gpu_support(vendor=AcceleratorVendor.NVIDIA, blackwell=True)
    jasna_app.update()

    assert not jasna_app._settings_panel._scroll._scrollbar.winfo_ismapped()
