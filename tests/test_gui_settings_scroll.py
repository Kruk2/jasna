from __future__ import annotations

from tkinter import TclError

import customtkinter as ctk
import pytest

from jasna.gui.components import CollapsibleSection
from jasna.gui.models import PresetManager
from jasna.gui.settings_panel import SettingsPanel
from jasna.gui.theme import Colors
from factories import ltx_models


def _settings_panel(root: ctk.CTk, tmp_path) -> SettingsPanel:
    panel = SettingsPanel(root, PresetManager(), ltx_models=ltx_models(root, tmp_path, installed=True))
    panel.pack(fill="both", expand=True)
    root.update()
    return panel


def _collapse_all_sections(panel: SettingsPanel) -> None:
    for section in panel._scroll.winfo_children():
        if isinstance(section, CollapsibleSection) and section._expanded:
            section._toggle()


def test_settings_scrollbar_thumb_contrasts_with_panel_without_hover(tmp_path) -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    try:
        panel = _settings_panel(root, tmp_path)

        assert panel._scroll.cget("scrollbar_button_color") == Colors.BORDER_LIGHT
        assert panel._scroll.cget("scrollbar_button_color") != Colors.BG_PANEL
    finally:
        root.destroy()


def test_settings_scrollbar_only_appears_when_sections_overflow(tmp_path) -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    try:
        root.geometry("420x480")
        panel = _settings_panel(root, tmp_path)

        assert panel._scroll._scrollbar.winfo_ismapped()

        _collapse_all_sections(panel)
        root.update()

        assert not panel._scroll._scrollbar.winfo_ismapped()
    finally:
        root.destroy()
