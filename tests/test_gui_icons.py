from __future__ import annotations

from tkinter import TclError
from unittest.mock import MagicMock

import customtkinter as ctk
import pytest

from jasna.gui import settings_panel
from jasna.gui.icons import render_icon, render_toggle
from jasna.gui.theme import Colors


@pytest.mark.parametrize("name", ["create", "delete", "folder", "reset", "save"])
def test_gui_icons_render_without_font_glyphs(name: str) -> None:
    image = render_icon(name, 18, Colors.TEXT_PRIMARY)

    assert image.mode == "RGBA"
    assert image.size == (18, 18)
    assert image.getchannel("A").getbbox() is not None


@pytest.mark.parametrize("selected", [False, True])
def test_toggle_switch_renders_without_customtkinter_shape_glyphs(selected: bool) -> None:
    image = render_toggle(
        selected,
        36,
        18,
        Colors.PRIMARY if selected else Colors.BORDER_LIGHT,
        Colors.TEXT_PRIMARY,
    )

    assert image.mode == "RGBA"
    assert image.size == (36, 18)
    assert image.getbbox() == (0, 0, 36, 18)


def test_compact_switch_uses_image_backed_control(monkeypatch) -> None:
    constructor = MagicMock(return_value=object())
    monkeypatch.setattr(settings_panel, "CompactSwitch", constructor)
    master = object()
    command = MagicMock()

    result = settings_panel.create_compact_switch(master, command, Colors.BG_CARD)

    assert result is constructor.return_value
    constructor.assert_called_once_with(master, command, Colors.BG_CARD)


def test_compact_switch_preserves_switch_state_and_callback() -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    try:
        callback = MagicMock()
        switch = settings_panel.CompactSwitch(root, callback, Colors.BG_PANEL)

        assert switch.get() == 0
        switch.select()
        assert switch.get() == 1
        switch.deselect()
        assert switch.get() == 0
        switch._toggle()
        assert switch.get() == 1
        callback.assert_called_once_with()
    finally:
        root.destroy()


def test_slider_value_uses_native_label_without_ctk_canvas(monkeypatch) -> None:
    constructor = MagicMock(return_value=object())
    monkeypatch.setattr(settings_panel.tk, "Label", constructor)
    master = object()

    result = settings_panel.create_slider_value_label(master, "90", 4)

    assert result is constructor.return_value
    constructor.assert_called_once_with(
        master,
        text="90",
        foreground=Colors.TEXT_PRIMARY,
        background=Colors.BG_PANEL,
        font=(settings_panel.Fonts.FAMILY, settings_panel.Fonts.SIZE_NORMAL),
        width=4,
        borderwidth=0,
        highlightthickness=0,
    )
