"""About dialog sizes to its content so the close button is never clipped."""

from __future__ import annotations

from tkinter import TclError

import customtkinter as ctk
import pytest

from jasna.gui import scaling
from jasna.gui.app import JasnaApp
from jasna.gui.locales import get_locale


@pytest.fixture
def root():
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    yield root
    root.destroy()


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_about_dialog_fits_content(root, lang):
    locale = get_locale()
    original = locale.current_language
    locale.set_language(lang)
    try:
        dialog = JasnaApp._show_about(root)
        try:
            dialog.update_idletasks()
            geometry_height = int(dialog.geometry().split("+")[0].split("x")[1])
            physical_height = scaling.to_physical(dialog, 0, geometry_height)[1]
            assert physical_height >= dialog.winfo_reqheight()
        finally:
            dialog.destroy()
    finally:
        locale.set_language(original)
