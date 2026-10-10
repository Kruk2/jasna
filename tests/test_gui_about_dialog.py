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
    # A bare CTk() can report a tiny geometry in a headless/virtual session, and
    # the dialog is deliberately clamped to its parent: give the parent room so
    # the assertion measures the dialog, not the parent's default size.
    root.geometry("1280x900")
    root.update_idletasks()
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
            required = dialog.winfo_reqheight()
            # The dialog is sized to its content, clamped to the monitor's work
            # area minus the placement margins - so that clamped height is the
            # expected geometry, on a normal desktop and on a small one alike.
            margin = scaling.to_physical(dialog, *scaling.SCREEN_MARGIN)[1]
            available = scaling.screen_rect(dialog)[3] - margin
            expected = max(1, min(required, available))
            assert abs(geometry_height - expected) <= 2  # physical<->logical rounding
        finally:
            dialog.destroy()
    finally:
        locale.set_language(original)
