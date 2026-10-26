"""About dialog sizes to its content so the close button is never clipped."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_about_dialog_fits_content(jasna_app, gui_language, lang, show_window):
    gui_language(lang)

    dialog = jasna_app._show_about()
    show_window(dialog)

    geometry_height = int(dialog.geometry().split("+")[0].split("x")[1])
    assert geometry_height >= dialog.winfo_reqheight()
