"""About dialog sizes to its content so the close button is never clipped."""

from __future__ import annotations

import pytest

from jasna.gui.locales import get_locale


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_about_dialog_fits_content(jasna_app, lang, show_window):
    locale = get_locale()
    original = locale.current_language
    locale.set_language(lang)
    try:
        dialog = jasna_app._show_about()
        show_window(dialog)
        geometry_height = int(dialog.geometry().split("+")[0].split("x")[1])
        assert geometry_height >= dialog.winfo_reqheight()
    finally:
        locale.set_language(original)
