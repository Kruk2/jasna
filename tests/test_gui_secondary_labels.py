import inspect

from jasna.gui.settings_sections.secondary import SecondarySection


def test_secondary_radio_labels_use_compact_hints() -> None:
    source = inspect.getsource(SecondarySection)

    assert "secondary_unet_4x_hint" in source
    assert "secondary_tvai_hint" in source
    assert "secondary_rtx_hint" in source


def test_english_secondary_hints_are_short() -> None:
    from jasna.gui.locales import TRANSLATIONS

    en = TRANSLATIONS["en"]
    assert en["secondary_unet_4x_hint"] == "high-quality, fast"
    assert en["secondary_tvai_hint"] == "high-quality, slow"
    assert en["secondary_rtx_hint"] == "ok quality, very fast"
    assert all(
        len(en[key]) <= 26
        for key in ("secondary_unet_4x_hint", "secondary_tvai_hint", "secondary_rtx_hint")
    )
