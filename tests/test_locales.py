from __future__ import annotations

import re

import pytest

from jasna.accelerator import AcceleratorVendor
from jasna.gui.locales import TRANSLATIONS
from jasna.media.encoder_settings import encoder_cq_spec

_TRANSLATED = sorted(set(TRANSLATIONS) - {"en"})
_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


@pytest.mark.parametrize("lang", _TRANSLATED)
def test_locale_key_set_matches_english(lang: str) -> None:
    en_keys = set(TRANSLATIONS["en"])
    lang_keys = set(TRANSLATIONS[lang])

    assert not en_keys - lang_keys, f"{lang} is missing keys: {sorted(en_keys - lang_keys)}"
    assert not lang_keys - en_keys, f"{lang} has keys absent from en: {sorted(lang_keys - en_keys)}"


@pytest.mark.parametrize("lang", sorted(TRANSLATIONS))
def test_locale_values_are_not_empty(lang: str) -> None:
    empty = [key for key, value in TRANSLATIONS[lang].items() if not value.strip()]
    assert not empty, f"{lang} has empty values for keys: {sorted(empty)}"


def _placeholders(text: str) -> list[str]:
    return sorted(set(_PLACEHOLDER_RE.findall(text)))


@pytest.mark.parametrize("lang", _TRANSLATED)
def test_format_placeholders_match(lang: str) -> None:
    mismatches = [
        f"  {key}: en={_placeholders(en_value)}, {lang}={_placeholders(TRANSLATIONS[lang][key])}"
        for key, en_value in TRANSLATIONS["en"].items()
        if _placeholders(en_value) != _placeholders(TRANSLATIONS[lang].get(key, en_value))
    ]

    assert not mismatches, f"Format placeholder mismatch between en and {lang}:\n" + "\n".join(mismatches)


@pytest.mark.parametrize("lang", sorted(TRANSLATIONS))
@pytest.mark.parametrize("key", ["btn_add_files", "btn_clear"])
def test_queue_button_text_does_not_depend_on_emoji_fonts(lang: str, key: str) -> None:
    assert not TRANSLATIONS[lang][key].startswith(("📁", "🗑"))


@pytest.mark.parametrize("lang", sorted(TRANSLATIONS))
def test_cq_tooltip_quotes_the_native_cq_ranges(lang: str) -> None:
    tip = TRANSLATIONS[lang]["tip_encoder_cq"]
    for vendor in (AcceleratorVendor.NVIDIA, AcceleratorVendor.AMD):
        for codec in ("h264", "hevc", "av1"):
            spec = encoder_cq_spec(codec, vendor)
            assert str(spec.default) in tip
            assert str(spec.minimum) in tip
            assert str(spec.maximum) in tip


def test_first_run_language_follows_system_locale(monkeypatch, tmp_path) -> None:
    import locale

    import jasna.gui.locales as locales
    from jasna import user_settings

    monkeypatch.setattr(user_settings, "get_settings_path", lambda: tmp_path / "missing.json")
    monkeypatch.setattr(locale, "getdefaultlocale", lambda: ("ja_JP", "UTF-8"))

    assert locales.LocaleManager().current_language == "ja"


@pytest.mark.parametrize(
    ("saved", "system", "expected"),
    [(None, ("zh_CN", "UTF-8"), "zh"), (None, ("de_DE", "UTF-8"), "en"), (None, (None, None), "en"),
     ({"language": "ko"}, ("zh_CN", "UTF-8"), "ko"), ({}, ("zh_CN", "UTF-8"), "en")],
    ids=["system-zh", "system-unsupported", "system-unknown", "saved", "saved-without-language"],
)
def test_user_language_prefers_the_saved_choice_over_the_system_locale(monkeypatch, tmp_path, saved, system, expected) -> None:
    import json
    import locale

    from jasna import user_settings

    path = tmp_path / "settings.json"
    if saved is not None:
        path.write_text(json.dumps(saved), encoding="utf-8")
    monkeypatch.setattr(user_settings, "get_settings_path", lambda: path)
    monkeypatch.setattr(locale, "getdefaultlocale", lambda: system)

    assert user_settings.user_language() == expected
