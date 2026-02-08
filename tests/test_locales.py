from __future__ import annotations

import re

import pytest

from jasna.gui.locales import TRANSLATIONS

_FULL_LOCALES = ["zh", "ja"]
_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


@pytest.mark.parametrize("lang", _FULL_LOCALES)
def test_locale_has_all_en_keys(lang: str) -> None:
    en_keys = set(TRANSLATIONS["en"].keys())
    lang_keys = set(TRANSLATIONS[lang].keys())

    missing = en_keys - lang_keys
    assert not missing, f"{lang} is missing translation keys: {sorted(missing)}"


@pytest.mark.parametrize("lang", _FULL_LOCALES)
def test_locale_has_no_extra_keys(lang: str) -> None:
    en_keys = set(TRANSLATIONS["en"].keys())
    lang_keys = set(TRANSLATIONS[lang].keys())

    extra = lang_keys - en_keys
    assert not extra, f"{lang} has extra keys not in English (en): {sorted(extra)}"


@pytest.mark.parametrize("lang", _FULL_LOCALES)
def test_locale_values_are_not_empty(lang: str) -> None:
    empty = [key for key, value in TRANSLATIONS[lang].items() if not value.strip()]
    assert not empty, f"{lang} has empty values for keys: {sorted(empty)}"


@pytest.mark.parametrize("lang", _FULL_LOCALES)
def test_format_placeholders_match(lang: str) -> None:
    mismatches: list[str] = []
    for key in TRANSLATIONS["en"]:
        en_val = TRANSLATIONS["en"][key]
        lang_val = TRANSLATIONS[lang].get(key)
        if lang_val is None:
            continue

        en_placeholders = set(_PLACEHOLDER_RE.findall(en_val))
        lang_placeholders = set(_PLACEHOLDER_RE.findall(lang_val))

        if en_placeholders != lang_placeholders:
            mismatches.append(
                f"  {key}: en={sorted(en_placeholders)}, {lang}={sorted(lang_placeholders)}"
            )

    assert not mismatches, (
        f"Format placeholder mismatch between en and {lang}:\n" + "\n".join(mismatches)
    )
