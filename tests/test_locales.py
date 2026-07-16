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


_LICENSE_KEYS = {
    "supporter_title",
    "supporter_blurb",
    "supporter_perks",
    "license_email_placeholder",
    "license_key_placeholder",
    "license_activate",
    "license_active",
    "license_crypto_info",
    "license_chip_inactive",
    "license_chip_active",
}


_IMAGE_RESTORE_TOOLTIP_KEYS = {
    "tip_image_restore_steps",
    "tip_image_restore_strength",
    "tip_image_restore_variants",
    "tip_image_restore_seed",
    "tip_image_restore_freeu",
}

_POST_EXPORT_KEYS = {
    "section_post_export_action",
    "post_export_action",
    "post_export_none",
    "post_export_shutdown",
    "post_export_command",
    "post_export_command_placeholder",
    "tip_post_export_action",
    "error_post_export_command_required",
}

_FRAME_RATE_KEYS = {"retarget_high_fps", "tip_retarget_high_fps"}


@pytest.mark.parametrize("lang", sorted(TRANSLATIONS))
def test_all_languages_define_license_keys(lang: str) -> None:
    missing = _LICENSE_KEYS - TRANSLATIONS[lang].keys()
    assert not missing, f"{lang} missing license keys: {sorted(missing)}"


@pytest.mark.parametrize("lang", sorted(TRANSLATIONS))
def test_all_languages_define_image_restore_tooltips(lang: str) -> None:
    missing = _IMAGE_RESTORE_TOOLTIP_KEYS - TRANSLATIONS[lang].keys()
    assert not missing, f"{lang} missing image restore tooltip keys: {sorted(missing)}"


@pytest.mark.parametrize("lang", sorted(TRANSLATIONS))
def test_all_languages_define_support_button_labels(lang: str) -> None:
    translations = TRANSLATIONS[lang]
    assert translations["bmc_support"].strip()
    # Unifans is a brand name — kept verbatim across every locale.
    assert translations["unifans_support"] == "Unifans"


@pytest.mark.parametrize("lang", sorted(TRANSLATIONS))
def test_all_languages_define_post_export_keys(lang: str) -> None:
    missing = _POST_EXPORT_KEYS - TRANSLATIONS[lang].keys()
    assert not missing, f"{lang} missing post-export keys: {sorted(missing)}"


@pytest.mark.parametrize("lang", sorted(TRANSLATIONS))
def test_all_languages_define_frame_rate_retarget_keys(lang: str) -> None:
    missing = _FRAME_RATE_KEYS - TRANSLATIONS[lang].keys()
    assert not missing, f"{lang} missing frame-rate keys: {sorted(missing)}"


def test_english_activation_copy_uses_app_activation_language() -> None:
    en = TRANSLATIONS["en"]
    assert en["supporter_title"] == "Activate Jasna"
    assert en["license_chip_inactive"] == "Activate Jasna"
    assert en["secondary_unet_4x_hint"] == "high-quality, fast"
    assert "supporter" not in en["supporter_title"].lower()
    assert "license" not in en["supporter_title"].lower()


@pytest.mark.parametrize("lang", sorted(TRANSLATIONS))
def test_codec_tooltip_covers_all_three_codecs(lang: str) -> None:
    tip = TRANSLATIONS[lang].get("tip_codec")
    if tip is None:
        pytest.skip(f"{lang} has no tip_codec override")
    assert "AV1" in tip
    assert "H.264" in tip
    assert "H.265" in tip


@pytest.mark.parametrize("lang", sorted(TRANSLATIONS))
def test_cq_tooltip_mentions_codec_relative_values(lang: str) -> None:
    tip = TRANSLATIONS[lang].get("tip_encoder_cq")
    if tip is None:
        pytest.skip(f"{lang} has no tip_encoder_cq override")
    assert "CQ" in tip
    assert "AV1" in tip
    assert "29" in tip


@pytest.mark.parametrize("lang", sorted(TRANSLATIONS))
def test_activation_benefits_include_sd15_image_restoration(lang: str) -> None:
    perks = TRANSLATIONS[lang]["supporter_perks"]
    assert "UNet 4x" in perks
    assert "SD 1.5" in perks
