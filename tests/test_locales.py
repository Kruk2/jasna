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


@pytest.mark.parametrize("lang", sorted(TRANSLATIONS))
@pytest.mark.parametrize("key", ["btn_add_files", "btn_clear"])
def test_queue_button_text_does_not_depend_on_emoji_fonts(lang: str, key: str) -> None:
    assert not TRANSLATIONS[lang][key].startswith(("📁", "🗑"))


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

_SEGMENT_EDITOR_KEYS = {
    "segments_full_video",
    "segments_media_info",
    "segments_previous_frame",
    "segments_next_frame",
    "segments_ranges",
    "segments_undo",
    "segments_redo",
    "segments_new_range",
    "segments_clear_all",
    "segments_clear_confirm",
    "segments_mark_in_short",
    "segments_mark_out_short",
    "segments_mark_in_hint",
    "segments_mark_out_hint",
    "segments_add_range",
    "segments_update_range",
    "segments_apply",
    "segments_invalid_range",
    "segments_time_out_of_bounds",
    "segments_merged",
    "segments_drag_hint",
    "segments_range_row",
    "segments_delete_range",
    "segments_zoom_out",
    "segments_zoom_in",
    "segments_fit",
    "segments_fit_hint",
    "segments_workload_full",
    "segments_workload",
    "segments_analysis_failed",
    "segments_smart_render_unavailable",
    "segments_smart_render_range_too_short",
    "segments_smart_render_before_first_keyframe",
    "segments_smart_render_whole_video",
    "segments_discard_title",
    "segments_discard_changes",
    "segments_edit_tooltip",
    "segments_summary_percent",
    "segments_scan_title",
    "segments_scan_subtitle",
    "segments_scan",
    "segments_scan_again",
    "segments_scan_stop",
    "segments_scan_stopping",
    "segments_scan_help_button",
    "segments_scan_model",
    "segments_scan_model_hint",
    "segments_scan_threshold",
    "segments_scan_threshold_hint",
    "segments_scan_interval",
    "segments_scan_interval_hint",
    "segments_scan_frequency_every_frame",
    "segments_scan_frequency_quarter",
    "segments_scan_frequency_half",
    "segments_scan_frequency_one",
    "segments_scan_frequency_two",
    "segments_scan_add",
    "segments_scan_add_hint",
    "segments_scan_overlay",
    "segments_scan_overlay_hint",
    "segments_scan_help_hint",
    "segments_scan_help_title",
    "segments_scan_help_body",
    "segments_scan_progress",
    "segments_scan_stopped",
    "segments_scan_low_vram",
    "segments_scan_low_vram_short",
    "segments_scan_failed",
    "segments_scan_mask_failed",
    "segments_scan_none",
    "segments_scan_result",
    "segments_scan_result_partial",
    "segments_scan_all_added",
    "segments_scan_model_changed",
    "segments_timeline_title",
    "segments_timeline_hint",
    "segments_legend_selected",
    "segments_legend_detected",
    "segments_legend_playhead",
}


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


@pytest.mark.parametrize("lang", sorted(TRANSLATIONS))
def test_all_languages_define_segment_editor_keys(lang: str) -> None:
    missing = _SEGMENT_EDITOR_KEYS - TRANSLATIONS[lang].keys()
    assert not missing, f"{lang} missing segment-editor keys: {sorted(missing)}"


def test_segment_editor_copy_hides_smart_render_assembly_details() -> None:
    translations = TRANSLATIONS["en"]
    copy = " ".join(
        translations[key]
        for key in _SEGMENT_EDITOR_KEYS
        if key in translations
    ).lower()

    assert "silent preview" not in copy
    assert "transition" not in copy
    assert "cut point" not in copy


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


_MASK_FEEDBACK_KEYS = [
    "segments_suggest_mask",
    "segments_suggest_mask_hint",
    "mask_editor_title",
    "mask_editor_instructions",
    "mask_editor_anonymous_note",
    "mask_editor_undo_point",
    "mask_editor_erase_all",
    "mask_editor_submit",
    "mask_editor_hide_shapes",
    "mask_editor_show_shapes",
    "mask_editor_quick_help",
    "mask_editor_opacity",
    "mask_editor_loading_frame",
    "mask_editor_frame_failed",
    "mask_feedback_uploaded",
    "mask_feedback_upload_failed",
]


@pytest.mark.parametrize("lang", sorted(TRANSLATIONS))
@pytest.mark.parametrize("key", _MASK_FEEDBACK_KEYS)
def test_all_languages_define_mask_feedback_keys(lang: str, key: str) -> None:
    assert TRANSLATIONS[lang][key].strip()
