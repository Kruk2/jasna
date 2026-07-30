from __future__ import annotations

from dataclasses import fields
from tkinter import TclError
from types import SimpleNamespace

import customtkinter as ctk
import pytest

from jasna import os_utils
from jasna.gui.models import AppSettings
from jasna.gui.settings_sections.advanced import AdvancedSection
from jasna.gui.settings_sections.basic import BasicSection
from jasna.gui.settings_sections.encoding import EncodingSection
from jasna.gui.settings_sections.image_restoration import ImageRestorationSection
from jasna.gui.settings_sections.post_export import PostExportSection
from jasna.gui.settings_sections.secondary import SecondarySection
from jasna.gui.settings_sections.widgets import ValueOptionMenu


class _FakeValueMenu:
    def __init__(self, options: dict[str, str], value: str):
        self._value_to_label = dict(options)
        self._label_to_value = {label: v for v, label in options.items()}
        self._label = self._value_to_label[value]

    def get(self) -> str:
        return self._label

    def set(self, label: str) -> None:
        self._label = label

    get_value = ValueOptionMenu.get_value
    set_value = ValueOptionMenu.set_value


class _FakeWidget:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


def test_value_option_menu_maps_between_values_and_labels() -> None:
    menu = _FakeValueMenu({"none": "なし", "low": "低"}, "none")

    assert menu.get_value() == "none"
    menu.set_value("low")
    assert menu.get() == "低"
    assert menu.get_value() == "low"


def test_value_option_menu_falls_back_to_first_option_for_unknown_value() -> None:
    menu = _FakeValueMenu({"auto_rename": "Rename", "overwrite": "Overwrite"}, "overwrite")

    menu.set_value("bogus")

    assert menu.get_value() == "auto_rename"


def _fake_section_widgets() -> dict:
    return {
        "max_clip_size": _FakeWidget(90),
        "fp16_mode": _FakeWidget(1),
        "detection_model": _FakeWidget("rfdetr-v6"),
        "detection_score_threshold": _FakeWidget(0.35),
        "compile_basicvsrpp": _FakeWidget(1),
        "file_conflict": _FakeValueMenu({"auto_rename": "A", "overwrite": "B", "skip": "C"}, "skip"),
        "temporal_overlap": _FakeWidget(8),
        "max_detection_gap": _FakeWidget(2),
        "min_detection_duration": _FakeWidget(2),
        "scene_detection": _FakeWidget(0),
        "enable_crossfade": _FakeWidget(0),
        "vr_mode": _FakeValueMenu({"auto": "自動", "off": "オフ"}, "off"),
        "denoise_strength": _FakeValueMenu({"none": "なし", "high": "高"}, "high"),
        "denoise_step": _FakeValueMenu({"after_primary": "一", "after_secondary": "二"}, "after_secondary"),
        "secondary_var": _FakeWidget("tvai"),
        "tvai_ffmpeg_path": _FakeWidget("/opt/tvai/ffmpeg"),
        "tvai_model": _FakeWidget("iris-3"),
        "tvai_scale": _FakeWidget("2x"),
        "tvai_workers": _FakeWidget(3),
        "tvai_denoise": _FakeWidget(1),
        "rtx_scale": _FakeWidget("4x"),
        "rtx_quality": _FakeWidget("Ultra"),
        "rtx_denoise": _FakeWidget("None"),
        "rtx_deblur": _FakeWidget("Low"),
        "image_restore_steps": _FakeWidget(30),
        "image_restore_strength": _FakeWidget(0.55),
        "image_restore_freeu": _FakeWidget(0),
        "image_restore_seed": _FakeWidget("not-a-number"),
        "image_restore_variants": _FakeWidget(2),
        "codec": _FakeValueMenu({"hevc": "HEVC (H.265)", "av1": "AV1"}, "av1"),
        "encoder_cq": _FakeWidget(29),
        "encoder_custom_args": _FakeWidget("cq=22"),
        "sharpen_strength": _FakeWidget(0.35),
        "retarget_high_fps": _FakeWidget(1),
        "fmp4": _FakeWidget(1),
        "lut_path": _FakeWidget(" /luts/a.cube "),
        "working_directory": _FakeWidget(""),
        "post_export_action": _FakeValueMenu({"none": "何も", "command": "コマンド"}, "command"),
        "post_export_command": _FakeWidget("echo done "),
    }


def _collect_all(widgets: dict) -> dict:
    fake = SimpleNamespace(_widgets=widgets)
    values: dict = {}
    for section in (
        BasicSection,
        AdvancedSection,
        SecondarySection,
        ImageRestorationSection,
        EncodingSection,
        PostExportSection,
    ):
        values.update(section.collect(fake))
    return values


def test_sections_collect_internal_values_without_translation_lookups() -> None:
    values = _collect_all(_fake_section_widgets())

    assert values["file_conflict"] == "skip"
    assert values["vr_mode"] == "off"
    assert values["denoise_strength"] == "high"
    assert values["denoise_step"] == "after_secondary"
    assert values["codec"] == "av1"
    assert values["post_export_action"] == "command"
    assert values["post_export_command"] == "echo done"
    assert values["secondary_restoration"] == "tvai"
    assert values["tvai_scale"] == 2
    assert values["tvai_denoise"] is True
    assert values["rtx_quality"] == "ultra"
    assert values["image_restore_seed"] == 0
    assert values["lut_path"] == "/luts/a.cube"
    assert values["enable_crossfade"] is False
    assert values["scene_detection"] is False
    assert values["retarget_high_fps"] is True
    assert values["fmp4"] is True
    assert values["sharpen_strength"] == 0.35


def test_sections_collect_covers_all_widget_backed_appsettings_fields() -> None:
    values = _collect_all(_fake_section_widgets())

    defaults_only = {
        "batch_size",
        "tvai_args",
        "vr_projection",
        "output_same_as_input",
        "output_folder",
        "output_pattern",
    }
    expected = {f.name for f in fields(AppSettings)} - defaults_only
    assert set(values) == expected

    settings = AppSettings(batch_size=4, **values)
    assert settings.codec == "av1"


@pytest.fixture
def _basic_section_panel(monkeypatch, tmp_path):
    monkeypatch.setattr(os_utils.sys, "platform", "linux", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    from jasna.mosaic import detection_registry

    monkeypatch.setattr(
        detection_registry,
        "detection_model_choices",
        lambda *a, **k: ["rfdetr-v6", "rfdetr-v6-large"],
    )

    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    from jasna.gui.settings_panel import SettingsPanel

    panel = SettingsPanel(root)
    try:
        yield panel, panel._sections[0]
    finally:
        root.destroy()


def test_switching_detection_model_applies_recommended_threshold(_basic_section_panel) -> None:
    panel, basic = _basic_section_panel

    basic._on_detection_model_changed("rfdetr-v6-large")
    assert panel._widgets["detection_score_threshold"].get() == pytest.approx(0.40)

    basic._on_detection_model_changed("rfdetr-v6")
    assert panel._widgets["detection_score_threshold"].get() == pytest.approx(0.35)


def test_apply_reverts_missing_model_to_default_recommended(_basic_section_panel) -> None:
    panel, basic = _basic_section_panel

    basic.apply(AppSettings(detection_model="rfdetr-v5", detection_score_threshold=0.6))

    assert panel._widgets["detection_model"].get() == "rfdetr-v6"
    assert panel._widgets["detection_score_threshold"].get() == pytest.approx(0.35)


def test_apply_keeps_installed_model_and_threshold(_basic_section_panel) -> None:
    panel, basic = _basic_section_panel

    basic.apply(AppSettings(detection_model="rfdetr-v6-large", detection_score_threshold=0.55))

    assert panel._widgets["detection_model"].get() == "rfdetr-v6-large"
    assert panel._widgets["detection_score_threshold"].get() == pytest.approx(0.55)


def test_default_encoder_cq_matches_hevc_encoder_default() -> None:
    # The GUI always sends an explicit cq, so a stale default here silently
    # overrides the measured encoder default for every GUI job.
    from jasna.media.video_encoder import DEFAULT_ENCODER_OPTIONS

    assert str(AppSettings().encoder_cq) == DEFAULT_ENCODER_OPTIONS["cq"]


def test_settings_panel_get_settings_is_locale_independent(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(os_utils.sys, "platform", "linux", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    from jasna.gui.locales import get_locale

    monkeypatch.setattr(get_locale(), "_current_lang", "ja")

    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    try:
        from jasna.gui.settings_panel import SettingsPanel

        panel = SettingsPanel(root)
        assert panel.get_settings() == AppSettings()
    finally:
        root.destroy()


def test_max_clip_size_slider_spans_ten_to_seven_hundred_twenty() -> None:
    """The sub-engines stopped depending on the clip length, so the slider is no
    longer capped at the largest clip an engine had been compiled for."""
    from jasna.gui.settings_sections import basic

    captured: dict = {}

    class _Slider:
        def __init__(self, _parent, **kwargs):
            captured.update(kwargs)

        def pack(self, **_kwargs) -> None:
            pass

        def set(self, _value) -> None:
            pass

    basic.build_max_clip_size_slider(_Slider, None, lambda _value: None)

    assert captured["from_"] == 10
    assert captured["to"] == 720
    assert (captured["to"] - captured["from_"]) % captured["number_of_steps"] == 0
    assert (captured["to"] - captured["from_"]) // captured["number_of_steps"] == 10
