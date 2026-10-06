from __future__ import annotations

from dataclasses import fields, replace
from tkinter import TclError
from types import SimpleNamespace
from unittest.mock import MagicMock

import customtkinter as ctk
import pytest

from jasna import os_utils
from jasna.accelerator import AcceleratorVendor
from jasna.gui.models import AppSettings, PresetManager
from jasna.gui.settings_sections.advanced import AdvancedSection
from jasna.gui.settings_sections.basic import BasicSection
from jasna.gui.settings_sections.encoding import EncodingSection
from jasna.gui.settings_sections.image_restoration import ImageRestorationSection
from jasna.gui.settings_sections.post_export import PostExportSection
from jasna.gui.settings_sections.restoration_model import RestorationModelSection
from jasna.gui.settings_sections.secondary import SecondarySection
from jasna.gui.settings_sections.widgets import ValueOptionMenu
from factories import ltx_models


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
        "restoration_model": _FakeWidget("ltx"),
        "ltx_model": _FakeValueMenu({"distilled": "Distilled", "undistilled": "Undistilled"}, "undistilled"),
        "ltx_seed": _FakeWidget(" 123 "),
        "ltx_fast": _FakeWidget(1),
        "ltx_large_canvas": _FakeWidget(0),
        "ltx_trial": _FakeWidget(1),
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
        "amd_upscale_engine": _FakeWidget("amf-sr"),
        "amd_upscale_scale": _FakeWidget("2x"),
        "amd_upscale_algorithm": _FakeWidget("sr1-1"),
        "amd_upscale_sharpness": _FakeWidget("0.5"),
        "amd_upscale_ffmpeg_path": _FakeWidget(" C:\\tools\\ffmpeg.exe "),
        "amd_upscale_timeout": _FakeWidget("90"),
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
        "post_export_video_command": _FakeWidget("remux {output} "),
    }


def _collect_all(widgets: dict) -> dict:
    fake = SimpleNamespace(_widgets=widgets)
    values: dict = {}
    for section in (
        RestorationModelSection,
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
    assert values["post_export_video_command"] == "remux {output}"
    assert values["secondary_restoration"] == "tvai"
    assert values["tvai_scale"] == 2
    assert values["tvai_denoise"] is True
    assert values["rtx_quality"] == "ultra"
    assert values["amd_upscale_engine"] == "amf-sr"
    assert values["amd_upscale_scale"] == 2
    assert values["amd_upscale_algorithm"] == "sr1-1"
    assert values["amd_upscale_sharpness"] == 0.5
    assert values["amd_upscale_ffmpeg_path"] == " C:\\tools\\ffmpeg.exe "
    assert values["amd_upscale_timeout_s"] == 90
    assert values["image_restore_seed"] == 0
    assert values["lut_path"] == "/luts/a.cube"
    assert values["enable_crossfade"] is False
    assert values["scene_detection"] is False
    assert values["retarget_high_fps"] is True
    assert values["fmp4"] is True
    assert values["sharpen_strength"] == 0.35
    assert values["restoration_model"] == "ltx"
    assert values["ltx_model"] == "undistilled"
    assert values["ltx_seed"] == 123
    assert values["ltx_fast"] is True
    assert values["ltx_large_canvas"] is False
    assert values["ltx_trial"] is True


def test_sections_collect_covers_all_widget_backed_appsettings_fields() -> None:
    values = _collect_all(_fake_section_widgets())

    defaults_only = {
        "batch_size",
        "tvai_args",
        "vr_projection",
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

    panel = SettingsPanel(root, PresetManager(), ltx_models=ltx_models(root, tmp_path / "ltx", installed=True))
    try:
        yield panel, next(section for section in panel._sections if isinstance(section, BasicSection))
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


def test_default_encoder_cq_uses_portable_native_default_sentinel() -> None:
    assert AppSettings().encoder_cq is None


def test_nvidia_codec_change_recalls_literal_cq() -> None:
    slider = MagicMock()
    slider.get.return_value = 30
    label = MagicMock()
    section = SimpleNamespace(
        _widgets={"encoder_cq": slider, "encoder_cq_val": label},
        _active_codec="hevc",
        _cq_vendor=AcceleratorVendor.NVIDIA,
        _cq_values={"hevc": 28, "h264": 25, "av1": 35},
        _on_modified=MagicMock(),
    )

    EncodingSection._on_codec_changed(section, "h264")

    assert section._cq_values["hevc"] == 30
    slider.configure.assert_called_once_with(
        from_=1,
        to=51,
        number_of_steps=50,
    )
    slider.set.assert_called_once_with(25)
    label.configure.assert_called_once_with(text="25")
    assert section._active_codec == "h264"

    slider.reset_mock()
    label.reset_mock()
    slider.get.return_value = 26
    EncodingSection._on_codec_changed(section, "hevc")

    assert section._cq_values["h264"] == 26
    slider.set.assert_called_once_with(30)
    label.configure.assert_called_once_with(text="30")


def test_nvidia_av1_codec_change_uses_native_range() -> None:
    slider = MagicMock()
    slider.get.return_value = 28
    label = MagicMock()
    section = SimpleNamespace(
        _widgets={"encoder_cq": slider, "encoder_cq_val": label},
        _active_codec="hevc",
        _cq_vendor=AcceleratorVendor.NVIDIA,
        _cq_values={"hevc": 28, "h264": 25, "av1": 35},
        _on_modified=MagicMock(),
    )

    EncodingSection._on_codec_changed(section, "av1")

    slider.configure.assert_called_once_with(
        from_=1,
        to=63,
        number_of_steps=62,
    )
    slider.set.assert_called_once_with(35)


def test_cq_slider_updates_only_active_codec() -> None:
    label = MagicMock()
    on_modified = MagicMock()
    section = SimpleNamespace(
        _widgets={"encoder_cq_val": label},
        _active_codec="h264",
        _cq_values={"hevc": 28, "h264": 25, "av1": 35},
        _on_modified=on_modified,
    )

    EncodingSection._on_cq_changed(section, 31.0)

    assert section._cq_values == {"hevc": 28, "h264": 31, "av1": 35}
    label.configure.assert_called_once_with(text="31")
    on_modified.assert_called_once_with()


def test_processor_passes_literal_cq_for_forced_codec(monkeypatch) -> None:
    import jasna.accelerator as accelerator
    from jasna.gui.processor import build_job_encoder_settings

    settings = AppSettings(codec="hevc", encoder_cq=28)
    monkeypatch.setattr(
        accelerator,
        "vendor_for_device",
        lambda: AcceleratorVendor.NVIDIA,
    )

    assert build_job_encoder_settings(settings, "h264") == {"cq": 28}


def test_processor_resolves_native_default_for_codec(monkeypatch) -> None:
    import jasna.accelerator as accelerator
    from jasna.gui.processor import build_job_encoder_settings

    settings = AppSettings(codec="h264")
    monkeypatch.setattr(
        accelerator,
        "vendor_for_device",
        lambda: AcceleratorVendor.NVIDIA,
    )

    assert build_job_encoder_settings(settings, "h264") == {"cq": 25}


def test_processor_rejects_cq_in_custom_args(monkeypatch) -> None:
    import jasna.accelerator as accelerator
    from jasna.gui.processor import build_job_encoder_settings

    settings = AppSettings(encoder_cq=28, encoder_custom_args="cq=22")
    monkeypatch.setattr(
        accelerator,
        "vendor_for_device",
        lambda: AcceleratorVendor.NVIDIA,
    )

    with pytest.raises(ValueError, match="CQ.*custom encoder settings"):
        build_job_encoder_settings(settings, "hevc")


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

        panel = SettingsPanel(root, PresetManager(), ltx_models=ltx_models(root, tmp_path / "ltx", installed=True))
        assert panel.get_settings() == replace(AppSettings(), encoder_cq=28)
        assert panel._saved_preset_settings == panel.get_settings()
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


def test_only_queue_wide_post_export_action_stays_editable_while_processing(_basic_section_panel) -> None:
    panel, _basic = _basic_section_panel

    panel.set_enabled(False)

    assert panel._widgets["post_export_action"].cget("state") == "normal"
    assert panel._widgets["post_export_command"].cget("state") == "normal"
    assert panel._widgets["post_export_video_command"].cget("state") == "disabled"
    assert panel._widgets["max_clip_size"].cget("state") == "disabled"


def test_parse_seed_falls_back_to_default() -> None:
    from jasna.gui.settings_sections.restoration_model import parse_seed
    from jasna.session_config import LTX_DEFAULT_SEED

    assert parse_seed(" 42 ") == 42
    assert parse_seed("abc") == LTX_DEFAULT_SEED


@pytest.fixture
def _panel_factory(monkeypatch, tmp_path):
    monkeypatch.setattr(os_utils.sys, "platform", "linux", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    from jasna.gui.settings_panel import SettingsPanel

    try:
        yield lambda installed: SettingsPanel(
            root, PresetManager(), ltx_models=ltx_models(root, tmp_path / "ltx", installed=installed)
        )
    finally:
        root.destroy()


def _select_ltx(panel) -> None:
    panel._model_section._cards["ltx"].radio.invoke()


def test_ltx_hides_standard_only_settings_and_restores_them(_panel_factory) -> None:
    panel = _panel_factory(True)
    secondary = panel._model_aware_sections[2]
    panel.update_idletasks()
    clip_row = panel._widgets["max_clip_size"].master
    overlap_row = panel._widgets["temporal_overlap"].master

    _select_ltx(panel)

    assert panel.get_settings().restoration_model == "ltx"
    assert not clip_row.winfo_manager()
    assert not panel._widgets["compile_basicvsrpp"].master.winfo_manager()
    assert not overlap_row.winfo_manager()
    assert panel._widgets["vr_mode"].master.winfo_manager()
    assert not secondary._section.winfo_manager()
    assert panel._model_section._ltx_options.winfo_manager()

    panel._model_section._cards["basicvsrpp"].radio.invoke()

    assert clip_row.winfo_manager() and overlap_row.winfo_manager()
    assert secondary._section.winfo_manager()
    assert not panel._model_section._ltx_options.winfo_manager()
    order = panel._scroll.pack_slaves()
    assert order.index(secondary._section) == order.index(secondary._previous_section) + 1


def test_ltx_without_model_files_runs_as_a_trial(_panel_factory) -> None:
    panel = _panel_factory(False)

    assert panel._model_section._cards["ltx"].radio.cget("state") == "normal"
    panel._model_section.apply(AppSettings(restoration_model="ltx"))
    settings = panel.get_settings()
    assert (settings.restoration_model, settings.ltx_trial) == ("ltx", True)


def test_gpu_support_disables_ltx_and_shows_fast_mode_on_blackwell(_panel_factory) -> None:
    panel = _panel_factory(True)
    section = panel._model_section
    section.apply(AppSettings(restoration_model="ltx", ltx_fast=True))

    section.set_gpu_support(nvidia=True, blackwell=True)
    assert section._fast_row.winfo_manager()
    assert panel.get_settings().ltx_fast is True

    section.set_gpu_support(nvidia=True, blackwell=False)
    assert not section._fast_row.winfo_manager()
    assert panel.get_settings().ltx_fast is False

    section.set_gpu_support(nvidia=False, blackwell=False)
    assert panel.get_settings().restoration_model == "basicvsrpp"
    assert section._cards["ltx"].radio.cget("state") == "disabled"


def test_ltx_settings_round_trip_through_panel(_panel_factory) -> None:
    panel = _panel_factory(True)
    ltx = replace(AppSettings(), restoration_model="ltx", ltx_seed=99, ltx_large_canvas=True, encoder_cq=28)

    for section in panel._sections:
        section.apply(ltx)

    assert panel.get_settings() == ltx


def test_disabling_settings_keeps_unavailable_ltx_disabled(_panel_factory) -> None:
    panel = _panel_factory(True)
    panel._model_section.set_gpu_support(nvidia=False, blackwell=False)

    panel.set_enabled(False)
    panel.set_enabled(True)

    assert panel._model_section._cards["ltx"].radio.cget("state") == "disabled"
    assert panel._model_section._cards["basicvsrpp"].radio.cget("state") == "normal"
