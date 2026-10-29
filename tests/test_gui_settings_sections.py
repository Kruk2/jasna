from __future__ import annotations

from dataclasses import fields, replace

import pytest

from jasna import os_utils
from jasna.accelerator import AcceleratorVendor
from jasna.gui.models import AppSettings, PresetManager
from jasna.gui.settings_sections.basic import BasicSection
from jasna.gui.settings_sections.encoding import EncodingSection
from jasna.gui.settings_sections.secondary import SecondarySection
from jasna.gui.settings_sections.widgets import ValueOptionMenu
from jasna.gui import ltx_models as ltx_models_module
from factories import ltx_models

pytestmark = pytest.mark.usefixtures("nvidia_vendor")


def test_value_option_menu_maps_values_to_labels_and_falls_back_to_the_first(tk_root) -> None:
    picked: list[str] = []
    menu = ValueOptionMenu(tk_root, options={"none": "なし", "low": "低"}, command=picked.append)

    menu.set_value("low")
    assert (menu.get(), menu.get_value()) == ("低", "low")
    menu.set_value("bogus")
    assert menu.get_value() == "none"
    menu._dropdown_callback("低")
    assert picked == ["low"]


@pytest.fixture
def panel_factory(monkeypatch, tmp_path, tk_root):
    from jasna.gui.settings_panel import SettingsPanel
    from jasna.mosaic import detection_registry

    monkeypatch.setattr(os_utils.sys, "platform", "linux", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(
        detection_registry, "detection_model_choices", lambda *a, **k: ["rfdetr-v6", "rfdetr-v6-large"]
    )
    return lambda installed=True: SettingsPanel(
        tk_root, PresetManager(), ltx_models=ltx_models(tk_root, tmp_path / "ltx", installed=installed)
    )


def _apply(panel, settings: AppSettings) -> None:
    for section in panel._sections:
        section.apply(settings)


_NON_DEFAULT_SETTINGS = AppSettings(
    max_clip_size=120,
    temporal_overlap=6,
    enable_crossfade=False,
    vr_mode="off",
    fp16_mode=False,
    denoise_strength="high",
    denoise_step="after_secondary",
    secondary_restoration="tvai",
    tvai_ffmpeg_path="/opt/tvai/ffmpeg",
    tvai_model="iris-3",
    tvai_scale=2,
    tvai_workers=3,
    tvai_denoise=True,
    rtx_scale=2,
    rtx_quality="ultra",
    rtx_denoise="none",
    rtx_deblur="low",
    detection_model="rfdetr-v6-large",
    detection_score_threshold=0.5,
    max_detection_gap=3,
    min_detection_duration=4,
    scene_detection=False,
    compile_basicvsrpp=False,
    image_restore_steps=30,
    image_restore_strength=0.55,
    image_restore_freeu=False,
    image_restore_seed=7,
    image_restore_variants=2,
    codec="av1",
    encoder_cq=40,
    encoder_custom_args="preset=p5",
    sharpen_strength=0.35,
    lut_path="/luts/a.cube",
    retarget_high_fps=True,
    fmp4=True,
    post_export_action="command",
    post_export_command="echo done",
    post_export_video_command="remux {output}",
    file_conflict="skip",
    working_directory="/scratch",
)


@pytest.mark.parametrize(
    "settings",
    [
        _NON_DEFAULT_SETTINGS,
        replace(AppSettings(), restoration_model="ltx", ltx_seed=99, ltx_large_canvas=True, ltx_less_overlap=False, encoder_cq=28),
    ],
    ids=["standard", "ltx"],
)
def test_settings_round_trip_through_panel(panel_factory, settings) -> None:
    panel = panel_factory()

    _apply(panel, settings)

    assert panel.get_settings() == settings


def test_sections_collect_every_setting_except_runtime_ones(panel_factory) -> None:
    panel = panel_factory()

    collected = {key for section in panel._sections for key in section.collect()}

    runtime_only = {"batch_size", "tvai_args", "vr_projection", "output_folder", "output_pattern"}
    assert collected == {f.name for f in fields(AppSettings)} - runtime_only


@pytest.fixture
def _basic_section_panel(panel_factory):
    panel = panel_factory()
    return panel, next(section for section in panel._sections if isinstance(section, BasicSection))


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


def test_codec_change_switches_cq_range_and_remembers_each_codec_value(panel_factory) -> None:
    panel = panel_factory()
    encoding = next(section for section in panel._sections if isinstance(section, EncodingSection))
    slider = panel._widgets["encoder_cq"]
    encoding.apply(AppSettings(codec="hevc", encoder_cq=30))

    encoding._on_codec_changed("h264")
    assert (slider.cget("to"), slider.get(), panel._widgets["encoder_cq_val"].cget("text")) == (51, 25, "25")
    slider.set(26)
    encoding._on_cq_changed(26.0)

    encoding._on_codec_changed("av1")
    assert (slider.cget("to"), slider.get()) == (63, 35)

    encoding._on_codec_changed("hevc")
    assert slider.get() == 30
    encoding._on_codec_changed("h264")
    assert slider.get() == 26


@pytest.mark.parametrize(
    ("settings", "expected"),
    [(AppSettings(codec="hevc", encoder_cq=28), {"cq": 28}), (AppSettings(codec="h264"), {"cq": 25})],
    ids=["literal-cq", "native-default"],
)
def test_job_encoder_settings_use_literal_or_native_default_cq(monkeypatch, settings, expected) -> None:
    import jasna.accelerator as accelerator
    from jasna.gui.processor import build_job_encoder_settings

    monkeypatch.setattr(accelerator, "vendor_for_device", lambda: AcceleratorVendor.NVIDIA)

    assert build_job_encoder_settings(settings, "h264") == expected


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


def test_settings_panel_get_settings_is_locale_independent(panel_factory, gui_language) -> None:
    gui_language("ja")

    panel = panel_factory()

    assert panel.get_settings() == replace(AppSettings(), encoder_cq=28)
    assert panel._saved_preset_settings == panel.get_settings()


def test_max_clip_size_slider_spans_ten_to_seven_hundred_twenty_in_steps_of_ten(_basic_section_panel) -> None:
    panel, _basic = _basic_section_panel
    slider = panel._widgets["max_clip_size"]

    slider.set(10)
    assert slider.get() == 10
    slider.set(723)
    assert slider.get() == 720
    slider.set(34)
    assert slider.get() == 30


def test_temporal_filter_sliders_stay_below_max_clip_size(_basic_section_panel) -> None:
    panel, basic = _basic_section_panel
    sliders = [panel._widgets[key] for key in ("max_detection_gap", "min_detection_duration")]
    for slider in sliders:
        slider.set(10)

    basic._on_max_clip_size_slider(10)
    assert [(slider.cget("to"), slider.get()) for slider in sliders] == [(9, 9), (9, 9)]
    assert panel._widgets["max_detection_gap_val"].cget("text") == "9"

    basic._on_max_clip_size_slider(90)
    assert [(slider.cget("to"), slider.get()) for slider in sliders] == [(10, 9), (10, 9)]


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


def _select_ltx(panel) -> None:
    panel._model_section._cards["ltx"].radio.invoke()


def test_ltx_hides_standard_only_settings_and_restores_them(panel_factory) -> None:
    panel = panel_factory(True)
    secondary = panel._secondary_section
    panel.update_idletasks()
    clip_row = panel._widgets["max_clip_size"].master
    overlap_row = panel._widgets["temporal_overlap"].master

    _select_ltx(panel)

    assert panel.get_settings().restoration_model == "ltx"
    assert not clip_row.winfo_manager()
    assert not panel._widgets["compile_basicvsrpp"].master.winfo_manager()
    assert not overlap_row.winfo_manager()
    assert panel._widgets["vr_mode"].master.winfo_manager()
    assert panel._widgets["max_detection_gap"].master.winfo_manager()
    assert panel._widgets["min_detection_duration"].master.winfo_manager()
    assert panel._model_section._ltx_options.winfo_manager()

    panel._model_section._cards["basicvsrpp"].radio.invoke()

    assert clip_row.winfo_manager() and overlap_row.winfo_manager()
    assert not panel._model_section._ltx_options.winfo_manager()


def test_ltx_offers_only_rtx_super_res_unless_auto_split(panel_factory) -> None:
    panel = panel_factory(True)
    radios = panel._secondary_section._engine_radios
    _select_ltx(panel)

    assert panel._secondary_section._section.winfo_manager()
    assert radios["tvai"].cget("state") == panel._widgets["rtx_scale"].cget("state") == "disabled"
    assert radios["rtx-super-res"].cget("state") == "normal"

    panel._widgets["ltx_auto_split"]._toggle()
    assert radios["tvai"].cget("state") == panel._widgets["rtx_scale"].cget("state") == "normal"


def test_amd_secondary_choices_and_busy_state(panel_factory, monkeypatch, tmp_path) -> None:
    from jasna import engine_paths

    model = tmp_path / "unet-4x.onnx"
    model.touch()
    monkeypatch.setattr(engine_paths, "UNET4X_ONNX_PATH", model)
    panel = panel_factory(True)
    panel.set_gpu_support(vendor=AcceleratorVendor.AMD, blackwell=False)
    radios = panel._secondary_section._engine_radios
    radios["unet-4x"].invoke()
    assert panel.get_settings().secondary_restoration == "unet-4x"
    assert radios["tvai"].cget("state") == radios["rtx-super-res"].cget("state") == "disabled"

    panel._secondary_section.set_enabled(False)
    panel.set_gpu_support(vendor=AcceleratorVendor.NVIDIA, blackwell=False)
    assert radios["unet-4x"].cget("state") == "disabled"
    panel._secondary_section.set_enabled(True)
    assert radios["tvai"].cget("state") == "normal"


def test_ltx_without_model_files_runs_as_a_trial(panel_factory, monkeypatch) -> None:
    monkeypatch.setattr(ltx_models_module, "protection_available", lambda: False)
    panel = panel_factory(False)

    assert panel._model_section._cards["ltx"].radio.cget("state") == "normal"
    panel._model_section.apply(AppSettings(restoration_model="ltx"))
    settings = panel.get_settings()
    assert (settings.restoration_model, settings.ltx_trial) == ("ltx", True)


def test_gpu_support_disables_ltx_and_shows_fast_mode_on_blackwell(panel_factory) -> None:
    panel = panel_factory(True)
    section = panel._model_section
    section.apply(AppSettings(restoration_model="ltx", ltx_fast=True))

    section.set_gpu_support(vendor=AcceleratorVendor.NVIDIA, blackwell=True)
    assert section._fast_row.winfo_manager()
    assert panel.get_settings().ltx_fast is True

    section.set_gpu_support(vendor=AcceleratorVendor.NVIDIA, blackwell=False)
    assert not section._fast_row.winfo_manager()
    assert panel.get_settings().ltx_fast is False

    section.set_gpu_support(vendor=AcceleratorVendor.CPU, blackwell=False)
    assert panel.get_settings().restoration_model == "basicvsrpp"
    assert section._cards["ltx"].radio.cget("state") == "disabled"


@pytest.mark.parametrize("downloadable", [True, False])
def test_amd_without_source_models_offers_downloads_or_a_locked_trial(panel_factory, monkeypatch, downloadable) -> None:
    from jasna.gui.locales import t

    monkeypatch.setattr(ltx_models_module, "protection_available", lambda: downloadable)
    panel = panel_factory(False)
    section = panel._model_section
    section.set_gpu_support(vendor=AcceleratorVendor.AMD, blackwell=False)

    _select_ltx(panel)

    settings = panel.get_settings()
    assert (settings.restoration_model, settings.ltx_trial) == ("ltx", not downloadable)
    assert section._widgets["ltx_trial"].cget("state") == ("normal" if downloadable else "disabled")
    if downloadable:
        assert not section._trial_status.winfo_manager()
    else:
        assert section._trial_status.cget("text") == t("model_ltx_trial_only")
    assert section.unavailable_reason() == ("model_ltx_not_downloaded" if downloadable else None)


def test_amd_runs_installed_source_models_without_the_trial(panel_factory) -> None:
    panel = panel_factory(True)
    section = panel._model_section
    section.set_gpu_support(vendor=AcceleratorVendor.AMD, blackwell=False)

    _select_ltx(panel)

    assert panel.get_settings().ltx_trial is False
    assert section.unavailable_reason() is None


def test_processing_locks_secondary_controls_and_keeps_missing_unet4x_disabled(
    panel_factory, monkeypatch, tmp_path, caplog
) -> None:
    from jasna import engine_paths

    monkeypatch.setattr(engine_paths, "UNET4X_ONNX_PATH", tmp_path / "missing.onnx")
    monkeypatch.setattr(engine_paths, "UNET4X_ONNX_ENC_PATH", tmp_path / "missing.onnx.enc")
    panel = panel_factory(True)
    secondary = next(section for section in panel._sections if isinstance(section, SecondarySection))
    controls = [*secondary._engine_radios.values(), secondary._tvai_denoise_switch]

    with caplog.at_level("DEBUG", logger="jasna.gui.settings_panel"):
        panel.set_enabled(False)
        locked = [control.cget("state") for control in controls]
        panel.set_enabled(True)

    assert locked == ["disabled"] * len(controls)
    assert {value: radio.cget("state") for value, radio in secondary._engine_radios.items()} == {
        "none": "normal",
        "unet-4x": "disabled",
        "tvai": "normal",
        "rtx-super-res": "normal",
    }
    assert secondary._tvai_denoise_switch.cget("state") == "normal"
    assert "does not support" not in caplog.text


def test_disabling_settings_keeps_unavailable_ltx_disabled(panel_factory) -> None:
    panel = panel_factory(True)
    panel._model_section.set_gpu_support(vendor=AcceleratorVendor.CPU, blackwell=False)

    panel.set_enabled(False)
    panel.set_enabled(True)

    assert panel._model_section._cards["ltx"].radio.cget("state") == "disabled"
    assert panel._model_section._cards["basicvsrpp"].radio.cget("state") == "normal"
