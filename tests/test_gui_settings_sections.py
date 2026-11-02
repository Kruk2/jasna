from __future__ import annotations

from dataclasses import fields, replace

import pytest

from jasna import os_utils
from jasna.accelerator import AcceleratorVendor
from jasna.gui.models import AppSettings, PresetManager
from jasna.gui.settings_sections.detection import DetectionSection
from jasna.gui.settings_sections.encoding import EncodingSection
from jasna.gui.settings_sections.expert import ExpertSection
from jasna.gui.settings_sections.image_restoration import ImageRestorationSection
from jasna.gui.settings_sections.post_export import PostExportSection
from jasna.gui.settings_sections.processing import ProcessingSection
from jasna.gui.settings_sections.restoration_model import RestorationModelSection
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
    vr_mode="off",
    fp16_mode=False,
    denoise_strength="high",
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
        replace(AppSettings(), restoration_model="ltx", ltx_seed=99, ltx_large_canvas=True, ltx_overlap=33, encoder_cq=28),
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


def _section(panel, section_type):
    return next(section for section in panel._sections if isinstance(section, section_type))


@pytest.fixture
def _detection_section_panel(panel_factory):
    panel = panel_factory()
    return panel, _section(panel, DetectionSection)


@pytest.fixture
def _expert_section_panel(panel_factory):
    panel = panel_factory()
    return panel, _section(panel, ExpertSection)


def test_switching_detection_model_applies_recommended_threshold(_detection_section_panel) -> None:
    panel, detection = _detection_section_panel

    detection._on_detection_model_changed("rfdetr-v6-large")
    assert panel._widgets["detection_score_threshold"].get() == pytest.approx(0.40)

    detection._on_detection_model_changed("rfdetr-v6")
    assert panel._widgets["detection_score_threshold"].get() == pytest.approx(0.35)


def test_apply_reverts_missing_model_to_default_recommended(_detection_section_panel) -> None:
    panel, detection = _detection_section_panel

    detection.apply(AppSettings(detection_model="rfdetr-v5", detection_score_threshold=0.6))

    assert panel._widgets["detection_model"].get() == "rfdetr-v6"
    assert panel._widgets["detection_score_threshold"].get() == pytest.approx(0.35)


def test_apply_keeps_installed_model_and_threshold(_detection_section_panel) -> None:
    panel, detection = _detection_section_panel

    detection.apply(AppSettings(detection_model="rfdetr-v6-large", detection_score_threshold=0.55))

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


def test_max_clip_size_slider_spans_ten_to_seven_hundred_twenty_in_steps_of_ten(_expert_section_panel) -> None:
    panel, _expert = _expert_section_panel
    slider = panel._widgets["max_clip_size"]

    slider.set(10)
    assert slider.get() == 10
    slider.set(723)
    assert slider.get() == 720
    slider.set(34)
    assert slider.get() == 30


def test_temporal_filter_sliders_stay_below_max_clip_size(_expert_section_panel) -> None:
    panel, expert = _expert_section_panel
    sliders = [panel._widgets[key] for key in ("max_detection_gap", "min_detection_duration")]
    for slider in sliders:
        slider.set(10)

    expert._on_max_clip_size_slider(10)
    assert [(slider.cget("to"), slider.get()) for slider in sliders] == [(9, 9), (9, 9)]
    assert panel._widgets["max_detection_gap_val"].cget("text") == "9"

    expert._on_max_clip_size_slider(90)
    assert [(slider.cget("to"), slider.get()) for slider in sliders] == [(10, 9), (10, 9)]


def test_only_queue_wide_post_export_action_stays_editable_while_processing(panel_factory) -> None:
    panel = panel_factory()

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


def _shown(widget) -> bool:
    return bool(widget.winfo_manager())


def _select_ltx(panel) -> None:
    panel._model_section._cards["ltx"].radio.invoke()


def test_ltx_hides_standard_only_settings_and_restores_them(panel_factory) -> None:
    panel = panel_factory(True)
    panel.set_view("advanced")
    processing = _section(panel, ProcessingSection).frame
    clip_row = panel._widgets["max_clip_size"].master
    ltx_overlap_row = panel._widgets["ltx_overlap"].master
    assert _shown(processing) and _shown(clip_row) and not _shown(ltx_overlap_row)

    _select_ltx(panel)

    assert panel.get_settings().restoration_model == "ltx"
    assert not _shown(processing)
    assert not _shown(clip_row)
    assert not _shown(panel._widgets["compile_basicvsrpp"].master)
    assert not _shown(panel._widgets["scene_detection"].master.master)
    assert _shown(ltx_overlap_row)
    assert _shown(panel._widgets["fp16_mode"].master)
    assert _shown(panel._widgets["max_detection_gap"].master)
    assert _shown(panel._model_section._ltx_options)

    panel._widgets["ltx_auto_split"]._toggle()
    assert _shown(processing) and _shown(clip_row) and _shown(ltx_overlap_row)
    panel._widgets["ltx_auto_split"]._toggle()
    assert not _shown(clip_row)

    panel._model_section._cards["basicvsrpp"].radio.invoke()

    assert _shown(processing) and _shown(clip_row) and not _shown(ltx_overlap_row)
    assert not _shown(panel._model_section._ltx_options)


def test_ltx_offers_only_rtx_super_res_unless_auto_split(panel_factory) -> None:
    panel = panel_factory(True)
    radios = panel._secondary_section._engine_radios
    _select_ltx(panel)
    panel.set_view("advanced")
    panel.show_tab("upscale")

    assert panel._secondary_section.frame.winfo_manager()
    assert radios["tvai"].cget("state") == panel._widgets["rtx_scale"].cget("state") == "disabled"
    assert radios["rtx-super-res"].cget("state") == "normal"
    assert panel._secondary_section._ltx_only_note.winfo_manager()

    panel._widgets["ltx_auto_split"]._toggle()
    assert radios["tvai"].cget("state") == panel._widgets["rtx_scale"].cget("state") == "normal"
    assert not panel._secondary_section._ltx_only_note.winfo_manager()


@pytest.mark.parametrize("auto_split,expected", [(False, "none"), (True, "unet-4x")])
def test_ltx_clears_an_unavailable_secondary_restorer(panel_factory, monkeypatch, tmp_path, auto_split, expected) -> None:
    from jasna import engine_paths

    model = tmp_path / "unet-4x.onnx"
    model.touch()
    monkeypatch.setattr(engine_paths, "UNET4X_ONNX_PATH", model)
    panel = panel_factory(True)
    if auto_split:
        panel._widgets["ltx_auto_split"]._toggle()
    panel._secondary_section._engine_radios["unet-4x"].invoke()

    _select_ltx(panel)

    assert panel.get_settings().secondary_restoration == expected


def test_applying_ltx_with_an_unavailable_secondary_restorer_clears_it(panel_factory, monkeypatch, tmp_path) -> None:
    from jasna import engine_paths

    model = tmp_path / "unet-4x.onnx"
    model.touch()
    monkeypatch.setattr(engine_paths, "UNET4X_ONNX_PATH", model)
    panel = panel_factory(True)

    _apply(panel, AppSettings(restoration_model="ltx", secondary_restoration="unet-4x"))

    assert panel.get_settings().secondary_restoration == "none"


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
    panel.set_view("advanced")
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


def test_simple_view_hides_tuning_rows_and_keeps_their_values(panel_factory) -> None:
    panel = panel_factory()
    panel.set_view("advanced")
    _apply(panel, _NON_DEFAULT_SETTINGS)

    panel.set_view("simple")

    assert panel.get_settings() == _NON_DEFAULT_SETTINGS
    assert _shown(panel._widgets["codec"].master) and _shown(panel._widgets["file_conflict"].master)
    assert _shown(panel._widgets["post_export_command"].master)
    assert not _shown(panel._widgets["sharpen_strength"].master)
    assert not _shown(panel._widgets["post_export_video_command"])
    assert not _shown(_section(panel, DetectionSection).frame)
    assert not _shown(panel._tab_bar)

    panel.set_view("advanced")
    panel.show_tab("output")

    assert _shown(panel._widgets["sharpen_strength"].master)
    assert _shown(panel._widgets["post_export_video_command"])


def test_simple_view_keeps_the_ltx_model_choice_and_hides_its_tuning(panel_factory) -> None:
    panel = panel_factory(True)
    _select_ltx(panel)

    panel.set_view("simple")
    assert _shown(panel._widgets["ltx_model"].master)
    assert not _shown(panel._widgets["ltx_seed"].master)
    assert not _shown(panel._widgets["ltx_auto_split"].master)

    panel.set_view("advanced")
    assert _shown(panel._widgets["ltx_seed"].master)
    assert _shown(panel._widgets["ltx_auto_split"].master)


@pytest.mark.parametrize(
    ("tab", "expected"),
    [
        ("model", [RestorationModelSection, ProcessingSection]),
        ("detection", [DetectionSection]),
        ("upscale", [SecondarySection]),
        ("output", [EncodingSection, PostExportSection]),
        ("images", [ImageRestorationSection]),
        ("expert", [ExpertSection]),
    ],
)
def test_advanced_view_shows_only_the_chosen_tab(panel_factory, tab, expected) -> None:
    panel = panel_factory()
    panel.set_view("advanced")

    panel.show_tab(tab)

    assert _shown(panel._tab_bar)
    assert [type(section) for section in panel._sections if _shown(section.frame)] == expected


def test_the_settings_view_is_remembered(panel_factory) -> None:
    assert not _shown(panel_factory()._tab_bar)

    panel_factory().set_view("advanced")

    assert _shown(panel_factory()._tab_bar)


@pytest.mark.parametrize(
    ("view", "chosen", "offered"),
    [
        ("simple", "distilled2", ["distilled2", "distilled4", "distilled"]),
        ("simple", "undistilled", ["distilled2", "distilled4", "distilled", "undistilled"]),
        ("advanced", "distilled2", ["distilled1", "distilled2", "distilled4", "distilled", "undistilled"]),
    ],
)
def test_simple_view_offers_three_ltx_models_and_keeps_another_chosen_one(panel_factory, view, chosen, offered) -> None:
    from jasna.gui.locales import t

    panel = panel_factory(True)
    panel.set_view(view)

    _apply(panel, AppSettings(restoration_model="ltx", ltx_model=chosen))

    assert panel._widgets["ltx_model"].cget("values") == [t(f"ltx_model_{model}") for model in offered]
    assert panel.get_settings().ltx_model == chosen
