from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tkinter import messagebox

import pytest

from factories import ltx_models, write_sparse_file
from jasna import os_utils
from jasna.accelerator import AcceleratorVendor
from jasna.gui import ltx_models as ltx_models_module
from jasna.gui.locales import t
from jasna.gui.ltx_models import (
    LtxModels,
    model_key,
    card_unavailable_reason,
    read_install_state,
    run_unavailable_reason,
    trial_only,
)
from jasna.gui.models import AppSettings, PresetManager
from jasna.ltx import model_files
from jasna.ltx.model_files import bundle_names
from jasna.segments import SegmentRange, SegmentRestoration
from jasna.session_config import LTX_DEFAULT_MODEL

pytestmark = pytest.mark.usefixtures("nvidia_vendor")

NVIDIA, AMD, CPU = AcceleratorVendor.NVIDIA, AcceleratorVendor.AMD, AcceleratorVendor.CPU


class _NowCalls:
    def post(self, callback) -> None:
        callback()


def _install(directory: Path, model: str, *, fast: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in bundle_names(model, fast=fast):
        (directory / name).touch()


@pytest.fixture(autouse=True)
def public_source(monkeypatch):
    monkeypatch.setattr(ltx_models_module, "protection_available", lambda: False)


@pytest.fixture
def downloadable(monkeypatch):
    monkeypatch.setattr(ltx_models_module, "protection_available", lambda: True)


def test_model_key_names_each_model_choice() -> None:
    assert model_key("basicvsrpp", "undistilled") == "basicvsrpp"
    assert model_key("ltx", "distilled") == "ltx_distilled"
    assert model_key("ltx", "distilled4") == "ltx_distilled4"
    assert model_key("ltx", "distilled1") == "ltx_distilled1"
    assert model_key("ltx", "distilled2") == "ltx_distilled2"
    assert model_key("ltx", "undistilled") == "ltx_undistilled"


def test_availability_without_downloads(tmp_path) -> None:
    _install(tmp_path, "distilled", fast=False)
    state = read_install_state(tmp_path, "auto", "auto")

    assert not state.downloadable
    assert state.installed("distilled", False) and not state.installed("distilled", True)
    assert card_unavailable_reason(NVIDIA) is None
    assert card_unavailable_reason(AMD) is None
    assert card_unavailable_reason(CPU) == "model_ltx_needs_nvidia"
    assert not trial_only(state) and trial_only(read_install_state(tmp_path / "empty", "auto", "auto"))
    assert run_unavailable_reason(state, "undistilled", False, NVIDIA, trial=False) == "model_ltx_not_installed"
    assert run_unavailable_reason(state, "undistilled", False, NVIDIA, trial=True) is None
    assert run_unavailable_reason(state, "distilled", False, None, trial=False) is None
    assert run_unavailable_reason(state, "distilled", True, NVIDIA, trial=False) == "model_ltx_not_installed"
    assert run_unavailable_reason(state, "distilled", True, CPU, trial=True) == "model_ltx_needs_nvidia"


def test_availability_with_downloads(tmp_path, downloadable) -> None:
    state = read_install_state(tmp_path, "auto", "auto")

    assert state.downloadable
    assert not trial_only(state)
    assert run_unavailable_reason(state, "undistilled", False, NVIDIA, trial=False) == "model_ltx_not_downloaded"


def test_amd_uses_normal_licensed_model_downloads(tmp_path, downloadable) -> None:
    state = read_install_state(tmp_path, "auto", "auto")
    assert not trial_only(state)
    assert state.usable("distilled2", False)
    assert run_unavailable_reason(state, "distilled2", False, AMD, trial=False) == "model_ltx_not_downloaded"


def test_vae_choice_decides_which_files_a_model_needs(tmp_path) -> None:
    _install(tmp_path, "distilled1", fast=False)
    models = LtxModels(tmp_path, _NowCalls(), lambda: None)
    assert models.state.installed("distilled1", False)
    assert not ltx_models_module.license_missing(tmp_path, "distilled1", False, "auto", "auto")

    models.set_vaes("full", "full")

    assert [file.name for file in models.state.missing[("distilled1", False)]] == ["vae-decoder-full.safetensors.enc"]
    assert ltx_models_module.license_missing(tmp_path, "distilled1", False, "full", "full")


def _answer(monkeypatch, yes: bool) -> list[str]:
    asked: list[str] = []
    monkeypatch.setattr(messagebox, "askyesno", lambda _title, message: asked.append(message) or yes)
    return asked


def _fake_download(monkeypatch, error: str | None) -> list:
    downloads = []

    def start(download, on_percent, on_done):
        downloads.append(download)
        on_percent(40)
        on_done(error)

    monkeypatch.setattr(ltx_models_module, "start_download", start)
    return downloads


def test_ensure_asks_and_downloads_the_missing_files(tmp_path, monkeypatch, downloadable) -> None:
    asked = _answer(monkeypatch, yes=True)
    downloads = _fake_download(monkeypatch, error=None)
    changes: list[int | None] = []
    models = LtxModels(tmp_path, _NowCalls(), lambda: changes.append(models.percent))
    ready: list[bool] = []

    assert models.ensure("undistilled", False, on_ready=lambda: ready.append(True))

    assert asked == [t("ltx_download_confirm", model=t("model_ltx_undistilled"), size="14.9 GB")]
    assert len(downloads) == 1
    assert changes == [0, 40, None]
    assert ready == [True]


def test_ensure_does_nothing_when_the_user_declines(tmp_path, monkeypatch, downloadable) -> None:
    _answer(monkeypatch, yes=False)
    downloads = _fake_download(monkeypatch, error=None)
    models = LtxModels(tmp_path, _NowCalls(), lambda: None)

    assert not models.ensure("distilled", False, on_ready=lambda: pytest.fail("not downloaded"))
    assert downloads == [] and not models.downloading


@pytest.mark.parametrize("fast,old_size", [(False, 13_368_048_749), (True, 7_733_294_997)])
def test_ensure_warns_before_replacing_models_from_older_release(
    tmp_path, monkeypatch, downloadable, fast, old_size, sparse_ltx_downloads
):
    from jasna.gui.locales import EN

    monkeypatch.setattr(model_files, "is_frozen", lambda: True)
    monkeypatch.setattr(ltx_models_module, "t", lambda key, **values: EN[key].format(**values))
    names = bundle_names("distilled2", fast=fast)
    for index, name in enumerate(names):
        write_sparse_file(tmp_path / (name + ".enc"), old_size if index == 0 else model_files.LTX_DOWNLOADS[name].size_bytes)
    asked = _answer(monkeypatch, yes=False)
    downloads = _fake_download(monkeypatch, error=None)
    models = LtxModels(tmp_path, _NowCalls(), lambda: None)

    assert not models.ensure("distilled2", fast, on_ready=lambda: pytest.fail("download declined"))
    assert len(asked) == 1 and "needs updating" in asked[0]
    assert "older Jasna" in asked[0] and "License code still works" in asked[0]
    assert len(models.state.missing[("distilled2", fast)]) == 1
    assert downloads == [] and (tmp_path / (names[0] + ".enc")).stat().st_size == old_size


def test_ensure_shows_the_download_error(tmp_path, monkeypatch, downloadable) -> None:
    _answer(monkeypatch, yes=True)
    _fake_download(monkeypatch, error="checksum failed")
    errors: list[str] = []
    monkeypatch.setattr(messagebox, "showerror", lambda _title, message: errors.append(message))
    models = LtxModels(tmp_path, _NowCalls(), lambda: None)

    assert models.ensure("distilled", False, on_ready=lambda: pytest.fail("download failed"))
    assert errors == [t("ltx_download_failed", message="checksum failed")]
    assert not models.downloading


def test_ensure_skips_installed_models_and_unavailable_downloads(tmp_path, monkeypatch) -> None:
    _answer(monkeypatch, yes=True)
    _install(tmp_path, "distilled", fast=False)
    models = LtxModels(tmp_path, _NowCalls(), lambda: None)

    assert models.ensure("distilled", False, on_ready=lambda: None)
    assert not models.ensure("undistilled", False, on_ready=lambda: None)


@pytest.fixture
def tk_root(tk_root, monkeypatch, tmp_path):
    monkeypatch.setattr(os_utils.sys, "platform", "linux", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    return tk_root


def _panel(root, models):
    from jasna.gui.settings_panel import SettingsPanel

    return SettingsPanel(root, PresetManager(), ltx_models=models)


def test_the_model_menu_round_trips_the_undistilled_model(tk_root, tmp_path) -> None:
    panel = _panel(tk_root, ltx_models(tk_root, tmp_path / "ltx", installed=True))
    undistilled = replace(AppSettings(), restoration_model="ltx", ltx_model="undistilled", ltx_seed=3, encoder_cq=28)

    for section in panel._sections:
        section.apply(undistilled)

    assert panel.get_settings() == undistilled
    assert set(panel._model_section._cards) == {"basicvsrpp", "ltx"}
    _pick_ltx_model(panel._model_section, "distilled")
    assert panel.get_settings().ltx_model == "distilled"
    panel._model_section._cards["basicvsrpp"].radio.invoke()
    assert (panel.get_settings().restoration_model, panel.get_settings().ltx_model) == ("basicvsrpp", "distilled")


def _pick_ltx_model(section, model: str) -> None:
    section._ltx_model.set_value(model)
    section._on_ltx_choice_changed()


def test_picking_a_missing_variant_never_asks_or_reverts(tk_root, tmp_path, monkeypatch, downloadable) -> None:
    asked = _answer(monkeypatch, yes=True)
    models = ltx_models(tk_root, tmp_path / "ltx", installed=False)
    panel = _panel(tk_root, models)
    section = panel._model_section
    section.set_gpu_support(vendor=NVIDIA, blackwell=True)

    section._cards["ltx"].radio.invoke()
    _pick_ltx_model(section, "undistilled")
    section._widgets["ltx_fast"]._toggle()

    settings = panel.get_settings()
    assert asked == [] and not models.downloading
    assert (settings.restoration_model, settings.ltx_model, settings.ltx_fast) == ("ltx", "undistilled", True)
    assert section._notice.cget("text") == t("model_ltx_not_downloaded")
    assert section._notice.winfo_manager()


def _queue_job(app, settings: AppSettings, **job_options) -> None:
    app._settings_panel.set_gpu_support(vendor=NVIDIA, blackwell=True, frame_generation=True)
    for section in app._settings_panel._sections:
        section.apply(settings)
    app._queue_panel.add_job(Path("a.mp4"))
    for name, value in job_options.items():
        setattr(app._queue_panel.get_jobs()[0], name, value)


@pytest.fixture
def validated(monkeypatch) -> list[AppSettings]:
    """Settings each Start reached validation with; validation then stops it before processing."""
    from jasna.gui import validation

    calls: list[AppSettings] = []

    def stop(settings, jobs, **_kwargs):
        calls.append(settings)
        return ["stopped before processing"]

    monkeypatch.setattr(validation, "validate_gui_start", stop)
    monkeypatch.setattr(messagebox, "showerror", lambda *_args, **_kwargs: None)
    return calls


@pytest.fixture
def ltx_installed(tmp_path) -> None:
    _install(tmp_path / "models" / "ltx", LTX_DEFAULT_MODEL, fast=False)


def test_start_downloads_only_the_fast_model_then_starts(
    downloadable, jasna_app, monkeypatch, validated, sparse_ltx_downloads
) -> None:
    _answer(monkeypatch, yes=True)
    monkeypatch.setattr(ltx_models_module, "license_missing", lambda *_args: False)
    downloaded: list[str] = []

    def download_files(directory, files, _progress, **_kwargs):
        directory.mkdir(parents=True, exist_ok=True)
        for file in files:
            downloaded.append(file.name)
            write_sparse_file(directory / file.name, file.size_bytes)

    monkeypatch.setattr(model_files, "download_files", download_files)
    monkeypatch.setattr(ltx_models_module, "start_download", lambda download, _on_percent, on_done: (download(None), on_done(None)))
    _queue_job(jasna_app, replace(AppSettings(), restoration_model="ltx", ltx_fast=True))

    jasna_app._on_start()
    jasna_app._main_thread._run_pending()

    assert sorted(downloaded) == sorted(f"{name}.enc" for name in bundle_names(LTX_DEFAULT_MODEL, fast=True))
    assert [settings.ltx_fast for settings in validated] == [True]


def test_declining_the_download_a_job_needs_does_not_start(downloadable, jasna_app, monkeypatch, validated) -> None:
    asked = _answer(monkeypatch, yes=False)
    _queue_job(
        jasna_app,
        replace(AppSettings(), ltx_model="undistilled"),
        segments=(SegmentRange(1, 2, SegmentRestoration("ltx", 5)),),
    )

    jasna_app._on_start()

    assert len(asked) == 1 and t("model_ltx_undistilled") in asked[0]
    assert validated == []
    assert not [entry for entry in jasna_app._log_panel._entries if entry[1] == "ERROR"]


def test_a_missing_downloadable_model_stays_selectable(tk_root, tmp_path, downloadable) -> None:
    panel = _panel(tk_root, ltx_models(tk_root, tmp_path / "ltx", installed=False))
    section = panel._model_section

    section.apply(replace(AppSettings(), restoration_model="ltx", ltx_model="undistilled"))

    assert section._cards["ltx"].radio.cget("state") == "normal"
    assert section.unavailable_reason() == "model_ltx_not_downloaded"


def test_a_running_download_locks_the_choice(tk_root, tmp_path, monkeypatch, downloadable) -> None:
    _answer(monkeypatch, yes=True)
    started = []
    monkeypatch.setattr(ltx_models_module, "start_download", lambda *args: started.append(args))
    models = ltx_models(tk_root, tmp_path / "ltx", installed=False, on_change=lambda: panel.refresh_ltx_models())
    panel = _panel(tk_root, models)
    panel._model_section._cards["ltx"].radio.invoke()

    models.ensure("distilled", False, on_ready=lambda: None)

    assert len(started) == 1
    assert panel.get_settings().restoration_model == "ltx"
    assert panel._model_section._download_row.winfo_manager()
    assert all(card.radio.cget("state") == "disabled" for card in panel._model_section._cards.values())
    assert panel._model_section._ltx_model.cget("state") == "disabled"


def test_a_missing_variant_without_downloads_stays_selected_with_a_note(tk_root, tmp_path) -> None:
    directory = tmp_path / "ltx"
    _install(directory, LTX_DEFAULT_MODEL, fast=False)
    panel = _panel(tk_root, ltx_models(tk_root, directory, installed=False))
    section = panel._model_section
    section.set_gpu_support(vendor=NVIDIA, blackwell=True)
    section._cards["ltx"].radio.invoke()
    assert not section._notice.winfo_manager()

    _pick_ltx_model(section, "undistilled")
    section._widgets["ltx_fast"]._toggle()

    settings = panel.get_settings()
    assert (settings.ltx_model, settings.ltx_fast, settings.ltx_trial) == ("undistilled", True, False)
    assert section.unavailable_reason() == "model_ltx_not_installed"
    assert section._notice.cget("text") == t("model_ltx_not_installed")


def test_every_model_option_has_a_tooltip(tk_root, tmp_path, monkeypatch) -> None:
    from jasna.gui.settings_sections import restoration_model, widgets

    tips: dict[str, list] = {}
    for tooltip_class in {restoration_model.Tooltip, widgets.Tooltip}:
        original = tooltip_class.__init__

        def record(self, widget, text, original=original):
            tips.setdefault(text, []).append(widget)
            original(self, widget, text)

        monkeypatch.setattr(tooltip_class, "__init__", record)
    panel = _panel(tk_root, ltx_models(tk_root, tmp_path / "ltx", installed=True))
    section = panel._model_section

    for key, card in section._cards.items():
        assert card.radio in tips[t(f"tip_model_{key}")]
    assert section._ltx_model in tips[t("tip_ltx_model")]
    assert section._new_seed_btn in tips[t("tip_ltx_new_seed")]
    for key in ("ltx_seed", "ltx_fast", "ltx_large_canvas", "ltx_overlap", "ltx_trial"):
        assert section._widgets[key] in tips[t(f"tip_{key}")]


def test_the_trial_switch_round_trips_and_skips_the_download(tk_root, tmp_path, monkeypatch, downloadable) -> None:
    asked = _answer(monkeypatch, yes=True)
    panel = _panel(tk_root, ltx_models(tk_root, tmp_path / "ltx", installed=False))
    section = panel._model_section
    trial = replace(AppSettings(), restoration_model="ltx", ltx_model="undistilled", ltx_trial=True)

    section.apply(trial)

    assert panel.get_settings().ltx_trial is True
    assert section.unavailable_reason() is None
    assert section._trial_status.cget("text") == t("ltx_trial_notice")
    _pick_ltx_model(section, "distilled")
    assert asked == [] and panel.get_settings().ltx_model == "distilled"


def test_turning_the_trial_off_never_asks_for_the_missing_model(tk_root, tmp_path, monkeypatch, downloadable) -> None:
    asked = _answer(monkeypatch, yes=True)
    panel = _panel(tk_root, ltx_models(tk_root, tmp_path / "ltx", installed=False))
    section = panel._model_section
    section.apply(replace(AppSettings(), restoration_model="ltx", ltx_trial=True))
    assert not section._notice.winfo_manager()

    section._widgets["ltx_trial"]._toggle()

    assert asked == []
    assert panel.get_settings().ltx_trial is False
    assert section._notice.cget("text") == t("model_ltx_not_downloaded")


def test_without_any_ltx_model_only_a_locked_trial_is_possible(tk_root, tmp_path) -> None:
    panel = _panel(tk_root, ltx_models(tk_root, tmp_path / "ltx", installed=False))
    section = panel._model_section

    section._cards["ltx"].radio.invoke()

    settings = panel.get_settings()
    assert (settings.restoration_model, settings.ltx_trial) == ("ltx", True)
    assert section._widgets["ltx_trial"].cget("cursor") == ""
    section._widgets["ltx_trial"]._toggle()
    assert panel.get_settings().ltx_trial is True
    assert section._trial_status.cget("text") == t("model_ltx_trial_only")
    panel.set_enabled(False)
    panel.set_enabled(True)
    assert section._widgets["ltx_trial"].cget("cursor") == ""


@pytest.mark.parametrize("accept", [True, False])
def test_start_offers_a_trial_when_the_license_is_missing(ltx_installed, jasna_app, monkeypatch, validated, accept) -> None:
    monkeypatch.setattr(ltx_models_module, "license_missing", lambda *_args: True)
    asked = _answer(monkeypatch, yes=accept)
    _queue_job(jasna_app, replace(AppSettings(), restoration_model="ltx"))

    jasna_app._on_start()

    assert asked == [t("ltx_license_trial_confirm")]
    assert [settings.ltx_trial for settings in validated] == ([True] if accept else [])


def test_start_explains_an_ltx_lock_instead_of_offering_a_trial(ltx_installed, jasna_app, monkeypatch, validated) -> None:
    monkeypatch.setattr(ltx_models_module, "license_blocked", lambda: True)
    asked = _answer(monkeypatch, yes=True)
    warned: list[str] = []
    monkeypatch.setattr(messagebox, "showwarning", lambda _title, message: warned.append(message))
    _queue_job(jasna_app, replace(AppSettings(), restoration_model="ltx"))

    jasna_app._on_start()

    assert (warned, asked, validated) == ([t("ltx_blocked_text")], [], [])


def test_start_with_a_trial_skips_the_license_offer(ltx_installed, jasna_app, monkeypatch, validated) -> None:
    monkeypatch.setattr(ltx_models_module, "license_missing", lambda *_args: True)
    asked = _answer(monkeypatch, yes=True)
    _queue_job(jasna_app, replace(AppSettings(), restoration_model="ltx", ltx_trial=True))

    jasna_app._on_start()

    assert asked == []
    assert len(validated) == 1


def test_trial_session_config_and_key(monkeypatch, tmp_path) -> None:
    from jasna.gui.video_session import video_session_config, video_session_key

    detection_path = tmp_path / "rfdetr-v6.onnx"
    detection_path.touch()
    monkeypatch.setattr("jasna.mosaic.detection_registry.require_detection_model_weights", lambda _name: detection_path)
    real = replace(AppSettings(), restoration_model="ltx")
    trial = replace(real, ltx_trial=True)

    assert video_session_config(trial, codec="hevc", encoder_settings={}).ltx_trial is True
    assert video_session_config(real, codec="hevc", encoder_settings={}).ltx_trial is False
    assert video_session_key(trial) != video_session_key(real)


def test_a_trial_seed_preview_never_reuses_a_real_prepared_range() -> None:
    from jasna.gui.ltx_seed_preview import prepared_key

    segment = SegmentRange(1, 2, SegmentRestoration("ltx", 5))
    real = replace(AppSettings(), restoration_model="ltx")

    assert prepared_key(segment, real) != prepared_key(segment, replace(real, ltx_trial=True))


def test_license_missing_only_for_encrypted_models(tmp_path, monkeypatch, downloadable) -> None:
    pytest.importorskip("jasna.protection.license_store")
    from jasna.protection import LicenseError

    directory = tmp_path / "ltx"
    _install(directory, "distilled", fast=False)
    assert not ltx_models_module.license_missing(directory, "distilled", False, "auto", "auto")

    def refuse(*_args, **_kwargs):
        raise LicenseError("No license found.")

    monkeypatch.setattr(model_files, "needs_license", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(ltx_models_module, "read_license_info", lambda: (True, True))
    monkeypatch.setattr("jasna.protection.protected_tensors.check_saved_ltx_license", refuse)
    assert ltx_models_module.license_missing(directory, "distilled", False, "auto", "auto")


def test_the_progress_stage_is_marked_as_a_trial(tk_root) -> None:
    from jasna.gui.control_bar import ControlBar, status_text
    from jasna.progressbar import StageProgress

    bar = ControlBar(tk_root)
    stage = StageProgress("denoise", 3, 4, 0.0, 0.0)
    bar.set_trial(True)
    bar.update_progress(percent=10, stage=stage)
    assert bar._fps_label.cget("text") == t("ltx_trial_stage", stage=status_text(0.0, stage, trial=False))
    bar.set_trial(False)
    bar.update_progress(percent=10, stage=stage)
    assert bar._fps_label.cget("text") == status_text(0.0, stage, trial=False)
