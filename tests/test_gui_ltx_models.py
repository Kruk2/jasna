from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tkinter import TclError, messagebox
from types import SimpleNamespace

import customtkinter as ctk
import pytest

from factories import ltx_models
from jasna import os_utils
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
from jasna.gui.models import AppSettings, JobItem, PresetManager
from jasna.ltx import model_files
from jasna.ltx.model_files import bundle_names
from jasna.segments import SegmentRange, SegmentRestoration
from jasna.session_config import LTX_DEFAULT_MODEL


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
    assert model_key("ltx", "distilled2") == "ltx_distilled2"
    assert model_key("ltx", "undistilled") == "ltx_undistilled"


def test_availability_without_downloads(tmp_path) -> None:
    _install(tmp_path, "distilled", fast=False)
    state = read_install_state(tmp_path)

    assert not state.downloadable
    assert state.installed("distilled", False) and not state.installed("distilled", True)
    assert card_unavailable_reason(nvidia=True) is None
    assert card_unavailable_reason(nvidia=False) == "model_ltx_needs_nvidia"
    assert not trial_only(state) and trial_only(read_install_state(tmp_path / "empty"))
    assert run_unavailable_reason(state, "undistilled", False, nvidia=True, trial=False) == "model_ltx_not_installed"
    assert run_unavailable_reason(state, "undistilled", False, nvidia=True, trial=True) is None
    assert run_unavailable_reason(state, "distilled", False, nvidia=None, trial=False) is None
    assert run_unavailable_reason(state, "distilled", True, nvidia=True, trial=False) == "model_ltx_not_installed"
    assert run_unavailable_reason(state, "distilled", True, nvidia=False, trial=True) == "model_ltx_needs_nvidia"


def test_availability_with_downloads(tmp_path, downloadable) -> None:
    state = read_install_state(tmp_path)

    assert state.downloadable
    assert not trial_only(state)
    assert run_unavailable_reason(state, "undistilled", False, nvidia=True, trial=False) == "model_ltx_not_downloaded"


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
def tk_root(monkeypatch, tmp_path):
    monkeypatch.setattr(os_utils.sys, "platform", "linux", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    yield root
    root.destroy()


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
    section.set_gpu_support(nvidia=True, blackwell=True)

    section._cards["ltx"].radio.invoke()
    _pick_ltx_model(section, "undistilled")
    section._widgets["ltx_fast"]._toggle()

    settings = panel.get_settings()
    assert asked == [] and not models.downloading
    assert (settings.restoration_model, settings.ltx_model, settings.ltx_fast) == ("ltx", "undistilled", True)
    assert section._notice.cget("text") == t("model_ltx_not_downloaded")
    assert section._notice.winfo_manager()


def test_start_downloads_only_the_fast_model(tk_root, tmp_path, monkeypatch, downloadable) -> None:
    from jasna.gui.app import JasnaApp

    _answer(monkeypatch, yes=True)
    downloaded: list[str] = []
    monkeypatch.setattr(
        model_files, "download_files", lambda _directory, files, _progress, **_kw: downloaded.extend(f.name for f in files)
    )
    monkeypatch.setattr(ltx_models_module, "start_download", lambda download, _on_percent, on_done: (download(None), on_done(None)))
    models = LtxModels(tmp_path / "ltx", _NowCalls(), lambda: None)
    panel = _panel(tk_root, models)
    section = panel._model_section
    section.set_gpu_support(nvidia=True, blackwell=True)
    section._cards["ltx"].radio.invoke()
    section._widgets["ltx_fast"]._toggle()
    restarted: list[bool] = []
    app = SimpleNamespace(
        _preview_gpu_busy=False,
        _processor=None,
        _queue_panel=SimpleNamespace(
            get_jobs=lambda: [JobItem(Path("a.mp4"))], get_output_folder=lambda: "", get_output_pattern=lambda: ""
        ),
        _settings_panel=panel,
        _ltx_models=models,
        _on_start=lambda: restarted.append(True),
    )

    JasnaApp._on_start(app)

    assert sorted(downloaded) == sorted(f"{name}.enc" for name in bundle_names(LTX_DEFAULT_MODEL, fast=True))
    assert restarted == [True]


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
    models = ltx_models(tk_root, tmp_path / "ltx", installed=False)
    panel = _panel(tk_root, models)
    models._on_change = panel.refresh_ltx_models
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
    section.set_gpu_support(nvidia=True, blackwell=True)
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
    for key in ("ltx_seed", "ltx_fast", "ltx_large_canvas", "ltx_less_overlap", "ltx_trial"):
        assert section._widgets[key] in tips[t(f"tip_{key}")]


def test_start_downloads_a_missing_ltx_model_first(monkeypatch) -> None:
    from jasna.gui.app import JasnaApp

    ensured: list[tuple] = []
    settings = replace(AppSettings(), ltx_model="undistilled")
    job = JobItem(Path("a.mp4"), segments=(SegmentRange(1, 2, SegmentRestoration("ltx", 5)),))
    app = SimpleNamespace(
        _preview_gpu_busy=False,
        _processor=None,
        _queue_panel=SimpleNamespace(get_jobs=lambda: [job], get_output_folder=lambda: "", get_output_pattern=lambda: ""),
        _settings_panel=SimpleNamespace(
            get_settings=lambda: settings, ltx_unavailable_reason=lambda: "model_ltx_not_downloaded"
        ),
        _ltx_models=SimpleNamespace(ensure=lambda model, fast, on_ready: ensured.append((model, fast)) or True),
    )
    app._on_start = lambda: None

    JasnaApp._on_start(app)

    assert ensured == [("undistilled", False)]


def test_declining_the_ltx_download_at_start_shows_no_error(tk_root, tmp_path, monkeypatch, downloadable) -> None:
    from jasna.gui.app import JasnaApp

    asked = _answer(monkeypatch, yes=False)
    errors: list[str] = []
    monkeypatch.setattr(messagebox, "showerror", lambda _title, message: errors.append(message))
    settings = replace(AppSettings(), restoration_model="ltx")
    app = SimpleNamespace(
        _preview_gpu_busy=False,
        _processor=None,
        _queue_panel=SimpleNamespace(
            get_jobs=lambda: [JobItem(Path("a.mp4"))], get_output_folder=lambda: "", get_output_pattern=lambda: ""
        ),
        _settings_panel=SimpleNamespace(
            get_settings=lambda: settings, ltx_unavailable_reason=lambda: "model_ltx_not_downloaded"
        ),
        _ltx_models=ltx_models(tk_root, tmp_path / "ltx", installed=False),
        _log_panel=SimpleNamespace(error=errors.append),
    )
    app._on_start = lambda: None

    JasnaApp._on_start(app)

    assert len(asked) == 1
    assert errors == []


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


def _start_app(settings: AppSettings, *, license_missing: bool, yes: bool, monkeypatch):
    from jasna.gui.app import JasnaApp

    monkeypatch.setattr(ltx_models_module, "license_missing", lambda directory, model, fast: license_missing)
    asked = _answer(monkeypatch, yes=yes)
    trials: list[bool] = []
    restarted: list[bool] = []
    app = SimpleNamespace(
        _preview_gpu_busy=False,
        _processor=None,
        _queue_panel=SimpleNamespace(get_jobs=lambda: [JobItem(Path("a.mp4"))], get_output_folder=lambda: "", get_output_pattern=lambda: ""),
        _settings_panel=SimpleNamespace(
            get_settings=lambda: settings, ltx_unavailable_reason=lambda: None, set_ltx_trial=trials.append
        ),
        _ltx_models=SimpleNamespace(directory=Path("ltx")),
        _on_start=lambda: restarted.append(True),
    )
    return app, asked, trials, restarted, JasnaApp._on_start


def test_start_offers_a_trial_when_the_license_is_missing(monkeypatch) -> None:
    settings = replace(AppSettings(), restoration_model="ltx")
    app, asked, trials, restarted, on_start = _start_app(settings, license_missing=True, yes=True, monkeypatch=monkeypatch)

    on_start(app)

    assert asked == [t("ltx_license_trial_confirm")]
    assert trials == [True] and restarted == [True]


def test_declining_the_trial_offer_does_not_start(monkeypatch) -> None:
    settings = replace(AppSettings(), restoration_model="ltx")
    app, asked, trials, restarted, on_start = _start_app(settings, license_missing=True, yes=False, monkeypatch=monkeypatch)

    on_start(app)

    assert len(asked) == 1 and trials == [] and restarted == []


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
    from jasna.protection import LicenseError

    directory = tmp_path / "ltx"
    _install(directory, "distilled", fast=False)
    assert not ltx_models_module.license_missing(directory, "distilled", False)

    def refuse(*_args, **_kwargs):
        raise LicenseError("No license found.")

    monkeypatch.setattr(model_files.LtxModelFiles, "from_dir", refuse)
    assert ltx_models_module.license_missing(directory, "distilled", False)


def test_start_with_a_trial_skips_the_license_offer(monkeypatch) -> None:
    from jasna.gui import validation

    class Validated(Exception):
        pass

    def validated(*_args, **_kwargs):
        raise Validated

    settings = replace(AppSettings(), restoration_model="ltx", ltx_trial=True)
    app, asked, trials, restarted, on_start = _start_app(settings, license_missing=True, yes=True, monkeypatch=monkeypatch)
    monkeypatch.setattr(validation, "validate_gui_start", validated)

    with pytest.raises(Validated):
        on_start(app)

    assert asked == [] and trials == [] and restarted == []


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
