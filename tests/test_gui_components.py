from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import customtkinter as ctk
import pytest
from tkinter import TclError

from jasna.gui import app as app_module
from jasna.gui import components, job_list_item
from jasna.gui.app import JasnaApp
from jasna.gui.components import StatusPill
from jasna.gui.job_list_item import JobListItem
from jasna.gui.control_bar import ControlBar, status_text
from jasna.gui.locales import t
from jasna.gui.locales.th import TH
from jasna.progressbar import StageProgress


@pytest.fixture
def tk_root():
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    yield root
    root.destroy()


def _job_item(root, **callbacks) -> JobListItem:
    handlers = {
        name: MagicMock()
        for name in (
            "on_remove", "on_drag_start", "on_drag_move", "on_drag_end", "on_edit_segments",
            "on_play", "on_open_containing_folder", "on_copy_path", "on_open_restored_output", "on_requeue",
        )
    }
    handlers.update(callbacks)
    return JobListItem(root, filename="clip.mp4", duration="1m 0s", status="pending", **handlers)


class _RecordingMenu:
    instances: list["_RecordingMenu"] = []

    def __init__(self, *_args, **_kwargs):
        self.commands = []
        self.popup = None
        _RecordingMenu.instances.append(self)

    def add_command(self, **kwargs):
        self.commands.append(kwargs)

    def add_separator(self):
        self.commands.append(None)

    def bind(self, *_args):
        pass

    def delete(self, *_args):
        self.commands.clear()

    def tk_popup(self, x, y):
        self.popup = (x, y)


@pytest.fixture
def recording_menu(monkeypatch):
    _RecordingMenu.instances = []
    monkeypatch.setattr(job_list_item.tkinter, "Menu", _RecordingMenu)
    monkeypatch.setattr(job_list_item, "t", lambda key, **_kwargs: key)
    return _RecordingMenu.instances


def test_segment_tooltips_hide_before_editor_opens(tk_root) -> None:
    on_edit_segments = MagicMock()
    item = _job_item(tk_root, on_edit_segments=on_edit_segments)
    item._segment_tooltips = [MagicMock(), MagicMock()]

    item._handle_edit_segments()

    for tooltip in item._segment_tooltips:
        tooltip.hide.assert_called_once_with()
    on_edit_segments.assert_called_once_with()


def test_queue_overflow_menu_uses_button_and_right_click_coordinates(tk_root, recording_menu) -> None:
    item = _job_item(tk_root)
    tk_root.update()
    button = item._overflow_btn

    assert item._show_action_menu() == "break"
    assert recording_menu[0].popup == (button.winfo_rootx(), button.winfo_rooty() + button.winfo_height())
    assert recording_menu[0].commands[0]["label"] == "open_containing_folder"

    assert item._show_action_menu(SimpleNamespace(x_root=30, y_root=40)) == "break"
    assert recording_menu[0].popup == (30, 40)


def test_queue_overflow_menu_includes_completed_actions(tk_root, recording_menu) -> None:
    item = _job_item(tk_root)
    item.set_action_options(has_restored_output=True, requeueable=True)

    item._show_action_menu()

    assert [entry["label"] for entry in recording_menu[0].commands if entry] == [
        "open_containing_folder",
        "copy_path",
        "open_restored_output",
        "requeue",
    ]


def test_queue_overflow_menu_is_suppressed_when_hidden(tk_root, recording_menu) -> None:
    item = _job_item(tk_root)
    item.set_action_menu_visible(False)

    assert item._show_action_menu() == "break"
    assert recording_menu == []


def test_conflict_dot_toggles(tk_root) -> None:
    item = _job_item(tk_root)

    item.set_conflict(True)
    assert item._conflict_dot.winfo_manager() == "pack"
    item.set_conflict(False)
    assert item._conflict_dot.winfo_manager() == ""


def test_enabling_start_button_hides_disabled_tooltip() -> None:
    control_bar = object.__new__(ControlBar)
    tooltip = MagicMock()
    control_bar._start_disabled_tooltip = tooltip
    control_bar._start_btn = MagicMock()
    control_bar._start_btn_normal_fg = "normal"
    control_bar._start_btn_normal_hover = "hover"

    ControlBar.set_start_enabled(control_bar, True)

    tooltip.hide.assert_called_once_with()
    control_bar._start_btn.configure.assert_called_once_with(
        state="normal",
        fg_color="normal",
        hover_color="hover",
    )


def test_updating_disabled_start_button_hides_previous_tooltip() -> None:
    control_bar = object.__new__(ControlBar)
    tooltip = MagicMock()
    control_bar._start_disabled_tooltip = tooltip
    control_bar._start_btn = MagicMock()

    ControlBar.set_start_enabled(control_bar, False)

    tooltip.hide.assert_called_once_with()


@pytest.mark.parametrize("confirmed", [True, False])
def test_stop_button_stops_only_after_confirmation(tk_root, monkeypatch, confirmed) -> None:
    control_bar = ControlBar(tk_root)
    control_bar._on_stop = MagicMock()
    monkeypatch.setattr("jasna.gui.control_bar.messagebox.askyesno", lambda *a, **k: confirmed)

    ControlBar._handle_stop(control_bar)

    assert control_bar._on_stop.called is confirmed


def test_stop_confirmed_after_queue_finished_keeps_completed_state(tk_root, monkeypatch) -> None:
    processor = MagicMock()
    processor.is_running.return_value = True
    app = SimpleNamespace(
        _processor=processor,
        _log_panel=MagicMock(),
        _status_pill=MagicMock(),
        _control_bar=MagicMock(),
        _settings_panel=MagicMock(),
        _queue_panel=MagicMock(),
    )
    control_bar = ControlBar(tk_root)
    control_bar._on_stop = lambda: JasnaApp._on_stop(app)

    def queue_finishes_while_dialog_is_open(*_args, **_kwargs):
        processor.is_running.return_value = False
        return True

    monkeypatch.setattr("jasna.gui.control_bar.messagebox.askyesno", queue_finishes_while_dialog_is_open)

    ControlBar._handle_stop(control_bar)

    processor.stop.assert_not_called()
    app._log_panel.info.assert_not_called()
    app._control_bar.reset.assert_not_called()
    app._control_bar.set_start_enabled.assert_not_called()


def test_completed_job_combines_status_and_elapsed_time(tk_root) -> None:
    item = _job_item(tk_root)
    item.set_fps_eta(30.0, 10.0, None)

    item.set_completed(2.6)

    assert item._status_label.cget("text") == f"{t('completed_in')} 2s"
    assert item._fps_label.cget("text") == ""
    assert item._eta_label.cget("text") == ""


def test_status_pill_sizes_to_localized_content(monkeypatch) -> None:
    translations = {
        "status_idle": "พร้อม",
        "status_processing": "กำลังประมวลผล",
        "status_paused": "หยุดชั่วคราว",
        "status_completed": "เสร็จสิ้น",
        "status_error": "ข้อผิดพลาด",
    }
    monkeypatch.setattr(components, "t", translations.__getitem__)
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    try:
        pill = StatusPill(root)
        pill.pack()
        widths = []
        for status in ("IDLE", "PROCESSING"):
            pill.set_status(status, "#ffffff")
            root.update_idletasks()
            widths.append(pill.winfo_reqwidth())

        assert max(widths) < 180
        assert pill._label.cget("text") == translations["status_processing"].upper()
    finally:
        root.destroy()


def test_header_keeps_about_button_visible_at_default_width(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "t", TH.__getitem__)
    monkeypatch.setattr(components, "t", TH.__getitem__)
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    for name in (
        "_open_video_player",
        "_show_system_check",
        "_show_help",
        "_show_about",
        "_open_license_dialog",
        "_refresh_license_chip",
        "_on_language_changed",
    ):
        setattr(root, name, lambda *_args: None)

    try:
        root.geometry("1320x100")
        JasnaApp._build_header(root)
        root._status_pill.set_status("PROCESSING", "#ffffff")
        root.update()

        window_right = root.winfo_rootx() + root.winfo_width()
        about_right = root._about_btn.winfo_rootx() + root._about_btn.winfo_width()
        status_right = (
            root._status_pill.winfo_rootx() + root._status_pill.winfo_width()
        )
        header_right_left = root._lang_dropdown.master.winfo_rootx()
        assert root._about_btn.winfo_width() > 1
        assert about_right <= window_right
        assert status_right < header_right_left
    finally:
        root.destroy()


def test_grab_modal_waits_for_visibility_before_grabbing() -> None:
    from jasna.gui.components import grab_modal

    dialog = MagicMock()
    dialog.winfo_viewable.return_value = False
    grab_modal(dialog)

    grabbing = [call[0] for call in dialog.method_calls if call[0] in ("wait_visibility", "grab_set", "lift", "focus_force")]
    assert grabbing == ["wait_visibility", "grab_set", "lift", "focus_force"]


def test_grab_modal_restores_minimized_dialog_on_focus(tk_root) -> None:
    from jasna.gui.components import grab_modal

    dialog = ctk.CTkToplevel(tk_root)
    dialog.transient(tk_root)
    grab_modal(dialog)
    focus_bindings = dialog.bind("<FocusIn>")
    grab_modal(dialog)
    assert dialog.bind("<FocusIn>") == focus_bindings

    dialog.withdraw()
    dialog.event_generate("<FocusIn>")
    tk_root.update()
    assert dialog.winfo_ismapped()


def test_control_bar_shows_the_ltx_step_instead_of_fps(tk_root) -> None:
    control_bar = ControlBar(tk_root)
    stage = StageProgress("denoise", 3, 4, 0.0, 0.0)
    control_bar.update_progress(filename="a.mp4", percent=42.0, eta_seconds=90.0, stage=stage)
    assert control_bar._fps_label.cget("text") == status_text(0.0, stage, trial=False)
    control_bar.update_progress(filename="a.mp4", percent=50.0, fps=30.0)
    assert control_bar._fps_label.cget("text") == "FPS: 30.0"


def test_status_text_names_the_ltx_step_its_share_and_time_left() -> None:
    assert status_text(0.0, StageProgress("scan", 1, 4, 0.0, 0.0), trial=False) == "Step 1 of 4: Finding mosaics…"
    assert status_text(0.0, StageProgress("scan", 1, 4, 0.456, 0.0), trial=False) == "Step 1 of 4: Finding mosaics 45%"
    assert (
        status_text(0.0, StageProgress("denoise", 3, 4, 0.5, 130.0), trial=True)
        == "Step 3 of 4: Restoring 50%, 2m 10s left (trial)"
    )
    assert status_text(24.0, None, trial=False) == "FPS: 24.0"


def test_control_bar_progress_rows_fit_its_height(tk_root) -> None:
    tk_root.geometry("1400x200")
    control_bar = ControlBar(tk_root)
    control_bar.pack(fill="x")
    control_bar.update_progress(filename="a.mp4", percent=5.0, eta_seconds=60.0, stage=StageProgress("scan", 1, 4, 0.5, 30.0))
    tk_root.update()
    bottom = control_bar._eta_label.winfo_rooty() + control_bar._eta_label.winfo_height()
    assert bottom <= control_bar.winfo_rooty() + control_bar.winfo_height()
    assert control_bar._fps_label.winfo_ismapped() and control_bar._eta_label.winfo_ismapped()


def test_job_item_shows_the_ltx_step_in_place_of_fps(tk_root) -> None:
    item = _job_item(tk_root)
    item.set_fps_eta(0.0, 0.0, StageProgress("scan", 1, 4, 0.456, 30.0))
    assert item._fps_label.cget("text") == "Step 1/4: 45%"
    assert item._eta_label.cget("text") == ""
    item.set_fps_eta(30.0, 10.0, None)
    assert item._fps_label.cget("text") == "30.0fps"


@pytest.mark.parametrize("running", [True, False])
def test_window_close_asks_only_while_processing(monkeypatch, running) -> None:
    app = object.__new__(JasnaApp)
    app._processor = MagicMock(is_running=MagicMock(return_value=running))
    app._on_close = MagicMock()
    dialog = MagicMock()
    monkeypatch.setattr(app_module, "CloseWhileProcessingDialog", dialog)

    JasnaApp._on_close_requested(app)

    assert dialog.called is running
    assert app._on_close.called is not running


@pytest.mark.parametrize("button", ["minimize_button", "quit_button"])
def test_close_while_processing_dialog_runs_chosen_action(tk_root, button) -> None:
    on_minimize = MagicMock()
    on_quit = MagicMock()
    dialog = components.CloseWhileProcessingDialog(tk_root, on_minimize=on_minimize, on_quit=on_quit)

    getattr(dialog, button).invoke()

    assert not dialog.winfo_exists()
    assert on_minimize.called is (button == "minimize_button")
    assert on_quit.called is (button == "quit_button")
