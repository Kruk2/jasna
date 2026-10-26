from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import customtkinter as ctk
import pytest

from jasna.gui import app as app_module
from jasna.gui import components, job_list_item
from jasna.gui.components import StatusPill, grab_modal
from jasna.gui.job_list_item import JobListItem
from jasna.gui.control_bar import ControlBar, status_text
from jasna.gui.locales import t
from jasna.gui.locales.th import TH
from jasna.progressbar import StageProgress


def _open_tooltips(widget) -> list[ctk.CTkToplevel]:
    found = [widget] if isinstance(widget, ctk.CTkToplevel) and widget.wm_overrideredirect() else []
    for child in widget.winfo_children():
        found += _open_tooltips(child)
    return found


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
    opened_with_tooltips: list[int] = []
    item = _job_item(tk_root, on_edit_segments=lambda: opened_with_tooltips.append(len(_open_tooltips(tk_root))))
    item._segment_tooltips[0]._show()
    assert len(_open_tooltips(tk_root)) == 1

    item._handle_edit_segments()

    assert opened_with_tooltips == [0]


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


def test_start_button_drops_its_disabled_tooltip_when_state_changes(tk_root) -> None:
    control_bar = ControlBar(tk_root)

    control_bar.set_start_enabled(False, "Add files first")
    control_bar._start_disabled_tooltip._show()
    control_bar.set_start_enabled(False, "Restoring")
    assert _open_tooltips(tk_root) == []

    control_bar._start_disabled_tooltip._show()
    control_bar.set_start_enabled(True)
    assert _open_tooltips(tk_root) == []
    assert control_bar._start_btn.cget("state") == "normal"


@pytest.mark.parametrize("confirmed", [True, False])
def test_stop_button_stops_only_after_confirmation(tk_root, monkeypatch, confirmed) -> None:
    stopped: list[bool] = []
    control_bar = ControlBar(tk_root)
    control_bar.set_callbacks(on_stop=lambda: stopped.append(True))
    monkeypatch.setattr("jasna.gui.control_bar.messagebox.askyesno", lambda *a, **k: confirmed)

    control_bar._handle_stop()

    assert stopped == ([True] if confirmed else [])


def test_stop_confirmed_after_queue_finished_keeps_completed_state(jasna_app, monkeypatch) -> None:
    running = [True]
    stops: list[bool] = []
    monkeypatch.setattr(jasna_app._processor, "is_running", lambda: running[0])
    monkeypatch.setattr(jasna_app._processor, "stop", lambda: stops.append(True))

    def queue_finishes_while_dialog_is_open(*_args, **_kwargs):
        running[0] = False
        return True

    monkeypatch.setattr("jasna.gui.control_bar.messagebox.askyesno", queue_finishes_while_dialog_is_open)
    logged = len(jasna_app._log_panel._entries)

    jasna_app._control_bar._handle_stop()

    assert stops == []
    assert len(jasna_app._log_panel._entries) == logged


def test_completed_job_combines_status_and_elapsed_time(tk_root) -> None:
    item = _job_item(tk_root)
    item.set_fps_eta(30.0, 10.0, None)

    item.set_completed(2.6)

    assert item._status_label.cget("text") == f"{t('completed_in')} 2s"
    assert item._fps_label.cget("text") == ""
    assert item._eta_label.cget("text") == ""


def test_status_pill_sizes_to_localized_content(tk_root, monkeypatch) -> None:
    translations = {
        "status_idle": "พร้อม",
        "status_processing": "กำลังประมวลผล",
        "status_paused": "หยุดชั่วคราว",
        "status_completed": "เสร็จสิ้น",
        "status_error": "ข้อผิดพลาด",
    }
    monkeypatch.setattr(components, "t", translations.__getitem__)
    pill = StatusPill(tk_root)
    pill.pack()
    widths = []
    for status in ("IDLE", "PROCESSING"):
        pill.set_status(status, "#ffffff")
        tk_root.update_idletasks()
        widths.append(pill.winfo_reqwidth())

    assert max(widths) < 180
    assert pill._label.cget("text") == translations["status_processing"].upper()


@pytest.fixture
def thai_header(monkeypatch):
    monkeypatch.setattr(app_module, "t", TH.__getitem__)
    monkeypatch.setattr(components, "t", TH.__getitem__)


def test_header_keeps_about_button_visible_at_default_width(thai_header, jasna_app, show_window) -> None:
    jasna_app._status_pill.set_status("PROCESSING", "#ffffff")
    show_window(jasna_app)

    window_right = jasna_app.winfo_rootx() + jasna_app.winfo_width()
    about_right = jasna_app._about_btn.winfo_rootx() + jasna_app._about_btn.winfo_width()
    status_right = jasna_app._status_pill.winfo_rootx() + jasna_app._status_pill.winfo_width()
    assert jasna_app._about_btn.winfo_width() > 1
    assert about_right <= window_right
    assert status_right < jasna_app._lang_dropdown.master.winfo_rootx()


def test_grab_modal_grabs_a_new_dialog_and_restores_it_on_focus(tk_root) -> None:
    dialog = ctk.CTkToplevel(tk_root)
    dialog.transient(tk_root)
    grab_modal(dialog)
    assert dialog.grab_current() is dialog
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
def test_window_close_asks_only_while_processing(jasna_app, monkeypatch, running) -> None:
    monkeypatch.setattr(jasna_app._processor, "is_running", lambda: running)

    jasna_app._on_close_requested()

    if running:
        assert any(isinstance(w, components.CloseWhileProcessingDialog) for w in jasna_app.winfo_children())
    else:
        assert not jasna_app.children


@pytest.mark.parametrize("button", ["minimize_button", "quit_button"])
def test_close_while_processing_dialog_runs_chosen_action(tk_root, button) -> None:
    on_minimize = MagicMock()
    on_quit = MagicMock()
    dialog = components.CloseWhileProcessingDialog(tk_root, on_minimize=on_minimize, on_quit=on_quit)

    getattr(dialog, button).invoke()

    assert not dialog.winfo_exists()
    assert on_minimize.called is (button == "minimize_button")
    assert on_quit.called is (button == "quit_button")
