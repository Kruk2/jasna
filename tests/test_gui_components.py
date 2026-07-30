from __future__ import annotations

from unittest.mock import MagicMock

import customtkinter as ctk
import pytest
from tkinter import TclError

from jasna.gui import app as app_module
from jasna.gui import components
from jasna.gui.app import JasnaApp
from jasna.gui.components import JobListItem, StatusPill
from jasna.gui.control_bar import ControlBar
from jasna.gui.locales import t
from jasna.gui.locales.th import TH


def test_segment_tooltips_hide_before_editor_opens() -> None:
    item = object.__new__(JobListItem)
    item._segments_editable = True
    item._on_edit_segments = MagicMock()
    item._segment_tooltips = [MagicMock(), MagicMock()]

    JobListItem._handle_edit_segments(item)

    for tooltip in item._segment_tooltips:
        tooltip.hide.assert_called_once_with()
    item._on_edit_segments.assert_called_once_with()


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


def test_completed_job_combines_status_and_elapsed_time() -> None:
    item = object.__new__(JobListItem)
    item._status_label = MagicMock()
    item._fps_label = MagicMock()
    item._eta_label = MagicMock()

    JobListItem.set_completed(item, 2.6)

    item._status_label.configure.assert_called_once_with(
        text=f"{t('completed_in')} 2s",
    )
    item._fps_label.configure.assert_called_once_with(text="")
    item._eta_label.configure.assert_called_once_with(text="")


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
        for status in ("IDLE", "PROCESSING", "PAUSED", "COMPLETED", "ERROR"):
            pill.set_status(status, "#ffffff")
            root.update_idletasks()
            widths.append(pill.winfo_reqwidth())

        assert max(widths) < 180
        assert pill._label.cget("text") == translations["status_error"].upper()
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
