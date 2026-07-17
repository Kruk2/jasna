from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import TclError

import customtkinter as ctk
import pytest

from jasna.gui.app import JasnaApp
from jasna.gui.models import JobStatus
from jasna.gui.queue_panel import QueuePanel
from jasna.gui.theme import Colors, Sizing


def test_queue_footer_stacks_count_above_action_buttons() -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    try:
        root.geometry("320x800")
        panel = QueuePanel(root)
        panel.pack(fill="both", expand=True)
        root.update_idletasks()

        count_bottom = panel._queue_count.winfo_rooty() + panel._queue_count.winfo_height()
        actions_top = min(
            panel._clear_completed_btn.winfo_rooty(),
            panel._clear_btn.winfo_rooty(),
        )
        assert count_bottom <= actions_top

        empty_content_width = panel._empty_state.winfo_width() - 40
        assert panel._empty_label.winfo_reqwidth() <= empty_content_width
    finally:
        root.destroy()


def test_queue_scrollbar_only_appears_when_jobs_overflow() -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    try:
        root.geometry("420x800")
        panel = QueuePanel(root)
        panel.pack(fill="both", expand=True)
        root.update()

        assert not panel._list_frame._scrollbar.winfo_ismapped()

        for index in range(12):
            panel.add_job(Path(f"/tmp/video-{index}.mp4"))
        root.update()

        assert panel._list_frame._scrollbar.winfo_ismapped()
    finally:
        root.destroy()


def test_segment_button_only_appears_for_pending_video_jobs() -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    try:
        panel = QueuePanel(root)
        panel.pack(fill="both", expand=True)
        panel.add_job(Path("/tmp/video.mp4"))
        root.update()

        job = panel._jobs[0]
        widget = panel._job_widgets[0]
        assert widget._segments_btn.winfo_ismapped()

        panel.update_job_status(job.id, JobStatus.PROCESSING)
        root.update()
        assert not widget._segments_btn.winfo_ismapped()

        panel.update_job_status(job.id, JobStatus.COMPLETED)
        root.update()
        assert not widget._segments_btn.winfo_ismapped()

        panel.update_job_status(job.id, JobStatus.PENDING)
        root.update()
        assert widget._segments_btn.winfo_ismapped()
    finally:
        root.destroy()


def test_main_workspace_starts_wider_and_can_resize_queue_panel() -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    try:
        root.geometry("1200x800")
        root._on_jobs_changed = lambda: None
        root._open_interactive_image_restore = lambda: None
        root._processor = None
        root._set_preview_gpu_busy = lambda _busy: None
        root._on_output_changed = lambda *_args: None
        root.TkdndVersion = None

        JasnaApp._build_main_body(root)
        root.update_idletasks()

        assert isinstance(root._workspace, tk.PanedWindow)
        assert root._workspace.cget("background") == Colors.BORDER
        assert int(root._workspace.cget("sashwidth")) == 4
        assert root._queue_panel.winfo_width() >= Sizing.QUEUE_PANEL_WIDTH

        queue_width = root._queue_panel.winfo_width()
        settings_width = root._settings_panel.winfo_width()
        sash_x = root._workspace.sash_coord(0)[0] + 2
        root._workspace.event_generate("<ButtonPress-1>", x=sash_x, y=300)
        root.update()
        root._workspace.event_generate("<B1-Motion>", x=sash_x + 80, y=300)
        root.update()
        root._workspace.event_generate("<ButtonRelease-1>", x=sash_x + 80, y=300)
        root.update()

        assert root._queue_panel.winfo_width() > queue_width
        assert root._settings_panel.winfo_width() < settings_width
    finally:
        root.destroy()
