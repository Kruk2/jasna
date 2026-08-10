from __future__ import annotations

from pathlib import Path
from types import MethodType, SimpleNamespace
import tkinter as tk
from tkinter import TclError
from unittest.mock import MagicMock

import customtkinter as ctk
import pytest

from jasna.gui.app import JasnaApp
from jasna.gui.models import JobItem, JobStatus
from jasna.gui.queue_panel import QueuePanel
from jasna.segments import SegmentRange
from jasna.gui.theme import Colors, Sizing


def test_reset_jobs_for_run_prepares_every_status_and_preserves_job_options(
    tmp_path: Path,
) -> None:
    statuses = (
        JobStatus.PENDING,
        JobStatus.COMPLETED,
        JobStatus.ERROR,
        JobStatus.SKIPPED,
        JobStatus.PAUSED,
    )
    jobs = [
        JobItem(
            tmp_path / f"video-{index}.mp4",
            status=status,
            progress=0.75,
            error_message="old error",
            segments=(SegmentRange(index, index + 1),),
            detection_model="rfdetr-v6",
            detection_score_threshold=0.4,
            vr_projection="fisheye",
        )
        for index, status in enumerate(statuses)
    ]
    original_ids = [job.id for job in jobs]
    original_segments = [job.segments for job in jobs]
    widgets = [MagicMock() for _ in jobs]
    conflict_path = tmp_path / "video-0-restored.mp4"
    conflict_path.touch()

    panel = SimpleNamespace(
        _jobs=jobs,
        _job_widgets=widgets,
        _find_job_index_by_id=lambda job_id: next(
            index for index, job in enumerate(jobs) if job.id == job_id
        ),
        _get_output_path=lambda path: tmp_path / f"{path.stem}-restored.mp4",
    )
    panel.update_job_status = MethodType(QueuePanel.update_job_status, panel)
    panel._refresh_conflicts = MethodType(QueuePanel._refresh_conflicts, panel)

    QueuePanel.reset_jobs_for_run(panel)

    assert [job.id for job in jobs] == original_ids
    assert [job.segments for job in jobs] == original_segments
    assert all(job.status is JobStatus.PENDING for job in jobs)
    assert all(job.progress == 0.0 for job in jobs)
    assert all(job.error_message == "" for job in jobs)
    assert jobs[0].has_conflict
    assert not any(job.has_conflict for job in jobs[1:])
    for widget in widgets:
        widget.set_segments_editable.assert_called_with(True)


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


def test_queue_rows_offer_video_player_and_folder_actions() -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    try:
        panel = QueuePanel(root)
        panel.pack(fill="both", expand=True)
        played = MagicMock()
        panel.set_on_play(played)
        panel.add_job(Path("/tmp/video.mp4"))
        panel.add_job(Path("/tmp/image.png"))
        root.update()

        video, image = panel._job_widgets
        assert video._segments_btn.winfo_ismapped()
        assert video._play_btn.winfo_ismapped()
        assert video._overflow_btn.winfo_ismapped()
        assert not image._segments_btn.winfo_ismapped()
        assert not image._play_btn.winfo_ismapped()
        assert image._overflow_btn.winfo_ismapped()

        video._handle_play()
        played.assert_called_once_with(Path("/tmp/video.mp4"))

        panel.set_running(True, processing_job_id=panel._jobs[0].id)
        assert video._play_btn.cget("state") == "disabled"
    finally:
        root.destroy()


def test_video_added_during_processing_has_disabled_player() -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    try:
        panel = QueuePanel(root)
        panel.pack(fill="both", expand=True)
        played = MagicMock()
        panel.set_on_play(played)
        panel.set_running(True)

        panel.add_job(Path("/tmp/later.mp4"))
        video = panel._job_widgets[0]

        assert video._play_btn.cget("state") == "disabled"
        video._handle_play()
        played.assert_not_called()
    finally:
        root.destroy()


def test_same_as_input_clears_output_and_refreshes_conflicts(tmp_path: Path) -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    try:
        panel = QueuePanel(root)
        panel.pack(fill="both", expand=True)
        changed = MagicMock()
        panel.set_on_output_changed(changed)
        source = tmp_path / "clip.mp4"
        (tmp_path / "clip_restored.mp4").touch()
        panel.add_job(source)
        panel._set_output_folder(str(tmp_path / "elsewhere"))
        changed.reset_mock()

        panel._on_same_as_input()

        assert panel.get_output_folder() == ""
        assert panel._jobs[0].has_conflict
        assert panel._same_as_input_btn.cget("state") == "normal"
        changed.assert_called_once_with("", panel.get_output_pattern())

        panel.set_running(True, processing_job_id=panel._jobs[0].id)
        assert panel._same_as_input_btn.cget("state") == "disabled"
    finally:
        root.destroy()


def test_workspace_sash_cursor_stays_on_the_sash() -> None:
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

        assert root._workspace.cget("sashcursor") == ""
        assert root._workspace.cget("cursor") == "sb_h_double_arrow"
        assert root._queue_panel.cget("cursor") == "arrow"
        assert root._settings_panel.cget("cursor") == "arrow"
    finally:
        root.destroy()


def test_repeated_running_state_does_not_reconfigure_queue_rows() -> None:
    control_updates = []
    row_updates = []
    jobs = [
        SimpleNamespace(id=index, status=JobStatus.PENDING)
        for index in range(25)
    ]
    widgets = [
        SimpleNamespace(
            set_removable=lambda value: row_updates.append(("removable", value)),
            set_segments_editable=lambda value: row_updates.append(("segments", value)),
            set_player_enabled=lambda value: row_updates.append(("player", value)),
        )
        for _ in jobs
    ]

    def control():
        return SimpleNamespace(configure=lambda **kwargs: control_updates.append(kwargs))

    panel = SimpleNamespace(
        _running=False,
        _processing_job_id=None,
        _clear_btn=control(),
        _clear_completed_btn=control(),
        _output_browse_btn=control(),
        _same_as_input_btn=control(),
        _pattern_entry=control(),
        _add_files_btn=control(),
        _add_folder_btn=control(),
        _jobs=jobs,
        _job_widgets=widgets,
        _find_job_index_by_id=lambda job_id: job_id,
    )

    QueuePanel.set_running(panel, True, processing_job_id=0)
    control_updates.clear()
    row_updates.clear()

    QueuePanel.set_running(panel, True, processing_job_id=0)

    assert control_updates == []
    assert row_updates == []


@pytest.mark.parametrize("hidpi", [1.0, 1.25, 1.5], indirect=True)
def test_main_workspace_starts_wider_and_can_resize_queue_panel(hidpi) -> None:
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
        assert int(root._workspace.cget("sashwidth")) == int(4 * hidpi)
        assert root._queue_panel.winfo_width() >= int(Sizing.QUEUE_PANEL_WIDTH * hidpi)
        assert int(root._workspace.panecget(root._queue_panel, "minsize")) == int(
            Sizing.QUEUE_PANEL_MIN_WIDTH * hidpi
        )

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
