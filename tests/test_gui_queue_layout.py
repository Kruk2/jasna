from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from tkinterdnd2.TkinterDnD import DnDEvent

from factories import tk_event
from jasna.gui.models import JobItem, JobStatus
from jasna.gui import queue_panel as queue_panel_module
from jasna.gui.queue_panel import QueuePanel
from jasna.segments import SegmentRange
from jasna.gui.theme import Colors, Sizing


def test_reset_jobs_for_run_prepares_every_status_and_preserves_job_options(
    tk_root, tmp_path: Path,
) -> None:
    panel = QueuePanel(tk_root)
    panel.pack(fill="both", expand=True)
    panel._output_bar.set_folder(str(tmp_path))
    statuses = (JobStatus.PENDING, JobStatus.COMPLETED, JobStatus.ERROR, JobStatus.SKIPPED)
    for index, status in enumerate(statuses):
        panel.add_jobs([JobItem(path=tmp_path / f"video-{index}.mp4")])
        job = panel.get_jobs()[index]
        job.segments = (SegmentRange(index, index + 1),)
        job.detection_model = "rfdetr-v6"
        job.error_message = "old error"
        panel.update_job_status(job.id, status, 0.75)
    jobs = panel.get_jobs()
    (tmp_path / "video-0_restored.mp4").touch()

    panel.reset_jobs_for_run()
    tk_root.update()

    assert panel.get_jobs() == jobs
    assert [job.segments for job in jobs] == [(SegmentRange(index, index + 1),) for index in range(4)]
    assert all(job.detection_model == "rfdetr-v6" for job in jobs)
    assert all(job.status is JobStatus.PENDING for job in jobs)
    assert all(job.progress == 0.0 for job in jobs)
    assert all(job.error_message == "" for job in jobs)
    assert [job.has_conflict for job in jobs] == [True, False, False, False]
    assert all(widget._segments_btn.winfo_ismapped() for widget in panel._job_widgets)


def test_queue_footer_stacks_count_above_action_buttons(tk_root, show_window) -> None:
    tk_root.geometry("320x800")
    panel = QueuePanel(tk_root)
    panel.pack(fill="both", expand=True)
    show_window(tk_root)

    count_bottom = panel._queue_count.winfo_rooty() + panel._queue_count.winfo_height()
    actions_top = min(
        panel._clear_completed_btn.winfo_rooty(),
        panel._clear_btn.winfo_rooty(),
    )
    assert count_bottom <= actions_top

    empty_content_width = panel._empty_state.winfo_width() - 40
    assert panel._empty_label.winfo_reqwidth() <= empty_content_width


def test_queue_scrollbar_only_appears_when_jobs_overflow(tk_root) -> None:
    tk_root.geometry("420x800")
    panel = QueuePanel(tk_root)
    panel.pack(fill="both", expand=True)
    tk_root.update()

    assert not panel._list_frame._scrollbar.winfo_ismapped()

    for index in range(12):
        panel.add_jobs([JobItem(path=Path(f"/tmp/video-{index}.mp4"))])
    tk_root.update()

    assert panel._list_frame._scrollbar.winfo_ismapped()


def test_large_queue_renders_one_page_and_restores_job_state(tk_root) -> None:
    panel = QueuePanel(tk_root)
    panel.pack(fill="both", expand=True)
    for index in range(200):
        panel.add_jobs([JobItem(path=Path(f"/tmp/video-{index}.mp4"))])
    tk_root.update()

    assert sum(widget is not None for widget in panel._job_widgets) == 20
    assert panel._page_label.cget("text") == "1 / 10"

    def widget_count(widget):
        return 1 + sum(widget_count(child) for child in widget.winfo_children())

    assert widget_count(tk_root) < 3000

    processing_job = panel.get_jobs()[170]
    completed_job = panel.get_jobs()[171]
    panel.update_job_status(completed_job.id, JobStatus.COMPLETED, 1.0, elapsed_seconds=65)
    panel.set_running(True, processing_job_id=processing_job.id)
    panel._change_page(8)
    tk_root.update()

    assert not panel._job_widgets[170]._removable
    widget = panel._job_widgets[171]
    assert widget is not None
    assert widget._status_label.cget("text").endswith("1m 5s")
    assert widget._requeueable
    assert widget_count(tk_root) < 3000
    panel.set_running(False)
    panel._requeue_job(completed_job)
    assert panel.get_jobs()[-1] is completed_job
    panel._change_page(1)
    assert panel._job_widgets[-1] is not None
    assert panel._job_widgets[-1]._status_label.cget("text") == queue_panel_module.t("job_pending")


def _paged_panel(root, job_count: int) -> QueuePanel:
    panel = QueuePanel(root)
    panel.pack(fill="both", expand=True)
    for index in range(job_count):
        panel.add_jobs([JobItem(path=Path(f"/tmp/video-{index}.mp4"))])
    return panel


def _shown_jobs(panel: QueuePanel) -> list[JobItem]:
    shown = [job for job, widget in zip(panel.get_jobs(), panel._job_widgets) if widget is not None]
    packed = [panel._job_widgets.index(widget) for widget in panel._list_frame.pack_slaves() if widget in panel._job_widgets]
    assert [panel.get_jobs()[index] for index in packed] == shown
    return shown


def test_removing_a_row_keeps_other_rows_and_pulls_up_the_next_one(tk_root) -> None:
    panel = _paged_panel(tk_root, 50)
    panel._change_page(1)
    kept_rows = {id(panel.get_jobs()[index]): panel._job_widgets[index] for index in range(20, 40) if index != 23}

    panel._remove_job(panel.get_jobs()[23])

    assert _shown_jobs(panel) == panel.get_jobs()[20:40]
    for job, widget in zip(panel.get_jobs()[20:39], panel._job_widgets[20:39]):
        assert widget is kept_rows[id(job)]
    assert panel._page_label.cget("text") == "2 / 3"


def test_requeue_keeps_other_rows_and_moves_the_job_to_the_last_page(tk_root) -> None:
    panel = _paged_panel(tk_root, 50)
    job = panel.get_jobs()[2]
    panel.update_job_status(job.id, JobStatus.COMPLETED)
    kept_rows = {id(panel.get_jobs()[index]): panel._job_widgets[index] for index in range(20) if index != 2}

    panel._requeue_job(job)

    assert panel.get_jobs()[-1] is job
    assert _shown_jobs(panel) == panel.get_jobs()[:20]
    for shown_job, widget in zip(panel.get_jobs()[:19], panel._job_widgets[:19]):
        assert widget is kept_rows[id(shown_job)]
    panel._change_page(2)
    assert _shown_jobs(panel) == panel.get_jobs()[40:]
    assert panel._job_widgets[-1]._status_label.cget("text") == queue_panel_module.t("job_pending")


def test_removing_the_only_row_of_the_last_page_shows_the_previous_page(tk_root) -> None:
    panel = _paged_panel(tk_root, 21)
    panel._change_page(1)

    panel._remove_job(panel.get_jobs()[20])

    assert panel._page == 0
    assert _shown_jobs(panel) == panel.get_jobs()
    assert panel._page_label.cget("text") == "1 / 1"


def test_clear_completed_on_the_last_page_shows_the_new_last_page(tk_root) -> None:
    panel = _paged_panel(tk_root, 45)
    panel._change_page(2)
    for job in panel.get_jobs()[40:]:
        panel.update_job_status(job.id, JobStatus.COMPLETED)

    panel._on_clear_completed()

    assert panel._page == 1
    assert _shown_jobs(panel) == panel.get_jobs()[20:40]
    assert panel._page_label.cget("text") == "2 / 2"


def test_dragging_on_a_later_page_reorders_within_that_page(tk_root) -> None:
    panel = _paged_panel(tk_root, 50)
    panel._change_page(1)
    tk_root.update()
    dragged_job = panel.get_jobs()[20]
    dragged = panel._job_widgets[20]
    target = panel._job_widgets[22]
    event = tk_event(y_root=target.winfo_rooty() + target.winfo_height() // 2 + 1)

    panel._on_widget_drag_move(dragged, event)

    assert panel.get_jobs()[22] is dragged_job
    assert panel._job_widgets[22] is dragged
    assert _shown_jobs(panel) == panel.get_jobs()[20:40]


def test_segment_button_only_appears_for_pending_video_jobs(tk_root) -> None:
    panel = QueuePanel(tk_root)
    panel.pack(fill="both", expand=True)
    panel.add_jobs([JobItem(path=Path("/tmp/video.mp4"))])
    tk_root.update()

    job = panel.get_jobs()[0]
    widget = panel._job_widgets[0]
    assert widget._segments_btn.winfo_ismapped()

    panel.update_job_status(job.id, JobStatus.PROCESSING)
    tk_root.update()
    assert not widget._segments_btn.winfo_ismapped()

    panel.update_job_status(job.id, JobStatus.COMPLETED)
    tk_root.update()
    assert not widget._segments_btn.winfo_ismapped()

    panel.update_job_status(job.id, JobStatus.PENDING)
    tk_root.update()
    assert widget._segments_btn.winfo_ismapped()


def test_queue_rows_offer_video_player_and_folder_actions(tk_root) -> None:
    panel = QueuePanel(tk_root)
    panel.pack(fill="both", expand=True)
    played = MagicMock()
    panel.set_on_play(played)
    panel.add_jobs([JobItem(path=Path("/tmp/video.mp4"))])
    panel.add_jobs([JobItem(path=Path("/tmp/image.png"))])
    tk_root.update()

    video, image = panel._job_widgets
    assert video._segments_btn.winfo_ismapped()
    assert video._play_btn.winfo_ismapped()
    assert video._overflow_btn.winfo_ismapped()
    assert not image._segments_btn.winfo_ismapped()
    assert not image._play_btn.winfo_ismapped()
    assert image._overflow_btn.winfo_ismapped()

    video._handle_play()
    played.assert_called_once_with(Path("/tmp/video.mp4"))

    panel.set_running(True, processing_job_id=panel.get_jobs()[0].id)
    assert not video._play_btn.winfo_ismapped()
    assert not video._overflow_btn.winfo_ismapped()

    panel.set_running(False)
    tk_root.update()
    assert video._play_btn.winfo_ismapped()
    assert video._overflow_btn.winfo_ismapped()


def test_video_added_during_processing_has_disabled_player(tk_root) -> None:
    panel = QueuePanel(tk_root)
    panel.pack(fill="both", expand=True)
    played = MagicMock()
    panel.set_on_play(played)
    panel.set_running(True)

    panel.add_jobs([JobItem(path=Path("/tmp/later.mp4"))])
    video = panel._job_widgets[0]

    assert not video._play_btn.winfo_ismapped()
    video._handle_play()
    played.assert_not_called()


def test_queue_folder_action_uses_input_or_completed_output(tk_root, monkeypatch, tmp_path: Path) -> None:
    panel = QueuePanel(tk_root)
    panel.pack(fill="both", expand=True)
    opened = MagicMock()
    monkeypatch.setattr(queue_panel_module, "open_containing_folder", opened)
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    panel.add_jobs([JobItem(path=input_path)])
    job = panel.get_jobs()[0]
    widget = panel._job_widgets[0]

    widget._on_open_containing_folder()
    opened.assert_called_once_with(
        input_path, parent=panel.winfo_toplevel(), select_file=False
    )

    opened.reset_mock()
    job.output_path = output_path
    panel.update_job_status(job.id, JobStatus.PROCESSING)
    tk_root.update()
    assert not widget._overflow_btn.winfo_ismapped()

    panel.update_job_status(job.id, JobStatus.COMPLETED)
    tk_root.update()
    assert widget._overflow_btn.winfo_ismapped()
    widget._on_open_containing_folder()
    opened.assert_called_once_with(
        output_path, parent=panel.winfo_toplevel(), select_file=True
    )


def test_completed_job_menu_copies_output_and_requeues(tk_root, tmp_path: Path) -> None:
    panel = QueuePanel(tk_root)
    panel.pack(fill="both", expand=True)
    panel.add_jobs([JobItem(path=tmp_path / "first.mp4")])
    panel.add_jobs([JobItem(path=tmp_path / "second.mp4")])
    job = panel.get_jobs()[0]
    job.output_path = tmp_path / "first_restored.mp4"
    panel.update_job_status(job.id, JobStatus.COMPLETED)
    widget = panel._job_widgets[0]
    assert widget._has_restored_output
    assert widget._requeueable

    panel.set_running(True, processing_job_id=panel.get_jobs()[1].id)
    assert widget._requeueable
    panel.set_running(False)

    panel._copy_job_path(job)
    assert tk_root.clipboard_get() == str(job.output_path)

    panel._requeue_job(job)
    assert panel.get_jobs()[-1] is job
    assert job.status is JobStatus.PENDING
    assert job.output_path is None
    assert not panel._job_widgets[-1]._requeueable


def test_same_as_input_clears_output_and_refreshes_conflicts(tk_root, tmp_path: Path) -> None:
    panel = QueuePanel(tk_root)
    panel.pack(fill="both", expand=True)
    changed = MagicMock()
    panel.set_on_output_changed(changed)
    source = tmp_path / "clip.mp4"
    (tmp_path / "clip_restored.mp4").touch()
    panel.add_jobs([JobItem(path=source)])
    panel._output_bar.set_folder(str(tmp_path / "elsewhere"))
    changed.reset_mock()

    panel._output_bar._same_as_input_btn.invoke()

    assert panel.get_output_folder() == ""
    assert panel.get_jobs()[0].has_conflict
    assert panel._output_bar._same_as_input_btn.cget("state") == "normal"
    changed.assert_called_once_with("", panel.get_output_pattern())

    changed.reset_mock()
    panel._output_bar._output_entry.insert(0, str(tmp_path / "manual"))
    panel._output_bar._on_folder_typed()

    assert panel.get_output_folder() == str(tmp_path / "manual")
    assert panel._output_bar._same_as_input_btn.cget("fg_color") == Colors.BG_CARD
    changed.assert_called_once_with(str(tmp_path / "manual"), panel.get_output_pattern())

    panel.set_running(True, processing_job_id=panel.get_jobs()[0].id)
    assert panel._output_bar._same_as_input_btn.cget("state") == "disabled"


@pytest.mark.parametrize("hidpi", [1.0, 1.25, 1.5], indirect=True)
def test_main_workspace_starts_wider_and_can_resize_queue_panel(hidpi, jasna_app, show_window) -> None:
    show_window(jasna_app)

    assert int(jasna_app._workspace.cget("sashwidth")) == int(4 * hidpi)
    assert jasna_app._queue_panel.winfo_width() >= int(Sizing.QUEUE_PANEL_WIDTH * hidpi)
    assert int(jasna_app._workspace.panecget(jasna_app._queue_panel, "minsize")) == int(
        Sizing.QUEUE_PANEL_MIN_WIDTH * hidpi
    )

    queue_width = jasna_app._queue_panel.winfo_width()
    settings_width = jasna_app._settings_panel.winfo_width()
    sash_x = jasna_app._workspace.sash_coord(0)[0] + 2
    jasna_app._workspace.event_generate("<ButtonPress-1>", x=sash_x, y=300)
    jasna_app.update()
    jasna_app._workspace.event_generate("<B1-Motion>", x=sash_x + 80, y=300)
    jasna_app.update()
    jasna_app._workspace.event_generate("<ButtonRelease-1>", x=sash_x + 80, y=300)
    jasna_app.update()

    assert jasna_app._queue_panel.winfo_width() > queue_width
    assert jasna_app._settings_panel.winfo_width() < settings_width


def test_queue_output_path_keeps_image_extension(tk_root, tmp_path: Path) -> None:
    panel = QueuePanel(tk_root)
    panel._output_bar.set_folder(str(tmp_path))

    assert panel._get_output_path(Path("/in/photo.png")) == tmp_path / "photo_restored.png"
    assert panel._get_output_path(Path("/in/clip.mkv")) == tmp_path / "clip_restored.mp4"


def test_dropped_paths_with_spaces_and_braces_are_queued_as_dropped(tk_root, tmp_path: Path) -> None:
    folder = tmp_path / "my videos"
    folder.mkdir()
    dropped = [folder / "clip one.mp4", folder / "plain.mp4", folder / "with {braces}.mp4"]
    for path in dropped:
        path.touch()
    panel = QueuePanel(tk_root)

    drop = DnDEvent()
    drop.data = tk_root.tk.call("list", *map(str, dropped))
    panel._on_file_drop(drop)

    assert [job.path for job in panel.get_jobs()] == dropped


class _XTestPointer:
    """Drives the real X pointer, so the server's implicit button grab decides which window
    gets motion and release - the part ``event_generate`` skips."""

    def __init__(self, root) -> None:
        import ctypes
        import ctypes.util

        if root.tk.call("tk", "windowingsystem") != "x11" or not ctypes.util.find_library("Xtst"):
            pytest.skip("needs an X11 display with the XTEST extension")
        self._x11 = ctypes.CDLL(ctypes.util.find_library("X11"))
        self._xtst = ctypes.CDLL(ctypes.util.find_library("Xtst"))
        self._x11.XOpenDisplay.restype = ctypes.c_void_p
        self._x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self._x11.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        self._xtst.XTestFakeMotionEvent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
        self._xtst.XTestFakeButtonEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
        self._display = self._x11.XOpenDisplay(root.winfo_screen().encode())
        self._root = root

    def _sync(self) -> None:
        self._x11.XSync(self._display, 0)
        self._root.update()

    def move(self, x: int, y: int) -> None:
        self._xtst.XTestFakeMotionEvent(self._display, -1, x, y, 0)
        self._sync()

    def button(self, pressed: bool) -> None:
        self._xtst.XTestFakeButtonEvent(self._display, 1, int(pressed), 0)
        self._sync()

    def close(self) -> None:
        self._x11.XCloseDisplay(self._display)


@pytest.fixture
def x_pointer(tk_root):
    pointer = _XTestPointer(tk_root)
    yield pointer
    pointer.button(pressed=False)
    pointer.close()


def test_one_handle_drag_moves_the_bottom_job_to_the_top(tk_root, show_window, x_pointer) -> None:
    tk_root.geometry("320x800+0+0")
    panel = _paged_panel(tk_root, 4)
    show_window(tk_root)
    dragged_job = panel.get_jobs()[3]
    handle = panel._job_widgets[3]._handle
    x = handle.winfo_rootx() + handle.winfo_width() // 2
    start_y = handle.winfo_rooty() + handle.winfo_height() // 2
    top_y = panel._job_widgets[0].winfo_rooty()

    x_pointer.move(x, start_y)
    x_pointer.button(pressed=True)
    for y in range(start_y, top_y - 1, -8):
        x_pointer.move(x, y)
    x_pointer.button(pressed=False)

    assert panel.get_jobs()[0] is dragged_job
    assert _shown_jobs(panel)[0] is dragged_job


def test_reset_jobs_for_run_preserves_pending_elapsed_seconds_and_clears_finished(tk_root, tmp_path: Path) -> None:
    panel = QueuePanel(tk_root)
    panel.pack(fill="both", expand=True)
    paused = JobItem(path=tmp_path / "paused.mp4", elapsed_seconds=42.0)
    done = JobItem(path=tmp_path / "done.mp4")
    errored = JobItem(path=tmp_path / "err.mp4")
    skipped = JobItem(path=tmp_path / "skip.mp4")
    panel.add_jobs([paused, done, errored, skipped])
    panel.update_job_status(done.id, JobStatus.COMPLETED, 1.0, elapsed_seconds=18.0)
    errored.elapsed_seconds = 9.0
    panel.update_job_status(errored.id, JobStatus.ERROR)
    skipped.elapsed_seconds = 4.0
    panel.update_job_status(skipped.id, JobStatus.SKIPPED)

    panel.reset_jobs_for_run()

    assert paused.elapsed_seconds == 42.0
    assert done.elapsed_seconds is None
    assert errored.elapsed_seconds is None
    assert skipped.elapsed_seconds is None


@pytest.mark.usefixtures("nvidia_vendor")
@pytest.mark.parametrize("terminal_status", [JobStatus.ERROR, JobStatus.SKIPPED])
def test_requeue_after_error_or_skipped_clears_stale_start_time(
    jasna_app, monkeypatch, tmp_path: Path, terminal_status: JobStatus
) -> None:
    from types import SimpleNamespace

    from jasna.gui import app as app_module
    from jasna.gui.processor import ProgressUpdate

    clock = [10.0]
    monkeypatch.setattr(app_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    video = tmp_path / "clip.mp4"
    video.touch()
    jasna_app._queue_panel.add_jobs([JobItem(path=video)])
    job = jasna_app._queue_panel.get_jobs()[0]

    jasna_app._handle_progress(ProgressUpdate(job_id=job.id, status=JobStatus.PROCESSING, resumable=True))
    clock[0] = 30.0
    jasna_app._handle_progress(ProgressUpdate(job_id=job.id, status=terminal_status, message="failed"))
    assert job.id not in jasna_app._job_start_times

    jasna_app._queue_panel._requeue_job(job)
    assert job.status is JobStatus.PENDING
    assert job.elapsed_seconds is None

    clock[0] = 200.0
    jasna_app._handle_progress(ProgressUpdate(job_id=job.id, status=JobStatus.PROCESSING, resumable=True))
    clock[0] = 207.0
    jasna_app._handle_progress(ProgressUpdate(job_id=job.id, status=JobStatus.COMPLETED, progress=100.0))
    jasna_app._handle_complete(queue_finished=True)

    assert job.elapsed_seconds == pytest.approx(7.0)
    assert jasna_app._queue_panel._job_widgets[0]._status_label.cget("text").endswith("7s")
    assert jasna_app._control_bar._fps_label.cget("text").endswith("7s")


@pytest.mark.usefixtures("nvidia_vendor")
def test_requeue_completed_job_mid_run_excludes_superseded_time_from_control_bar(
    jasna_app, monkeypatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    from jasna.gui import app as app_module
    from jasna.gui.processor import ProgressUpdate

    clock = [10.0]
    monkeypatch.setattr(app_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    first, second = JobItem(path=tmp_path / "a.mp4"), JobItem(path=tmp_path / "b.mp4")
    jasna_app._queue_panel.add_jobs([first, second])

    jasna_app._handle_progress(ProgressUpdate(job_id=first.id, status=JobStatus.PROCESSING, resumable=True))
    clock[0] = 20.0
    jasna_app._handle_progress(ProgressUpdate(job_id=first.id, status=JobStatus.COMPLETED, progress=100.0))

    jasna_app._handle_progress(ProgressUpdate(job_id=second.id, status=JobStatus.PROCESSING, resumable=True))
    clock[0] = 25.0
    jasna_app._handle_progress(ProgressUpdate(job_id=second.id, status=JobStatus.PENDING))

    jasna_app._queue_panel._requeue_job(first)

    clock[0] = 50.0
    jasna_app._handle_progress(ProgressUpdate(job_id=second.id, status=JobStatus.PROCESSING, resumable=True))
    clock[0] = 53.0
    jasna_app._handle_progress(ProgressUpdate(job_id=second.id, status=JobStatus.COMPLETED, progress=100.0))

    jasna_app._handle_progress(ProgressUpdate(job_id=first.id, status=JobStatus.PROCESSING, resumable=True))
    clock[0] = 65.0
    jasna_app._handle_progress(ProgressUpdate(job_id=first.id, status=JobStatus.COMPLETED, progress=100.0))
    jasna_app._handle_complete(queue_finished=True)

    assert second.elapsed_seconds == pytest.approx(8.0)
    assert first.elapsed_seconds == pytest.approx(12.0)
    assert jasna_app._control_bar._fps_label.cget("text").endswith("20s")
