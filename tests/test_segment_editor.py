from __future__ import annotations

import queue
import threading
from pathlib import Path
from tkinter import TclError
from unittest.mock import MagicMock

import customtkinter as ctk
import pytest

from jasna.gui import segment_editor
from jasna.gui.locales import t
from jasna.gui.models import JobItem
from jasna.gui.segment_editor import SegmentEditor
from jasna.gui.segment_editor_state import SegmentEditorState


def test_out_of_bounds_range_uses_specific_message() -> None:
    editor = object.__new__(SegmentEditor)
    editor._state = SegmentEditorState(duration=10, fps=30)
    editor._start_entry = MagicMock()
    editor._start_entry.get.return_value = "9"
    editor._end_entry = MagicMock()
    editor._end_entry.get.return_value = "11"
    editor._refresh_notice = MagicMock()

    SegmentEditor._add_or_update(editor)

    assert editor._edit_notice == t("segments_time_out_of_bounds")
    editor._refresh_notice.assert_called_once_with()


def test_preview_surface_selects_original_or_restored_from_toggle_state() -> None:
    editor = object.__new__(SegmentEditor)
    editor._closed = threading.Event()
    editor._preview = MagicMock()
    editor._preview_source = MagicMock(name="original")
    editor._restored_source = MagicMock(name="restored")
    editor._fit_to_label = MagicMock(side_effect=["original-image", "restored-image"])
    editor._resize_after = "pending"
    editor._scan_overlay = False

    editor._restore_active = False
    SegmentEditor._refresh_preview_image(editor)

    editor._fit_to_label.assert_called_with(editor._preview, editor._preview_source)
    editor._preview.configure.assert_called_with(image="original-image", text="")

    editor._restore_active = True
    SegmentEditor._refresh_preview_image(editor)

    editor._fit_to_label.assert_called_with(editor._preview, editor._restored_source)
    editor._preview.configure.assert_called_with(image="restored-image", text="")


def test_segment_editor_maps_before_taking_modal_grab(monkeypatch) -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    worker = MagicMock()
    worker.events = queue.Queue()
    monkeypatch.setattr(
        segment_editor,
        "SegmentPreviewWorker",
        MagicMock(return_value=worker),
    )
    root.update()
    editor = None
    try:
        editor = SegmentEditor(
            root,
            JobItem(Path("video.mp4")),
            MagicMock(),
            lambda: False,
            MagicMock(),
            MagicMock(),
        )

        assert editor.winfo_viewable()
        assert editor.grab_current() == editor
    finally:
        if editor is not None:
            editor._finish_close()
        root.destroy()


def _fake_metadata() -> object:
    from fractions import Fraction

    from av.video.reformatter import ColorRange as AvColorRange
    from av.video.reformatter import Colorspace as AvColorspace

    from jasna.media import VideoMetadata

    return VideoMetadata(
        video_file="video.mp4",
        video_height=1080,
        video_width=1920,
        video_fps=30.0,
        average_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        codec_name="h264",
        duration=60.0,
        time_base=Fraction(1, 90000),
        start_pts=0,
        color_range=AvColorRange.MPEG,
        color_space=AvColorspace.ITU709,
        num_frames=1800,
        is_10bit=False,
    )


def _build_editor_with_ui(root, monkeypatch) -> SegmentEditor:
    worker = MagicMock()
    worker.events = queue.Queue()
    monkeypatch.setattr(
        segment_editor, "SegmentPreviewWorker", MagicMock(return_value=worker)
    )
    root.update()
    editor = SegmentEditor(
        root,
        JobItem(Path("video.mp4")),
        MagicMock(),
        lambda: False,
        MagicMock(),
        MagicMock(),
    )
    worker.events.put(segment_editor.PreviewLoaded(_fake_metadata()))
    editor._poll_workers()
    root.update()
    assert editor._state is not None
    return editor


def test_scan_bar_builds_with_editor(monkeypatch) -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    editor = None
    try:
        editor = _build_editor_with_ui(root, monkeypatch)
        assert editor._scan_btn.cget("text") == t("segments_scan")
        assert editor._scan_stop_btn.cget("state") == "disabled"
        assert editor._scan_add_btn.cget("state") == "disabled"
        assert editor._scan_interval.get() == "1s"
    finally:
        if editor is not None:
            editor._finish_close()
        root.destroy()


def test_scan_lock_disables_everything_but_stop(monkeypatch) -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    editor = None
    try:
        editor = _build_editor_with_ui(root, monkeypatch)
        editor._set_scan_locked(True)
        for widget in editor._scan_lockable_widgets():
            assert widget.cget("state") == "disabled"
        assert editor._scan_stop_btn.cget("state") == "normal"

        editor._set_scan_locked(False)
        assert editor._scan_btn.cget("state") == "normal"
        assert editor._scan_stop_btn.cget("state") == "disabled"
        assert editor._apply_btn.cget("state") == "normal"
    finally:
        if editor is not None:
            editor._finish_close()
        root.destroy()


def test_scan_completed_populates_detections_and_add_button(monkeypatch) -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    editor = None
    try:
        editor = _build_editor_with_ui(root, monkeypatch)
        result = segment_editor.MosaicScanResult(
            times=(0.0, 1.0, 2.0, 3.0),
            scores=(0.0, 0.8, 0.9, 0.0),
            masks=[None] * 4,
            stride=1.0,
            duration=60.0,
            completed_until=3.0,
        )
        editor._scan_threshold = 0.5
        editor._scan_worker = MagicMock()
        editor._set_scan_locked(True)
        editor._handle_scan_event(segment_editor.ScanCompleted(result, stopped=False))

        assert editor._scan_worker is None
        assert editor._scan_result is result
        assert editor._timeline._detections
        assert editor._scan_proposals
        assert editor._scan_add_btn.cget("state") == "normal"

        editor._add_detected_ranges()
        assert editor._state.segments
        assert editor._state.segments[0].start == pytest.approx(0.5)
        assert editor._state.segments[0].end == pytest.approx(3.5)
    finally:
        if editor is not None:
            editor._finish_close()
        root.destroy()
