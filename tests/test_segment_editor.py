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
