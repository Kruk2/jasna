from __future__ import annotations

import queue
import threading
from fractions import Fraction
from pathlib import Path
from tkinter import TclError
from unittest.mock import MagicMock

import customtkinter as ctk
import pytest
import torch
from PIL import Image

from jasna.gui import segment_editor
from jasna.gui.locales import t
from jasna.gui.models import AppSettings, JobItem
from jasna.gui.segment_editor import SegmentEditor
from jasna.gui.segment_editor_state import SegmentEditorState
from jasna.media.splice import KeyframeIndex


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
            lambda: AppSettings(),
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
        lambda: AppSettings(),
        lambda: False,
        MagicMock(),
        MagicMock(),
    )
    worker.events.put(segment_editor.PreviewLoaded(_fake_metadata()))
    editor._poll_workers()
    root.update()
    assert editor._state is not None
    return editor


def test_editor_height_grows_on_tall_screens() -> None:
    editor = object.__new__(SegmentEditor)
    editor.winfo_screenwidth = MagicMock(return_value=2560)
    editor.winfo_screenheight = MagicMock(return_value=1440)
    editor.geometry = MagicMock()
    editor.minsize = MagicMock()

    SegmentEditor._size_and_center(editor)

    editor.geometry.assert_called_once_with("1826x1240+367+100")
    editor.minsize.assert_called_once_with(900, 640)


def test_previous_frame_uses_exact_decoder_predecessor() -> None:
    editor = object.__new__(SegmentEditor)
    editor._state = SegmentEditorState(duration=10.0, fps=30.0)
    editor._current = 1.0
    editor._set_playing = MagicMock()
    editor._time_label = MagicMock()
    editor._time_text = MagicMock(return_value="time")
    editor._timeline = MagicMock()
    editor._refresh_timeline = MagicMock()
    editor._preview_worker = MagicMock()
    editor._preview_worker.previous_frame.return_value = 7
    editor._restore_active = False

    SegmentEditor._step(editor, -1)

    editor._preview_worker.previous_frame.assert_called_once_with(1.0)
    assert editor._preview_generation == 7
    assert editor._current == pytest.approx(29 / 30)


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
        assert not editor._scan_activity.winfo_ismapped()
        assert editor._scan_interval.get() == t("segments_scan_frequency_one")
        assert (
            editor._scan_interval.cget("values")[0]
            == t("segments_scan_frequency_every_frame")
        )
        assert editor._scan_model.get() == AppSettings().detection_model
        assert editor._scan_thr_label.cget("text") == "0.25"
        assert editor._scan_overlay
        assert editor._scan_overlay_toggle.get() == 1
    finally:
        if editor is not None:
            editor._finish_close()
        root.destroy()


def test_source_codec_notice_only_shows_for_selected_ranges(monkeypatch) -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    editor = None
    try:
        editor = _build_editor_with_ui(root, monkeypatch)
        assert not editor._codec_notice.winfo_ismapped()

        editor._state.add(1.0, 2.0)
        editor._refresh_workload()
        editor.geometry("900x640")
        editor.update_idletasks()

        assert editor._codec_notice.winfo_ismapped()
        assert editor._codec_notice.cget("text") == t(
            "segments_source_codec_notice",
            codec="H.264 (AVC)",
        )
        assert (
            editor._codec_notice.winfo_reqwidth()
            <= editor._codec_notice.master.winfo_width()
        )

        editor._state.clear()
        editor._refresh_workload()
        editor.update_idletasks()

        assert not editor._codec_notice.winfo_ismapped()
    finally:
        if editor is not None:
            editor._finish_close()
        root.destroy()


def test_scan_card_stays_compact_and_visible_at_minimum_size(monkeypatch) -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    editor = None
    try:
        editor = _build_editor_with_ui(root, monkeypatch)
        editor.geometry("900x640")
        editor.update()

        assert editor._scan_card.winfo_reqheight() < 100
        assert (
            editor._scan_model.winfo_rootx() + editor._scan_model.winfo_width()
            < editor._scan_interval.winfo_rootx()
        )
        assert (
            editor._scan_interval.winfo_rootx() + editor._scan_interval.winfo_width()
            < editor._scan_thr_slider.winfo_rootx()
        )
        assert (
            editor._scan_btn.winfo_rootx() + editor._scan_btn.winfo_width()
            <= editor._scan_card.winfo_rootx() + editor._scan_card.winfo_width()
        )
        assert (
            editor._apply_btn.winfo_rooty() + editor._apply_btn.winfo_height()
            <= editor.winfo_rooty() + editor.winfo_height()
        )
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
        editor.update_idletasks()
        for widget in editor._scan_lockable_widgets():
            assert widget.cget("state") == "disabled"
        assert editor._scan_stop_btn.cget("state") == "normal"
        assert editor._scan_stop_btn.winfo_ismapped()
        assert editor._scan_progress.winfo_ismapped()
        assert not editor._timeline._enabled

        editor._set_scan_locked(False)
        assert editor._scan_btn.cget("state") == "normal"
        assert editor._scan_stop_btn.cget("state") == "disabled"
        assert not editor._scan_activity.winfo_ismapped()
        assert editor._apply_btn.cget("state") == "normal"
        assert editor._timeline._enabled
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
        editor.update_idletasks()

        assert editor._scan_worker is not None
        assert editor._scan_result is result
        assert editor._timeline._detections
        assert editor._scan_proposals
        assert editor._scan_add_btn.cget("state") == "normal"
        assert editor._scan_activity.winfo_ismapped()
        assert editor._scan_add_btn.winfo_ismapped()
        assert editor._scan_btn.cget("text") == t("segments_scan_again")
        assert editor._scan_status.cget("text") == t(
            "segments_scan_result",
            count=1,
            duration="00:00:03",
        )

        editor._add_detected_ranges()
        assert editor._state.segments
        assert editor._state.segments[0].start == pytest.approx(0.5)
        assert editor._state.segments[0].end == pytest.approx(3.5)
        assert editor._scan_add_btn.cget("state") == "disabled"
        assert not editor._scan_add_btn.winfo_ismapped()
    finally:
        if editor is not None:
            editor._finish_close()
        root.destroy()


def test_scan_threshold_updates_ranges_and_add_button(monkeypatch) -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    editor = None
    try:
        editor = _build_editor_with_ui(root, monkeypatch)
        result = segment_editor.MosaicScanResult(
            times=(0.0, 1.0, 2.0),
            scores=(0.0, 0.8, 0.0),
            masks=torch.zeros((3, 90, 160), dtype=torch.uint8),
            stride=1.0,
            duration=60.0,
            completed_until=2.0,
        )
        editor._scan_threshold = 0.5
        editor._scan_worker = MagicMock()
        editor._handle_scan_event(segment_editor.ScanCompleted(result, stopped=False))
        assert editor._timeline._detections
        assert editor._scan_add_btn.cget("state") == "normal"

        editor._on_scan_threshold(0.9)
        editor.after_cancel(editor._scan_thr_after)
        editor._apply_scan_threshold()

        assert editor._timeline._detections == ()
        assert editor._scan_add_btn.cget("state") == "disabled"
        assert not editor._scan_add_btn.winfo_ismapped()
    finally:
        if editor is not None:
            editor._finish_close()
        root.destroy()


def test_scan_overlay_respects_dynamic_threshold() -> None:
    editor = object.__new__(SegmentEditor)
    editor._state = SegmentEditorState(duration=10.0, fps=30.0)
    editor._current = 1.0
    editor._scan_threshold = 0.8
    editor._scan_result = segment_editor.MosaicScanResult(
        times=(1.0,),
        scores=(0.7,),
        masks=torch.ones((1, 90, 160), dtype=torch.uint8),
        stride=1.0,
        duration=10.0,
        completed_until=1.0,
    )
    image = Image.new("RGB", (160, 90), "black")

    assert editor._apply_scan_overlay(image) is image

    editor._scan_threshold = 0.6
    overlaid = editor._apply_scan_overlay(image)
    assert overlaid.getpixel((80, 45))[0] > 0


def test_smart_render_error_is_explained_once(monkeypatch) -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    editor = None
    try:
        editor = _build_editor_with_ui(root, monkeypatch)
        editor._state.add(0.0, 58.0)
        editor._analysis_error = None
        editor._keyframe_index = KeyframeIndex(
            pts=(0,),
            time_base=Fraction(1, 1),
            start_pts=0,
            end_pts=60,
        )

        editor._refresh_workload()

        assert editor._workload.cget("text") != editor._notice.cget("text")
        assert editor._notice.cget("text") == t("segments_smart_render_whole_video")
        assert editor._apply_btn.cget("state") == "disabled"
    finally:
        if editor is not None:
            editor._finish_close()
        root.destroy()


def test_save_remembers_detection_settings_on_video(monkeypatch) -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    editor = None
    try:
        editor = _build_editor_with_ui(root, monkeypatch)
        editor._scan_model.set("lada-yolo-v4")
        editor._scan_threshold = 0.55
        editor._finish_close = MagicMock()

        editor._save()

        assert editor._job.detection_model == "lada-yolo-v4"
        assert editor._job.detection_score_threshold == 0.55
        editor._finish_close.assert_called_once_with()
    finally:
        if editor is not None and editor.winfo_exists():
            SegmentEditor._finish_close(editor)
        root.destroy()


def test_suggest_mask_button_locked_during_scan(monkeypatch) -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    editor = None
    try:
        editor = _build_editor_with_ui(root, monkeypatch)
        assert editor._suggest_btn.cget("text") == t("segments_suggest_mask")
        assert editor._suggest_btn in editor._scan_lockable_widgets()
        editor._set_scan_locked(True)
        assert editor._suggest_btn.cget("state") == "disabled"
        editor._set_scan_locked(False)
        assert editor._suggest_btn.cget("state") == "normal"
    finally:
        if editor is not None:
            editor._finish_close()
        root.destroy()


def test_suggest_mask_grabs_full_frame_and_opens_dialog(monkeypatch) -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    editor = None
    opened = {}

    class _FakeDialog:
        def __init__(self, master, image, on_submit, on_closed=None):
            opened.update(image=image, on_submit=on_submit, on_closed=on_closed)

    monkeypatch.setattr(segment_editor, "MaskSuggestDialog", _FakeDialog)
    try:
        editor = _build_editor_with_ui(root, monkeypatch)
        editor._suggest_mask()
        assert editor._suggest_busy
        assert editor._suggest_btn.cget("state") == "disabled"
        editor._preview_worker.grab_full.assert_called_once_with()

        from PIL import Image

        frame = Image.new("RGB", (1920, 1080))
        editor._open_mask_suggest(frame)
        assert opened["image"] is frame

        opened["on_closed"]()
        assert not editor._suggest_busy
        assert editor._suggest_btn.cget("state") == "normal"
    finally:
        if editor is not None:
            editor._finish_close()
        root.destroy()


def test_feedback_upload_events_show_toast(monkeypatch) -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    editor = None
    toasts = []
    try:
        editor = _build_editor_with_ui(root, monkeypatch)
        monkeypatch.setattr(
            type(editor),
            "_show_toast",
            lambda self, message, type_: toasts.append((message, type_)),
        )
        editor._handle_feedback_event(segment_editor.FeedbackUploadFinished(True, ""))
        editor._handle_feedback_event(
            segment_editor.FeedbackUploadFinished(False, "boom")
        )
        assert toasts[0] == (t("mask_feedback_uploaded"), "success")
        assert toasts[1][1] == "error"
        assert "boom" in toasts[1][0]
    finally:
        if editor is not None:
            editor._finish_close()
        root.destroy()


@pytest.mark.parametrize("scaling", [1.0, 1.5])
def test_fit_to_label_compensates_widget_scaling(monkeypatch, scaling: float) -> None:
    editor = object.__new__(SegmentEditor)
    label = MagicMock()
    label.winfo_width.return_value = 916
    label.winfo_height.return_value = 556
    monkeypatch.setattr(
        ctk.ScalingTracker, "get_widget_scaling", staticmethod(lambda widget: scaling)
    )
    source = Image.new("RGB", (1920, 1080))

    result = SegmentEditor._fit_to_label(editor, label, source)

    rendered = (
        round(result._size[0] * scaling),
        round(result._size[1] * scaling),
    )
    assert rendered[0] <= 900 and rendered[1] <= 540
    assert result._light_image.size == (900, 506)
