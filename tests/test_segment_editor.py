from __future__ import annotations

import queue
import threading
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from tkinter import TclError
from unittest.mock import MagicMock

import customtkinter as ctk
import pytest
import torch
from PIL import Image

from jasna.gui import segment_editor
from jasna.gui.locales import t
from jasna.gui.models import AppSettings, JobItem
from jasna.gui.mosaic_scan import MosaicScanResult, ScanCompleted
from jasna.gui.segment_editor import SegmentEditor
from jasna.gui.segment_editor_state import PREVIEW_ZOOM_MAX, PreviewView, SegmentEditorState
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
        editor.withdraw()
        editor.event_generate("<FocusIn>")
        root.update()
        assert editor.winfo_viewable()
    finally:
        if editor is not None:
            editor._finish_close()
        root.destroy()


def _fake_metadata(*, width: int = 1920, height: int = 1080) -> object:
    from fractions import Fraction

    from av.video.reformatter import ColorRange as AvColorRange
    from av.video.reformatter import Colorspace as AvColorspace

    from jasna.media.probe import VideoMetadata

    return VideoMetadata(
        video_file="video.mp4",
        video_height=height,
        video_width=width,
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


def _build_editor_with_ui(
    root,
    monkeypatch,
    *,
    metadata=None,
    path: Path = Path("video.mp4"),
) -> SegmentEditor:
    worker = MagicMock()
    worker.events = queue.Queue()
    monkeypatch.setattr(
        segment_editor, "SegmentPreviewWorker", MagicMock(return_value=worker)
    )
    root.update()
    editor = SegmentEditor(
        root,
        JobItem(path),
        lambda: AppSettings(),
        lambda: False,
        MagicMock(),
        MagicMock(),
    )
    worker.events.put(segment_editor.PreviewLoaded(metadata or _fake_metadata()))
    editor._poll_workers()
    root.update()
    assert editor._state is not None
    return editor


def test_preview_surface_selects_original_or_restored_from_toggle_state(monkeypatch) -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    editor = None
    try:
        editor = _build_editor_with_ui(root, monkeypatch)
        editor._scan_panel._overlay = False
        editor._preview_source = Image.new("RGB", (160, 90), "red")
        editor._restored_source = Image.new("RGB", (160, 90), "blue")

        editor._restore_active = False
        editor._refresh_preview_image()
        assert editor._preview_image._light_image.getpixel((0, 0)) == (255, 0, 0)

        editor._restore_active = True
        editor._refresh_preview_image()
        assert editor._preview_image._light_image.getpixel((0, 0)) == (0, 0, 255)
    finally:
        if editor is not None:
            editor._finish_close()
        root.destroy()


def test_projection_selector_is_disabled_for_non_vr_video(monkeypatch) -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    editor = None
    try:
        editor = _build_editor_with_ui(root, monkeypatch)

        assert editor._vr_projection_menu.cget("state") == "disabled"
    finally:
        if editor is not None:
            editor._finish_close()
        root.destroy()


def test_projection_selector_is_enabled_for_detected_vr_video(monkeypatch) -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    editor = None
    try:
        editor = _build_editor_with_ui(
            root,
            monkeypatch,
            metadata=_fake_metadata(width=3840, height=1920),
        )

        assert editor._vr_projection_menu.cget("state") == "normal"
        assert editor._vr_projection_menu.get_value() == "auto"
        assert editor._vr_projection_label.cget("text") == t(
            "segments_vr_projection_resolved",
            projection=t("segments_vr_projection_raw"),
        )
    finally:
        if editor is not None:
            editor._finish_close()
        root.destroy()


def test_projection_selector_is_next_to_restore_preview_with_long_filename(
    monkeypatch,
) -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    editor = None
    try:
        editor = _build_editor_with_ui(
            root,
            monkeypatch,
            metadata=_fake_metadata(width=3840, height=1920),
            path=Path(f"SAVR-{'x' * 180}.mp4"),
        )
        editor.geometry("900x640")
        editor.update()

        menu_right = (
            editor._vr_projection_menu.winfo_rootx()
            + editor._vr_projection_menu.winfo_width()
        )
        editor_right = editor.winfo_rootx() + editor.winfo_width()
        assert editor._vr_projection_menu.winfo_ismapped()
        assert editor._vr_projection_menu.cget("state") == "normal"
        assert menu_right <= editor_right
        assert (
            editor._vr_projection_menu.master.master
            is editor._restore_toggle.master.master
        )
    finally:
        if editor is not None:
            editor._finish_close()
        root.destroy()


def test_projection_change_restarts_active_restoration_preview() -> None:
    editor = object.__new__(SegmentEditor)
    editor._vr_projection = "auto"
    editor._restore_active = True
    editor._restored_clip = (object(),)
    editor._restored_source = object()
    editor._restore_play_pending = True
    editor._set_playing = MagicMock()
    editor._schedule_restoration_preview = MagicMock()

    editor._on_vr_projection_changed("fisheye")

    assert editor._vr_projection == "fisheye"
    assert editor._restored_clip == ()
    assert editor._restored_source is None
    assert not editor._restore_play_pending
    editor._set_playing.assert_called_once_with(False)
    editor._schedule_restoration_preview.assert_called_once_with()


def test_pan_zoom_controls_are_visible_and_explain_gestures(monkeypatch) -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    editor = None
    try:
        editor = _build_editor_with_ui(root, monkeypatch)

        assert editor._pan_zoom_hint.winfo_ismapped()
        assert editor._pan_zoom_hint.cget("text") == t(
            "segments_preview_pan_zoom_hint"
        )
        assert editor._zoom_out_btn.winfo_ismapped()
        assert editor._zoom_label.cget("text") == "100%"
        assert editor._zoom_in_btn.winfo_ismapped()
        assert not editor._reset_view_btn.winfo_ismapped()
        assert editor._reset_view_btn.cget("text") == t(
            "segments_preview_reset_view"
        )

        editor._adjust_preview_zoom(0.25)
        editor.update()
        assert editor._reset_view_btn.winfo_ismapped()

        editor._reset_preview_view()
        editor.update()
        assert not editor._reset_view_btn.winfo_ismapped()
    finally:
        if editor is not None:
            editor._finish_close()
        root.destroy()


def test_preview_crop_uses_zoom_and_clamps_pan_to_source() -> None:
    view = PreviewView(zoom=2.0, center=(0.0, 0.0))

    assert view.crop_box(400, 200) == (0, 0, 200, 100)
    assert view.center == pytest.approx((0.25, 0.25))


def test_preview_view_zoom_limits_and_reset() -> None:
    view = PreviewView()

    assert view.is_reset
    assert not view.zoom_to(0.5, None)
    assert view.zoom_to(1.5, None)
    assert view.zoomed and not view.is_reset
    assert view.zoom_to(100.0, None)
    assert view.zoom == PREVIEW_ZOOM_MAX

    view.center = (0.7, 0.6)
    view.reset()
    assert view.is_reset
    assert view.crop_box(400, 200) is None


def test_preview_zoom_keeps_anchor_point_still() -> None:
    view = PreviewView()

    view.zoom_to(2.0, (1.0, 0.5))

    assert view.center == pytest.approx((0.75, 0.5))


def test_dragging_zoomed_preview_pans_the_source() -> None:
    view = PreviewView(zoom=2.0)

    view.pan(0.1, 0.0)

    assert view.center[0] < 0.5
    assert view.center[1] == pytest.approx(0.5)


def test_editor_height_grows_on_tall_screens(monkeypatch) -> None:
    monkeypatch.setattr(segment_editor.scaling, "window_scaling", lambda _window: 1.0)
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
        assert editor._scan_panel._scan_btn.cget("text") == t("segments_scan")
        assert editor._scan_panel._stop_btn.cget("state") == "disabled"
        assert editor._scan_panel._add_btn.cget("state") == "disabled"
        assert not editor._scan_panel._activity.winfo_ismapped()
        assert editor._scan_panel._interval.get() == t("segments_scan_frequency_one")
        assert (
            editor._scan_panel._interval.cget("values")[0]
            == t("segments_scan_frequency_every_frame")
        )
        assert editor._scan_panel._model.get() == AppSettings().detection_model
        assert editor._scan_panel._threshold_label.cget("text") == f"{AppSettings().detection_score_threshold:.2f}"
        assert editor._scan_panel._overlay
        assert editor._scan_panel._overlay_toggle.get() == 1
    finally:
        if editor is not None:
            editor._finish_close()
        root.destroy()


def test_scan_model_change_applies_recommended_threshold(monkeypatch) -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    editor = None
    try:
        editor = _build_editor_with_ui(root, monkeypatch)

        editor._scan_panel._on_model_changed("rfdetr-v6-large")

        assert editor._scan_panel._threshold == pytest.approx(0.40)
        assert editor._scan_panel._threshold_slider.get() == pytest.approx(0.40)
        assert editor._scan_panel._threshold_label.cget("text") == "0.40"
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

        assert editor._scan_panel.winfo_reqheight() < 100
        assert (
            editor._scan_panel._model.winfo_rootx() + editor._scan_panel._model.winfo_width()
            < editor._scan_panel._interval.winfo_rootx()
        )
        assert (
            editor._scan_panel._interval.winfo_rootx() + editor._scan_panel._interval.winfo_width()
            < editor._scan_panel._threshold_slider.winfo_rootx()
        )
        assert (
            editor._scan_panel._scan_btn.winfo_rootx() + editor._scan_panel._scan_btn.winfo_width()
            <= editor._scan_panel.winfo_rootx() + editor._scan_panel.winfo_width()
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
        editor._scan_panel._set_locked(True)
        editor.update()
        for widget in (*editor._lockable_widgets(), *editor._scan_panel.lockable_widgets()):
            assert widget.cget("state") == "disabled"
        assert editor._scan_panel._stop_btn.cget("state") == "normal"
        assert editor._scan_panel._stop_btn.winfo_ismapped()
        assert editor._scan_panel._progress.winfo_ismapped()
        assert not editor._timeline._enabled

        editor._scan_panel._set_locked(False)
        editor.update()
        assert editor._scan_panel._scan_btn.cget("state") == "normal"
        assert editor._scan_panel._stop_btn.cget("state") == "disabled"
        assert not editor._scan_panel._activity.winfo_ismapped()
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
        result = MosaicScanResult(
            times=(0.0, 1.0, 2.0, 3.0),
            scores=(0.0, 0.8, 0.9, 0.0),
            masks=[None] * 4,
            stride=1.0,
            duration=60.0,
            completed_until=3.0,
        )
        editor._scan_panel._threshold = 0.5
        editor._scan_panel._worker = MagicMock()
        editor._scan_panel._set_locked(True)
        editor._scan_panel._handle_event(ScanCompleted(result, stopped=False))
        editor.update_idletasks()

        assert editor._scan_panel._worker is not None
        assert editor._scan_panel._result is result
        assert editor._timeline._detections
        assert editor._scan_panel._proposals
        assert editor._scan_panel._add_btn.cget("state") == "normal"
        assert editor._scan_panel._activity.winfo_ismapped()
        assert editor._scan_panel._add_btn.winfo_ismapped()
        assert editor._scan_panel._scan_btn.cget("text") == t("segments_scan_again")
        assert editor._scan_panel._status.cget("text") == t(
            "segments_scan_result",
            count=1,
            duration="00:00:03",
        )

        editor._scan_panel._add_detected_ranges()
        assert editor._state.segments
        assert editor._state.segments[0].start == pytest.approx(0.5)
        assert editor._state.segments[0].end == pytest.approx(3.5)
        assert editor._scan_panel._add_btn.cget("state") == "disabled"
        assert not editor._scan_panel._add_btn.winfo_ismapped()
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
        result = MosaicScanResult(
            times=(0.0, 1.0, 2.0),
            scores=(0.0, 0.8, 0.0),
            masks=torch.zeros((3, 90, 160), dtype=torch.uint8),
            stride=1.0,
            duration=60.0,
            completed_until=2.0,
        )
        editor._scan_panel._threshold = 0.5
        editor._scan_panel._worker = MagicMock()
        editor._scan_panel._handle_event(ScanCompleted(result, stopped=False))
        assert editor._timeline._detections
        assert editor._scan_panel._add_btn.cget("state") == "normal"

        editor._scan_panel._on_threshold(0.9)
        editor._scan_panel.after_cancel(editor._scan_panel._threshold_after)
        editor._scan_panel._apply_threshold()

        assert editor._timeline._detections == ()
        assert editor._scan_panel._add_btn.cget("state") == "disabled"
        assert not editor._scan_panel._add_btn.winfo_ismapped()
    finally:
        if editor is not None:
            editor._finish_close()
        root.destroy()


def test_scan_overlay_respects_dynamic_threshold(monkeypatch) -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    editor = None
    try:
        editor = _build_editor_with_ui(root, monkeypatch)
        panel = editor._scan_panel
        editor._current = 1.0
        panel._threshold = 0.8
        panel._result = MosaicScanResult(
            times=(1.0,),
            scores=(0.7,),
            masks=torch.ones((1, 90, 160), dtype=torch.uint8),
            stride=1.0,
            duration=10.0,
            completed_until=1.0,
        )
        image = Image.new("RGB", (160, 90), "black")

        assert panel.apply_overlay(image) is image

        panel._threshold = 0.6
        overlaid = panel.apply_overlay(image)
        assert overlaid.getpixel((80, 45))[0] > 0
    finally:
        if editor is not None:
            editor._finish_close()
        root.destroy()


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


def test_save_remembers_detection_and_projection_settings_on_video(monkeypatch) -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    editor = None
    try:
        editor = _build_editor_with_ui(root, monkeypatch)
        editor._scan_panel._model.set("lada-yolo-v4")
        editor._scan_panel._threshold = 0.55
        editor._vr_projection = "gnomonic"
        editor._finish_close = MagicMock()

        editor._save()

        assert editor._job.detection_model == "lada-yolo-v4"
        assert editor._job.detection_score_threshold == 0.55
        assert editor._job.vr_projection == "gnomonic"
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
        assert editor._suggest_btn in editor._lockable_widgets()
        editor._scan_panel._set_locked(True)
        assert editor._suggest_btn.cget("state") == "disabled"
        editor._scan_panel._set_locked(False)
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
