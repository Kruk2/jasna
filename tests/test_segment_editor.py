from __future__ import annotations

import queue
from dataclasses import replace
import threading
from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock

import customtkinter as ctk
import pytest
import torch
from PIL import Image

from jasna.gui import (
    ltx_seed_preview,
    segment_editor,
    segment_range_panel,
    segment_restore_preview,
    segment_scan_panel,
    segment_seed_panel,
)
from jasna.gui.locales import t
from jasna.gui.ltx_models import model_key
from jasna.gui.ltx_seed_preview import LtxSeedPreviewWorker, SeedFrame, SeedProgress, SeedReady
from jasna.gui.models import AppSettings, JobItem
from jasna.gui.mosaic_scan import MosaicScanResult, ScanCompleted
from jasna.gui.segment_editor import SegmentEditor
from jasna.gui.restoration_preview import RestorationClip, RestorationFrame, RestorationPreviewWorker, RestoredClipFrame
from jasna.ltx.inference import Cancelled
from jasna.media.splice import KeyframeIndex
from jasna.segments import SegmentRange, SegmentRestoration
from jasna.session_config import LTX_DEFAULT_MODEL
from jasna.vr180 import resolve_vr_mode
from factories import scheduled_after_ids, shown_message, shown_pixel


def test_out_of_bounds_range_uses_specific_message(build_editor) -> None:
    editor = build_editor()
    editor._ranges._start_entry.insert(0, "59")
    editor._ranges._end_entry.insert(0, "61")

    editor._ranges.add_or_update()

    assert editor._notice.cget("text") == t("segments_time_out_of_bounds")
    assert editor._state.segments == ()


def test_segment_editor_maps_before_taking_modal_grab(build_editor, tk_root) -> None:
    editor = build_editor()

    assert editor.winfo_viewable()
    assert editor.grab_current() == editor
    editor.withdraw()
    editor.event_generate("<FocusIn>")
    tk_root.update()
    assert editor.winfo_viewable()


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


def _build_editor_with_ui(root, monkeypatch, *,
    metadata=None,
    path: Path = Path("video.mp4"),
    settings: AppSettings = AppSettings(),
    job: JobItem | None = None,
    ltx_unavailable_reason: str | None = None,
    set_preview_gpu_busy=None,
) -> SegmentEditor:
    worker = MagicMock()
    worker.events = queue.Queue()
    monkeypatch.setattr(
        segment_editor, "SegmentPreviewWorker", MagicMock(return_value=worker)
    )
    root.update()
    editor = SegmentEditor(
        root,
        job or JobItem(path),
        lambda: settings,
        lambda: False,
        set_preview_gpu_busy or MagicMock(),
        ltx_unavailable_reason,
        MagicMock(),
        MagicMock(),
    )
    metadata = metadata or _fake_metadata()
    worker.events.put(segment_editor.PreviewLoaded(metadata, resolve_vr_mode(settings.vr_mode, metadata, path)))
    editor._poll_workers()
    root.update()
    assert editor._state is not None
    return editor


@pytest.fixture
def build_editor(tk_root, monkeypatch):
    """Build loaded segment editors over ``tk_root``; any still open are closed after the test."""
    editors: list[SegmentEditor] = []

    def build(**kwargs) -> SegmentEditor:
        editors.append(_build_editor_with_ui(tk_root, monkeypatch, **kwargs))
        return editors[-1]

    yield build
    for editor in editors:
        if editor.winfo_exists():
            editor._finish_close()


def _load_keyframes(editor: SegmentEditor, index: KeyframeIndex = KeyframeIndex((0,), Fraction(1, 1000), 0, 60_000)) -> None:
    """Hand the editor the keyframe index its preview worker found."""
    editor._preview_worker.events.put(segment_editor.PreviewKeyframes(index, None))
    editor._poll_workers()


def _scan_result(scores: tuple[float, ...], *, sizes: tuple[float, ...] | None = None, masks=None) -> MosaicScanResult:
    """A scan sampled once a second from 0 s with ``scores``."""
    count = len(scores)
    return MosaicScanResult(
        times=tuple(float(second) for second in range(count)),
        scores=scores,
        masks=torch.zeros((count, 90, 160), dtype=torch.uint8) if masks is None else masks,
        sizes=sizes or (0.0,) * count,
        stride=1.0,
        completed_until=float(count - 1),
    )


@pytest.fixture
def finish_scan(monkeypatch):
    """Press the editor's scan button and let a fake scan worker finish with ``result``."""

    def finish(editor: SegmentEditor, result: MosaicScanResult) -> MagicMock:
        worker = MagicMock(events=queue.Queue())
        monkeypatch.setattr(segment_scan_panel, "MosaicScanWorker", MagicMock(return_value=worker))
        editor._scan_panel._scan_btn.invoke()
        worker.events.put(ScanCompleted(result, stopped=False))
        editor._poll_workers()
        return worker

    return finish


@pytest.fixture
def restoring_editor(build_editor, monkeypatch):
    """A loaded editor with restore preview on; its restoration worker is a mock numbering each request."""
    worker = MagicMock(events=queue.Queue())
    worker.request.side_effect = lambda *_args, **_kwargs: worker.request.call_count
    monkeypatch.setattr(segment_restore_preview, "RestorationPreviewWorker", MagicMock(return_value=worker))
    editor = build_editor()
    editor._restore.toggle()
    return editor, worker


def test_play_requests_restored_playback_when_nothing_is_buffered(restoring_editor) -> None:
    editor, worker = restoring_editor
    editor._seek(2.0)

    editor._toggle_play()

    assert worker.request.call_args.args[0] == 2.0
    assert worker.request.call_args.kwargs["playback"] is True


def test_restored_playback_continues_with_the_next_window(restoring_editor) -> None:
    editor, worker = restoring_editor
    editor._seek(1.0)
    editor._toggle_play()
    frames = tuple(RestoredClipFrame(seconds, Image.new("RGB", (16, 9))) for seconds in (1.0, 2.0))
    editor._restore.handle_event(RestorationClip(frames, worker.request.call_count))
    editor._current = 2.0

    editor._request_next_frame()

    assert worker.request.call_args.args[0] == pytest.approx(2.0 + 1 / 30)
    assert worker.request.call_args.kwargs["playback"] is True


def test_play_right_after_a_seed_render_took_the_gpu_restarts_restored_playback(restoring_editor) -> None:
    editor, worker = restoring_editor
    editor._claim_gpu_for_seed()
    editor._seek(2.0)

    editor._toggle_play()

    assert worker.request.call_args.args[0] == 2.0
    assert worker.request.call_args.kwargs["playback"] is True


def test_a_closed_restoration_worker_leaves_the_gpu_claimed_by_a_seed_render(build_editor, monkeypatch) -> None:
    gpu_owners: set[object] = set()

    def set_preview_gpu_busy(owner, busy):
        (gpu_owners.add if busy else gpu_owners.discard)(owner)

    stopped_callbacks = []
    monkeypatch.setattr(
        segment_restore_preview,
        "RestorationPreviewWorker",
        lambda path, metadata, *, on_stopped: stopped_callbacks.append(on_stopped) or MagicMock(events=queue.Queue()),
    )
    editor = build_editor(set_preview_gpu_busy=set_preview_gpu_busy)
    editor._restore.toggle()
    editor._claim_gpu_for_seed()

    stopped_callbacks[0]()

    assert gpu_owners


def test_app_preview_gpu_stays_busy_until_every_owner_gave_it_back(jasna_app) -> None:
    jasna_app._set_preview_gpu_busy("restore preview", True)
    jasna_app._set_preview_gpu_busy("seed render", True)

    jasna_app._set_preview_gpu_busy("restore preview", False)
    assert jasna_app._preview_gpu_busy
    jasna_app._set_preview_gpu_busy("seed render", False)
    assert not jasna_app._preview_gpu_busy


def test_preview_surface_selects_original_or_restored_from_toggle_state(restoring_editor) -> None:
    editor, worker = restoring_editor
    editor._preview_worker.events.put(
        segment_editor.PreviewFrame(0.0, Image.new("RGB", (160, 90), "red"), editor._preview_generation)
    )
    editor._poll_workers()
    editor._restore.handle_event(RestorationFrame(0.0, Image.new("RGB", (160, 90), "blue"), worker.request.call_count))
    assert shown_pixel(editor._preview, 0, 0) == (0, 0, 255)

    editor._restore.toggle()
    assert shown_pixel(editor._preview, 0, 0) == (255, 0, 0)


def test_projection_selector_is_disabled_for_non_vr_video(build_editor) -> None:
    editor = build_editor()

    assert editor._preview_card._vr_projection_menu.cget("state") == "disabled"


def test_projection_selector_is_enabled_for_detected_vr_video(build_editor) -> None:
    editor = build_editor(metadata=_fake_metadata(width=3840, height=1920),
    )

    assert editor._preview_card._vr_projection_menu.cget("state") == "normal"
    assert editor._preview_card._vr_projection_menu.get_value() == "auto"
    assert editor._preview_card._vr_projection_label.cget("text") == t(
        "segments_vr_projection_resolved",
        projection=t("segments_vr_projection_raw"),
    )


def test_projection_selector_is_next_to_restore_preview_with_long_filename(build_editor) -> None:
    editor = build_editor(metadata=_fake_metadata(width=3840, height=1920),
        path=Path(f"SAVR-{'x' * 180}.mp4"),
    )
    editor.geometry("900x640")
    editor.update()

    menu_right = (
        editor._preview_card._vr_projection_menu.winfo_rootx()
        + editor._preview_card._vr_projection_menu.winfo_width()
    )
    editor_right = editor.winfo_rootx() + editor.winfo_width()
    assert editor._preview_card._vr_projection_menu.winfo_ismapped()
    assert editor._preview_card._vr_projection_menu.cget("state") == "normal"
    assert menu_right <= editor_right
    assert (
        editor._preview_card._vr_projection_menu.master.master
        is editor._preview_card.restore_toggle.master.master
    )


def test_projection_change_restarts_active_restoration_preview(restoring_editor) -> None:
    editor, worker = restoring_editor
    editor._restore.handle_event(RestorationFrame(0.0, Image.new("RGB", (16, 9)), worker.request.call_count))

    editor._on_vr_projection_changed("fisheye")
    assert editor._restore._restored_source is None
    editor._restore.request()

    assert worker.request.call_args.kwargs["projection"] == "fisheye"


def test_pan_zoom_controls_are_visible_and_explain_gestures(build_editor) -> None:
    editor = build_editor()

    assert editor._preview_card._pan_zoom_hint.winfo_ismapped()
    assert editor._preview_card._pan_zoom_hint.cget("text") == t(
        "segments_preview_pan_zoom_hint"
    )
    assert editor._preview_card._zoom_out_btn.winfo_ismapped()
    assert editor._preview_card._zoom_label.cget("text") == "100%"
    assert editor._preview_card._zoom_in_btn.winfo_ismapped()
    assert not editor._preview_card._reset_view_btn.winfo_ismapped()
    assert editor._preview_card._reset_view_btn.cget("text") == t(
        "segments_preview_reset_view"
    )

    editor._preview_card._zoom_in_btn.invoke()
    editor.update()
    assert editor._preview_card._reset_view_btn.winfo_ismapped()

    editor._preview_card._reset_view()
    editor.update()
    assert not editor._preview_card._reset_view_btn.winfo_ismapped()


@pytest.mark.parametrize("factor", [1.0, 1.25, 1.5, 2.0, 3.0])
def test_editor_fits_the_screen_and_grows_on_tall_screens(build_editor, monkeypatch, factor) -> None:
    editor = build_editor()
    monkeypatch.setattr(segment_editor.scaling, "window_scaling", lambda _window: factor)
    monkeypatch.setattr(segment_editor.scaling, "screen_rect", lambda _window: (0, 0, 2560, 1440))
    requested: list[str] = []
    minimum: list[tuple[int, int]] = []
    monkeypatch.setattr(editor, "geometry", requested.append)
    monkeypatch.setattr(editor, "minsize", lambda width, height: minimum.append((width, height)))

    editor._size_and_center()

    size = requested[-1].split("+")[0]
    width, height = (round(int(value) * factor) for value in size.split("x"))
    assert width <= 2560 and height <= 1440
    assert all(round(value * factor) <= limit for value, limit in zip(minimum[-1], (width, height)))
    if factor == 1.0:
        assert requested == ["1826x1240+367+100"]
        assert minimum == [(900, 640)]


def test_previous_frame_uses_exact_decoder_predecessor(build_editor) -> None:
    editor = build_editor()
    editor._seek(1.0)

    editor._step(-1)

    editor._preview_worker.previous_frame.assert_called_once_with(1.0)
    assert editor._current == pytest.approx(29 / 30)


def test_scan_bar_builds_with_editor(build_editor) -> None:
    editor = build_editor()
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


def test_scan_model_change_applies_recommended_threshold(build_editor) -> None:
    editor = build_editor()

    editor._scan_panel._on_model_changed("rfdetr-v6-large")

    assert editor._scan_panel._threshold == pytest.approx(0.40)
    assert editor._scan_panel._threshold_slider.get() == pytest.approx(0.40)
    assert editor._scan_panel._threshold_label.cget("text") == "0.40"


def test_source_codec_notice_only_shows_for_selected_ranges(build_editor) -> None:
    editor = build_editor()
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


def test_scan_card_stays_compact_and_visible_at_minimum_size(build_editor) -> None:
    editor = build_editor()
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


def test_scan_lock_disables_everything_but_stop(build_editor) -> None:
    editor = build_editor()
    editor.geometry("900x640")
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


def test_scan_completed_populates_detections_and_add_button(build_editor, finish_scan) -> None:
    editor = build_editor()

    finish_scan(editor, _scan_result((0.0, 0.8, 0.9, 0.0)))
    editor.update_idletasks()

    assert editor._scan_panel._worker is not None
    assert editor._timeline._detections
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


def test_scan_ltx_auto_split_adds_big_mosaics_as_ltx(build_editor, finish_scan) -> None:
    editor = build_editor(settings=AppSettings(ltx_seed=5))
    _load_keyframes(editor, KeyframeIndex((0, 45, 120, 180), Fraction(1, 30), 0, 1800))
    panel = editor._scan_panel
    finish_scan(editor, _scan_result((0.0, 0.8, 0.9, 0.0), sizes=(0.0, 0.05, 0.3, 0.0)))
    editor.geometry("900x640")
    editor.update()
    assert panel._ltx_auto_row.winfo_ismapped()

    panel._ltx_auto_toggle.select()
    panel._toggle_ltx_auto()
    panel._add_detected_ranges()

    assert editor._state.segments == (
        SegmentRange(0.5, 1.5, SegmentRestoration("basicvsrpp", None)),
        SegmentRange(1.5, 3.5, SegmentRestoration("ltx", 5)),
    )


def test_scan_ltx_auto_split_is_hidden_when_ltx_cannot_run(build_editor, finish_scan) -> None:
    editor = build_editor(ltx_unavailable_reason="model_ltx_not_downloaded")
    panel = editor._scan_panel
    finish_scan(editor, _scan_result((0.9, 0.9)))
    editor.update_idletasks()
    assert panel._add_btn.winfo_ismapped()
    assert not panel._ltx_auto_row.winfo_ismapped()


def test_scan_finishes_without_ltx_split_when_ltx_cannot_run(build_editor, finish_scan) -> None:
    editor = build_editor(
        settings=AppSettings(ltx_auto_split=True),
        ltx_unavailable_reason="model_ltx_not_downloaded",
    )
    _load_keyframes(editor, KeyframeIndex((0, 45, 120, 180), Fraction(1, 30), 0, 1800))
    panel = editor._scan_panel

    finish_scan(editor, _scan_result((0.0, 0.8, 0.9, 0.0), sizes=(0.0, 0.05, 0.3, 0.0)))
    editor.update_idletasks()

    assert not panel.active
    assert panel._status.cget("text") == t("segments_scan_result", count=1, duration="00:00:03")
    panel._add_detected_ranges()
    assert editor._state.segments == (SegmentRange(0.5, 3.5, SegmentRestoration("basicvsrpp", None)),)


def test_scan_threshold_updates_ranges_and_add_button(build_editor, finish_scan) -> None:
    editor = build_editor()
    finish_scan(editor, _scan_result((0.0, 0.8, 0.0)))
    assert editor._timeline._detections
    assert editor._scan_panel._add_btn.cget("state") == "normal"

    editor._scan_panel._on_threshold(0.9)
    editor._scan_panel.after_cancel(editor._scan_panel._threshold_after)
    editor._scan_panel._apply_threshold()

    assert editor._timeline._detections == ()
    assert editor._scan_panel._add_btn.cget("state") == "disabled"
    assert not editor._scan_panel._add_btn.winfo_ismapped()


def test_scan_overlay_respects_dynamic_threshold(build_editor, finish_scan) -> None:
    editor = build_editor()
    panel = editor._scan_panel
    editor._seek(1.0)
    panel._on_threshold(0.8)
    finish_scan(editor, _scan_result((0.0, 0.7), masks=torch.ones((2, 90, 160), dtype=torch.uint8)))
    image = Image.new("RGB", (160, 90), "black")

    assert panel.apply_overlay(image) is image

    panel._on_threshold(0.6)
    overlaid = panel.apply_overlay(image)
    assert overlaid.getpixel((80, 45))[0] > 0


def test_smart_render_error_is_explained_once(build_editor) -> None:
    editor = build_editor()
    editor._state.add(0.0, 58.0)

    _load_keyframes(editor, KeyframeIndex(pts=(0,), time_base=Fraction(1, 1), start_pts=0, end_pts=60))

    assert editor._workload.cget("text") != editor._notice.cget("text")
    assert editor._notice.cget("text") == t("segments_smart_render_whole_video")
    assert editor._apply_btn.cget("state") == "disabled"


def test_save_remembers_detection_and_projection_settings_on_video(build_editor) -> None:
    job = JobItem(Path("video.mp4"))
    editor = build_editor(job=job)
    editor._scan_panel._model.set("lada-yolo-v4")
    editor._scan_panel._on_threshold(0.55)
    editor._on_vr_projection_changed("gnomonic")

    editor._save()

    assert job.detection_model == "lada-yolo-v4"
    assert job.detection_score_threshold == 0.55
    assert job.vr_projection == "gnomonic"
    assert not editor.winfo_exists()


def test_suggest_mask_button_locked_during_scan(build_editor) -> None:
    editor = build_editor()
    assert editor._preview_card.suggest_button.cget("text") == t("segments_suggest_mask")
    assert editor._preview_card.suggest_button in editor._lockable_widgets()
    editor._scan_panel._set_locked(True)
    assert editor._preview_card.suggest_button.cget("state") == "disabled"
    editor._scan_panel._set_locked(False)
    assert editor._preview_card.suggest_button.cget("state") == "normal"


def test_suggest_mask_grabs_full_frame_and_opens_dialog(build_editor, monkeypatch) -> None:
    opened = {}

    class _FakeDialog:
        def __init__(self, master, image, on_submit, on_closed=None):
            opened.update(image=image, on_submit=on_submit, on_closed=on_closed)

    monkeypatch.setattr(segment_editor, "MaskSuggestDialog", _FakeDialog)
    editor = build_editor()
    editor._suggest_mask()
    assert editor._suggest_busy
    assert editor._preview_card.suggest_button.cget("state") == "disabled"
    editor._preview_worker.grab_full.assert_called_once_with()

    from PIL import Image

    frame = Image.new("RGB", (1920, 1080))
    editor._open_mask_suggest(frame)
    assert opened["image"] is frame

    opened["on_closed"]()
    assert not editor._suggest_busy
    assert editor._preview_card.suggest_button.cget("state") == "normal"


def test_feedback_upload_events_show_toast(build_editor, monkeypatch) -> None:
    toasts = []
    editor = build_editor()
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


def test_inspector_sets_the_model_and_seed_of_the_selected_range(build_editor, tk_root) -> None:
    editor = build_editor(settings=AppSettings(ltx_seed=42))
    editor._ranges.create(1.0, 2.0)
    editor._ranges.create(4.0, 5.0)
    tk_root.update()
    assert editor._inspector.winfo_manager() == "grid"
    assert editor._inspector._range_model.get() == "basicvsrpp"
    assert not editor._inspector._seed_entry.winfo_manager()

    editor._inspector._model_radios["ltx"].invoke()
    tk_root.update()
    assert editor._state.selected_segment.restoration == SegmentRestoration("ltx", 42)
    assert editor._inspector._seed_entry.get() == "42"

    editor._inspector._seed_entry.delete(0, "end")
    editor._inspector._seed_entry.insert(0, "777")
    editor._inspector._commit_seed_entry()
    assert editor._state.selected_segment.restoration == SegmentRestoration("ltx", 777)

    editor._inspector._use_for_all_btn.invoke()
    assert {s.restoration for s in editor._state.segments} == {SegmentRestoration("ltx", 777)}
    assert editor._timeline._segments == editor._state.segments

    editor._ranges.undo()
    assert editor._state.segments[0].restoration == SegmentRestoration("basicvsrpp", None)


def test_new_ranges_follow_the_job_model_and_rows_name_it(build_editor, tk_root) -> None:
    editor = build_editor(settings=AppSettings(restoration_model="ltx", ltx_seed=9))
    editor._ranges.create(1.0, 2.0)
    tk_root.update()
    assert editor._state.segments[0].restoration == SegmentRestoration("ltx", 9)
    texts = [w.cget("text") for w in editor._ranges._segment_action_widgets if isinstance(w, ctk.CTkButton)]
    assert any(f"{t('model_' + model_key('ltx', LTX_DEFAULT_MODEL))} 9" in text for text in texts)


def test_ltx_is_blocked_with_a_reason_when_it_cannot_run(build_editor, tk_root) -> None:
    job = JobItem(Path("video.mp4"), segments=(SegmentRange(4.0, 5.0, SegmentRestoration("ltx", 3)),))
    editor = build_editor(job=job, ltx_unavailable_reason="model_ltx_not_installed"
    )
    assert editor._notice.cget("text") == t("segments_ltx_ranges_unavailable")
    editor._ranges.create(1.0, 2.0)
    tk_root.update()
    assert editor._inspector._model_radios["ltx"].cget("state") == "disabled"
    editor._ranges.select(1)
    assert editor._inspector._model_radios["ltx"].cget("state") == "normal"


class _FakeSeedWorker:
    instances: list = []

    def __init__(self, path, metadata, index, work_dir, *, on_stopped):
        self.events = queue.Queue()
        self.requests = []
        self.released = 0
        self.closed = False
        self.work_dir = work_dir
        _FakeSeedWorker.instances.append(self)

    def start(self):
        pass

    def try_seed(self, segment, seed, settings):
        self.requests.append((segment, seed, settings))
        return len(self.requests)

    def cancel(self):
        pass

    def release(self):
        self.released += 1

    def close(self):
        self.closed = True


def _seed_chip(editor, text):
    return next(widget for widget in editor._seeds._chip_widgets if widget.cget("text") == text)


def _seed_frames(tmp_path, seed, color):
    frames = []
    for seconds in (1.0, 1.5):
        path = tmp_path / f"{seed}-{seconds}.jpg"
        Image.new("RGB", (160, 90), color).save(path)
        frames.append(SeedFrame(seconds, path))
    return tuple(frames)


def test_restore_preview_renders_an_ltx_range_then_lets_the_user_compare_and_pick_seeds(build_editor, tk_root, monkeypatch, tmp_path) -> None:
    _FakeSeedWorker.instances = []
    monkeypatch.setattr(segment_seed_panel, "LtxSeedPreviewWorker", _FakeSeedWorker)
    monkeypatch.setattr(segment_seed_panel, "ltx_render_seconds", lambda settings, metadata, segment: 60.0)
    job = JobItem(tmp_path / "video.mp4", segments=(SegmentRange(1.0, 2.0, SegmentRestoration("ltx", 5)),))
    editor = build_editor(job=job)
    _load_keyframes(editor)
    editor._seek(1.5)
    editor._preview_source = Image.new("RGB", (160, 90), "red")
    tk_root.update()

    editor._restore.toggle()
    worker = _FakeSeedWorker.instances[0]
    assert worker.work_dir == tmp_path
    assert worker.requests[0][:2] == (job.segments[0], 5)
    assert editor._restore._active
    assert shown_message(editor._preview) == t("segments_preview_ltx_rendering", seed=5)
    assert not editor._seeds._render_btn.winfo_manager()
    worker.events.put(SeedProgress(5, "denoise", 0.5, 70.0, 1))
    editor._poll_workers()
    assert editor._seeds._status.cget("text") == t(
        "segments_seed_progress", seed=5, stage=t("ltx_stage_denoise"), percent=50.0, eta="1m 10s"
    )

    worker.events.put(SeedReady(5, _seed_frames(tmp_path, 5, "blue"), 1))
    editor._poll_workers()
    tk_root.update()
    assert 1.0 <= editor._current < 2.0
    assert editor._preview_display_source().getpixel((0, 0))[2] > 200

    editor._inspector._new_range_seed()
    new_seed = editor._state.selected_segment.restoration.ltx_seed
    tk_root.update()
    assert shown_message(editor._preview) == t("segments_preview_ltx_not_rendered", seed=new_seed)
    assert editor._seeds._render_btn.winfo_manager()
    editor._seeds._render_btn.invoke()
    assert worker.requests[1][:2] == (editor._state.selected_segment, new_seed)
    worker.events.put(SeedReady(new_seed, _seed_frames(tmp_path, new_seed, "green"), 2))
    editor._poll_workers()
    chips = [w.cget("text") for w in editor._seeds._chip_widgets if isinstance(w, ctk.CTkButton)]
    assert chips == ["5", f"{new_seed} ✓"]
    assert editor._preview_display_source().getpixel((0, 0))[1] > 100

    _seed_chip(editor, "5").invoke()
    assert editor._preview_display_source().getpixel((0, 0))[2] > 200
    editor._seeds._chip_widgets[-1].invoke()
    assert editor._state.selected_segment.restoration == SegmentRestoration("ltx", 5)

    editor._restore.toggle()
    assert editor._preview_display_source().getpixel((0, 0))[0] > 200

    editor._claim_gpu_for_scan()
    assert worker.released == 1
    editor._finish_close()
    assert worker.closed


def test_playback_over_a_rendered_ltx_range_shows_each_seed_frame(build_editor, tk_root, monkeypatch, tmp_path) -> None:
    _FakeSeedWorker.instances = []
    monkeypatch.setattr(segment_seed_panel, "LtxSeedPreviewWorker", _FakeSeedWorker)
    monkeypatch.setattr(segment_seed_panel, "ltx_render_seconds", lambda settings, metadata, segment: 60.0)
    job = JobItem(tmp_path / "video.mp4", segments=(SegmentRange(1.0, 2.0, SegmentRestoration("ltx", 5)),))
    editor = build_editor(job=job)
    _load_keyframes(editor)
    editor._seek(1.0)
    editor._restore.toggle()
    frames = []
    for seconds, color in ((1.0, "blue"), (1.5, "green")):
        path = tmp_path / f"{seconds}.jpg"
        Image.new("RGB", (160, 90), color).save(path)
        frames.append(SeedFrame(seconds, path))
    _FakeSeedWorker.instances[0].events.put(SeedReady(5, tuple(frames), 1))
    editor._poll_workers()
    tk_root.update()

    editor._toggle_play()
    editor._preview_worker.events.put(
        segment_editor.PreviewFrame(1.5, Image.new("RGB", (160, 90), "red"), editor._preview_generation)
    )
    editor._poll_workers()

    red, green, blue = shown_pixel(editor._preview, 0, 0)
    assert green > 100 and red < 100 and blue < 100


@pytest.mark.parametrize("answer", [False, True])
def test_restore_preview_asks_before_a_long_ltx_render(build_editor, monkeypatch, tmp_path, answer) -> None:
    _FakeSeedWorker.instances = []
    asked = []
    monkeypatch.setattr(segment_seed_panel, "LtxSeedPreviewWorker", _FakeSeedWorker)
    monkeypatch.setattr(segment_seed_panel, "ltx_render_seconds", lambda settings, metadata, segment: 360.0)
    monkeypatch.setattr(segment_editor.messagebox, "askyesno", lambda title, text, parent: asked.append(text) or answer)
    job = JobItem(tmp_path / "video.mp4", segments=(SegmentRange(1.0, 2.0, SegmentRestoration("ltx", 5)),))
    editor = build_editor(job=job)
    _load_keyframes(editor)
    editor._seek(1.5)

    editor._restore.toggle()

    assert asked == [t("segments_ltx_render_confirm", seed=5, duration="6m 0s")]
    assert editor._restore._active is answer
    assert bool(_FakeSeedWorker.instances) is answer


class _BlockingSeedRenderer(ltx_seed_preview.SeedRenderer):
    started = threading.Event()

    def render(self, segment, seed, settings, *, writer_for, report, cancel):
        self.started.set()
        cancel.wait(5)
        raise Cancelled()


class _BlockingSeedWorker(LtxSeedPreviewWorker):
    def _make_renderer(self):
        return _BlockingSeedRenderer(self.path, self.metadata, self.index, self.work_dir)


@pytest.mark.parametrize("stop_render", [True, False])
def test_closing_the_app_closes_the_segment_editor_and_its_seed_frames(jasna_app, monkeypatch, tmp_path, stop_render) -> None:
    preview = MagicMock()
    preview.events = queue.Queue()
    monkeypatch.setattr(segment_editor, "SegmentPreviewWorker", MagicMock(return_value=preview))
    monkeypatch.setattr(segment_seed_panel, "LtxSeedPreviewWorker", _BlockingSeedWorker)
    monkeypatch.setattr(segment_seed_panel, "ltx_render_seconds", lambda settings, metadata, segment: 0.0)
    asked: list[str] = []
    monkeypatch.setattr(
        segment_editor.messagebox, "askyesno", lambda _title, message, parent: asked.append(message) or stop_render
    )
    _BlockingSeedRenderer.started.clear()
    monkeypatch.setattr(jasna_app._settings_panel, "get_settings", AppSettings)
    monkeypatch.setattr(jasna_app._settings_panel, "ltx_unavailable_reason", lambda: None)
    work_dir = tmp_path / "videos"
    work_dir.mkdir()
    job = JobItem(work_dir / "video.mp4", segments=(SegmentRange(1.0, 2.0, SegmentRestoration("ltx", 5)),))
    jasna_app._edit_segments(job)
    editor = jasna_app._segment_editor
    metadata = _fake_metadata()
    preview.events.put(segment_editor.PreviewLoaded(metadata, resolve_vr_mode("auto", metadata, job.path)))
    editor._poll_workers()
    _load_keyframes(editor)
    editor._seek(1.5)
    editor._restore.toggle()
    assert _BlockingSeedRenderer.started.wait(5)

    jasna_app._on_close()

    assert asked == [t("segments_seed_stop_confirm")]
    assert (not jasna_app.children) is stop_render
    assert (not list(work_dir.glob(".jasna-seeds-*"))) is stop_render
    if not stop_render:
        editor._finish_close()


def test_seed_controls_have_tooltips_and_name_the_job_ltx_model(build_editor, tk_root, monkeypatch, tmp_path) -> None:
    tips: dict[str, list] = {}
    original = segment_editor.Tooltip.__init__

    def record(self, widget, text):
        tips.setdefault(text, []).append(widget)
        original(self, widget, text)

    monkeypatch.setattr(segment_editor.Tooltip, "__init__", record)
    _FakeSeedWorker.instances = []
    monkeypatch.setattr(segment_seed_panel, "LtxSeedPreviewWorker", _FakeSeedWorker)
    monkeypatch.setattr(segment_seed_panel, "ltx_render_seconds", lambda settings, metadata, segment: 60.0)
    job = JobItem(tmp_path / "video.mp4", segments=(SegmentRange(1.0, 2.0, SegmentRestoration("ltx", 5)),))
    settings = replace(AppSettings(), ltx_model="undistilled")
    editor = build_editor(job=job, settings=settings)
    _load_keyframes(editor)
    editor._ranges.select(0)
    editor._preview_source = Image.new("RGB", (160, 90), "red")
    tk_root.update()
    editor._seek(1.5)
    editor._restore.toggle()
    worker = _FakeSeedWorker.instances[0]
    worker.events.put(SeedReady(5, _seed_frames(tmp_path, 5, "blue"), 1))
    editor._poll_workers()
    editor._inspector._new_range_seed()
    _seed_chip(editor, "5").invoke()
    tk_root.update()

    assert editor._inspector._model_radios["ltx"].cget("text") == t("model_ltx_undistilled")
    expected = {
        "tip_segments_restore_with": editor._inspector._restore_with_label,
        "tip_model_basicvsrpp": editor._inspector._model_radios["basicvsrpp"],
        "tip_segments_use_for_all": editor._inspector._use_for_all_btn,
        "tip_ltx_seed": editor._inspector._seed_entry,
        "tip_ltx_new_seed": editor._inspector._new_seed_btn,
        "tip_segments_ltx_render": editor._seeds._render_btn,
        "tip_segments_seed_cancel": editor._seeds._cancel_btn,
    }
    for key, widget in expected.items():
        assert widget in tips[t(key)], key
    chips = [w for w in editor._seeds._chip_widgets if w.cget("text") == "5"]
    use = [w for w in editor._seeds._chip_widgets if w.cget("text") == t("segments_seed_use")]
    assert chips and chips[0] in tips[t("tip_segments_seed_chip")]
    assert use and use[0] in tips[t("tip_segments_seed_use")]
    assert editor._inspector._ltx_radio_tooltip._text == t("tip_model_ltx_undistilled")


def test_range_controls_fit_their_content_while_the_list_keeps_its_rows() -> None:
    assert segment_range_panel.range_controls_height(500, 200, 156) == 200
    assert segment_range_panel.range_controls_height(320, 200, 156) == 164
    assert segment_range_panel.range_controls_height(200, 200, 156) == 100


def _visible_in(widget, viewport) -> bool:
    top, bottom = viewport.winfo_rooty(), viewport.winfo_rooty() + viewport.winfo_height()
    return widget.winfo_ismapped() and top <= widget.winfo_rooty() and widget.winfo_rooty() + widget.winfo_height() <= bottom


def test_range_list_keeps_three_rows_beside_a_busy_seed_inspector(build_editor, tk_root) -> None:
    editor = build_editor()
    editor.geometry("1856x880")
    for start in (1.0, 4.0, 7.0, 10.0):
        editor._ranges.create(start, start + 1.0)
    editor._ranges.select(0)
    editor._inspector._model_radios["ltx"].invoke()
    key = segment_seed_panel.range_key(editor._state.selected_segment)
    editor._seeds._tried[key] = {1: (), 2: ()}
    editor._seeds._shown = (key, 2)
    editor._seeds._running = (key, 1)
    editor._seeds._status.configure(text="Seed 1: Restoring 30% · about 15s left")
    editor._seeds.refresh()
    for _ in range(5):
        tk_root.update()

    rows = editor._ranges._segment_list.winfo_children()
    viewport = editor._ranges._segment_list._parent_canvas
    assert all(_visible_in(row, viewport) for row in rows[:3])
    assert editor._ranges.controls._parent_canvas.yview() == (0.0, 1.0)
    assert _visible_in(editor._seeds._chips, editor._ranges.controls._parent_canvas)
    assert editor._ranges._range_action.grid_info()["row"] == 0
    editor._seeds._running = None


def test_range_times_stack_and_controls_scroll_at_minimum_size(build_editor, tk_root) -> None:
    editor = build_editor()
    editor.geometry("900x640")
    editor._ranges.create(1.0, 2.0)
    editor._inspector._model_radios["ltx"].invoke()
    for _ in range(5):
        tk_root.update()

    assert editor._ranges._end_entry.grid_info()["row"] == 1
    assert editor._ranges._range_action.grid_info()["row"] == 2
    assert editor._ranges.controls._parent_canvas.yview()[1] < 1.0
    assert editor._ranges._segment_list._parent_canvas.winfo_height() > 0


def test_range_list_keeps_its_rows_on_select_seeks_and_scrolls_to_the_selection(build_editor, tk_root) -> None:
    ranges = tuple(SegmentRange(float(start), start + 0.5) for start in range(0, 40, 2))
    editor = build_editor(job=JobItem(Path("video.mp4"), segments=ranges))
    editor.geometry("1400x1000")
    tk_root.update()
    rows = list(editor._ranges._segment_rows)
    canvas = editor._ranges._segment_list._parent_canvas
    assert canvas.yview()[0] == 0.0

    editor._ranges._select_from_list(19)
    tk_root.update()

    assert editor._ranges._segment_rows == rows
    assert editor._current == pytest.approx(38.0)
    assert rows[19].cget("fg_color") == segment_editor.Colors.PRIMARY_DARK
    assert rows[0].cget("fg_color") == segment_editor.Colors.BG_CARD
    top, bottom = canvas.yview()
    height = editor._ranges._segment_list.winfo_height()
    assert top > 0.0
    assert top <= rows[19].winfo_y() / height
    assert (rows[19].winfo_y() + rows[19].winfo_height()) / height <= bottom + 1e-6


def test_footer_shows_the_ltx_share_only_when_a_range_uses_ltx(build_editor, tk_root) -> None:
    standard, ltx = SegmentRestoration("basicvsrpp", None), SegmentRestoration("ltx", 5)
    job = JobItem(Path("video.mp4"), segments=(SegmentRange(0.0, 30.0, standard), SegmentRange(40.0, 50.0, ltx)))
    editor = build_editor(job=job)
    tk_root.update()
    assert editor._model_split_bar.winfo_ismapped()
    assert editor._model_split_bar.get() == pytest.approx(0.25)
    assert editor._model_split.cget("text") == t(
        "segments_model_split", ltx="00:00:10", percent=25.0, standard="00:00:30"
    )

    editor._ranges.select(1)
    editor._ranges._delete_range(1)
    tk_root.update()
    assert not editor._model_split_bar.winfo_ismapped()
    assert not editor._model_split.winfo_ismapped()


def test_restore_preview_uses_basicvsrpp_outside_an_ltx_range(build_editor, monkeypatch, tmp_path) -> None:
    preview_worker = MagicMock(events=queue.Queue())
    monkeypatch.setattr(segment_restore_preview, "RestorationPreviewWorker", MagicMock(return_value=preview_worker))
    monkeypatch.setattr(segment_seed_panel, "ltx_render_seconds", lambda settings, metadata, segment: 60.0)
    job = JobItem(tmp_path / "video.mp4", segments=(SegmentRange(1.0, 2.0, SegmentRestoration("ltx", 5)),))
    editor = build_editor(job=job)
    editor._seek(5.0)
    editor._restore.toggle()
    assert preview_worker.request.call_args[0][1].restoration_model == "basicvsrpp"

    editor._seek(1.5)
    editor._restore.request()
    assert preview_worker.request.call_count == 1
    assert shown_message(editor._preview) == t("segments_preview_ltx_not_rendered", seed=5)
    assert editor._seeds._render_btn.winfo_manager()


def test_ltx_render_button_enables_once_the_keyframes_arrive(build_editor, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(segment_seed_panel, "ltx_render_seconds", lambda settings, metadata, segment: 60.0)
    job = JobItem(tmp_path / "video.mp4", segments=(SegmentRange(1.0, 2.0, SegmentRestoration("ltx", 5)),))
    editor = build_editor(job=job)
    editor._seek(1.5)
    editor._restore.activate()
    editor._restore.request()
    assert editor._seeds._render_btn.cget("state") == "disabled"

    _load_keyframes(editor)

    assert editor._seeds._render_btn.cget("state") == "normal"


def test_ltx_render_estimate_is_worked_out_once_per_range(build_editor, monkeypatch, tmp_path) -> None:
    estimated: list[SegmentRange] = []
    monkeypatch.setattr(
        segment_seed_panel, "ltx_render_seconds", lambda settings, metadata, segment: estimated.append(segment) or 60.0
    )
    job = JobItem(tmp_path / "video.mp4", segments=(SegmentRange(1.0, 2.0, SegmentRestoration("ltx", 5)),))
    editor = build_editor(job=job)
    editor._seek(1.5)
    editor._restore.activate()

    for _ in range(3):
        editor._restore.request()

    assert estimated == [job.segments[0]]


def test_scan_says_it_loads_the_detection_model(build_editor, monkeypatch) -> None:
    worker = MagicMock(events=queue.Queue())
    monkeypatch.setattr(segment_scan_panel, "MosaicScanWorker", MagicMock(return_value=worker))
    editor = build_editor()
    panel = editor._scan_panel
    panel._start()
    assert panel._status.cget("text") == t("segments_scan_loading_model")

    worker.events.put(segment_scan_panel.ScanStatus("loading_models"))
    panel.poll()
    assert panel._status.cget("text") == t("segments_scan_loading_model")


@pytest.mark.parametrize(("auto", "percent", "color"), [(False, 12, "BG_PANEL"), (True, 12, "MODEL_LTX"), (True, 60, "PRIMARY")])
def test_scan_overlay_tags_each_mosaic_with_its_size(build_editor, finish_scan, auto, percent, color) -> None:
    editor = build_editor()
    panel = editor._scan_panel
    editor._seek(1.0)
    masks = torch.zeros((2, 90, 160), dtype=torch.uint8)
    masks[1, 45:90, 80:120] = 1
    finish_scan(editor, _scan_result((0.0, 0.9), sizes=(0.0, 0.5), masks=masks))
    if auto:
        panel._ltx_auto_toggle.select()
        panel._toggle_ltx_auto()
    panel._on_ltx_percent(percent)

    overlaid = panel.apply_overlay(Image.new("RGB", (1600, 900), "black"))

    expected = Image.new("RGB", (1, 1), getattr(segment_editor.Colors, color)).getpixel((0, 0))
    assert overlaid.getpixel((802, 447)) == expected


def test_closing_the_editor_cancels_its_pending_timers(restoring_editor, tk_root) -> None:
    editor, _worker = restoring_editor
    editor._restore.schedule()
    editor._scan_panel._on_threshold(0.5)
    pending = {editor._poll_after, editor._restore._after, editor._scan_panel._threshold_after}

    editor._finish_close()

    assert not pending & scheduled_after_ids(tk_root)


class _SlowStoppingRestorationWorker(RestorationPreviewWorker):
    """Runs on after being closed until ``finish`` is set."""

    finish = threading.Event()

    def _run(self) -> None:
        self._closed.wait()
        self.finish.wait(5)


def test_join_workers_waits_up_to_the_timeout_for_workers_closed_with_the_editor(build_editor, monkeypatch) -> None:
    monkeypatch.setattr(segment_restore_preview, "RestorationPreviewWorker", _SlowStoppingRestorationWorker)
    _SlowStoppingRestorationWorker.finish.clear()
    editor = build_editor()
    editor._restore.toggle()
    worker = editor._restore._worker
    editor._finish_close()

    editor.join_workers(timeout=0.01)
    assert worker._thread.is_alive()

    _SlowStoppingRestorationWorker.finish.set()
    editor.join_workers(timeout=5)
    assert not worker._thread.is_alive()
