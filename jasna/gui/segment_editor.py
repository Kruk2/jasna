from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections.abc import Callable

import customtkinter as ctk
from PIL import Image
from tkinter import messagebox

from jasna.gui.locales import t
from jasna.gui.models import AppSettings, JobItem
from jasna.gui.components import Tooltip
from jasna.gui.icons import create_compact_switch
from jasna.gui.restoration_preview import (
    RestorationClip,
    RestorationFailed,
    RestorationFrame,
    RestorationPreviewWorker,
    RestorationStatus,
    RestoredClipFrame,
)
from jasna.gui.mosaic_scan import (
    SCAN_SCORE_FLOOR,
    MosaicScanResult,
    MosaicScanWorker,
    ScanCompleted,
    ScanFailed,
    ScanProgress,
    ScanStatus,
    segments_from_scores,
)
from jasna.gui.segment_editor_state import SegmentEditorState
from jasna.gui.segment_preview import (
    PreviewEnded,
    PreviewFailed,
    PreviewFrame,
    PreviewLoaded,
    SegmentPreviewWorker,
)
from jasna.gui.segment_timeline import SegmentTimeline
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.media import VideoMetadata
from jasna.segments import SegmentRange, format_timestamp, parse_timestamp


class SegmentEditor(ctk.CTkToplevel):
    """Modal, frame-aware editor for per-job restoration ranges."""

    def __init__(
        self,
        master,
        job: JobItem,
        get_settings: Callable[[], AppSettings],
        is_gpu_busy: Callable[[], bool],
        set_preview_gpu_busy: Callable[[bool], None],
        on_saved: Callable[[tuple[SegmentRange, ...]], None],
        on_closed: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master)
        self._job = job
        self._get_settings = get_settings
        self._is_gpu_busy = is_gpu_busy
        self._set_preview_gpu_busy = set_preview_gpu_busy
        self._on_saved = on_saved
        self._on_closed = on_closed
        self._state: SegmentEditorState | None = None
        self._metadata: VideoMetadata | None = None
        self._current = 0.0
        self._playing = False
        self._closed = threading.Event()
        self._saved = False
        self._next_frame_after: str | None = None
        self._resize_after: str | None = None
        self._preview_source: Image.Image | None = None
        self._preview_image = None
        self._preview_generation = 0
        self._restore_active = False
        self._restore_after: str | None = None
        self._restore_toggle_blocked = False
        self._restoration_worker: RestorationPreviewWorker | None = None
        self._restored_source: Image.Image | None = None
        self._restored_clip: tuple[RestoredClipFrame, ...] = ()
        self._restore_play_pending = False
        self._restore_generation = 0
        self._keyframe_index = None
        self._analysis_error: str | None = None
        self._compatibility_error: str | None = None
        self._edit_notice: str | None = None
        self._edit_notice_warning = False
        self._analysis_events: queue.Queue[object] = queue.Queue()
        self._scan_worker: MosaicScanWorker | None = None
        self._scan_result: MosaicScanResult | None = None
        self._scan_proposals: tuple = ()
        self._scan_overlay = False
        self._scan_threshold = min(
            0.9, max(SCAN_SCORE_FLOOR, float(get_settings().detection_score_threshold))
        )
        self._scan_thr_after: str | None = None

        self.title(t("segments_title"))
        self.configure(fg_color=Colors.BG_MAIN)
        self.transient(master.winfo_toplevel())
        self.protocol("WM_DELETE_WINDOW", self._request_close)
        self._size_and_center()
        self._build_loading()
        self._bind_shortcuts()
        self.update_idletasks()
        self.wait_visibility()
        self._take_focus()

        self._preview_worker = SegmentPreviewWorker(job.path)
        self._preview_worker.start()
        self.after(25, self._poll_workers)

    def _size_and_center(self) -> None:
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        height = max(560, screen_h - 200)
        width = min(max(1, screen_w - 48), max(800, round(height * 1060 / 720)))
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.minsize(min(800, width), min(560, height))

    def _take_focus(self) -> None:
        if self._closed.is_set():
            return
        try:
            self.grab_set()
            self.lift()
            self.focus_force()
        except tk.TclError:
            pass

    def _build_loading(self) -> None:
        self._loading = ctk.CTkFrame(self, fg_color="transparent")
        self._loading.pack(fill="both", expand=True, padx=24, pady=24)
        ctk.CTkLabel(
            self._loading,
            text=f"{t('segments_title')} — {self._job.filename}",
            font=(Fonts.FAMILY, Fonts.SIZE_LARGE, "bold"),
            text_color=Colors.TEXT_PRIMARY,
        ).pack(pady=(80, 12))
        self._loading_bar = ctk.CTkProgressBar(self._loading, mode="indeterminate", width=260)
        self._loading_bar.pack()
        self._loading_bar.start()
        self._loading_label = ctk.CTkLabel(
            self._loading,
            text=t("segments_loading_preview"),
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.STATUS_PENDING,
        )
        self._loading_label.pack(pady=12)
        ctk.CTkButton(
            self._loading,
            text=t("segments_cancel"),
            fg_color=Colors.BG_CARD,
            hover_color=Colors.BORDER_LIGHT,
            command=self._finish_close,
        ).pack(pady=12)

    def _build_editor(self, metadata: VideoMetadata) -> None:
        self._loading_bar.stop()
        self._loading.destroy()
        self._metadata = metadata
        self._state = SegmentEditorState(
            duration=float(metadata.duration),
            fps=max(1.0, float(metadata.video_fps)),
            segments=self._job.snapshot_segments(),
        )
        self._job.duration_seconds = self._state.duration

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=16, pady=(12, 8))
        title_column = ctk.CTkFrame(header, fg_color="transparent")
        title_column.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(
            title_column,
            text=self._job.filename,
            font=(Fonts.FAMILY, Fonts.SIZE_LARGE, "bold"),
            text_color=Colors.TEXT_PRIMARY,
            anchor="w",
        ).pack(fill="x")
        ctk.CTkLabel(
            title_column,
            text=t(
                "segments_media_info",
                width=metadata.video_width,
                height=metadata.video_height,
                fps=metadata.video_fps,
                duration=format_timestamp(metadata.duration, milliseconds=False),
            ),
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.STATUS_PENDING,
            anchor="w",
        ).pack(fill="x")

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=16)
        body.grid_columnconfigure(0, weight=7, uniform="editor")
        body.grid_columnconfigure(1, weight=4, uniform="editor")
        body.grid_rowconfigure(0, weight=1)

        preview_card = ctk.CTkFrame(
            body,
            fg_color=Colors.BG_CARD,
            corner_radius=Sizing.BORDER_RADIUS,
        )
        preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        preview_card.grid_rowconfigure(0, weight=1)
        preview_card.grid_columnconfigure(0, weight=1)
        self._preview = ctk.CTkLabel(
            preview_card,
            text=t("segments_loading_preview"),
            fg_color=Colors.BG_PANEL,
            text_color=Colors.STATUS_PENDING,
            corner_radius=Sizing.BORDER_RADIUS,
        )
        self._preview.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 4))
        self._preview.bind("<Configure>", self._preview_resized)

        transport = ctk.CTkFrame(preview_card, fg_color="transparent")
        transport.grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 8))
        self._step_back = ctk.CTkButton(
            transport,
            text="|◀",
            width=44,
            fg_color=Colors.BG_PANEL,
            hover_color=Colors.BORDER_LIGHT,
            command=lambda: self._step(-1),
        )
        self._step_back.pack(side="left")
        Tooltip(self._step_back, t("segments_previous_frame"))
        self._play = ctk.CTkButton(
            transport,
            text="▶",
            width=48,
            command=self._toggle_play,
        )
        self._play.pack(side="left", padx=6)
        self._step_forward = ctk.CTkButton(
            transport,
            text="▶|",
            width=44,
            fg_color=Colors.BG_PANEL,
            hover_color=Colors.BORDER_LIGHT,
            command=lambda: self._step(1),
        )
        self._step_forward.pack(side="left")
        Tooltip(self._step_forward, t("segments_next_frame"))
        self._time_label = ctk.CTkLabel(
            transport,
            text=self._time_text(),
            font=(Fonts.FAMILY_MONO, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_PRIMARY,
        )
        self._time_label.pack(side="left", padx=12)
        restore_control = ctk.CTkFrame(transport, fg_color="transparent")
        restore_control.pack(side="right")
        self._restore_toggle = create_compact_switch(
            restore_control,
            self._toggle_restoration_preview,
            Colors.BG_CARD,
        )
        self._restore_toggle.pack(side="right")
        ctk.CTkLabel(
            restore_control,
            text=t("segments_restore_preview"),
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
        ).pack(side="right", padx=(0, 6))
        self._restore_toggle_tooltip = Tooltip(self._restore_toggle, t("segments_restore_preview_hint"))
        range_panel = ctk.CTkFrame(
            body,
            fg_color=Colors.BG_CARD,
            corner_radius=Sizing.BORDER_RADIUS,
        )
        range_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        range_panel.grid_columnconfigure(0, weight=1)
        range_panel.grid_rowconfigure(1, weight=1)
        range_header = ctk.CTkFrame(range_panel, fg_color="transparent")
        range_header.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))
        ctk.CTkLabel(
            range_header,
            text=t("segments_ranges"),
            font=(Fonts.FAMILY, Fonts.SIZE_HEADING, "bold"),
            text_color=Colors.TEXT_PRIMARY,
        ).pack(side="left")
        self._undo_btn = ctk.CTkButton(
            range_header,
            text="↶",
            width=30,
            height=26,
            fg_color=Colors.BG_PANEL,
            hover_color=Colors.BORDER_LIGHT,
            command=self._undo,
        )
        self._undo_btn.pack(side="right")
        Tooltip(self._undo_btn, t("segments_undo"))
        self._redo_btn = ctk.CTkButton(
            range_header,
            text="↷",
            width=30,
            height=26,
            fg_color=Colors.BG_PANEL,
            hover_color=Colors.BORDER_LIGHT,
            command=self._redo,
        )
        self._redo_btn.pack(side="right", padx=4)
        Tooltip(self._redo_btn, t("segments_redo"))

        self._segment_list = ctk.CTkScrollableFrame(
            range_panel,
            fg_color=Colors.BG_PANEL,
            corner_radius=Sizing.BORDER_RADIUS,
        )
        self._segment_list.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)

        list_actions = ctk.CTkFrame(range_panel, fg_color="transparent")
        list_actions.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        self._new_btn = ctk.CTkButton(
            list_actions,
            text=t("segments_new_range"),
            height=28,
            command=self._new_range,
        )
        self._new_btn.pack(side="left")
        self._clear_btn = ctk.CTkButton(
            list_actions,
            text=t("segments_clear_all"),
            height=28,
            fg_color=Colors.BG_PANEL,
            hover_color=Colors.STATUS_ERROR,
            command=self._clear_ranges,
        )
        self._clear_btn.pack(side="right")

        editor = ctk.CTkFrame(range_panel, fg_color="transparent")
        editor.grid(row=3, column=0, sticky="ew", padx=8, pady=(4, 8))
        editor.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            editor,
            text=t("segments_start"),
            text_color=Colors.TEXT_PRIMARY,
        ).grid(row=0, column=0, sticky="w", padx=(0, 6), pady=3)
        self._start_entry = ctk.CTkEntry(editor, font=(Fonts.FAMILY_MONO, Fonts.SIZE_SMALL))
        self._start_entry.grid(row=0, column=1, sticky="ew", pady=3)
        self._mark_in_btn = ctk.CTkButton(
            editor,
            text=t("segments_mark_in_short"),
            width=42,
            command=self._set_mark_in,
        )
        self._mark_in_btn.grid(row=0, column=2, padx=(5, 0), pady=3)
        Tooltip(self._mark_in_btn, t("segments_mark_in_hint"))
        ctk.CTkLabel(
            editor,
            text=t("segments_end"),
            text_color=Colors.TEXT_PRIMARY,
        ).grid(row=1, column=0, sticky="w", padx=(0, 6), pady=3)
        self._end_entry = ctk.CTkEntry(editor, font=(Fonts.FAMILY_MONO, Fonts.SIZE_SMALL))
        self._end_entry.grid(row=1, column=1, sticky="ew", pady=3)
        self._mark_out_btn = ctk.CTkButton(
            editor,
            text=t("segments_mark_out_short"),
            width=42,
            command=self._set_mark_out,
        )
        self._mark_out_btn.grid(row=1, column=2, padx=(5, 0), pady=3)
        Tooltip(self._mark_out_btn, t("segments_mark_out_hint"))
        self._range_action = ctk.CTkButton(
            editor,
            text=t("segments_add_range"),
            command=self._add_or_update,
        )
        self._range_action.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(5, 0))

        scan_bar = ctk.CTkFrame(self, fg_color="transparent")
        scan_bar.pack(fill="x", padx=16, pady=(8, 0))
        self._scan_btn = ctk.CTkButton(
            scan_bar,
            text=t("segments_scan"),
            height=24,
            width=130,
            command=self._start_scan,
        )
        self._scan_btn.pack(side="left")
        self._scan_stop_btn = ctk.CTkButton(
            scan_bar,
            text=t("segments_scan_stop"),
            height=24,
            width=90,
            fg_color=Colors.BG_CARD,
            hover_color=Colors.STATUS_ERROR,
            state="disabled",
            command=self._stop_scan,
        )
        self._scan_stop_btn.pack(side="left", padx=(6, 0))
        ctk.CTkLabel(
            scan_bar,
            text=t("segments_scan_interval"),
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_PRIMARY,
        ).pack(side="left", padx=(12, 4))
        self._scan_interval = ctk.CTkOptionMenu(
            scan_bar,
            values=["0.25s", "0.5s", "1s", "2s"],
            width=70,
            height=24,
        )
        self._scan_interval.set("1s")
        self._scan_interval.pack(side="left")
        ctk.CTkLabel(
            scan_bar,
            text=t("segments_scan_threshold"),
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_PRIMARY,
        ).pack(side="left", padx=(12, 4))
        self._scan_thr_slider = ctk.CTkSlider(
            scan_bar,
            from_=SCAN_SCORE_FLOOR,
            to=0.9,
            width=110,
            command=self._on_scan_threshold,
        )
        self._scan_thr_slider.set(self._scan_threshold)
        self._scan_thr_slider.pack(side="left")
        self._scan_thr_label = ctk.CTkLabel(
            scan_bar,
            text=f"{self._scan_threshold:.2f}",
            font=(Fonts.FAMILY_MONO, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_PRIMARY,
            width=36,
        )
        self._scan_thr_label.pack(side="left", padx=(4, 0))
        self._scan_add_btn = ctk.CTkButton(
            scan_bar,
            text=t("segments_scan_add", count=0),
            height=24,
            state="disabled",
            command=self._add_detected_ranges,
        )
        self._scan_add_btn.pack(side="left", padx=(12, 0))
        overlay_box = ctk.CTkFrame(scan_bar, fg_color="transparent")
        overlay_box.pack(side="right")
        self._scan_overlay_toggle = create_compact_switch(
            overlay_box,
            self._toggle_scan_overlay,
            Colors.BG_MAIN,
        )
        self._scan_overlay_toggle.pack(side="right")
        ctk.CTkLabel(
            overlay_box,
            text=t("segments_scan_overlay"),
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_PRIMARY,
        ).pack(side="right", padx=(0, 6))
        self._scan_progress = ctk.CTkProgressBar(scan_bar, width=120, height=8)
        self._scan_progress.set(0.0)
        self._scan_status = ctk.CTkLabel(
            scan_bar,
            text="",
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.STATUS_PENDING,
        )
        self._scan_status.pack(side="left", padx=(12, 0))

        timeline_header = ctk.CTkFrame(self, fg_color="transparent")
        timeline_header.pack(fill="x", padx=16, pady=(4, 2))
        for text, command, tip in (
            ("−", lambda: self._timeline.zoom_out(), "segments_zoom_out"),
            (t("segments_fit"), lambda: self._timeline.fit(), "segments_fit_hint"),
            ("+", lambda: self._timeline.zoom_in(), "segments_zoom_in"),
        ):
            button = ctk.CTkButton(
                timeline_header,
                text=text,
                width=34 if len(text) == 1 else 48,
                height=24,
                fg_color=Colors.BG_CARD,
                hover_color=Colors.BORDER_LIGHT,
                command=command,
            )
            button.pack(side="right", padx=(4, 0))
            Tooltip(button, t(tip))

        self._timeline = SegmentTimeline(
            self,
            duration=self._state.duration,
            fps=self._state.fps,
            on_seek=self._seek,
            on_create=self._timeline_create,
            on_select=self._select_range,
            on_adjust=self._timeline_adjust,
        )
        self._timeline.pack(fill="x", padx=16)

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill="x", padx=16, pady=(3, 12))
        info_column = ctk.CTkFrame(footer, fg_color="transparent")
        info_column.pack(side="left", fill="x", expand=True)
        self._workload = ctk.CTkLabel(
            info_column,
            text="",
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_PRIMARY,
            anchor="w",
        )
        self._workload.pack(fill="x")
        self._notice = ctk.CTkLabel(
            info_column,
            text="",
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.STATUS_PENDING,
            anchor="w",
        )
        self._notice.pack(fill="x")
        self._cancel_btn = ctk.CTkButton(
            footer,
            text=t("segments_cancel"),
            fg_color=Colors.BG_CARD,
            hover_color=Colors.BORDER_LIGHT,
            command=self._request_close,
        )
        self._cancel_btn.pack(side="right")
        self._apply_btn = ctk.CTkButton(
            footer,
            text=t("segments_apply"),
            command=self._save,
        )
        self._apply_btn.pack(side="right", padx=8)

        initial = self._state.selected_segment
        if initial is not None:
            self._current = initial.start
        self._refresh_all()
        self.update_idletasks()
        self._preview_generation = self._preview_worker.seek(self._current)
        self._start_keyframe_probe(metadata)

    def _poll_workers(self) -> None:
        if self._closed.is_set():
            return
        try:
            while True:
                event = self._preview_worker.events.get_nowait()
                if isinstance(event, PreviewLoaded):
                    if self._state is None:
                        self._build_editor(event.metadata)
                elif isinstance(event, PreviewFrame):
                    self._show_frame(event)
                elif isinstance(event, PreviewEnded):
                    if event.generation == self._preview_generation:
                        self._set_playing(False)
                elif isinstance(event, PreviewFailed):
                    self._show_preview_error(event.message)
        except queue.Empty:
            pass

        try:
            while True:
                result = self._analysis_events.get_nowait()
                if isinstance(result, Exception):
                    self._analysis_error = str(result)
                else:
                    self._keyframe_index = result
                    self._analysis_error = None
                self._refresh_workload()
        except queue.Empty:
            pass

        if self._restoration_worker is not None:
            try:
                while True:
                    self._handle_restoration_event(self._restoration_worker.events.get_nowait())
            except queue.Empty:
                pass
        scan_worker = self._scan_worker
        if scan_worker is not None:
            try:
                while True:
                    self._handle_scan_event(scan_worker.events.get_nowait())
            except queue.Empty:
                pass
        if self._state is not None:
            self._refresh_restore_toggle()
        self.after(25, self._poll_workers)

    def _start_keyframe_probe(self, metadata: VideoMetadata) -> None:
        def _probe() -> None:
            try:
                from jasna.media.splice import probe_keyframes

                result = probe_keyframes(self._job.path, metadata)
            except Exception as exc:
                result = exc
            if not self._closed.is_set():
                self._analysis_events.put(result)

        threading.Thread(
            target=_probe,
            name=f"segment-keyframes-{self._job.filename}",
            daemon=True,
        ).start()

    def _show_frame(self, event: PreviewFrame) -> None:
        if self._state is None or event.generation != self._preview_generation:
            return
        self._current = min(self._state.duration, max(0.0, event.seconds))
        self._preview_source = event.image
        if self._restore_active and self._playing and self._restored_clip:
            self._show_restored_clip_frame(self._current)
        elif not self._restore_active:
            self._refresh_preview_image()
        self._time_label.configure(text=self._time_text())
        self._timeline.reveal(self._current)
        self._refresh_timeline()
        if self._playing:
            delay = max(10, round(1000 / min(60.0, self._state.fps)))
            self._next_frame_after = self.after(delay, self._request_next_frame)

    def _preview_resized(self, _event=None) -> None:
        if self._resize_after is not None:
            try:
                self.after_cancel(self._resize_after)
            except tk.TclError:
                pass
        self._resize_after = self.after(60, self._refresh_preview_image)

    def _fit_to_label(self, label: ctk.CTkLabel, source: Image.Image) -> ctk.CTkImage:
        width = max(2, label.winfo_width() - 16)
        height = max(2, label.winfo_height() - 16)
        source_width, source_height = source.size
        scale = min(width / source_width, height / source_height)
        target_size = (
            max(2, round(source_width * scale)),
            max(2, round(source_height * scale)),
        )
        image = source.resize(target_size, Image.Resampling.LANCZOS)
        return ctk.CTkImage(image, size=image.size)

    def _refresh_preview_image(self) -> None:
        self._resize_after = None
        source = self._restored_source if self._restore_active else self._preview_source
        if source is None or self._closed.is_set():
            return
        if self._scan_overlay and not self._restore_active:
            source = self._apply_scan_overlay(source)
        self._preview_image = self._fit_to_label(self._preview, source)
        self._preview.configure(image=self._preview_image, text="")

    def _show_preview_error(self, message: str) -> None:
        self._set_playing(False)
        if self._state is None:
            self._loading_bar.stop()
            self._loading_label.configure(text=message, text_color=Colors.STATUS_ERROR)
            return
        self._preview.configure(image=None, text=message, text_color=Colors.STATUS_ERROR)

    def _time_text(self) -> str:
        if self._state is None:
            return format_timestamp(0)
        return f"{format_timestamp(self._current)} / {format_timestamp(self._state.duration)}"

    def _seek(self, seconds: float) -> None:
        state = self._require_state()
        self._set_playing(False)
        self._current = state.snap(seconds)
        self._time_label.configure(text=self._time_text())
        self._timeline.reveal(self._current)
        self._refresh_timeline()
        self._preview_generation = self._preview_worker.seek(self._current)
        if self._restore_active:
            self._restored_clip = ()
            self._restored_source = None
            self._preview_image = None
            self._restore_play_pending = False
            self._preview.configure(
                image=None,
                text=t("segments_restore_restoring"),
                text_color=Colors.STATUS_PENDING,
            )
            self._schedule_restoration_preview()

    def _step(self, frames: int) -> None:
        state = self._require_state()
        self._seek(self._current + int(frames) / state.fps)

    def _toggle_play(self) -> None:
        state = self._require_state()
        if self._restore_play_pending:
            self._restore_play_pending = False
            self._play.configure(text="▶")
            self._request_restoration_preview()
            return
        if self._playing:
            self._set_playing(False)
            return
        if self._current >= state.duration - 1 / state.fps:
            self._current = 0.0
        if self._restore_active:
            if self._restored_clip_covers(self._current):
                self._start_restored_playback(self._current)
            else:
                self._request_restoration_playback(self._current)
            return
        self._set_playing(True)
        self._preview_generation = self._preview_worker.seek(self._current)

    def _set_playing(self, playing: bool) -> None:
        self._playing = bool(playing)
        if hasattr(self, "_play"):
            self._play.configure(text="⏸" if self._playing else "▶")
        if not self._playing and self._next_frame_after is not None:
            try:
                self.after_cancel(self._next_frame_after)
            except tk.TclError:
                pass
            self._next_frame_after = None

    def _request_next_frame(self) -> None:
        self._next_frame_after = None
        if self._playing and not self._closed.is_set():
            if self._restore_active and self._restored_clip:
                state = self._require_state()
                last_seconds = self._restored_clip[-1].seconds
                frame_duration = 1 / state.fps
                if self._current >= last_seconds - frame_duration / 2:
                    self._set_playing(False)
                    next_seconds = last_seconds + frame_duration
                    if next_seconds < state.duration - frame_duration / 2:
                        self._request_restoration_playback(next_seconds)
                    return
            self._preview_worker.next_frame()

    def _toggle_restoration_preview(self) -> None:
        self._require_state()
        if self._restore_active:
            self._deactivate_restoration_preview()
            return
        if self._is_gpu_busy():
            self._restore_toggle.deselect()
            return
        self._restore_active = True
        self._set_playing(False)
        self._restore_toggle.select()
        self._restored_source = None
        self._preview_image = None
        self._preview.configure(
            image=None,
            text=t("segments_restore_restoring"),
            text_color=Colors.STATUS_PENDING,
        )
        if self._restoration_worker is None:
            self._set_preview_gpu_busy(True)
            try:
                self._restoration_worker = RestorationPreviewWorker(
                    self._job.path,
                    self._metadata,
                    on_stopped=lambda: self._set_preview_gpu_busy(False),
                )
                self._restoration_worker.start()
            except Exception:
                self._restoration_worker = None
                self._set_preview_gpu_busy(False)
                self._restore_active = False
                self._restore_toggle.deselect()
                self._refresh_preview_image()
                raise
        self._request_restoration_preview()

    def _deactivate_restoration_preview(self) -> None:
        self._restore_active = False
        if self._restoration_worker is not None:
            self._restoration_worker.cancel()
        self._restore_play_pending = False
        self._restored_clip = ()
        self._set_playing(False)
        if self._restore_after is not None:
            try:
                self.after_cancel(self._restore_after)
            except tk.TclError:
                pass
            self._restore_after = None
        self._restored_source = None
        self._preview_image = None
        self._restore_toggle.deselect()
        self._refresh_preview_image()

    def _schedule_restoration_preview(self) -> None:
        self._restore_generation = -1
        if self._restore_after is not None:
            try:
                self.after_cancel(self._restore_after)
            except tk.TclError:
                pass
        self._restore_after = self.after(400, self._request_restoration_preview)

    def _request_restoration_preview(self) -> None:
        self._restore_after = None
        if not self._restore_active or self._closed.is_set():
            return
        self._restore_play_pending = False
        self._restored_clip = ()
        self._play.configure(text="▶")
        self._restore_generation = self._restoration_worker.request(
            self._current,
            self._get_settings(),
        )
        if self._restored_source is None:
            self._preview.configure(
                image=None,
                text=t("segments_restore_restoring"),
                text_color=Colors.STATUS_PENDING,
            )

    def _request_restoration_playback(self, start_seconds: float) -> None:
        if not self._restore_active or self._closed.is_set():
            return
        if self._restore_after is not None:
            try:
                self.after_cancel(self._restore_after)
            except tk.TclError:
                pass
            self._restore_after = None
        self._set_playing(False)
        self._restore_play_pending = True
        self._restored_clip = ()
        self._restored_source = None
        self._preview_image = None
        self._play.configure(text="…")
        self._preview.configure(
            image=None,
            text=t("segments_restore_restoring"),
            text_color=Colors.STATUS_PENDING,
        )
        self._restore_generation = self._restoration_worker.request(
            start_seconds,
            self._get_settings(),
            playback=True,
        )

    def _handle_restoration_event(self, event) -> None:
        if not self._restore_active or event.generation != self._restore_generation:
            return
        if isinstance(event, RestorationStatus):
            if self._restored_source is None:
                self._preview.configure(
                    image=None,
                    text=self._restoration_status_text(event.message),
                    text_color=Colors.STATUS_PENDING,
                )
        elif isinstance(event, RestorationFrame):
            self._restored_clip = ()
            self._restored_source = event.image
            self._refresh_preview_image()
        elif isinstance(event, RestorationClip):
            self._restored_clip = event.frames
            if not event.frames:
                self._restore_play_pending = False
                self._play.configure(text="▶")
                return
            self._restored_source = event.frames[0].image
            self._refresh_preview_image()
            if self._restore_play_pending:
                self._restore_play_pending = False
                self._start_restored_playback(event.frames[0].seconds)
        elif isinstance(event, RestorationFailed):
            self._restore_play_pending = False
            self._play.configure(text="▶")
            self._restored_clip = ()
            self._restored_source = None
            self._preview_image = None
            self._preview.configure(
                image=None,
                text=t("segments_restore_failed", message=event.message),
                text_color=Colors.STATUS_ERROR,
            )

    def _restored_clip_covers(self, seconds: float) -> bool:
        if len(self._restored_clip) < 2 or self._state is None:
            return False
        tolerance = 0.75 / self._state.fps
        return (
            self._restored_clip[0].seconds - tolerance
            <= seconds
            < self._restored_clip[-1].seconds - tolerance
        )

    def _start_restored_playback(self, seconds: float) -> None:
        self._show_restored_clip_frame(seconds)
        self._set_playing(True)
        self._preview_generation = self._preview_worker.seek(seconds)

    def _show_restored_clip_frame(self, seconds: float) -> None:
        frame = min(self._restored_clip, key=lambda item: abs(item.seconds - seconds))
        self._restored_source = frame.image
        self._refresh_preview_image()

    @staticmethod
    def _restoration_status_text(message: str) -> str:
        if message == "loading_models":
            return t("segments_restore_loading_models")
        if message == "restoring":
            return t("segments_restore_restoring")
        return message

    def _refresh_restore_toggle(self) -> None:
        busy = bool(self._is_gpu_busy())
        if busy == self._restore_toggle_blocked:
            return
        self._restore_toggle_blocked = busy
        if busy and self._restore_active:
            self._deactivate_restoration_preview()
        self._restore_toggle.configure(state="disabled" if busy else "normal")
        self._restore_toggle_tooltip.set_text(
            t("segments_restore_gpu_busy") if busy else t("segments_restore_preview_hint")
        )

    def _start_scan(self) -> None:
        self._require_state()
        if self._scan_worker is not None:
            return
        if self._is_gpu_busy():
            self._scan_status.configure(
                text=t("segments_restore_gpu_busy"), text_color=Colors.STATUS_ERROR
            )
            return
        if self._restore_active:
            self._deactivate_restoration_preview()
        if self._restoration_worker is not None:
            self._restoration_worker.close()
            self._restoration_worker = None
        self._set_playing(False)
        self._set_preview_gpu_busy(True)
        stride_seconds = float(self._scan_interval.get().rstrip("s"))
        try:
            worker = MosaicScanWorker(
                self._job.path,
                self._metadata,
                self._get_settings(),
                stride_seconds=stride_seconds,
                on_stopped=lambda: self._set_preview_gpu_busy(False),
            )
            worker.start()
        except Exception:
            self._set_preview_gpu_busy(False)
            raise
        self._scan_worker = worker
        self._scan_result = None
        self._scan_proposals = ()
        self._timeline.set_detections(())
        self._scan_progress.set(0.0)
        self._scan_status.configure(
            text=t("segments_restore_loading_models"), text_color=Colors.STATUS_PENDING
        )
        self._set_scan_locked(True)

    def _stop_scan(self) -> None:
        if self._scan_worker is None:
            return
        self._scan_worker.stop()
        self._scan_stop_btn.configure(state="disabled")

    def _scan_lockable_widgets(self) -> tuple:
        return (
            self._scan_btn,
            self._scan_interval,
            self._scan_thr_slider,
            self._scan_add_btn,
            self._apply_btn,
            self._cancel_btn,
            self._new_btn,
            self._clear_btn,
            self._undo_btn,
            self._redo_btn,
            self._range_action,
            self._mark_in_btn,
            self._mark_out_btn,
            self._step_back,
            self._step_forward,
            self._play,
        )

    def _set_scan_locked(self, locked: bool) -> None:
        state = "disabled" if locked else "normal"
        for widget in self._scan_lockable_widgets():
            widget.configure(state=state)
        self._scan_stop_btn.configure(state="normal" if locked else "disabled")
        if locked:
            self._scan_progress.pack(side="left", padx=(12, 0))
        else:
            self._scan_progress.pack_forget()
            self._refresh_all()
            self._refresh_scan_view()

    def _handle_scan_event(self, event) -> None:
        if isinstance(event, ScanStatus):
            self._scan_status.configure(
                text=self._restoration_status_text(event.message),
                text_color=Colors.STATUS_PENDING,
            )
        elif isinstance(event, ScanProgress):
            self._scan_progress.set(event.fraction)
            self._scan_status.configure(
                text=(
                    f"{event.fps:.0f} fps · "
                    f"ETA {format_timestamp(event.eta_seconds, milliseconds=False)}"
                ),
                text_color=Colors.STATUS_PENDING,
            )
        elif isinstance(event, ScanFailed):
            self._scan_worker = None
            self._scan_status.configure(
                text=t("segments_scan_failed", message=event.message),
                text_color=Colors.STATUS_ERROR,
            )
            self._set_scan_locked(False)
        elif isinstance(event, ScanCompleted):
            self._scan_worker = None
            self._scan_result = event.result
            self._scan_status.configure(
                text=t("segments_scan_stopped") if event.stopped else "",
                text_color=Colors.STATUS_PENDING,
            )
            self._set_scan_locked(False)

    def _refresh_scan_view(self) -> None:
        result = self._scan_result
        if result is None or self._state is None:
            return
        runs = segments_from_scores(
            result.times,
            result.scores,
            threshold=self._scan_threshold,
            stride=result.stride,
            duration=self._state.duration,
            pad=0.0,
        )
        self._timeline.set_detections(runs)
        self._scan_proposals = segments_from_scores(
            result.times,
            result.scores,
            threshold=self._scan_threshold,
            stride=result.stride,
            duration=self._state.duration,
        )
        count = len(self._scan_proposals)
        self._scan_add_btn.configure(
            text=t("segments_scan_add", count=count),
            state="normal" if count and self._scan_worker is None else "disabled",
        )
        if not count and self._scan_worker is None:
            self._scan_status.configure(
                text=t("segments_scan_none"), text_color=Colors.STATUS_PENDING
            )
        if self._scan_overlay:
            self._refresh_preview_image()

    def _on_scan_threshold(self, value: float) -> None:
        self._scan_threshold = float(value)
        self._scan_thr_label.configure(text=f"{self._scan_threshold:.2f}")
        if self._scan_thr_after is not None:
            try:
                self.after_cancel(self._scan_thr_after)
            except tk.TclError:
                pass
        self._scan_thr_after = self.after(60, self._apply_scan_threshold)

    def _apply_scan_threshold(self) -> None:
        self._scan_thr_after = None
        self._refresh_scan_view()

    def _add_detected_ranges(self) -> None:
        state = self._require_state()
        if not self._scan_proposals:
            return
        added = state.add_many(self._scan_proposals)
        self._set_edit_result_notice(0)
        if not added:
            self._edit_notice = t("segments_scan_none")
            self._edit_notice_warning = True
        self._refresh_all()

    def _toggle_scan_overlay(self) -> None:
        self._scan_overlay = not self._scan_overlay
        if self._scan_overlay:
            self._scan_overlay_toggle.select()
        else:
            self._scan_overlay_toggle.deselect()
        self._refresh_preview_image()

    def _apply_scan_overlay(self, image: Image.Image) -> Image.Image:
        result = self._scan_result
        if result is None:
            return image
        mask = result.mask_at(self._current)
        if mask is None:
            return image
        mask_np = mask.numpy()
        if not mask_np.any():
            return image
        alpha = Image.fromarray((mask_np * 130).astype("uint8"), "L").resize(
            image.size, Image.Resampling.NEAREST
        )
        overlay = Image.new("RGB", image.size, "#ef4444")
        composed = image.copy()
        composed.paste(overlay, (0, 0), alpha)
        return composed

    def _new_range(self) -> None:
        state = self._require_state()
        state.select(None)
        state.clear_marks()
        start = state.snap(self._current)
        end = state.snap(min(state.duration, start + max(1.0, 1 / state.fps)))
        if end <= start:
            start = state.snap(max(0.0, state.duration - 1.0))
            end = state.duration
        self._set_entry(self._start_entry, start)
        self._set_entry(self._end_entry, end)
        self._edit_notice = None
        self._refresh_all()
        self._start_entry.focus_set()

    def _set_mark_in(self) -> None:
        state = self._require_state()
        self._set_entry(self._start_entry, state.set_mark_in(self._current))
        self._edit_notice = None
        self._refresh_notice()

    def _set_mark_out(self) -> None:
        state = self._require_state()
        self._set_entry(self._end_entry, state.set_mark_out(self._current))
        self._edit_notice = None
        self._refresh_notice()

    def _add_or_update(self) -> None:
        state = self._require_state()
        try:
            start = parse_timestamp(self._start_entry.get())
            end = parse_timestamp(self._end_entry.get())
            if start > state.duration or end > state.duration + 1e-6:
                self._edit_notice = t("segments_time_out_of_bounds")
                self._edit_notice_warning = False
                self._refresh_notice()
                return
            if state.selected_segment is None:
                result = state.add(start, end)
            else:
                result = state.replace_selected(start, end)
        except ValueError:
            self._edit_notice = t("segments_invalid_range")
            self._edit_notice_warning = False
            self._refresh_notice()
            return
        self._set_edit_result_notice(result.merged_count)
        selected = state.selected_segment
        if selected is not None:
            self._current = selected.start
            self._preview_generation = self._preview_worker.seek(self._current)
        self._refresh_all()

    def _timeline_create(self, start: float, end: float) -> None:
        state = self._require_state()
        result = state.add(start, end)
        self._set_edit_result_notice(result.merged_count)
        self._refresh_all()

    def _timeline_adjust(self, index: int, start: float, end: float) -> None:
        state = self._require_state()
        state.select(index)
        result = state.adjust_selected(start, end)
        self._set_edit_result_notice(result.merged_count)
        self._refresh_all()

    def _select_range(self, index: int) -> None:
        state = self._require_state()
        state.select(index)
        segment = state.selected_segment
        if segment is not None:
            self._set_entry(self._start_entry, segment.start)
            self._set_entry(self._end_entry, segment.end)
        self._refresh_all()

    def _delete_range(self, index: int) -> None:
        state = self._require_state()
        state.select(index)
        state.delete_selected()
        self._edit_notice = None
        self._refresh_all()

    def _clear_ranges(self) -> None:
        state = self._require_state()
        if not state.segments:
            return
        if not messagebox.askyesno(
            t("segments_clear_all"),
            t("segments_clear_confirm"),
            parent=self,
        ):
            return
        state.clear()
        self._edit_notice = None
        self._refresh_all()

    def _undo(self) -> None:
        if self._require_state().undo():
            self._edit_notice = None
            self._refresh_all()

    def _redo(self) -> None:
        if self._require_state().redo():
            self._edit_notice = None
            self._refresh_all()

    def _render_segment_list(self) -> None:
        state = self._require_state()
        for child in self._segment_list.winfo_children():
            child.destroy()
        if not state.segments:
            ctk.CTkLabel(
                self._segment_list,
                text=t("segments_drag_hint"),
                text_color=Colors.STATUS_PENDING,
                font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
                wraplength=230,
            ).pack(fill="x", padx=8, pady=14)
            return
        for index, segment in enumerate(state.segments):
            selected = index == state.selected_index
            row = ctk.CTkFrame(
                self._segment_list,
                fg_color=Colors.PRIMARY_DARK if selected else Colors.BG_CARD,
                corner_radius=Sizing.BORDER_RADIUS,
            )
            row.pack(fill="x", pady=2)
            label = ctk.CTkButton(
                row,
                text=t(
                    "segments_range_row",
                    index=index + 1,
                    start=format_timestamp(segment.start),
                    end=format_timestamp(segment.end),
                    duration=segment.duration,
                ),
                anchor="w",
                fg_color="transparent",
                hover_color=Colors.BORDER_LIGHT,
                text_color=Colors.TEXT_PRIMARY,
                font=(Fonts.FAMILY_MONO, Fonts.SIZE_TINY),
                command=lambda i=index: self._select_range(i),
            )
            label.pack(side="left", fill="x", expand=True)
            delete = ctk.CTkButton(
                row,
                text="✕",
                width=28,
                height=28,
                fg_color="transparent",
                hover_color=Colors.STATUS_ERROR,
                command=lambda i=index: self._delete_range(i),
            )
            delete.pack(side="right", padx=3)
            Tooltip(delete, t("segments_delete_range"))

    def _refresh_all(self) -> None:
        if self._state is None:
            return
        state = self._state
        if state.selected_segment is not None:
            self._set_entry(self._start_entry, state.selected_segment.start)
            self._set_entry(self._end_entry, state.selected_segment.end)
        self._range_action.configure(
            text=(
                t("segments_update_range")
                if state.selected_segment is not None
                else t("segments_add_range")
            )
        )
        self._apply_btn.configure(text=t("segments_apply"))
        self._undo_btn.configure(state="normal" if state.can_undo else "disabled")
        self._redo_btn.configure(state="normal" if state.can_redo else "disabled")
        self._render_segment_list()
        self._refresh_timeline()
        self._refresh_workload()
        self._refresh_notice()
        self._update_apply_state()

    def _refresh_timeline(self) -> None:
        state = self._require_state()
        self._timeline.set_data(
            segments=state.segments,
            selected_index=state.selected_index,
            playhead=self._current,
        )

    def _refresh_workload(self) -> None:
        if self._state is None:
            return
        state = self._state
        if not state.segments:
            self._compatibility_error = None
            self._workload.configure(
                text=t("segments_workload_full", duration=format_timestamp(state.duration, milliseconds=False)),
                text_color=Colors.TEXT_PRIMARY,
            )
            self._refresh_timeline()
            self._update_apply_state()
            return
        if self._analysis_error is not None:
            self._compatibility_error = t("segments_analysis_failed")
            self._workload.configure(text=t("segments_smart_render_unavailable"), text_color=Colors.STATUS_ERROR)
            self._update_apply_state()
            return
        if self._keyframe_index is None:
            self._compatibility_error = None
            self._workload.configure(
                text=t(
                    "segments_workload",
                    selected=format_timestamp(state.selected_duration, milliseconds=False),
                    percent=state.selected_duration / state.duration * 100,
                ),
                text_color=Colors.TEXT_PRIMARY,
            )
            self._update_apply_state()
            return
        try:
            from jasna.media.splice import build_splice_plan

            build_splice_plan(state.segments, self._keyframe_index, duration=state.duration)
            percent = state.selected_duration / state.duration * 100
            self._compatibility_error = None
            self._workload.configure(
                text=t(
                    "segments_workload",
                    selected=format_timestamp(state.selected_duration, milliseconds=False),
                    percent=percent,
                ),
                text_color=Colors.TEXT_PRIMARY,
            )
        except Exception:
            self._compatibility_error = t("segments_smart_render_unavailable")
            self._workload.configure(
                text=t("segments_smart_render_unavailable"),
                text_color=Colors.STATUS_ERROR,
            )
        self._refresh_notice()
        self._update_apply_state()

    def _refresh_notice(self) -> None:
        if self._state is None:
            return
        if self._compatibility_error:
            self._notice.configure(text=self._compatibility_error, text_color=Colors.STATUS_ERROR)
        elif self._edit_notice:
            self._notice.configure(
                text=self._edit_notice,
                text_color=Colors.STATUS_PAUSED if self._edit_notice_warning else Colors.STATUS_ERROR,
            )
        else:
            self._notice.configure(text="")

    def _update_apply_state(self) -> None:
        if self._state is None or not hasattr(self, "_apply_btn"):
            return
        enabled = not self._compatibility_error
        self._apply_btn.configure(state="normal" if enabled else "disabled")

    def _save(self) -> None:
        state = self._require_state()
        if not self._job.try_set_segments(state.output_segments):
            self._edit_notice = t("segments_job_started")
            self._edit_notice_warning = False
            self._refresh_notice()
            return
        self._saved = True
        self._on_saved(state.output_segments)
        self._finish_close()

    def _request_close(self) -> None:
        if self._scan_worker is not None:
            self._stop_scan()
            return
        if (
            not self._saved
            and self._state is not None
            and self._state.dirty
            and not messagebox.askyesno(
                t("segments_discard_title"),
                t("segments_discard_changes"),
                parent=self,
            )
        ):
            return
        self._finish_close()

    def _finish_close(self) -> None:
        if self._closed.is_set():
            return
        self._set_playing(False)
        self._closed.set()
        self._preview_worker.close()
        if self._restoration_worker is not None:
            self._restoration_worker.close()
        if self._scan_worker is not None:
            self._scan_worker.stop()
        try:
            self.grab_release()
        except tk.TclError:
            pass
        if self._on_closed is not None:
            self._on_closed()
        self.destroy()

    def _bind_shortcuts(self) -> None:
        self.bind("<space>", lambda event: self._shortcut(event, self._toggle_play))
        self.bind("<KeyPress-i>", lambda event: self._shortcut(event, self._set_mark_in))
        self.bind("<KeyPress-o>", lambda event: self._shortcut(event, self._set_mark_out))
        self.bind("<Return>", lambda event: self._shortcut(event, self._add_or_update))
        self.bind("<Delete>", lambda event: self._shortcut(event, self._delete_selected))
        self.bind("<Left>", lambda event: self._shortcut_step(event, -1))
        self.bind("<Right>", lambda event: self._shortcut_step(event, 1))
        self.bind("<Control-z>", lambda event: self._shortcut(event, self._undo, allow_entry=False))
        self.bind("<Control-y>", lambda event: self._shortcut(event, self._redo, allow_entry=False))
        self.bind("<Escape>", lambda _event: self._request_close())

    def _shortcut(self, event, action: Callable[[], None], *, allow_entry: bool = False):
        if self._state is None or self._scan_worker is not None:
            return "break"
        if not allow_entry and self._is_text_entry(event.widget):
            return None
        action()
        return "break"

    def _shortcut_step(self, event, direction: int):
        if self._state is None or self._scan_worker is not None or self._is_text_entry(event.widget):
            return None
        if int(getattr(event, "state", 0)) & 0x0001:
            self._seek(self._current + direction)
        else:
            self._step(direction)
        return "break"

    def _delete_selected(self) -> None:
        state = self._require_state()
        if state.delete_selected():
            self._edit_notice = None
            self._refresh_all()

    def _set_edit_result_notice(self, merged_count: int) -> None:
        self._edit_notice = (
            t("segments_merged", count=merged_count) if merged_count else None
        )
        self._edit_notice_warning = bool(merged_count)

    @staticmethod
    def _is_text_entry(widget) -> bool:
        try:
            return widget.winfo_class() in {"Entry", "TEntry", "Text"}
        except tk.TclError:
            return False

    @staticmethod
    def _set_entry(entry: ctk.CTkEntry, seconds: float) -> None:
        entry.delete(0, "end")
        entry.insert(0, format_timestamp(seconds))

    def _require_state(self) -> SegmentEditorState:
        if self._state is None:
            raise RuntimeError("segment editor is still loading")
        return self._state
