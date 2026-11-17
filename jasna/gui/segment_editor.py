from __future__ import annotations

import logging
import threading
import time
import tkinter as tk
from collections.abc import Callable
from dataclasses import replace

import customtkinter as ctk
from PIL import Image
from tkinter import messagebox

from jasna.gui import scaling
from jasna.gui.locales import t
from jasna.gui.models import AppSettings, JobItem
from jasna.gui.components import Tooltip, grab_modal
from jasna.gui.ltx_models import model_key
from jasna import __version__
from jasna.gui.components import Toast
from jasna.gui.mask_feedback import (
    FeedbackUploadFinished,
    MaskFeedbackWorker,
    MaskSuggestDialog,
)
from jasna.gui.queues import queued_now
from jasna.gui.segment_editor_state import SegmentEditorState, smart_render_error_key
from jasna.gui.segment_scan_panel import ScanPanel
from jasna.gui.segment_seed_panel import SeedPanel
from jasna.gui.segment_preview_card import PreviewCard
from jasna.gui.segment_inspector import RestorationInspector
from jasna.gui.segment_range_panel import RangePanel
from jasna.gui.segment_restore_preview import RestorePreview
from jasna.gui.segment_preview import (
    PreviewEnded,
    PreviewFailed,
    PreviewFrame,
    PreviewFullFrame,
    PreviewKeyframes,
    PreviewLoaded,
    SegmentPreviewWorker,
)
from jasna.gui.settings_sections.encoding import CODEC_CANONICAL_TO_LABEL
from jasna.gui.segment_timeline import SegmentTimeline
from jasna.gui.theme import Colors, Fonts
from jasna.media.probe import VideoMetadata
from jasna.media.splice import canonical_codec
from jasna.segments import (
    SegmentRange,
    format_timestamp,
    job_restoration,
)

logger = logging.getLogger(__name__)


class SegmentEditor(ctk.CTkToplevel):
    """Modal, frame-aware editor for per-job restoration ranges."""

    def __init__(
        self,
        master,
        job: JobItem,
        get_settings: Callable[[], AppSettings],
        is_gpu_busy: Callable[[], bool],
        set_preview_gpu_busy: Callable[[object, bool], None],
        ltx_unavailable_reason: str | None,
        on_saved: Callable[[tuple[SegmentRange, ...]], None],
        on_closed: Callable[[], None],
    ) -> None:
        super().__init__(master)
        self._job = job
        self._get_settings = get_settings
        self._is_gpu_busy = is_gpu_busy
        self._set_preview_gpu_busy = set_preview_gpu_busy
        self._ltx_unavailable_reason = ltx_unavailable_reason
        self._ltx_key = model_key("ltx", get_settings().ltx_model)
        self._ltx_label = t(f"model_{self._ltx_key}")
        self._on_saved = on_saved
        self._on_closed = on_closed
        self._state: SegmentEditorState | None = None
        self._metadata: VideoMetadata | None = None
        self._current = 0.0
        self._playing = False
        self._closed = threading.Event()
        self._saved = False
        self._next_frame_after: str | None = None
        self._preview_source: Image.Image | None = None
        self._preview_generation = 0
        self._preview_stereo = "off"
        self._vr_resolution = None
        self._vr_projection = job.vr_projection or "auto"
        self._restore: RestorePreview | None = None
        self._keyframe_index = None
        self._analysis_error: str | None = None
        self._compatibility_error: str | None = None
        self._edit_notice: str | None = None
        self._edit_notice_warning = False
        self._scan_panel: ScanPanel | None = None
        self._preview_card: PreviewCard | None = None
        self._timeline_zoom_buttons: list = []
        self._mask_feedback_worker = MaskFeedbackWorker()
        self._suggest_busy = False
        self._seeds: SeedPanel | None = None

        self.title(t("segments_title"))
        self.configure(fg_color=Colors.BG_MAIN)
        self.transient(master.winfo_toplevel())
        self.protocol("WM_DELETE_WINDOW", self.request_close)
        self._size_and_center()
        self._build_loading()
        self._bind_shortcuts()
        self.update_idletasks()
        self._take_focus()

        self._preview_worker = SegmentPreviewWorker(
            job.path,
            vr_mode=get_settings().vr_mode,
        )
        self._preview_worker.start()
        self._poll_after = self.after(25, self._poll_workers)

    def _size_and_center(self) -> None:
        rect = scaling.screen_rect(self)
        screen_w, screen_h = rect[2], rect[3]
        min_width, min_height = scaling.to_physical(self, 900, 640)
        side_margin, work_margin = scaling.to_physical(self, 48, 200)
        _, chrome_margin = scaling.to_physical(self, 0, 72)
        height = min(max(min_height, screen_h - work_margin), max(1, screen_h - chrome_margin))
        width = min(max(1, screen_w - side_margin), max(min_width, round(height * 1060 / 720)))
        scaling.apply_geometry(self, width, height, *scaling.centered_position((width, height), rect, rect))
        logical_width, logical_height = scaling.to_logical(self, width, height)
        scaling.apply_minsize(self, min(900, logical_width), min(640, logical_height))

    def _take_focus(self) -> None:
        if self._closed.is_set():
            return
        try:
            grab_modal(self)
        except tk.TclError:
            logger.debug("Segment editor could not grab focus yet", exc_info=True)

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
            text=t("btn_cancel"),
            fg_color=Colors.BG_CARD,
            hover_color=Colors.BORDER_LIGHT,
            command=self._finish_close,
        ).pack(pady=12)

    def _build_editor(self, metadata: VideoMetadata) -> None:
        self._loading_bar.stop()
        self._loading.destroy()
        self._metadata = metadata
        settings = self._get_settings()
        self._job_ltx_seed = int(settings.ltx_seed)
        self._state = SegmentEditorState(
            duration=float(metadata.duration),
            fps=max(1.0, float(metadata.video_fps)),
            segments=self._job.snapshot_segments(),
            default_restoration=job_restoration(settings.restoration_model, settings.ltx_seed),
        )
        self._job.duration_seconds = self._state.duration

        self._build_header(metadata)
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=16)
        body.grid_columnconfigure(0, weight=7, uniform="editor")
        body.grid_columnconfigure(1, weight=4, uniform="editor")
        body.grid_rowconfigure(0, weight=1)
        # Let the preview/range area absorb small-window height changes instead
        # of allowing its requested size to push the timeline/footer off-screen.
        body.grid_propagate(False)

        self._build_preview_card(body)
        self._build_range_panel(body)
        self._build_restore_preview()
        self._build_scan_and_timeline(metadata)
        self._build_footer()

        initial = self._state.selected_segment
        if initial is not None:
            self._current = initial.start
        self._refresh_all()
        self.update_idletasks()
        self._preview_generation = self._preview_worker.seek(self._current)

    def _build_header(self, metadata: VideoMetadata) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=16, pady=(12, 8))
        title_column = ctk.CTkFrame(header, fg_color="transparent")
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
        self._codec_notice = ctk.CTkLabel(
            title_column,
            text="",
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.STATUS_WARNING,
            anchor="w",
            justify="left",
            wraplength=840,
        )
        title_column.pack(side="left", fill="x", expand=True)

    def _build_preview_card(self, body: ctk.CTkFrame) -> None:
        self._preview_card = PreviewCard(
            body,
            source=self._preview_display_source,
            vr_resolution=self._vr_resolution,
            vr_projection=self._vr_projection,
            time_text=self._time_text(),
            on_step=self._step,
            on_toggle_play=self._toggle_play,
            on_suggest_mask=self._suggest_mask,
            on_toggle_restore=lambda: self._restore.toggle(),
            on_projection_changed=self._on_vr_projection_changed,
        )
        self._preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self._preview = self._preview_card.canvas

    def _build_range_panel(self, body: ctk.CTkFrame) -> None:
        self._ranges = RangePanel(
            body,
            state=self._state,
            ltx_label=self._ltx_label,
            current_seconds=lambda: self._current,
            scanning=self._scanning,
            on_seek=self._seek,
            on_show=self._show_preview_at,
            on_notice=self._set_edit_notice,
            on_changed=self._refresh_all,
        )
        self._ranges.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self._build_restoration_inspector(self._ranges.controls)

    def _build_restore_preview(self) -> None:
        self._restore = RestorePreview(
            self._preview_card,
            video_path=self._job.path,
            metadata=self._metadata,
            state=self._state,
            seeds=self._seeds,
            closed=self._closed,
            current_seconds=lambda: self._current,
            settings=self._standard_preview_settings,
            projection=lambda: self._vr_projection,
            is_gpu_busy=self._is_gpu_busy,
            set_gpu_busy=self._set_preview_gpu_busy,
            set_playing=self._set_playing,
            play_from=self._play_from,
        )

    def _build_restoration_inspector(self, parent: ctk.CTkFrame) -> None:
        self._inspector = RestorationInspector(
            parent,
            state=self._state,
            job_ltx_seed=self._job_ltx_seed,
            ltx_key=self._ltx_key,
            ltx_label=self._ltx_label,
            ltx_blocked_reason=self._ltx_unavailable_reason,
            scanning=self._scanning,
            seed_panel=self._build_seed_panel,
            on_edited=self._on_restoration_edited,
            on_selected_changed=self._on_selected_restoration_changed,
        )
        self._seeds = self._inspector.seeds

    def _build_seed_panel(self, inspector: ctk.CTkFrame, on_use_seed: Callable[[int], None]) -> SeedPanel:
        return SeedPanel(
            inspector,
            render_parent=self._preview_card.options,
            video_path=self._job.path,
            metadata=self._metadata,
            state=self._state,
            ltx_blocked_reason=self._ltx_unavailable_reason,
            keyframe_index=lambda: self._keyframe_index,
            base_settings=self._get_settings,
            video_settings=lambda: replace(self._current_video_settings(), vr_projection=self._vr_projection),
            current_seconds=lambda: self._current,
            restore_active=lambda: self._restore.active,
            is_gpu_busy=self._is_gpu_busy,
            scanning=self._scanning,
            claim_gpu=self._claim_gpu_for_seed,
            release_gpu=lambda: self._set_preview_gpu_busy(self._seeds, False),
            on_started=self._on_seed_started,
            on_cancelled=self._on_seed_cancelled,
            on_show_frames=self._show_ltx_frames,
            on_use_seed=on_use_seed,
        )

    def _on_restoration_edited(self) -> None:
        self._edit_notice = None
        self._refresh_all()

    def _on_selected_restoration_changed(self) -> None:
        if self._restore.active:
            self._restore.request()

    def _keyframe_seconds(self) -> tuple[float, ...] | None:
        index = self._keyframe_index
        if index is None:
            return None
        return index.keyframe_seconds()

    def _claim_gpu_for_seed(self) -> None:
        self._restore.close_worker()
        self._set_playing(False)
        self._set_preview_gpu_busy(self._seeds, True)

    def _on_seed_started(self) -> None:
        self._refresh_all()
        self._restore.show_ltx_state()

    def _on_seed_cancelled(self) -> None:
        if self._restore.active:
            self._restore.request()

    def _show_ltx_frames(self, key: tuple[float, float]) -> None:
        """Turn the restore preview on over the rendered LTX range ``key``."""
        if not key[0] <= self._current < key[1]:
            self._seek(key[0])
        self._restore.activate()
        self._refresh_all()
        self._restore.request()

    def _build_scan_and_timeline(self, metadata: VideoMetadata) -> None:
        self._timeline = SegmentTimeline(
            self,
            duration=self._state.duration,
            fps=self._state.fps,
            on_seek=self._seek,
            on_create=self._ranges.create,
            on_select=self._ranges.select,
            on_adjust=self._ranges.adjust,
        )
        settings = self._get_settings()
        self._scan_panel = ScanPanel(
            self,
            video_path=self._job.path,
            metadata=metadata,
            state=self._state,
            timeline=self._timeline,
            detection_model=self._job.detection_model or str(settings.detection_model),
            threshold=(
                self._job.detection_score_threshold
                if self._job.detection_score_threshold is not None
                else settings.detection_score_threshold
            ),
            stereo=self._preview_stereo,
            ltx_seed=None if self._ltx_unavailable_reason is not None else self._job_ltx_seed,
            keyframes=self._keyframe_seconds,
            base_settings=self._get_settings,
            current_seconds=lambda: self._current,
            is_gpu_busy=self._is_gpu_busy,
            claim_gpu=self._claim_gpu_for_scan,
            release_gpu=lambda: self._set_preview_gpu_busy(self._scan_panel, False),
            on_lock_changed=self._on_scan_lock_changed,
            on_ranges_added=self._on_scan_ranges_added,
            on_preview_changed=self._preview.redraw,
            on_detection_changed=self._on_scan_detection_changed,
        )
        self._scan_panel.pack(fill="x", padx=16, pady=(8, 0))

        timeline_header = ctk.CTkFrame(self, fg_color="transparent")
        timeline_header.pack(fill="x", padx=16, pady=(6, 2))
        timeline_heading = ctk.CTkFrame(timeline_header, fg_color="transparent")
        timeline_heading.pack(side="left")
        ctk.CTkLabel(
            timeline_heading,
            text=t("segments_timeline_title"),
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL, "bold"),
            text_color=Colors.TEXT_PRIMARY,
        ).pack(side="left")
        ctk.CTkLabel(
            timeline_heading,
            text=t("segments_timeline_hint"),
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.STATUS_PENDING,
        ).pack(side="left", padx=(8, 0))
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
            self._timeline_zoom_buttons.append(button)

        self._timeline.pack(fill="x", padx=16)

        legend = ctk.CTkFrame(self, fg_color="transparent")
        legend.pack(fill="x", padx=20, pady=(0, 2))
        for color, label in (
            (Colors.PRIMARY, t("model_basicvsrpp")),
            (Colors.MODEL_LTX, self._ltx_label),
            (Colors.STATUS_WARNING, t("segments_legend_detected")),
            ("#f8fafc", t("segments_legend_playhead")),
        ):
            item = ctk.CTkFrame(legend, fg_color="transparent")
            item.pack(side="left", padx=(0, 16))
            ctk.CTkFrame(
                item,
                width=12,
                height=12,
                fg_color=color,
                corner_radius=2,
            ).pack(side="left", padx=(0, 5))
            ctk.CTkLabel(
                item,
                text=label,
                font=(Fonts.FAMILY, Fonts.SIZE_TINY),
                text_color=Colors.STATUS_PENDING,
            ).pack(side="left")

    def _build_footer(self) -> None:
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill="x", padx=16, pady=(3, 12))
        info_column = ctk.CTkFrame(footer, fg_color="transparent")
        info_column.pack(side="left", fill="x", expand=True)
        workload_row = ctk.CTkFrame(info_column, fg_color="transparent")
        workload_row.pack(fill="x")
        self._workload = ctk.CTkLabel(
            workload_row,
            text="",
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_PRIMARY,
            anchor="w",
        )
        self._workload.pack(side="left")
        self._model_split_bar = ctk.CTkProgressBar(
            workload_row,
            width=120,
            height=8,
            fg_color=Colors.PRIMARY,
            progress_color=Colors.MODEL_LTX,
        )
        self._model_split = ctk.CTkLabel(
            workload_row,
            text="",
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.STATUS_PENDING,
            anchor="w",
        )
        for widget in (self._model_split_bar, self._model_split):
            Tooltip(widget, t("segments_model_split_hint"))
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
            text=t("btn_cancel"),
            fg_color=Colors.BG_CARD,
            hover_color=Colors.BORDER_LIGHT,
            command=self.request_close,
        )
        self._cancel_btn.pack(side="right")
        self._apply_btn = ctk.CTkButton(
            footer,
            text=t("segments_apply"),
            command=self._save,
        )
        self._apply_btn.pack(side="right", padx=8)

    def _poll_workers(self) -> None:
        if self._closed.is_set():
            return
        for event in queued_now(self._preview_worker.events):
            if isinstance(event, PreviewLoaded):
                if self._state is None:
                    self._vr_resolution = event.vr_resolution
                    self._preview_stereo = event.vr_resolution.resolved
                    self._build_editor(event.metadata)
            elif isinstance(event, PreviewKeyframes):
                self._keyframe_index = event.index
                self._analysis_error = event.error
                self._refresh_workload()
                if self._scan_panel is not None:
                    self._scan_panel.refresh_view()
                    self._seeds.refresh_render_button()
            elif isinstance(event, PreviewFrame):
                self._show_frame(event)
            elif isinstance(event, PreviewFullFrame):
                self._open_mask_suggest(event.image)
            elif isinstance(event, PreviewEnded):
                if event.generation == self._preview_generation:
                    self._set_playing(False)
            elif isinstance(event, PreviewFailed):
                self._show_preview_error(event.message)

        if self._restore is not None:
            self._restore.poll()
        if self._seeds is not None:
            self._seeds.poll()
        if self._scan_panel is not None:
            self._scan_panel.poll()
        for event in queued_now(self._mask_feedback_worker.events):
            self._handle_feedback_event(event)
        if self._state is not None:
            self._restore.refresh_toggle()
        self._poll_after = self.after(25, self._poll_workers)

    def _show_frame(self, event: PreviewFrame) -> None:
        if self._state is None or event.generation != self._preview_generation:
            return
        self._current = min(self._state.duration, max(0.0, event.seconds))
        self._preview_source = event.image
        if self._playing and self._restore.playing_clip():
            self._restore.show_clip_frame(self._current)
        elif not self._restore.active or self._seeds.ltx_range_at_playhead() is not None:
            self._preview.redraw()
        self._preview_card.time_label.configure(text=self._time_text())
        self._timeline.set_playhead(self._current)
        if self._playing:
            delay = max(10, round(1000 / min(60.0, self._state.fps)))
            self._next_frame_after = self.after(delay, self._request_next_frame)

    def _active_preview_source(self) -> Image.Image | None:
        seed_image = self._seeds.view_image()
        if seed_image is not None:
            return seed_image
        if self._restore.active and self._seeds.ltx_range_at_playhead() is None:
            return self._restore.source
        return None if self._restore.active else self._preview_source

    def _preview_display_source(self) -> Image.Image | None:
        source = self._active_preview_source()
        if source is None or self._closed.is_set():
            return None
        return self._scan_panel.apply_overlay(source) if source is self._preview_source else source

    def _show_preview_error(self, message: str) -> None:
        self._set_playing(False)
        if self._state is None:
            self._loading_bar.stop()
            self._loading_label.configure(text=message, text_color=Colors.STATUS_ERROR)
            return
        self._preview.show_message(message, Colors.STATUS_ERROR)

    def _time_text(self) -> str:
        if self._state is None:
            return format_timestamp(0)
        return f"{format_timestamp(self._current)} / {format_timestamp(self._state.duration)}"

    def _seek(self, seconds: float) -> None:
        snapped = self._require_state().snap(seconds)
        self._move_playhead(snapped, self._preview_worker.seek(snapped))

    def _step(self, frames: int) -> None:
        state = self._require_state()
        if int(frames) == -1:
            self._move_playhead(
                state.snap(self._current - 1 / state.fps), self._preview_worker.previous_frame(self._current)
            )
            return
        self._seek(self._current + int(frames) / state.fps)

    def _move_playhead(self, seconds: float, generation: int) -> None:
        self._set_playing(False)
        self._current = seconds
        self._preview_card.time_label.configure(text=self._time_text())
        self._timeline.set_playhead(self._current)
        self._preview_generation = generation
        self._restore.after_seek()

    def _show_preview_at(self, seconds: float) -> None:
        self._current = seconds
        self._preview_generation = self._preview_worker.seek(seconds)

    def _play_from(self, seconds: float) -> None:
        self._set_playing(True)
        self._preview_generation = self._preview_worker.seek(seconds)

    def _toggle_play(self) -> None:
        state = self._require_state()
        if self._restore.play_pending:
            self._restore.cancel_pending_play()
            return
        if self._playing:
            self._set_playing(False)
            return
        if self._current >= state.duration - 1 / state.fps:
            self._current = 0.0
        if self._restore.active and state.restoration_at(self._current).model != "ltx":
            self._restore.play(self._current)
            return
        self._play_from(self._current)

    def _set_playing(self, playing: bool) -> None:
        self._playing = bool(playing)
        if self._preview_card is not None:
            self._preview_card.play_button.configure(text="⏸" if self._playing else "▶")
        if not self._playing and self._next_frame_after is not None:
            self.after_cancel(self._next_frame_after)
            self._next_frame_after = None

    def _request_next_frame(self) -> None:
        self._next_frame_after = None
        if self._playing and not self._closed.is_set():
            if self._restore.finish_clip_at(self._current):
                return
            self._preview_worker.next_frame()

    def _on_vr_projection_changed(self, projection: str) -> None:
        if projection == self._vr_projection:
            return
        self._vr_projection = projection
        self._restore.projection_changed()

    def _claim_gpu_for_scan(self) -> None:
        self._seeds.release()
        if self._restore.active:
            self._restore.deactivate()
        self._restore.close_worker()
        self._set_playing(False)
        self._set_preview_gpu_busy(self._scan_panel, True)

    def _lockable_widgets(self) -> tuple:
        return (
            self._apply_btn,
            self._cancel_btn,
            *self._ranges.lockable_widgets(),
            *self._preview_card.lockable_widgets(),
            *self._inspector.lockable_widgets(),
            *self._timeline_zoom_buttons,
        )

    def _on_scan_lock_changed(self, locked: bool) -> None:
        state = "disabled" if locked else "normal"
        for widget in self._lockable_widgets():
            widget.configure(state=state)
        self._timeline.set_enabled(not locked)
        if not locked:
            self._refresh_all()

    def _on_scan_ranges_added(self, added: int) -> None:
        self._edit_notice = None if added else t("segments_scan_all_added")
        self._edit_notice_warning = not added
        self._refresh_all()

    def _on_scan_detection_changed(self) -> None:
        if self._restore.active:
            self._restore.schedule()

    def _scanning(self) -> bool:
        return self._scan_panel is not None and self._scan_panel.active

    def _current_video_settings(self) -> AppSettings:
        return self._scan_panel.video_settings()

    def _standard_preview_settings(self) -> AppSettings:
        return replace(self._current_video_settings(), restoration_model="basicvsrpp")

    def _suggest_mask(self) -> None:
        self._require_state()
        if self._suggest_busy:
            return
        self._suggest_busy = True
        self._set_playing(False)
        self._preview_card.suggest_button.configure(state="disabled", text=t("mask_editor_loading_frame"))
        self._preview_worker.grab_full()

    def _open_mask_suggest(self, image: Image.Image) -> None:
        if not self._suggest_busy or self._closed.is_set():
            return
        self._preview_card.suggest_button.configure(text=t("segments_suggest_mask"))
        MaskSuggestDialog(
            self,
            image,
            on_submit=lambda polygons: self._submit_mask_feedback(image, polygons),
            on_closed=self._mask_suggest_closed,
        )

    def _mask_suggest_closed(self) -> None:
        self._suggest_busy = False
        if not self._scanning():
            self._preview_card.suggest_button.configure(state="normal")
        self._take_focus()

    def _submit_mask_feedback(self, image: Image.Image, polygons: tuple) -> None:
        self._mask_feedback_worker.upload(
            image,
            polygons,
            str(self._current_video_settings().detection_model),
            __version__,
        )

    def _handle_feedback_event(self, event) -> None:
        if isinstance(event, FeedbackUploadFinished):
            if event.ok:
                self._show_toast(t("mask_feedback_uploaded"), "success")
            else:
                self._show_toast(
                    t("mask_feedback_upload_failed", message=event.message), "error"
                )

    def _show_toast(self, message: str, type_: str) -> None:
        if self._closed.is_set():
            return
        Toast.show(self, message, type_, rely=0.92)

    def _refresh_all(self) -> None:
        if self._state is None:
            return
        self._ranges.refresh()
        self._apply_btn.configure(text=t("segments_apply"))
        self._inspector.refresh()
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
        self._refresh_model_split()
        if not state.segments:
            self._codec_notice.pack_forget()
            self._compatibility_error = None
            self._workload.configure(
                text=t("segments_workload_full", duration=format_timestamp(state.duration, milliseconds=False)),
                text_color=Colors.TEXT_PRIMARY,
            )
            self._update_apply_state()
            return
        codec = canonical_codec(self._metadata.codec_name)
        self._codec_notice.configure(
            text=t(
                "segments_source_codec_notice",
                codec=CODEC_CANONICAL_TO_LABEL.get(codec, codec.upper()),
            )
        )
        if not self._codec_notice.winfo_manager():
            self._codec_notice.pack(fill="x", pady=(2, 0))
        self._workload.configure(
            text=t(
                "segments_workload",
                selected=format_timestamp(state.selected_duration, milliseconds=False),
                percent=state.selected_duration / state.duration * 100,
            ),
            text_color=Colors.TEXT_PRIMARY,
        )
        if self._analysis_error is not None:
            self._compatibility_error = t("segments_analysis_failed")
            self._refresh_notice()
            self._update_apply_state()
            return
        if self._keyframe_index is None:
            self._compatibility_error = None
            self._refresh_notice()
            self._update_apply_state()
            return
        error_key = smart_render_error_key(state.segments, self._keyframe_index, state.duration)
        self._compatibility_error = None if error_key is None else t(error_key)
        self._refresh_notice()
        self._update_apply_state()

    def _refresh_model_split(self) -> None:
        """Show how the selected time splits between LTX and BasicVSR++ once a range uses LTX."""
        state = self._require_state()
        ltx = state.ltx_duration
        if not ltx:
            self._model_split_bar.pack_forget()
            self._model_split.pack_forget()
            return
        total = state.selected_duration
        self._model_split_bar.set(ltx / total)
        self._model_split.configure(
            text=t(
                "segments_model_split",
                ltx=format_timestamp(ltx, milliseconds=False),
                percent=ltx / total * 100,
                standard=format_timestamp(total - ltx, milliseconds=False),
            )
        )
        if not self._model_split_bar.winfo_manager():
            self._model_split_bar.pack(side="left", padx=(14, 8))
            self._model_split.pack(side="left")

    def _refresh_notice(self) -> None:
        if self._state is None:
            return
        if self._compatibility_error:
            self._notice.configure(text=self._compatibility_error, text_color=Colors.STATUS_ERROR)
        elif self._edit_notice:
            self._notice.configure(
                text=self._edit_notice,
                text_color=Colors.STATUS_WARNING if self._edit_notice_warning else Colors.STATUS_ERROR,
            )
        elif self._ltx_unavailable_reason is not None and any(
            segment.restoration.model == "ltx" for segment in self._state.segments
        ):
            self._notice.configure(text=t("segments_ltx_ranges_unavailable"), text_color=Colors.STATUS_WARNING)
        else:
            self._notice.configure(text="")

    def _update_apply_state(self) -> None:
        if self._state is None:
            return
        enabled = not self._compatibility_error and not self._scanning()
        self._apply_btn.configure(state="normal" if enabled else "disabled")

    def _save(self) -> None:
        state = self._require_state()
        if not self._job.try_set_video_options(
            state.segments,
            detection_model=self._scan_panel.detection_model,
            detection_score_threshold=self._scan_panel.threshold,
            vr_projection=self._vr_projection,
        ):
            self._edit_notice = t("segments_job_started")
            self._edit_notice_warning = False
            self._refresh_notice()
            return
        self._saved = True
        self._on_saved(state.segments)
        self._finish_close()

    def request_close(self) -> None:
        if self._scanning():
            self._scan_panel.stop()
            return
        if self._seeds is not None and self._seeds.rendering_seed is not None and not messagebox.askyesno(
            t("segments_ltx_render_confirm_title"),
            t("segments_seed_stop_confirm"),
            parent=self,
        ):
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
        self.after_cancel(self._poll_after)
        self._preview_worker.close()
        if self._restore is not None:
            self._restore.close()
        if self._seeds is not None:
            self._seeds.close()
        if self._scan_panel is not None:
            self._scan_panel.close()
        self.grab_release()
        self._on_closed()
        self.destroy()

    def join_workers(self, timeout: float) -> None:
        """Wait up to ``timeout`` seconds in all for the GPU workers stopped by closing."""
        deadline = time.monotonic() + timeout
        for part in (self._seeds, self._restore, self._scan_panel):
            if part is not None:
                part.join(max(0.0, deadline - time.monotonic()))

    def _bind_shortcuts(self) -> None:
        self.bind("<space>", lambda event: self._shortcut(event, self._toggle_play))
        self.bind("<KeyPress-i>", lambda event: self._shortcut(event, self._ranges.set_mark_in))
        self.bind("<KeyPress-o>", lambda event: self._shortcut(event, self._ranges.set_mark_out))
        self.bind("<Return>", lambda event: self._shortcut(event, self._ranges.add_or_update))
        self.bind("<Delete>", lambda event: self._shortcut(event, self._ranges.delete_selected))
        self.bind("<Left>", lambda event: self._shortcut_step(event, -1))
        self.bind("<Right>", lambda event: self._shortcut_step(event, 1))
        self.bind("<Control-z>", lambda event: self._shortcut(event, self._ranges.undo))
        self.bind("<Control-y>", lambda event: self._shortcut(event, self._ranges.redo))
        self.bind("<Escape>", lambda _event: self.request_close())

    def _shortcut(self, event, action: Callable[[], None]):
        if self._state is None or self._scanning():
            return "break"
        if self._is_text_entry(event.widget):
            return None
        action()
        return "break"

    def _shortcut_step(self, event, direction: int):
        if self._state is None or self._scanning() or self._is_text_entry(event.widget):
            return None
        if event.state & 0x0001:
            self._seek(self._current + direction)
        else:
            self._step(direction)
        return "break"

    def _set_edit_notice(self, notice: str | None, warning: bool) -> None:
        self._edit_notice = notice
        self._edit_notice_warning = warning
        self._refresh_notice()

    @staticmethod
    def _is_text_entry(widget) -> bool:
        try:
            return widget.winfo_class() in {"Entry", "TEntry", "Text"}
        except tk.TclError:
            return False

    def _require_state(self) -> SegmentEditorState:
        if self._state is None:
            raise RuntimeError("segment editor is still loading")
        return self._state
