from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from collections.abc import Callable

import customtkinter as ctk
from PIL import Image
from tkinter import messagebox

from jasna.gui import scaling
from jasna.gui.locales import t
from jasna.gui.models import AppSettings, JobItem
from jasna.gui.components import Tooltip, grab_modal
from jasna.gui.icons import CompactSwitch
from jasna.gui.restoration_preview import (
    RestorationClip,
    RestorationFailed,
    RestorationFrame,
    RestorationPreviewWorker,
    RestorationStatus,
    RestoredClipFrame,
)
from jasna import __version__
from jasna.gui.components import Toast
from jasna.gui.mask_feedback import (
    FeedbackUploadFinished,
    MaskFeedbackWorker,
    MaskSuggestDialog,
)
from jasna.gui.segment_editor_state import (
    PREVIEW_ZOOM_MAX,
    PREVIEW_ZOOM_STEP,
    PreviewView,
    SegmentEditorState,
    smart_render_error_key,
)
from jasna.gui.segment_scan_panel import ScanPanel, worker_status_text
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
from jasna.gui.settings_sections.widgets import ValueOptionMenu
from jasna.gui.segment_timeline import SegmentTimeline
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.media.probe import VideoMetadata
from jasna.media.splice import canonical_codec
from jasna.segments import SegmentRange, format_timestamp, parse_timestamp

logger = logging.getLogger(__name__)


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
        self._preview_left_eye = False
        self._preview_view = PreviewView()
        self._preview_pan_anchor: tuple[int, int] | None = None
        self.vr_resolution = None
        self._vr_projection = job.vr_projection or "auto"
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
        self._scan_panel: ScanPanel | None = None
        self._play: ctk.CTkButton | None = None
        self._segment_action_widgets: list = []
        self._timeline_zoom_buttons: list = []
        self._mask_feedback_worker = MaskFeedbackWorker()
        self._suggest_busy = False

        self.title(t("segments_title"))
        self.configure(fg_color=Colors.BG_MAIN)
        self.transient(master.winfo_toplevel())
        self.protocol("WM_DELETE_WINDOW", self._request_close)
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
        self.after(25, self._poll_workers)

    def _size_and_center(self) -> None:
        rect = scaling.screen_rect(self)
        screen_w, screen_h = rect[2], rect[3]
        min_width, min_height = scaling.to_physical(self, 900, 640)
        side_margin, work_margin = scaling.to_physical(self, 48, 200)
        _, chrome_margin = scaling.to_physical(self, 0, 72)
        height = min(max(min_height, screen_h - work_margin), max(1, screen_h - chrome_margin))
        width = min(max(1, screen_w - side_margin), max(min_width, round(height * 1060 / 720)))
        x = rect[0] + max(0, (screen_w - width) // 2)
        y = rect[1] + max(0, (screen_h - height) // 2)
        scaling.apply_geometry(self, width, height, x, y)
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
        self._state = SegmentEditorState(
            duration=float(metadata.duration),
            fps=max(1.0, float(metadata.video_fps)),
            segments=self._job.snapshot_segments(),
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
        self._preview.bind("<MouseWheel>", self._preview_mousewheel)
        self._preview.bind("<Button-4>", self._preview_mousewheel)
        self._preview.bind("<Button-5>", self._preview_mousewheel)
        self._preview.bind("<ButtonPress-1>", self._preview_pan_start)
        self._preview.bind("<B1-Motion>", self._preview_pan_drag)
        self._preview.bind("<ButtonRelease-1>", self._preview_pan_end)
        self._preview.bind("<Double-Button-1>", self._reset_preview_view)

        transport = ctk.CTkFrame(preview_card, fg_color="transparent")
        transport.grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 2))
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
        self._suggest_btn = ctk.CTkButton(
            transport,
            text=t("segments_suggest_mask"),
            height=28,
            fg_color=Colors.BG_PANEL,
            hover_color=Colors.BORDER_LIGHT,
            command=self._suggest_mask,
        )
        self._suggest_btn.pack(side="left", padx=(12, 0))
        Tooltip(self._suggest_btn, t("segments_suggest_mask_hint"))

        view_controls = ctk.CTkFrame(preview_card, fg_color="transparent")
        view_controls.grid(row=2, column=0, sticky="ew", padx=8, pady=2)
        self._pan_zoom_hint = ctk.CTkLabel(
            view_controls,
            text=t("segments_preview_pan_zoom_hint"),
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.STATUS_PENDING,
        )
        self._pan_zoom_hint.pack(side="left")
        Tooltip(self._pan_zoom_hint, t("segments_preview_pan_zoom_hint"))
        zoom_controls = ctk.CTkFrame(view_controls, fg_color="transparent")
        zoom_controls.pack(side="right")
        self._zoom_out_btn = ctk.CTkButton(
            zoom_controls,
            text="−",
            width=30,
            height=26,
            fg_color=Colors.BG_PANEL,
            hover_color=Colors.BORDER_LIGHT,
            command=lambda: self._adjust_preview_zoom(-PREVIEW_ZOOM_STEP),
        )
        self._zoom_out_btn.pack(side="left")
        Tooltip(self._zoom_out_btn, t("segments_preview_zoom_out"))
        self._zoom_label = ctk.CTkLabel(
            zoom_controls,
            text="100%",
            width=48,
            font=(Fonts.FAMILY_MONO, Fonts.SIZE_TINY),
            text_color=Colors.TEXT_PRIMARY,
        )
        self._zoom_label.pack(side="left", padx=3)
        self._zoom_in_btn = ctk.CTkButton(
            zoom_controls,
            text="+",
            width=30,
            height=26,
            fg_color=Colors.BG_PANEL,
            hover_color=Colors.BORDER_LIGHT,
            command=lambda: self._adjust_preview_zoom(PREVIEW_ZOOM_STEP),
        )
        self._zoom_in_btn.pack(side="left")
        Tooltip(self._zoom_in_btn, t("segments_preview_zoom_in"))
        self._reset_view_btn = ctk.CTkButton(
            zoom_controls,
            text=t("segments_preview_reset_view"),
            width=84,
            height=26,
            fg_color=Colors.BG_PANEL,
            hover_color=Colors.BORDER_LIGHT,
            command=self._reset_preview_view,
        )
        Tooltip(
            self._reset_view_btn,
            t("segments_preview_reset_view_hint"),
        )
        self._update_preview_zoom_controls()

        preview_options = ctk.CTkFrame(preview_card, fg_color="transparent")
        preview_options.grid(row=3, column=0, sticky="ew", padx=8, pady=(2, 8))
        restore_control = ctk.CTkFrame(preview_options, fg_color="transparent")
        restore_control.pack(side="right")
        self._restore_toggle = CompactSwitch(
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
        projection_names = {
            "raw": t("segments_vr_projection_raw"),
            "fisheye": t("segments_vr_projection_fisheye"),
            "gnomonic": t("segments_vr_projection_gnomonic"),
        }
        projection_control = ctk.CTkFrame(preview_options, fg_color="transparent")
        projection_control.pack(side="right", padx=(0, 16))
        projection_label_text = t("segments_vr_projection")
        if self.vr_resolution.is_sbs:
            projection_label_text = t(
                "segments_vr_projection_resolved",
                projection=projection_names[self.vr_resolution.projection],
            )
        self._vr_projection_label = ctk.CTkLabel(
            projection_control,
            text=projection_label_text,
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.STATUS_PENDING,
        )
        self._vr_projection_label.pack(side="left", padx=(0, 6))
        self._vr_projection_menu = ValueOptionMenu(
            projection_control,
            options={
                "auto": t("segments_vr_projection_auto"),
                **projection_names,
            },
            command=self._on_vr_projection_changed,
            fg_color=Colors.BG_PANEL,
            button_color=Colors.BG_PANEL,
            button_hover_color=Colors.BORDER_LIGHT,
            dropdown_fg_color=Colors.BG_CARD,
            dropdown_hover_color=Colors.PRIMARY,
            text_color=Colors.TEXT_PRIMARY,
            width=150,
            state="normal" if self.vr_resolution.is_sbs else "disabled",
        )
        self._vr_projection_menu.pack(side="left")
        self._vr_projection_menu.set_value(self._vr_projection)
        Tooltip(self._vr_projection_label, t("segments_vr_projection_hint"))
        Tooltip(self._vr_projection_menu, t("segments_vr_projection_hint"))

    def _build_range_panel(self, body: ctk.CTkFrame) -> None:
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

    def _build_scan_and_timeline(self, metadata: VideoMetadata) -> None:
        self._timeline = SegmentTimeline(
            self,
            duration=self._state.duration,
            fps=self._state.fps,
            on_seek=self._seek,
            on_create=self._timeline_create,
            on_select=self._select_range,
            on_adjust=self._timeline_adjust,
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
            left_eye_only=self._preview_left_eye,
            base_settings=self._get_settings,
            current_seconds=lambda: self._current,
            is_gpu_busy=self._is_gpu_busy,
            claim_gpu=self._claim_gpu_for_scan,
            release_gpu=lambda: self._set_preview_gpu_busy(False),
            on_lock_changed=self._on_scan_lock_changed,
            on_ranges_added=self._on_scan_ranges_added,
            on_preview_changed=self._refresh_preview_image,
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
            (Colors.PRIMARY, t("segments_legend_selected")),
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
            text=t("btn_cancel"),
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

    def _poll_workers(self) -> None:
        if self._closed.is_set():
            return
        try:
            while True:
                event = self._preview_worker.events.get_nowait()
                if isinstance(event, PreviewLoaded):
                    if self._state is None:
                        from jasna.vr180 import resolve_vr_mode

                        self.vr_resolution = resolve_vr_mode(
                            self._get_settings().vr_mode,
                            event.metadata,
                            self._job.path,
                        )
                        self._preview_left_eye = self.vr_resolution.is_sbs
                        self._build_editor(event.metadata)
                elif isinstance(event, PreviewKeyframes):
                    self._keyframe_index = event.index
                    self._analysis_error = event.error
                    self._refresh_workload()
                elif isinstance(event, PreviewFrame):
                    self._show_frame(event)
                elif isinstance(event, PreviewFullFrame):
                    self._open_mask_suggest(event.image)
                elif isinstance(event, PreviewEnded):
                    if event.generation == self._preview_generation:
                        self._set_playing(False)
                elif isinstance(event, PreviewFailed):
                    self._show_preview_error(event.message)
        except queue.Empty:
            pass

        if self._restoration_worker is not None:
            try:
                while True:
                    self._handle_restoration_event(self._restoration_worker.events.get_nowait())
            except queue.Empty:
                pass
        if self._scan_panel is not None:
            self._scan_panel.poll()
        try:
            while True:
                self._handle_feedback_event(self._mask_feedback_worker.events.get_nowait())
        except queue.Empty:
            pass
        if self._state is not None:
            self._refresh_restore_toggle()
        self.after(25, self._poll_workers)

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
            self.after_cancel(self._resize_after)
        self._resize_after = self.after(60, self._refresh_preview_image)

    def _active_preview_source(self) -> Image.Image | None:
        return self._restored_source if self._restore_active else self._preview_source

    def _preview_image_geometry(
        self,
        source: Image.Image,
    ) -> tuple[float, float, float, float]:
        widget_width = max(2, self._preview.winfo_width())
        widget_height = max(2, self._preview.winfo_height())
        available_width = max(2, widget_width - 16)
        available_height = max(2, widget_height - 16)
        scale = min(
            available_width / source.width,
            available_height / source.height,
        )
        display_width = max(1.0, source.width * scale)
        display_height = max(1.0, source.height * scale)
        return (
            (widget_width - display_width) / 2,
            (widget_height - display_height) / 2,
            display_width,
            display_height,
        )

    def _preview_crop(self, source: Image.Image) -> Image.Image:
        box = self._preview_view.crop_box(source.width, source.height)
        return source if box is None else source.crop(box)

    def _update_preview_zoom_controls(self) -> None:
        view = self._preview_view
        self._zoom_label.configure(text=f"{round(view.zoom * 100)}%")
        self._zoom_out_btn.configure(state="normal" if view.zoomed else "disabled")
        self._zoom_in_btn.configure(state="disabled" if view.zoom >= PREVIEW_ZOOM_MAX else "normal")
        reset_needed = not view.is_reset
        if reset_needed and not self._reset_view_btn.winfo_manager():
            self._reset_view_btn.pack(side="left", padx=(6, 0))
        elif not reset_needed and self._reset_view_btn.winfo_manager():
            self._reset_view_btn.pack_forget()
        self._preview.configure(cursor="fleur" if view.zoomed else "")

    def _set_preview_zoom(
        self,
        zoom: float,
        *,
        anchor: tuple[float, float] | None = None,
    ) -> None:
        anchor_fraction = None
        source = self._active_preview_source() if anchor is not None else None
        if source is not None:
            left, top, width, height = self._preview_image_geometry(source)
            anchor_fraction = (
                min(1.0, max(0.0, (anchor[0] - left) / width)),
                min(1.0, max(0.0, (anchor[1] - top) / height)),
            )
        changed = self._preview_view.zoom_to(zoom, anchor_fraction)
        self._update_preview_zoom_controls()
        if changed:
            self._refresh_preview_image()

    def _adjust_preview_zoom(self, amount: float) -> None:
        self._set_preview_zoom(self._preview_view.zoom + float(amount))

    def _reset_preview_view(self, event=None):
        self._preview_view.reset()
        self._preview_pan_anchor = None
        self._update_preview_zoom_controls()
        self._refresh_preview_image()
        return "break" if event is not None else None

    def _preview_mousewheel(self, event):
        button = int(getattr(event, "num", 0))
        delta = int(getattr(event, "delta", 0))
        direction = 1 if button == 4 or delta > 0 else -1 if button == 5 or delta < 0 else 0
        if not direction:
            return None
        self._set_preview_zoom(
            self._preview_view.zoom + direction * PREVIEW_ZOOM_STEP,
            anchor=(float(event.x), float(event.y)),
        )
        return "break"

    def _preview_pan_start(self, event):
        if not self._preview_view.zoomed:
            self._preview_pan_anchor = None
            return None
        self._preview_pan_anchor = (int(event.x), int(event.y))
        return "break"

    def _preview_pan_drag(self, event):
        anchor = self._preview_pan_anchor
        source = self._active_preview_source()
        if anchor is None or source is None or not self._preview_view.zoomed:
            return None
        _, _, display_width, display_height = self._preview_image_geometry(source)
        self._preview_view.pan(
            (int(event.x) - anchor[0]) / display_width,
            (int(event.y) - anchor[1]) / display_height,
        )
        self._preview_pan_anchor = (int(event.x), int(event.y))
        self._refresh_preview_image()
        return "break"

    def _preview_pan_end(self, _event=None):
        was_panning = self._preview_pan_anchor is not None
        self._preview_pan_anchor = None
        return "break" if was_panning else None

    def _fit_to_label(self, label: ctk.CTkLabel, source: Image.Image) -> ctk.CTkImage:
        # winfo_* measures physical pixels while CTkImage's size is multiplied
        # by the widget scaling factor at render time; divide it back out so
        # HiDPI displays do not overflow and clip the preview (issue #229).
        widget_scaling = scaling.widget_scaling(label)
        width = max(2, label.winfo_width() - 16)
        height = max(2, label.winfo_height() - 16)
        source_width, source_height = source.size
        scale = min(width / source_width, height / source_height)
        pixel_size = (
            max(2, round(source_width * scale)),
            max(2, round(source_height * scale)),
        )
        image = source.resize(pixel_size, Image.Resampling.LANCZOS)
        return ctk.CTkImage(
            image,
            size=(
                max(1, round(pixel_size[0] / widget_scaling)),
                max(1, round(pixel_size[1] / widget_scaling)),
            ),
        )

    def _refresh_preview_image(self) -> None:
        self._resize_after = None
        source = self._active_preview_source()
        if source is None or self._closed.is_set():
            return
        if not self._restore_active:
            source = self._scan_panel.apply_overlay(source)
        source = self._preview_crop(source)
        self._preview_image = self._fit_to_label(self._preview, source)
        self._preview.configure(image=self._preview_image, text="")

    def _show_preview_message(self, text: str, color: str) -> None:
        # CTkLabel.configure(image=None) updates its Python-side image reference
        # but does not clear the underlying tkinter.Label image. Clear that
        # first so releasing our CTkImage cannot leave Tk pointing at a deleted
        # ``pyimage`` and abort the callback before background work starts.
        self._preview._label.configure(image="")
        self._preview.configure(image=None, text=text, text_color=color)
        self._preview_image = None

    def _show_preview_error(self, message: str) -> None:
        self._set_playing(False)
        if self._state is None:
            self._loading_bar.stop()
            self._loading_label.configure(text=message, text_color=Colors.STATUS_ERROR)
            return
        self._show_preview_message(message, Colors.STATUS_ERROR)

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
        self._prepare_restoration_after_seek()

    def _step(self, frames: int) -> None:
        state = self._require_state()
        if int(frames) == -1:
            current = self._current
            self._set_playing(False)
            self._current = state.snap(current - 1 / state.fps)
            self._time_label.configure(text=self._time_text())
            self._timeline.reveal(self._current)
            self._refresh_timeline()
            self._preview_generation = self._preview_worker.previous_frame(current)
            self._prepare_restoration_after_seek()
            return
        self._seek(self._current + int(frames) / state.fps)

    def _prepare_restoration_after_seek(self) -> None:
        if not self._restore_active:
            return
        self._restored_clip = ()
        self._restored_source = None
        self._restore_play_pending = False
        self._show_preview_message(
            t("segments_restore_restoring"),
            Colors.STATUS_PENDING,
        )
        self._schedule_restoration_preview()

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
        if self._play is not None:
            self._play.configure(text="⏸" if self._playing else "▶")
        if not self._playing and self._next_frame_after is not None:
            self.after_cancel(self._next_frame_after)
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
        self._show_preview_message(
            t("segments_restore_restoring"),
            Colors.STATUS_PENDING,
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
            self.after_cancel(self._restore_after)
            self._restore_after = None
        self._restored_source = None
        self._preview_image = None
        self._restore_toggle.deselect()
        self._refresh_preview_image()

    def _schedule_restoration_preview(self) -> None:
        self._restore_generation = -1
        if self._restore_after is not None:
            self.after_cancel(self._restore_after)
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
            self._current_video_settings(),
            projection=self._vr_projection,
        )
        if self._restored_source is None:
            self._show_preview_message(
                t("segments_restore_restoring"),
                Colors.STATUS_PENDING,
            )

    def _request_restoration_playback(self, start_seconds: float) -> None:
        if not self._restore_active or self._closed.is_set():
            return
        if self._restore_after is not None:
            self.after_cancel(self._restore_after)
            self._restore_after = None
        self._set_playing(False)
        self._restore_play_pending = True
        self._restored_clip = ()
        self._restored_source = None
        self._play.configure(text="…")
        self._show_preview_message(
            t("segments_restore_restoring"),
            Colors.STATUS_PENDING,
        )
        self._restore_generation = self._restoration_worker.request(
            start_seconds,
            self._current_video_settings(),
            projection=self._vr_projection,
            playback=True,
        )

    def _on_vr_projection_changed(self, projection: str) -> None:
        if projection == self._vr_projection:
            return
        self._vr_projection = projection
        if not self._restore_active:
            return
        self._set_playing(False)
        self._restore_play_pending = False
        self._restored_clip = ()
        self._restored_source = None
        self._schedule_restoration_preview()

    def _handle_restoration_event(self, event) -> None:
        if not self._restore_active or event.generation != self._restore_generation:
            return
        if isinstance(event, RestorationStatus):
            if self._restored_source is None:
                self._show_preview_message(
                    worker_status_text(event.message),
                    Colors.STATUS_PENDING,
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
            self._show_preview_message(
                t("segments_restore_failed", message=event.message),
                Colors.STATUS_ERROR,
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

    def _claim_gpu_for_scan(self) -> None:
        if self._restore_active:
            self._deactivate_restoration_preview()
        if self._restoration_worker is not None:
            self._restoration_worker.close()
            self._restoration_worker = None
        self._set_playing(False)
        self._set_preview_gpu_busy(True)

    def _lockable_widgets(self) -> tuple:
        return (
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
            self._restore_toggle,
            self._start_entry,
            self._end_entry,
            self._suggest_btn,
            *self._timeline_zoom_buttons,
            *self._segment_action_widgets,
        )

    def _on_scan_lock_changed(self, locked: bool) -> None:
        state = "disabled" if locked else "normal"
        for widget in self._lockable_widgets():
            widget.configure(state=state)
        self._timeline.set_enabled(not locked)
        if not locked:
            self._refresh_all()

    def _on_scan_ranges_added(self, added: int) -> None:
        self._set_edit_result_notice(0)
        if not added:
            self._edit_notice = t("segments_scan_all_added")
            self._edit_notice_warning = True
        self._refresh_all()

    def _on_scan_detection_changed(self) -> None:
        if self._restore_active:
            self._schedule_restoration_preview()

    def _scanning(self) -> bool:
        return self._scan_panel is not None and self._scan_panel.active

    def _current_video_settings(self) -> AppSettings:
        return self._scan_panel.video_settings()

    def _suggest_mask(self) -> None:
        self._require_state()
        if self._suggest_busy:
            return
        self._suggest_busy = True
        self._set_playing(False)
        self._suggest_btn.configure(state="disabled", text=t("mask_editor_loading_frame"))
        self._preview_worker.grab_full()

    def _open_mask_suggest(self, image: Image.Image) -> None:
        if not self._suggest_busy or self._closed.is_set():
            return
        self._suggest_btn.configure(text=t("segments_suggest_mask"))
        MaskSuggestDialog(
            self,
            image,
            on_submit=lambda polygons: self._submit_mask_feedback(image, polygons),
            on_closed=self._mask_suggest_closed,
        )

    def _mask_suggest_closed(self) -> None:
        self._suggest_busy = False
        if not self._scanning():
            self._suggest_btn.configure(state="normal")
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
        Toast(self, message, type_).place(relx=0.5, rely=0.92, anchor="center")

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
        self._segment_action_widgets = []
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
            label.configure(state="disabled" if self._scanning() else "normal")
            self._segment_action_widgets.append(label)
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
            delete.configure(state="disabled" if self._scanning() else "normal")
            self._segment_action_widgets.append(delete)
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
        self._undo_btn.configure(
            state="normal" if state.can_undo and not self._scanning() else "disabled"
        )
        self._redo_btn.configure(
            state="normal" if state.can_redo and not self._scanning() else "disabled"
        )
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
            self._codec_notice.pack_forget()
            self._compatibility_error = None
            self._workload.configure(
                text=t("segments_workload_full", duration=format_timestamp(state.duration, milliseconds=False)),
                text_color=Colors.TEXT_PRIMARY,
            )
            self._refresh_timeline()
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
            state.output_segments,
            detection_model=self._scan_panel.detection_model,
            detection_score_threshold=self._scan_panel.threshold,
            vr_projection=self._vr_projection,
        ):
            self._edit_notice = t("segments_job_started")
            self._edit_notice_warning = False
            self._refresh_notice()
            return
        self._saved = True
        self._on_saved(state.output_segments)
        self._finish_close()

    def _request_close(self) -> None:
        if self._scanning():
            self._scan_panel.stop()
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
        if self._scan_panel is not None:
            self._scan_panel.close()
        self.grab_release()
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
        if self._state is None or self._scanning():
            return "break"
        if not allow_entry and self._is_text_entry(event.widget):
            return None
        action()
        return "break"

    def _shortcut_step(self, event, direction: int):
        if self._state is None or self._scanning() or self._is_text_entry(event.widget):
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
