"""Mosaic scan card of the segment editor: finds likely ranges and paints detections on the preview."""

from __future__ import annotations

import queue
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk
from PIL import Image

from jasna.gui.components import Tooltip
from jasna.gui.icons import CompactSwitch
from jasna.gui.locales import t
from jasna.gui.models import AppSettings
from jasna.gui.mosaic_scan import (
    SCAN_SCORE_FLOOR,
    MosaicScanResult,
    MosaicScanWorker,
    ScanCompleted,
    ScanFailed,
    ScanMaskFailed,
    ScanMaskReady,
    ScanProgress,
    ScanStatus,
    ScanStorageSpilled,
    segments_from_scores,
)
from jasna.gui.segment_editor_state import SegmentEditorState
from jasna.gui.segment_timeline import SegmentTimeline
from jasna.gui.settings_sections.widgets import ValueOptionMenu
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.media.probe import VideoMetadata
from jasna.segments import SegmentRange, format_timestamp

_MASK_CACHE_SIZE = 256


def worker_status_text(message: str) -> str:
    if message == "loading_models":
        return t("segments_restore_loading_models")
    if message == "restoring":
        return t("segments_restore_restoring")
    return message


def clamp_scan_threshold(value: float) -> float:
    return min(1.0, max(SCAN_SCORE_FLOOR, float(value)))


class ScanPanel(ctk.CTkFrame):
    def __init__(
        self,
        master,
        *,
        video_path: Path,
        metadata: VideoMetadata,
        state: SegmentEditorState,
        timeline: SegmentTimeline,
        detection_model: str,
        threshold: float,
        left_eye_only: bool,
        base_settings: Callable[[], AppSettings],
        current_seconds: Callable[[], float],
        is_gpu_busy: Callable[[], bool],
        claim_gpu: Callable[[], None],
        release_gpu: Callable[[], None],
        on_lock_changed: Callable[[bool], None],
        on_ranges_added: Callable[[int], None],
        on_preview_changed: Callable[[], None],
        on_detection_changed: Callable[[], None],
    ) -> None:
        super().__init__(master, fg_color=Colors.BG_CARD, corner_radius=Sizing.BORDER_RADIUS)
        self._video_path = video_path
        self._metadata = metadata
        self._state = state
        self._timeline = timeline
        self._left_eye_only = left_eye_only
        self._base_settings = base_settings
        self._current_seconds = current_seconds
        self._is_gpu_busy = is_gpu_busy
        self._claim_gpu = claim_gpu
        self._release_gpu = release_gpu
        self._on_lock_changed = on_lock_changed
        self._on_ranges_added = on_ranges_added
        self._on_preview_changed = on_preview_changed
        self._on_detection_changed = on_detection_changed

        self._worker: MosaicScanWorker | None = None
        self._active = False
        self._low_vram = False
        self._was_stopped = False
        self._result: MosaicScanResult | None = None
        self._proposals: tuple[SegmentRange, ...] = ()
        self._overlay = True
        self._threshold = clamp_scan_threshold(threshold)
        self._threshold_after: str | None = None
        self._mask_generation = 0
        self._mask_requested_key: int | None = None
        self._mask_cache: OrderedDict[int, tuple[float, float, object]] = OrderedDict()

        self._build(detection_model)

    @property
    def active(self) -> bool:
        return self._active

    @property
    def detection_model(self) -> str:
        return self._model.get()

    @property
    def threshold(self) -> float:
        return self._threshold

    def video_settings(self) -> AppSettings:
        return replace(
            self._base_settings(),
            detection_model=self.detection_model,
            detection_score_threshold=self._threshold,
        )

    def lockable_widgets(self) -> tuple:
        return (
            self._scan_btn,
            self._info_btn,
            self._model,
            self._interval,
            self._threshold_slider,
            self._add_btn,
            self._overlay_toggle,
        )

    def _build(self, detection_model: str) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(7, 0))
        ctk.CTkLabel(
            header,
            text=t("segments_scan_title"),
            font=(Fonts.FAMILY, Fonts.SIZE_HEADING, "bold"),
            text_color=Colors.TEXT_PRIMARY,
            anchor="w",
            height=26,
        ).pack(side="left")
        ctk.CTkLabel(
            header,
            text=t("segments_scan_subtitle"),
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.STATUS_PENDING,
            anchor="w",
            height=26,
        ).pack(side="left", fill="x", expand=True, padx=(8, 8))
        self._info_btn = ctk.CTkButton(
            header,
            text=t("segments_scan_help_button"),
            width=104,
            height=26,
            fg_color="transparent",
            hover_color=Colors.BORDER_LIGHT,
            border_width=1,
            border_color=Colors.BORDER_LIGHT,
            command=self._show_help,
        )
        self._info_btn.pack(side="right")
        Tooltip(self._info_btn, t("segments_scan_help_hint"))

        from jasna.mosaic.detection_registry import detection_model_choices

        available_models = detection_model_choices()
        if detection_model not in available_models:
            available_models.insert(0, detection_model)

        settings = ctk.CTkFrame(self, fg_color="transparent")
        settings.pack(fill="x", padx=12, pady=(3, 7))
        settings.grid_columnconfigure(3, weight=1)

        model_field = ctk.CTkFrame(settings, fg_color="transparent")
        model_field.grid(row=0, column=0, sticky="w", padx=(0, 16))
        model_label = ctk.CTkLabel(
            model_field,
            text=t("segments_scan_model"),
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.STATUS_PENDING,
            height=16,
        )
        model_label.pack(anchor="w", pady=(0, 1))
        self._model = ctk.CTkOptionMenu(
            model_field,
            values=available_models,
            width=170,
            height=26,
            command=self._on_model_changed,
        )
        self._model.set(detection_model)
        self._model.pack(anchor="w")
        Tooltip(model_label, t("segments_scan_model_hint"))
        Tooltip(self._model, t("segments_scan_model_hint"))

        frequency_field = ctk.CTkFrame(settings, fg_color="transparent")
        frequency_field.grid(row=0, column=1, sticky="w", padx=(0, 16))
        frequency_label = ctk.CTkLabel(
            frequency_field,
            text=t("segments_scan_interval"),
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.STATUS_PENDING,
            height=16,
        )
        frequency_label.pack(anchor="w", pady=(0, 1))
        self._interval = ValueOptionMenu(
            frequency_field,
            options={
                "0": t("segments_scan_frequency_every_frame"),
                "0.25": t("segments_scan_frequency_quarter"),
                "0.5": t("segments_scan_frequency_half"),
                "1": t("segments_scan_frequency_one"),
                "2": t("segments_scan_frequency_two"),
            },
            width=210,
            height=26,
        )
        self._interval.set_value("1")
        self._interval.pack(anchor="w")
        Tooltip(frequency_label, t("segments_scan_interval_hint"))
        Tooltip(self._interval, t("segments_scan_interval_hint"))

        confidence_field = ctk.CTkFrame(settings, fg_color="transparent")
        confidence_field.grid(row=0, column=2, sticky="w")
        confidence_label = ctk.CTkLabel(
            confidence_field,
            text=t("segments_scan_threshold"),
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.STATUS_PENDING,
            height=16,
        )
        confidence_label.pack(anchor="w", pady=(0, 1))
        confidence_control = ctk.CTkFrame(confidence_field, fg_color="transparent")
        confidence_control.pack(anchor="w")
        self._threshold_slider = ctk.CTkSlider(
            confidence_control,
            from_=SCAN_SCORE_FLOOR,
            to=1.0,
            width=150,
            command=self._on_threshold,
        )
        self._threshold_slider.set(self._threshold)
        self._threshold_slider.pack(side="left")
        self._threshold_label = ctk.CTkLabel(
            confidence_control,
            text=f"{self._threshold:.2f}",
            font=(Fonts.FAMILY_MONO, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_PRIMARY,
            width=42,
            height=26,
        )
        self._threshold_label.pack(side="left", padx=(6, 0))
        Tooltip(confidence_label, t("segments_scan_threshold_hint"))
        Tooltip(self._threshold_slider, t("segments_scan_threshold_hint"))

        self._scan_btn = ctk.CTkButton(
            settings,
            text=t("segments_scan"),
            height=28,
            width=130,
            command=self._start,
        )
        self._scan_btn.grid(row=0, column=4, sticky="se", padx=(16, 0))

        self._activity = ctk.CTkFrame(
            self,
            fg_color=Colors.BG_PANEL,
            corner_radius=Sizing.BORDER_RADIUS,
        )
        activity_row = ctk.CTkFrame(self._activity, fg_color="transparent")
        activity_row.pack(fill="x", padx=10, pady=8)
        self._status_dot = ctk.CTkLabel(
            activity_row,
            text="●",
            width=14,
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.STATUS_PENDING,
        )
        self._status_dot.pack(side="left", padx=(0, 5))
        self._status = ctk.CTkLabel(
            activity_row,
            text="",
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.STATUS_PENDING,
            anchor="w",
        )
        self._status.pack(side="left", fill="x", expand=True)

        self._stop_btn = ctk.CTkButton(
            activity_row,
            text=t("segments_scan_stop"),
            height=28,
            width=90,
            fg_color=Colors.STATUS_ERROR,
            hover_color="#e11d48",
            state="disabled",
            command=self.stop,
        )

        self._add_btn = ctk.CTkButton(
            activity_row,
            text=t("segments_scan_add", count=0),
            height=28,
            width=150,
            state="disabled",
            command=self._add_detected_ranges,
        )
        Tooltip(self._add_btn, t("segments_scan_add_hint"))

        self._overlay_box = ctk.CTkFrame(activity_row, fg_color="transparent")
        self._overlay_toggle = CompactSwitch(
            self._overlay_box,
            self._toggle_overlay,
            Colors.BG_PANEL,
        )
        self._overlay_toggle.pack(side="right")
        self._overlay_toggle.select()
        ctk.CTkLabel(
            self._overlay_box,
            text=t("segments_scan_overlay"),
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_PRIMARY,
        ).pack(side="right", padx=(0, 6))
        Tooltip(self._overlay_box, t("segments_scan_overlay_hint"))

        self._progress = ctk.CTkProgressBar(self._activity, height=7)
        self._progress.set(0.0)

    def poll(self) -> None:
        worker = self._worker
        if worker is None:
            return
        try:
            while True:
                self._handle_event(worker.events.get_nowait())
        except queue.Empty:
            pass

    def stop(self) -> None:
        if self._worker is None or not self._active:
            return
        self._worker.stop()
        self._stop_btn.configure(state="disabled")
        self._set_status(t("segments_scan_stopping"), dot_color=Colors.STATUS_WARNING)

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()

    def _set_status(
        self,
        text: str,
        *,
        color: str = Colors.STATUS_PENDING,
        dot_color: str | None = None,
    ) -> None:
        self._status.configure(text=text, text_color=color)
        self._status_dot.configure(text_color=dot_color or color)

    def _set_ui_state(self, state: str) -> None:
        self._stop_btn.pack_forget()
        self._add_btn.pack_forget()
        self._overlay_box.pack_forget()
        self._progress.pack_forget()

        if state == "idle":
            self._activity.pack_forget()
            self._scan_btn.configure(text=t("segments_scan"))
            return

        self._activity.pack(fill="x", padx=12, pady=(0, 10))
        if state == "scanning":
            self._scan_btn.configure(text=t("segments_scan"))
            self._stop_btn.pack(side="right", padx=(10, 0))
            self._progress.pack(fill="x", padx=10, pady=(0, 8))
            return

        self._scan_btn.configure(
            text=t("segments_scan_again") if self._result is not None else t("segments_scan")
        )
        if state == "results":
            if self._proposals:
                self._add_btn.pack(side="right", padx=(10, 0))
            self._overlay_box.pack(side="right", padx=(12, 0))

    def _start(self) -> None:
        if self._active:
            return
        if self._is_gpu_busy():
            self._set_status(t("segments_restore_gpu_busy"), color=Colors.STATUS_ERROR)
            self._set_ui_state("message")
            return
        if self._worker is not None:
            self._worker.close()
            self._worker.join()
            self._worker = None
        self._claim_gpu()
        try:
            worker = MosaicScanWorker(
                self._video_path,
                self._metadata,
                self.video_settings(),
                stride_seconds=float(self._interval.get_value()),
                on_stopped=self._release_gpu,
            )
            worker.start()
        except Exception:
            self._release_gpu()
            raise
        self._worker = worker
        self._result = None
        self._low_vram = False
        self._was_stopped = False
        self._proposals = ()
        self._mask_cache.clear()
        self._mask_requested_key = None
        self._timeline.set_detections(())
        self._progress.set(0.0)
        self._set_status(t("segments_restore_loading_models"), dot_color=Colors.STATUS_PROCESSING)
        self._set_locked(True)

    def _set_locked(self, locked: bool) -> None:
        self._active = locked
        state = "disabled" if locked else "normal"
        for widget in self.lockable_widgets():
            widget.configure(state=state)
        self._stop_btn.configure(state="normal" if locked else "disabled")
        self._on_lock_changed(locked)
        if locked:
            self._set_ui_state("scanning")
        else:
            self.refresh_view()
            if self._result is None:
                self._set_ui_state("idle")

    def _handle_event(self, event) -> None:
        if isinstance(event, ScanStatus):
            self._set_status(worker_status_text(event.message), dot_color=Colors.STATUS_PROCESSING)
        elif isinstance(event, ScanProgress):
            self._progress.set(event.fraction)
            status = t(
                "segments_scan_progress",
                percent=round(event.fraction * 100),
                fps=round(event.fps),
                eta=format_timestamp(event.eta_seconds, milliseconds=False),
            )
            if self._low_vram:
                status = f"{status} · {t('segments_scan_low_vram_short')}"
            self._set_status(status, dot_color=Colors.STATUS_PROCESSING)
        elif isinstance(event, ScanStorageSpilled):
            self._low_vram = True
            self._set_status(t("segments_scan_low_vram"), dot_color=Colors.STATUS_WARNING)
        elif isinstance(event, ScanFailed):
            if self._worker is not None:
                self._worker.close()
            self._worker = None
            self._set_locked(False)
            self._set_status(t("segments_scan_failed", message=event.message), color=Colors.STATUS_ERROR)
            self._set_ui_state("message")
        elif isinstance(event, ScanCompleted):
            self._result = event.result
            self._was_stopped = event.stopped
            self._set_locked(False)
        elif isinstance(event, ScanMaskReady):
            self._cache_mask(event.seconds, event.score, event.mask)
            if event.generation == self._mask_generation:
                self._mask_requested_key = None
                self._on_preview_changed()
        elif isinstance(event, ScanMaskFailed):
            if event.generation == self._mask_generation:
                self._mask_requested_key = None
                self._set_status(
                    t("segments_scan_mask_failed", message=event.message),
                    color=Colors.STATUS_ERROR,
                )

    def refresh_view(self) -> None:
        result = self._result
        if result is None:
            return
        state = self._state
        runs = segments_from_scores(
            result.times,
            result.scores,
            threshold=self._threshold,
            stride=result.stride,
            duration=state.duration,
            pad=0.0,
        )
        self._timeline.set_detections(runs)
        proposals = segments_from_scores(
            result.times,
            result.scores,
            threshold=self._threshold,
            stride=result.stride,
            duration=state.duration,
        )
        self._proposals = tuple(
            proposal
            for proposal in proposals
            if not any(
                selected.start <= proposal.start and selected.end >= proposal.end
                for selected in state.segments
            )
        )
        count = len(self._proposals)
        self._add_btn.configure(
            text=t("segments_scan_add", count=count),
            state="normal" if count and not self._active else "disabled",
        )
        total_seconds = sum(proposal.duration for proposal in proposals)
        if not proposals:
            self._set_status(t("segments_scan_none"), dot_color=Colors.STATUS_PENDING)
        elif not count:
            self._set_status(t("segments_scan_all_added"), dot_color=Colors.STATUS_COMPLETED)
        else:
            summary_key = "segments_scan_result_partial" if self._was_stopped else "segments_scan_result"
            self._set_status(
                t(
                    summary_key,
                    count=len(proposals),
                    duration=format_timestamp(total_seconds, milliseconds=False),
                ),
                dot_color=Colors.STATUS_WARNING,
            )
        self._set_ui_state("results")
        if self._overlay:
            self._on_preview_changed()

    def _on_threshold(self, value: float) -> None:
        self._threshold = float(value)
        self._threshold_label.configure(text=f"{self._threshold:.2f}")
        if self._threshold_after is not None:
            self.after_cancel(self._threshold_after)
        self._threshold_after = self.after(60, self._apply_threshold)

    def _apply_threshold(self) -> None:
        self._threshold_after = None
        self.refresh_view()
        self._on_detection_changed()

    def _add_detected_ranges(self) -> None:
        if not self._proposals:
            return
        self._on_ranges_added(self._state.add_many(self._proposals))
        self.refresh_view()

    def _toggle_overlay(self) -> None:
        self._overlay = bool(self._overlay_toggle.get())
        self._on_preview_changed()

    def apply_overlay(self, image: Image.Image) -> Image.Image:
        result = self._result
        if not self._overlay or result is None:
            return image
        seconds = self._current_seconds()
        sample = result.sample_at(seconds, tolerance=0.51 / self._state.fps)
        if sample is None:
            sample = self._cached_mask(seconds)
        if sample is None:
            self._request_mask(seconds)
            return image
        _, score, mask = sample
        if score < self._threshold:
            return image
        mask_np = mask.numpy()
        if self._left_eye_only:
            mask_np = mask_np[:, : mask_np.shape[1] // 2]
        if not mask_np.any():
            return image
        alpha = Image.fromarray((mask_np * 130).astype("uint8"), "L").resize(
            image.size, Image.Resampling.NEAREST
        )
        overlay = Image.new("RGB", image.size, "#ef4444")
        composed = image.copy()
        composed.paste(overlay, (0, 0), alpha)
        return composed

    def _request_mask(self, seconds: float) -> None:
        worker = self._worker
        if worker is None or self._active:
            return
        key = self._mask_key(seconds)
        if self._mask_requested_key == key:
            return
        self._mask_requested_key = key
        self._mask_generation = worker.request_mask(seconds)

    def _cache_mask(self, seconds: float, score: float, mask) -> None:
        key = self._mask_key(seconds)
        self._mask_cache[key] = (float(seconds), float(score), mask)
        self._mask_cache.move_to_end(key)
        while len(self._mask_cache) > _MASK_CACHE_SIZE:
            self._mask_cache.popitem(last=False)

    def _cached_mask(self, seconds: float):
        key = self._mask_key(seconds)
        sample = self._mask_cache.get(key)
        if sample is not None:
            self._mask_cache.move_to_end(key)
        return sample

    def _mask_key(self, seconds: float) -> int:
        return round(float(seconds) * self._state.fps)

    def _on_model_changed(self, model: str) -> None:
        from jasna.mosaic.detection_registry import recommended_score_threshold

        self._threshold = clamp_scan_threshold(recommended_score_threshold(str(model)))
        self._threshold_slider.set(self._threshold)
        self._threshold_label.configure(text=f"{self._threshold:.2f}")
        if self._worker is not None:
            self._worker.close()
            self._worker = None
        self._result = None
        self._proposals = ()
        self._mask_cache.clear()
        self._mask_requested_key = None
        self._timeline.set_detections(())
        self._add_btn.configure(text=t("segments_scan_add", count=0), state="disabled")
        self._set_status(t("segments_scan_model_changed"), dot_color=Colors.STATUS_PENDING)
        self._set_ui_state("message")
        self._on_preview_changed()
        self._on_detection_changed()

    def _show_help(self) -> None:
        messagebox.showinfo(
            t("segments_scan_help_title"),
            t("segments_scan_help_body"),
            parent=self.winfo_toplevel(),
        )
