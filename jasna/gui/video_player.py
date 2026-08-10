from __future__ import annotations

import math
import queue
import threading
import time
import tkinter as tk
from dataclasses import replace
from pathlib import Path

import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox

from jasna.gui import scaling
from jasna.gui.locales import t
from jasna.gui.models import AppSettings
from jasna.gui.raw_player import (
    PREROLL_SECONDS,
    PlaybackClock,
    PlayerEnded,
    PlayerFailed,
    PlayerStatus,
    RawPlayerWorker,
    SoftwareClock,
    TimestampFrameBuffer,
    VlcAudioClock,
    VlcUnavailableError,
    source_has_audio,
)
from jasna.gui.settings_sections.widgets import ValueOptionMenu
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.media import VideoMetadata, get_video_meta_data
from jasna.media.media_files import VIDEO_EXTENSIONS
from jasna.mosaic.detection_registry import (
    detection_model_choices,
    recommended_score_threshold,
)

_TICK_SECONDS = 1 / 30
_TICK_MS = round(_TICK_SECONDS * 1000)
_FRAME_TOLERANCE = 1 / 60
_PLAYER_MIN_SIZE = (820, 620)
_PLAYER_ASPECT = (16, 9)
_PLAYER_CHROME_SIZE = (32, 270)
_SEEK_STEP_SECONDS = 30.0
_FULLSCREEN_EDGE_PX = 8
_BUFFER_STATUS_INTERVAL_SECONDS = 0.25


def next_player_tick(previous_deadline: float, now: float) -> tuple[float, int]:
    deadline = previous_deadline + _TICK_SECONDS
    if deadline <= now:
        missed_ticks = math.floor((now - deadline) / _TICK_SECONDS) + 1
        deadline += missed_ticks * _TICK_SECONDS
    delay_ms = max(1, math.ceil((deadline - now) * 1000))
    return deadline, delay_ms


def fit_player_view_size(
    bounds: tuple[int, int],
    aspect: tuple[float, float] = _PLAYER_ASPECT,
) -> tuple[int, int]:
    width, height = max(1, int(bounds[0])), max(1, int(bounds[1]))
    aspect_width = max(1.0, float(aspect[0]))
    aspect_height = max(1.0, float(aspect[1]))
    view_width = min(width, round(height * aspect_width / aspect_height))
    view_height = min(
        height,
        round(view_width * aspect_height / aspect_width),
    )
    return max(1, view_width), max(1, view_height)


def video_display_aspect(metadata: VideoMetadata | None) -> tuple[float, float]:
    if metadata is None:
        return _PLAYER_ASPECT
    sample_aspect = float(metadata.sample_aspect_ratio)
    if sample_aspect <= 0:
        sample_aspect = 1.0
    return (
        max(1.0, float(metadata.video_width) * sample_aspect),
        max(1.0, float(metadata.video_height)),
    )


def player_dialog_size(
    *,
    screen_size: tuple[int, int],
    chrome_size: tuple[int, int],
    screen_margin: tuple[int, int],
    minimum_size: tuple[int, int],
) -> tuple[int, int]:
    max_width = max(1, screen_size[0] - screen_margin[0])
    max_height = max(1, screen_size[1] - screen_margin[1])
    view_bounds = (
        max(1, max_width - chrome_size[0]),
        max(1, max_height - chrome_size[1]),
    )
    view_width, view_height = fit_player_view_size(view_bounds)
    return (
        min(max_width, max(minimum_size[0], view_width + chrome_size[0])),
        min(max_height, max(minimum_size[1], view_height + chrome_size[1])),
    )


def available_player_secondary_restorations() -> tuple[str, ...]:
    from jasna.accelerator import is_nvidia_device
    from jasna.engine_paths import UNET4X_ONNX_ENC_PATH, UNET4X_ONNX_PATH

    values = ["none"]
    if is_nvidia_device():
        values.append("rtx-super-res")
        if UNET4X_ONNX_PATH.exists() or UNET4X_ONNX_ENC_PATH.exists():
            values.append("unet-4x")
    return tuple(values)


def format_player_time(seconds: float) -> str:
    total = max(0, round(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class VideoPlayerDialog(ctk.CTkToplevel):
    def __init__(
        self,
        master,
        settings: AppSettings,
        *,
        initial_path: Path | None = None,
        on_closed,
    ) -> None:
        super().__init__(master)
        self.withdraw()
        self._base_settings = settings
        self._on_closed = on_closed
        self._path: Path | None = None
        self._metadata: VideoMetadata | None = None
        self._probe_generation = 0
        self._worker: RawPlayerWorker | None = None
        self._frame_buffer: TimestampFrameBuffer | None = None
        self._clock: PlaybackClock | None = None
        self._has_audio = False
        self._generation = 0
        self._aligned_generation = -1
        self._desired_playing = False
        self._playing = False
        self._buffering = False
        self._eof = False
        self._current_seconds = 0.0
        self._seek_dragging = False
        self._seek_target = 0.0
        self._photo = None
        self._photo_size: tuple[int, int] | None = None
        self._fullscreen = False
        self._fullscreen_controls_visible = False
        self._windowed_geometry: str | None = None
        self._closed = False
        self._stopping = False
        self._choose_after_stop = False
        self._close_after_stop = False
        self._failed_message = ""
        self._last_frame_image = None
        self._view_size = (2, 2)
        self._last_status: tuple[str, str] | None = None
        self._last_time_text: str | None = None
        self._next_buffer_status_at = 0.0

        self.title(t("player_title"))
        self.configure(fg_color=Colors.BG_MAIN)
        self.transient(master)
        self.protocol("WM_DELETE_WINDOW", self.request_close)
        self._build_ui()
        self.update_idletasks()
        self._show_centered(master)
        self.grab_set()
        self.lift()
        self.focus_force()
        self.bind("<F11>", self._toggle_fullscreen)
        self.bind("<Escape>", self._exit_fullscreen)
        self.bind("<Left>", lambda event: self._seek_relative(-_SEEK_STEP_SECONDS, event))
        self.bind("<Right>", lambda event: self._seek_relative(_SEEK_STEP_SECONDS, event))
        self.bind("<space>", self._space_pressed)
        self.bind("<Motion>", self._fullscreen_mouse_moved, add="+")
        self._next_tick_at = time.monotonic() + _TICK_SECONDS
        self.after(_TICK_MS, self._tick)
        if initial_path is not None:
            self._load_path(initial_path)

    def _size_and_center(self, master) -> None:
        rect = scaling.screen_rect(master)
        screen_size = rect[2:]
        size = player_dialog_size(
            screen_size=screen_size,
            chrome_size=scaling.to_physical(self, *_PLAYER_CHROME_SIZE),
            screen_margin=scaling.to_physical(self, *scaling.SCREEN_MARGIN),
            minimum_size=scaling.to_physical(self, *_PLAYER_MIN_SIZE),
        )
        x = rect[0] + (screen_size[0] - size[0]) // 2
        y = rect[1] + (screen_size[1] - size[1]) // 2
        scaling.apply_geometry(self, *size, x, y)
        logical_size = scaling.to_logical(self, *size)
        scaling.apply_minsize(
            self,
            min(_PLAYER_MIN_SIZE[0], logical_size[0]),
            min(_PLAYER_MIN_SIZE[1], logical_size[1]),
        )

    def _show_centered(self, master) -> None:
        self._size_and_center(master)
        self.deiconify()
        self.wait_visibility()
        self._size_and_center(master)
        self.update_idletasks()

    def _build_ui(self) -> None:
        self._outer = ctk.CTkFrame(self, fg_color="transparent")
        self._outer.pack(fill="both", expand=True, padx=16, pady=16)
        self._outer.grid_columnconfigure(0, weight=1)
        self._outer.grid_rowconfigure(1, weight=1)

        self._file_row = ctk.CTkFrame(self._outer, fg_color="transparent")
        self._file_row.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self._file_label = ctk.CTkLabel(
            self._file_row,
            text=t("player_no_video"),
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            text_color=Colors.STATUS_PENDING,
            anchor="w",
        )
        self._file_label.pack(side="left", fill="x", expand=True)

        self._video_area = ctk.CTkFrame(
            self._outer,
            fg_color="#000000",
            corner_radius=0,
            height=1,
        )
        self._video_area.grid(row=1, column=0, sticky="nsew")
        self._video_surface = tk.Label(
            self._video_area,
            text=t("player_choose_prompt"),
            foreground=Colors.STATUS_PENDING,
            background="#000000",
            font=(Fonts.FAMILY, scaling.raw_tk_font_size(self, Fonts.SIZE_NORMAL)),
            borderwidth=0,
            highlightthickness=0,
        )
        self._video_surface.place(relx=0.5, rely=0.5, anchor="center")
        self._video_area.bind("<Configure>", self._video_area_resized)

        self._bottom_panel = ctk.CTkFrame(self._outer, fg_color="transparent")
        self._bottom_panel.grid(row=2, column=0, sticky="ew")

        transport = ctk.CTkFrame(self._bottom_panel, fg_color="transparent")
        transport.pack(fill="x", padx=10, pady=(10, 4))
        self._play_btn = ctk.CTkButton(
            transport,
            text="▶",
            width=48,
            state="disabled",
            command=self._toggle_play,
        )
        self._play_btn.pack(side="left")
        self._seek = ctk.CTkSlider(
            transport,
            from_=0,
            to=1,
            number_of_steps=1,
            state="disabled",
            command=self._seek_moved,
        )
        self._seek.pack(side="left", fill="x", expand=True, padx=10)
        self._seek.bind("<ButtonRelease-1>", self._seek_released)
        self._time_label = ctk.CTkLabel(
            transport,
            text="0:00 / 0:00",
            width=125,
            font=(Fonts.FAMILY_MONO, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_PRIMARY,
        )
        self._time_label.pack(side="left")
        self._fullscreen_btn = ctk.CTkButton(
            transport,
            text=t("player_fullscreen"),
            width=110,
            fg_color=Colors.BG_CARD,
            hover_color=Colors.BORDER_LIGHT,
            command=self._toggle_fullscreen,
        )
        self._fullscreen_btn.pack(side="right", padx=(10, 0))

        audio = ctk.CTkFrame(self._bottom_panel, fg_color="transparent")
        audio.pack(fill="x", padx=10, pady=(0, 8))
        self._mute = ctk.CTkSwitch(
            audio,
            text=t("player_mute"),
            state="disabled",
            command=self._mute_changed,
        )
        self._mute.pack(side="left")
        ctk.CTkLabel(
            audio,
            text=t("player_volume"),
            text_color=Colors.TEXT_PRIMARY,
        ).pack(side="left", padx=(14, 6))
        self._volume = ctk.CTkSlider(
            audio,
            from_=0,
            to=100,
            number_of_steps=100,
            width=150,
            state="disabled",
            command=self._volume_changed,
        )
        self._volume.set(80)
        self._volume.pack(side="left")
        self._status = ctk.CTkLabel(
            audio,
            text=t("player_idle"),
            text_color=Colors.STATUS_PENDING,
            anchor="e",
        )
        self._status.pack(side="right", fill="x", expand=True)

        self._settings_card = ctk.CTkFrame(
            self._outer,
            fg_color=Colors.BG_CARD,
            corner_radius=Sizing.BORDER_RADIUS,
        )
        self._settings_card.grid(row=3, column=0, sticky="ew", pady=(2, 10))
        self._settings_card.grid_columnconfigure(1, weight=1)
        self._settings_card.grid_columnconfigure(3, weight=1)
        ctk.CTkLabel(
            self._settings_card,
            text=t("player_detection_model"),
            text_color=Colors.TEXT_PRIMARY,
        ).grid(row=0, column=0, padx=(12, 6), pady=10, sticky="w")
        models = detection_model_choices()
        self._model = ctk.CTkOptionMenu(
            self._settings_card,
            values=models,
            command=self._model_changed,
            fg_color=Colors.BG_PANEL,
            button_color=Colors.BG_PANEL,
            button_hover_color=Colors.BORDER_LIGHT,
            dropdown_fg_color=Colors.BG_PANEL,
            dropdown_hover_color=Colors.PRIMARY,
            text_color=Colors.TEXT_PRIMARY,
            width=160,
        )
        selected_model = (
            self._base_settings.detection_model
            if self._base_settings.detection_model in models
            else models[0]
        )
        self._model.set(selected_model)
        self._model.grid(row=0, column=1, padx=(0, 14), pady=10, sticky="w")

        ctk.CTkLabel(
            self._settings_card,
            text=t("player_secondary"),
            text_color=Colors.TEXT_PRIMARY,
        ).grid(row=0, column=2, padx=(0, 6), pady=10, sticky="w")
        secondary_values = available_player_secondary_restorations()
        self._secondary = ValueOptionMenu(
            self._settings_card,
            options={
                value: {
                    "none": t("secondary_none"),
                    "rtx-super-res": t("secondary_rtx_super_res"),
                    "unet-4x": t("secondary_unet_4x"),
                }[value]
                for value in secondary_values
            },
            command=self._secondary_changed,
            fg_color=Colors.BG_PANEL,
            button_color=Colors.BG_PANEL,
            button_hover_color=Colors.BORDER_LIGHT,
            dropdown_fg_color=Colors.BG_PANEL,
            dropdown_hover_color=Colors.PRIMARY,
            text_color=Colors.TEXT_PRIMARY,
            width=180,
        )
        initial_secondary = (
            self._base_settings.secondary_restoration
            if self._base_settings.secondary_restoration in secondary_values
            else "none"
        )
        self._secondary.set_value(initial_secondary)
        self._secondary.grid(row=0, column=3, padx=(0, 14), pady=10, sticky="w")

        ctk.CTkLabel(
            self._settings_card,
            text=t("player_detection_threshold"),
            text_color=Colors.TEXT_PRIMARY,
        ).grid(row=1, column=0, padx=(12, 6), pady=(0, 10), sticky="w")
        threshold_row = ctk.CTkFrame(self._settings_card, fg_color="transparent")
        threshold_row.grid(
            row=1,
            column=1,
            columnspan=3,
            padx=(0, 14),
            pady=(0, 10),
            sticky="ew",
        )
        self._threshold = ctk.CTkSlider(
            threshold_row,
            from_=0,
            to=1,
            number_of_steps=20,
            command=self._threshold_changed,
        )
        self._threshold.pack(side="left", fill="x", expand=True)
        self._threshold.set(self._base_settings.detection_score_threshold)
        self._threshold.bind("<ButtonRelease-1>", self._threshold_released)
        self._threshold_value = ctk.CTkLabel(
            threshold_row,
            text=f"{self._base_settings.detection_score_threshold:.2f}",
            width=42,
            text_color=Colors.TEXT_PRIMARY,
        )
        self._threshold_value.pack(side="left", padx=(8, 0))

        self._actions = ctk.CTkFrame(self._outer, fg_color="transparent")
        self._actions.grid(row=4, column=0, sticky="ew")
        self._choose_btn = ctk.CTkButton(
            self._actions,
            text=t("player_choose_video"),
            command=self._choose_video_action,
        )
        self._choose_btn.pack(side="left")
        ctk.CTkButton(
            self._actions,
            text=t("btn_close"),
            width=90,
            fg_color=Colors.BG_CARD,
            hover_color=Colors.BORDER_LIGHT,
            command=self.request_close,
        ).pack(side="right")

    def _choose_video_action(self) -> None:
        if self._worker is None:
            self._choose_video()
            return
        self._choose_after_stop = True
        self._begin_stop()

    def _choose_video(self) -> None:
        patterns = " ".join(f"*{extension}" for extension in sorted(VIDEO_EXTENSIONS))
        selected = filedialog.askopenfilename(
            parent=self,
            title=t("player_choose_video"),
            filetypes=[
                (t("player_video_files"), patterns),
                (t("player_all_files"), "*.*"),
            ],
        )
        if not selected:
            return
        self._load_path(Path(selected))

    def _load_path(self, path: Path) -> None:
        self._path = path
        self._metadata = None
        self._probe_generation += 1
        generation = self._probe_generation
        self._file_label.configure(
            text=self._path.name,
            text_color=Colors.TEXT_PRIMARY,
        )
        self._video_surface.configure(image="", text=t("player_loading_video"))
        self._photo = None
        self._photo_size = None
        self._last_frame_image = None
        self._play_btn.configure(state="disabled", text="▶")
        self._set_status(t("player_loading_video"), Colors.STATUS_PENDING)

        def probe() -> None:
            try:
                metadata = get_video_meta_data(str(path))
                result = (metadata, "")
            except Exception as exc:
                result = (None, str(exc))
            self._ui_after(lambda: self._video_probed(generation, *result))

        threading.Thread(
            target=probe,
            name=f"player-probe-{path.name}",
            daemon=True,
        ).start()

    def _video_probed(
        self,
        generation: int,
        metadata: VideoMetadata | None,
        error: str,
    ) -> None:
        if generation != self._probe_generation or self._closed:
            return
        if metadata is None:
            self._set_status(
                t("player_video_failed", message=error),
                Colors.STATUS_ERROR,
            )
            return
        self._metadata = metadata
        self._seek.configure(
            to=max(0.001, float(metadata.duration)),
            number_of_steps=max(1, int(metadata.duration * 10)),
        )
        self._seek.set(0)
        self._update_time_label(0)
        self._video_surface.configure(
            image="",
            text=t("player_ready_to_play"),
        )
        self._play_btn.configure(state="normal")
        self._fit_video_surface(
            self._video_area.winfo_width(),
            self._video_area.winfo_height(),
        )
        self._set_status(t("player_ready"), Colors.STATUS_COMPLETED)

    def _start_playback(
        self,
        *,
        seconds: float = 0.0,
        autoplay: bool = True,
    ) -> None:
        if self._path is None or self._metadata is None or self._stopping:
            return
        seconds = min(float(self._metadata.duration), max(0.0, float(seconds)))
        settings = self._playback_settings()
        try:
            self._has_audio = source_has_audio(self._path)
            self._clock = VlcAudioClock(self._path) if self._has_audio else SoftwareClock()
            if self._has_audio:
                self._clock.set_volume(round(self._volume.get()))
                self._clock.set_muted(self._mute.get() == 1)
        except (OSError, VlcUnavailableError) as exc:
            self._clock = None
            message = t("player_vlc_missing")
            self._set_status(message, Colors.STATUS_ERROR)
            messagebox.showerror(t("player_title"), f"{message}\n\n{exc}", parent=self)
            return

        self._frame_buffer = TimestampFrameBuffer()
        self._worker = RawPlayerWorker(
            self._path,
            self._metadata,
            settings,
            self._frame_buffer,
            max_size=self._surface_size(),
        )
        self._worker.start()
        self._generation = self._worker.play_from(seconds)
        self._aligned_generation = -1
        self._desired_playing = bool(autoplay)
        self._playing = False
        self._buffering = bool(autoplay)
        self._eof = False
        self._current_seconds = seconds
        self._next_buffer_status_at = 0.0
        self._play_btn.configure(state="normal", text="⏸" if autoplay else "▶")
        self._seek.configure(state="normal")
        self._mute.configure(state="normal" if self._has_audio else "disabled")
        self._volume.configure(state="normal" if self._has_audio else "disabled")
        self._set_status(t("player_loading_models"), Colors.STATUS_PENDING)

    def _playback_settings(self) -> AppSettings:
        return replace(
            self._base_settings,
            detection_model=self._model.get(),
            detection_score_threshold=float(self._threshold.get()),
            secondary_restoration=self._secondary.get_value(),
        )

    def _tick(self) -> None:
        if self._closed:
            return
        self._poll_worker_events()
        self._tick_playback()
        self._next_tick_at, delay_ms = next_player_tick(
            self._next_tick_at,
            time.monotonic(),
        )
        self.after(delay_ms, self._tick)

    def _poll_worker_events(self) -> None:
        if self._worker is None:
            return
        try:
            while True:
                event = self._worker.events.get_nowait()
                if event.generation != self._generation:
                    continue
                if isinstance(event, PlayerStatus):
                    if event.message == "loading_models":
                        message = t("player_loading_models")
                    else:
                        message = t("player_restoring")
                    self._set_status(message, Colors.STATUS_PENDING)
                elif isinstance(event, PlayerEnded):
                    self._eof = True
                elif isinstance(event, PlayerFailed):
                    self._failed_message = t("player_failed", message=event.message)
                    self._set_status(self._failed_message, Colors.STATUS_ERROR)
                    self._begin_stop()
        except queue.Empty:
            pass

    def _tick_playback(self) -> None:
        if (
            self._frame_buffer is None
            or self._clock is None
            or self._metadata is None
            or self._stopping
        ):
            return
        first = self._frame_buffer.peek()
        if first is not None and self._aligned_generation != self._generation:
            self._show_frame(first)
            self._clock.seek(first.seconds)
            self._current_seconds = first.seconds
            self._aligned_generation = self._generation
            self._update_time_label(first.seconds)
            if not self._desired_playing:
                self._set_status(t("player_paused"), Colors.STATUS_PAUSED)

        if self._desired_playing and not self._playing:
            if self._frame_buffer.ready(PREROLL_SECONDS):
                try:
                    self._clock.play()
                except VlcUnavailableError as exc:
                    self._failed_message = t("player_failed", message=str(exc))
                    self._begin_stop()
                    return
                self._playing = True
                self._buffering = False
                self._play_btn.configure(text="⏸")
                self._update_playing_buffer_status(
                    self._current_seconds,
                    force=True,
                )
            elif first is not None:
                self._buffering = True
                self._set_status(t("player_buffering"), Colors.STATUS_PAUSED)

        if not self._playing:
            return
        try:
            now = min(float(self._metadata.duration), self._clock.seconds())
        except VlcUnavailableError as exc:
            self._failed_message = t("player_failed", message=str(exc))
            self._begin_stop()
            return
        due = self._frame_buffer.pop_due(now, _FRAME_TOLERANCE)
        if due is not None:
            self._show_frame(due)
        self._current_seconds = now
        self._update_playing_buffer_status(now)
        if not self._seek_dragging:
            self._seek.set(now)
            self._update_time_label(now)

        if self._frame_buffer.empty():
            self._clock.pause()
            self._playing = False
            if self._eof or self._frame_buffer.eof:
                self._desired_playing = False
                self._buffering = False
                self._play_btn.configure(text="▶")
                self._set_status(t("player_finished"), Colors.STATUS_COMPLETED)
            else:
                self._buffering = True
                self._set_status(t("player_buffering"), Colors.STATUS_PAUSED)

    def _show_frame(self, frame) -> None:
        if frame.generation != self._generation:
            return
        self._last_frame_image = frame.image
        self._display_image(frame.image, self._surface_size())

    def _display_image(
        self,
        image: Image.Image,
        size: tuple[int, int] | None = None,
    ) -> None:
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.BILINEAR)
        if self._photo is not None and self._photo_size == image.size:
            self._photo.paste(image)
            return
        self._photo = ImageTk.PhotoImage(image)
        self._photo_size = image.size
        self._video_surface.configure(image=self._photo, text="")

    def _toggle_play(self) -> None:
        if self._clock is None or self._frame_buffer is None:
            if self._metadata is not None:
                self._start_playback()
            return
        if self._desired_playing:
            self._desired_playing = False
            self._playing = False
            self._buffering = False
            self._clock.pause()
            self._play_btn.configure(text="▶")
            self._set_status(t("player_paused"), Colors.STATUS_PAUSED)
            return
        if (self._eof or self._frame_buffer.eof) and self._frame_buffer.empty():
            self._desired_playing = True
            self._seek_to(0)
            return
        self._desired_playing = True
        self._buffering = not self._frame_buffer.ready(PREROLL_SECONDS)
        self._next_buffer_status_at = 0.0
        self._play_btn.configure(text="⏸")

    def _space_pressed(self, _event=None):
        self._toggle_play()
        return "break"

    def _seek_relative(self, offset: float, _event=None):
        if self._worker is not None and self._metadata is not None:
            target = min(
                float(self._metadata.duration),
                max(0.0, self._current_seconds + float(offset)),
            )
            self._seek_to(target)
        return "break"

    def _seek_moved(self, value: float) -> None:
        if self._worker is None or self._metadata is None:
            return
        self._seek_dragging = True
        self._seek_target = min(float(self._metadata.duration), max(0.0, float(value)))
        self._update_time_label(self._seek_target)

    def _seek_released(self, _event) -> None:
        if not self._seek_dragging:
            return
        self._seek_dragging = False
        self._seek_to(self._seek_target)

    def _seek_to(self, seconds: float) -> None:
        if self._worker is None or self._clock is None:
            return
        self._clock.pause()
        self._clock.seek(seconds)
        self._playing = False
        self._buffering = self._desired_playing
        self._eof = False
        self._current_seconds = seconds
        self._next_buffer_status_at = 0.0
        self._generation = self._worker.play_from(seconds)
        self._aligned_generation = -1
        self._video_surface.configure(image="", text=t("player_restoring"))
        self._photo = None
        self._photo_size = None
        self._last_frame_image = None
        self._set_status(t("player_restoring"), Colors.STATUS_PENDING)

    def _model_changed(self, model: str) -> None:
        threshold = recommended_score_threshold(model)
        self._threshold.set(threshold)
        self._threshold_value.configure(text=f"{threshold:.2f}")
        self._request_pipeline_reload()

    def _threshold_changed(self, value: float) -> None:
        self._threshold_value.configure(text=f"{float(value):.2f}")

    def _threshold_released(self, _event=None) -> None:
        self._request_pipeline_reload()

    def _secondary_changed(self, _value: str) -> None:
        self._request_pipeline_reload()

    def _request_pipeline_reload(self) -> None:
        if self._worker is None or self._clock is None or self._stopping:
            return
        seconds = self._current_seconds
        self._clock.pause()
        self._clock.seek(seconds)
        self._playing = False
        self._buffering = self._desired_playing
        self._eof = False
        self._next_buffer_status_at = 0.0
        self._generation = self._worker.reload_from(
            self._playback_settings(),
            seconds,
        )
        self._aligned_generation = -1
        self._play_btn.configure(text="⏸" if self._desired_playing else "▶")
        self._set_status(t("player_loading_models"), Colors.STATUS_PENDING)

    def _volume_changed(self, value: float) -> None:
        if isinstance(self._clock, VlcAudioClock):
            self._clock.set_volume(round(value))

    def _mute_changed(self) -> None:
        if isinstance(self._clock, VlcAudioClock):
            self._clock.set_muted(self._mute.get() == 1)

    def _video_area_resized(self, event) -> None:
        self._fit_video_surface(event.width, event.height)

    def _fit_video_surface(self, area_width: int, area_height: int) -> None:
        width, height = fit_player_view_size(
            (area_width, area_height),
            video_display_aspect(self._metadata),
        )
        self._view_size = (width, height)
        self._video_surface.place_configure(width=width, height=height)
        if self._worker is not None:
            self._worker.set_max_size((width, height))
        if self._last_frame_image is not None:
            self._display_image(self._last_frame_image, (width, height))

    def _toggle_fullscreen(self, _event=None):
        if self._fullscreen:
            return self._exit_fullscreen(_event)
        self._windowed_geometry = self.geometry()
        self._file_row.grid_remove()
        self._settings_card.grid_remove()
        self._actions.grid_remove()
        self._bottom_panel.grid_remove()
        self._outer.pack_configure(padx=0, pady=0)
        self.attributes("-fullscreen", True)
        self._fullscreen = True
        self._fullscreen_controls_visible = False
        self._fullscreen_btn.configure(text=t("player_exit_fullscreen"))
        return "break" if _event is not None else None

    def _exit_fullscreen(self, _event=None):
        if not self._fullscreen:
            return None
        self._hide_fullscreen_controls()
        self.attributes("-fullscreen", False)
        self._file_row.grid()
        self._bottom_panel.configure(fg_color="transparent")
        self._bottom_panel.grid()
        self._settings_card.grid()
        self._actions.grid()
        self._outer.pack_configure(padx=16, pady=16)
        if self._windowed_geometry is not None:
            self.geometry(self._windowed_geometry)
        self._fullscreen = False
        self._fullscreen_btn.configure(text=t("player_fullscreen"))
        return "break" if _event is not None else None

    def _fullscreen_mouse_moved(self, event) -> None:
        if not self._fullscreen:
            return
        if not self._fullscreen_controls_visible:
            window_bottom = self.winfo_rooty() + self.winfo_height()
            if event.y_root >= window_bottom - _FULLSCREEN_EDGE_PX:
                self._show_fullscreen_controls()
            return
        left = self._bottom_panel.winfo_rootx()
        top = self._bottom_panel.winfo_rooty()
        right = left + self._bottom_panel.winfo_width()
        bottom = top + self._bottom_panel.winfo_height()
        if not (left <= event.x_root < right and top <= event.y_root < bottom):
            self._hide_fullscreen_controls()

    def _show_fullscreen_controls(self) -> None:
        if not self._fullscreen or self._fullscreen_controls_visible:
            return
        self._bottom_panel.configure(fg_color=Colors.BG_PANEL)
        self._bottom_panel.place(
            relx=0,
            rely=1,
            relwidth=1,
            anchor="sw",
        )
        self._bottom_panel.lift()
        self._fullscreen_controls_visible = True

    def _hide_fullscreen_controls(self) -> None:
        self._bottom_panel.place_forget()
        self._fullscreen_controls_visible = False

    def _surface_size(self) -> tuple[int, int]:
        return self._view_size

    def _begin_stop(self) -> None:
        if self._stopping:
            return
        if self._worker is None:
            self._finish_stop()
            return
        self._stopping = True
        self._desired_playing = False
        self._playing = False
        if self._clock is not None:
            self._clock.close()
            self._clock = None
        worker = self._worker
        worker.close()
        self._set_status(t("player_stopping"), Colors.STATUS_PENDING)
        self._set_config_enabled(False)
        self._play_btn.configure(state="disabled")
        self._seek.configure(state="disabled")

        def join_worker() -> None:
            worker.join()
            self._ui_after(self._finish_stop)

        threading.Thread(target=join_worker, name="player-stop", daemon=True).start()

    def _finish_stop(self) -> None:
        self._worker = None
        self._frame_buffer = None
        self._stopping = False
        self._desired_playing = False
        self._playing = False
        self._buffering = False
        self._eof = False
        self._play_btn.configure(
            state="normal" if self._metadata is not None else "disabled",
            text="▶",
        )
        self._seek.configure(state="disabled")
        self._mute.configure(state="disabled")
        self._volume.configure(state="disabled")
        self._set_config_enabled(True)
        if self._failed_message:
            self._set_status(self._failed_message, Colors.STATUS_ERROR)
            self._failed_message = ""
        else:
            self._set_status(t("player_ready"), Colors.STATUS_COMPLETED)
        if self._close_after_stop:
            self._destroy_dialog()
        elif self._choose_after_stop:
            self._choose_after_stop = False
            self._choose_video()

    def _set_config_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self._choose_btn.configure(state=state)
        self._model.configure(state=state)
        self._threshold.configure(state=state)
        self._secondary.configure(state=state)

    def _set_status(self, text: str, color: str) -> None:
        value = (text, color)
        if value == self._last_status:
            return
        self._last_status = value
        self._status.configure(text=text, text_color=color)

    def _update_playing_buffer_status(
        self,
        seconds: float,
        *,
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        if not force and now < self._next_buffer_status_at:
            return
        self._next_buffer_status_at = now + _BUFFER_STATUS_INTERVAL_SECONDS
        buffered = self._frame_buffer.buffered_ahead(seconds)
        self._set_status(
            t("player_playing_buffer", seconds=f"{buffered:.1f}"),
            Colors.STATUS_COMPLETED,
        )

    def _update_time_label(self, seconds: float) -> None:
        duration = float(self._metadata.duration) if self._metadata is not None else 0.0
        text = f"{format_player_time(seconds)} / {format_player_time(duration)}"
        if text == self._last_time_text:
            return
        self._last_time_text = text
        self._time_label.configure(text=text)

    def request_close(self) -> None:
        if self._closed:
            return
        if self._worker is not None:
            self._close_after_stop = True
            self._begin_stop()
        else:
            self._destroy_dialog()

    def _destroy_dialog(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._probe_generation += 1
        try:
            self.grab_release()
        except tk.TclError:
            pass
        self.destroy()
        self._on_closed()

    def _ui_after(self, callback) -> None:
        if self._closed:
            return
        try:
            self.after(0, callback)
        except (tk.TclError, RuntimeError):
            pass
