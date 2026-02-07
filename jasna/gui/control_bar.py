"""Control bar - bottom playback controls and progress display."""

import customtkinter as ctk
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.gui.locales import t


class ControlBar(ctk.CTkFrame):
    """Bottom control bar with playback controls and progress display."""
    
    def __init__(self, master, **kwargs):
        super().__init__(
            master,
            fg_color=Colors.BG_PANEL,
            corner_radius=0,
            height=Sizing.CONTROL_BAR_HEIGHT,
            **kwargs
        )
        self.pack_propagate(False)
        
        self._on_start: callable = None
        self._on_stop: callable = None
        self._on_toggle_logs: callable = None
        
        self._is_running = False
        
        self._build_controls()
        self._build_progress()
        self._build_stats()
        
    def _build_controls(self):
        controls = ctk.CTkFrame(self, fg_color="transparent", width=100)
        controls.pack(side="left", padx=Sizing.PADDING_MEDIUM, pady=Sizing.PADDING_MEDIUM)
        controls.pack_propagate(False)
        
        # Start button (shown when idle)
        self._start_btn = ctk.CTkButton(
            controls,
            text="▶",
            font=(Fonts.FAMILY, 20),
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
            text_color=Colors.TEXT_PRIMARY,
            width=48,
            height=48,
            corner_radius=24,
            command=self._handle_start,
        )
        self._start_btn.pack(side="left")
        
        # Stop button (shown when running)
        self._stop_btn = ctk.CTkButton(
            controls,
            text="■",
            font=(Fonts.FAMILY, 20),
            fg_color=Colors.STATUS_ERROR,
            hover_color="#b91c1c",
            text_color=Colors.TEXT_PRIMARY,
            width=48,
            height=48,
            corner_radius=24,
            command=self._handle_stop,
        )
        
    def _build_progress(self):
        progress_area = ctk.CTkFrame(self, fg_color="transparent")
        progress_area.pack(side="left", fill="both", expand=True, padx=Sizing.PADDING_MEDIUM, pady=Sizing.PADDING_MEDIUM)
        
        # Filename
        self._filename_label = ctk.CTkLabel(
            progress_area,
            text=t("no_file_processing"),
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            text_color=Colors.TEXT_PRIMARY,
            anchor="w",
        )
        self._filename_label.pack(fill="x")
        
        # Progress bar row
        bar_row = ctk.CTkFrame(progress_area, fg_color="transparent")
        bar_row.pack(fill="x", pady=(4, 0))
        
        self._progress_bar = ctk.CTkProgressBar(
            bar_row,
            height=8,
            fg_color=Colors.BG_CARD,
            progress_color=Colors.PRIMARY,
            corner_radius=4,
        )
        self._progress_bar.pack(side="left", fill="x", expand=True, padx=(0, 12))
        self._progress_bar.set(0)
        
        self._percent_label = ctk.CTkLabel(
            bar_row,
            text="0%",
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL, "bold"),
            text_color=Colors.TEXT_PRIMARY,
            width=50,
        )
        self._percent_label.pack(side="right")
        
        # Stats row
        stats_row = ctk.CTkFrame(progress_area, fg_color="transparent")
        stats_row.pack(fill="x", pady=(4, 0))
        
        self._fps_label = ctk.CTkLabel(
            stats_row,
            text="FPS: --",
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_MUTED,
        )
        self._fps_label.pack(side="left")
        
        self._eta_label = ctk.CTkLabel(
            stats_row,
            text="ETA: --",
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_MUTED,
        )
        self._eta_label.pack(side="right")
        
    def _build_stats(self):
        stats = ctk.CTkFrame(self, fg_color="transparent", width=180)
        stats.pack(side="right", padx=Sizing.PADDING_MEDIUM, pady=Sizing.PADDING_MEDIUM)
        stats.pack_propagate(False)
        
        # Queue progress
        queue_frame = ctk.CTkFrame(stats, fg_color="transparent")
        queue_frame.pack(side="left", padx=(0, 16))
        
        queue_label = ctk.CTkLabel(
            queue_frame,
            text=t("queue_label"),
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.TEXT_MUTED,
        )
        queue_label.pack()
        
        self._queue_progress = ctk.CTkLabel(
            queue_frame,
            text="0 / 0",
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL, "bold"),
            text_color=Colors.TEXT_PRIMARY,
        )
        self._queue_progress.pack()
        
        # Logs toggle
        self._logs_btn = ctk.CTkButton(
            stats,
            text=t("logs_btn"),
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            fg_color=Colors.BG_CARD,
            hover_color=Colors.BORDER_LIGHT,
            text_color=Colors.TEXT_PRIMARY,
            height=32,
            width=80,
            command=self._handle_toggle_logs,
        )
        self._logs_btn.pack(side="right")
        
    def _handle_start(self):
        if self._on_start:
            self._on_start()
            
    def _handle_stop(self):
        if self._on_stop:
            self._on_stop()
            
    def _handle_toggle_logs(self):
        if self._on_toggle_logs:
            self._on_toggle_logs()
            
    def set_callbacks(self, on_start=None, on_stop=None, on_toggle_logs=None):
        self._on_start = on_start
        self._on_stop = on_stop
        self._on_toggle_logs = on_toggle_logs
        
    def set_running(self, running: bool, paused: bool = False):
        self._is_running = running
        
        if running:
            self._start_btn.pack_forget()
            self._stop_btn.pack(side="left")
        else:
            self._stop_btn.pack_forget()
            self._start_btn.pack(side="left")
            
    def update_progress(
        self,
        filename: str = "",
        percent: float = 0.0,
        fps: float = 0.0,
        eta_seconds: float = 0.0,
        queue_current: int = 0,
        queue_total: int = 0,
    ):
        self._filename_label.configure(text=filename or t("no_file_processing"))
        self._progress_bar.set(percent / 100.0)
        self._percent_label.configure(text=f"{int(percent)}%")
        self._fps_label.configure(text=f"FPS: {fps:.1f}" if fps > 0 else "FPS: --")
        
        if eta_seconds > 0:
            mins, secs = divmod(int(eta_seconds), 60)
            hours, mins = divmod(mins, 60)
            if hours:
                eta_str = f"{hours}h {mins}m"
            elif mins:
                eta_str = f"{mins}m {secs}s"
            else:
                eta_str = f"{secs}s"
            self._eta_label.configure(text=f"ETA: {eta_str}")
        else:
            self._eta_label.configure(text="ETA: --")
            
        self._queue_progress.configure(text=f"{queue_current} / {queue_total}")
        
    def reset(self):
        self.set_running(False)
        self.update_progress()
