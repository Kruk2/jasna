"""Control bar - bottom playback controls and progress display."""

import customtkinter as ctk
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.gui.locales import t
from jasna.gui.system_stats import SystemStats


_METRIC_WIDTH = 48

# Green -> Yellow -> Red gradient stops (0% -> 50% -> 100%)
_COLOR_STOPS = (
    (0.0, (52, 211, 153)),   # emerald-400
    (0.5, (251, 191, 36)),   # amber-400
    (1.0, (251, 113, 133)),  # rose-400
)


def _color_for_percent(pct: int) -> str:
    t = max(0.0, min(1.0, pct / 100.0))
    for i in range(len(_COLOR_STOPS) - 1):
        t0, c0 = _COLOR_STOPS[i]
        t1, c1 = _COLOR_STOPS[i + 1]
        if t <= t1:
            f = (t - t0) / (t1 - t0) if t1 != t0 else 0.0
            r = int(c0[0] + (c1[0] - c0[0]) * f)
            g = int(c0[1] + (c1[1] - c0[1]) * f)
            b = int(c0[2] + (c1[2] - c0[2]) * f)
            return f"#{r:02x}{g:02x}{b:02x}"
    _, c = _COLOR_STOPS[-1]
    return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"


class _SystemMetric(ctk.CTkFrame):
    def __init__(self, master, *, label: str):
        super().__init__(master, fg_color="transparent", width=_METRIC_WIDTH)
        self.pack_propagate(False)

        self._bar_value = 0.0
        self._anim_after_id = None

        self._label = ctk.CTkLabel(
            self,
            text=str(label).upper(),
            font=(Fonts.FAMILY, 10, "bold"),
            text_color=Colors.TEXT_MUTED,
            anchor="w",
        )
        self._label.pack(anchor="w")

        value_row = ctk.CTkFrame(self, fg_color="transparent")
        value_row.pack(anchor="w", pady=(1, 0))

        self._value = ctk.CTkLabel(
            value_row,
            text="--",
            font=(Fonts.FAMILY_MONO, 14),
            text_color=Colors.TEXT_PRIMARY,
        )
        self._value.pack(side="left")

        self._unit = ctk.CTkLabel(
            value_row,
            text="%",
            font=(Fonts.FAMILY, 10),
            text_color=Colors.TEXT_MUTED,
        )
        self._unit.pack(side="left", padx=(1, 0), pady=(2, 0))

        self._bar_track = ctk.CTkFrame(
            self,
            fg_color=Colors.BORDER,
            height=2,
            corner_radius=1,
        )
        self._bar_track.pack(fill="x", pady=(2, 0))
        self._bar_track.pack_propagate(False)

        self._bar_fill = ctk.CTkFrame(
            self._bar_track,
            fg_color=_color_for_percent(0),
            height=2,
            corner_radius=1,
        )
        self._bar_fill.place(relx=0.0, rely=0.0, relheight=1.0, relwidth=0.0)

    def _set_bar_value(self, value: float, color: str) -> None:
        v = max(0.0, min(1.0, float(value)))
        self._bar_value = v
        self._bar_fill.place_configure(relwidth=v)
        self._bar_fill.configure(fg_color=color)

    def set_percent(self, value: int | None) -> None:
        if value is None:
            self._value.configure(text="--", text_color=Colors.TEXT_PRIMARY)
            if self._anim_after_id is not None:
                try:
                    self.after_cancel(self._anim_after_id)
                except Exception:
                    pass
                self._anim_after_id = None
            self._set_bar_value(0.0, _color_for_percent(0))
            return

        pct = max(0, min(100, int(value)))
        color = _color_for_percent(pct)

        self._value.configure(text=str(pct), text_color=color)

        target = pct / 100.0
        start = float(self._bar_value)

        if self._anim_after_id is not None:
            try:
                self.after_cancel(self._anim_after_id)
            except Exception:
                pass
            self._anim_after_id = None

        steps = 6
        step_ms = 30

        def _step(i: int) -> None:
            if i >= steps:
                self._set_bar_value(target, color)
                self._anim_after_id = None
                return
            frac = (i + 1) / steps
            self._set_bar_value(start + (target - start) * frac, color)
            self._anim_after_id = self.after(step_ms, lambda: _step(i + 1))

        _step(0)


class SystemStatusPanel(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(
            master,
            fg_color=Colors.BG_MAIN,
            border_color=Colors.BORDER,
            border_width=1,
            corner_radius=Sizing.BORDER_RADIUS,
            **kwargs,
        )

        self._last = None

        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(padx=8, pady=5)

        self._gpu = _SystemMetric(content, label="GPU")
        self._gpu.pack(side="left")

        self._vram = _SystemMetric(content, label="VRAM")
        self._vram.pack(side="left", padx=(6, 0))

        sep = ctk.CTkFrame(content, fg_color=Colors.BORDER, width=1, height=32)
        sep.pack(side="left", padx=6)
        sep.pack_propagate(False)

        self._ram = _SystemMetric(content, label="RAM")
        self._ram.pack(side="left")

        self._cpu = _SystemMetric(content, label="CPU")
        self._cpu.pack(side="left", padx=(6, 0))

    def set_stats(self, stats: SystemStats) -> None:
        if stats == self._last:
            return
        self._last = stats

        self._gpu.set_percent(stats.gpu_util)
        self._vram.set_percent(stats.vram_util)
        self._ram.set_percent(stats.ram_util)
        self._cpu.set_percent(stats.cpu_util)


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
        stats = ctk.CTkFrame(self, fg_color="transparent")
        stats.pack(side="right", padx=Sizing.PADDING_MEDIUM, pady=Sizing.PADDING_MEDIUM)

        # Logs toggle (rightmost)
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

        # Queue progress (just left of logs)
        queue_frame = ctk.CTkFrame(stats, fg_color="transparent")
        queue_frame.pack(side="right", padx=(0, 16))

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

        # System status panel (left of queue)
        self._system_status = SystemStatusPanel(stats)
        self._system_status.pack(side="right", padx=(0, 12))
        
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

    def set_system_stats(self, stats: SystemStats) -> None:
        self._system_status.set_stats(stats)
