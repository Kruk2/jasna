"""Main Jasna GUI application window."""

import customtkinter as ctk
import logging
import os
from pathlib import Path
import platform
import sys
import threading
import time
import tkinter as tk

from tkinterdnd2 import TkinterDnD, DND_FILES

from jasna import __version__
from jasna import startup_timing
from jasna.gui.branding import (
    HEADER_LOGO_SIZE,
    create_header_logo,
    install_window_icon,
)
from jasna.gui import scaling
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.gui.components import StatusPill, BuyMeCoffeeButton, UnifansButton, Toast, LicenseDialog
from jasna.gui.icons import create_icon, create_native_icon_image
from jasna.gui.queue_panel import QueuePanel
from jasna.gui.settings_panel import SettingsPanel
from jasna.engine_paths import UNET4X_ONNX_ENC_PATH
from jasna.gui.control_bar import ControlBar
from jasna.gui.log_panel import LogPanel
from jasna.gui.log_filter import runtime_log_level_for_filter
from jasna.gui.run_log import AsyncRunLog, RunTelemetrySampler
from jasna.gui.processor import Processor, ProgressUpdate, _is_linux_amd_runtime
from jasna.gui.models import JobStatus, PresetManager
from jasna.gui.locales import get_locale, t, LANGUAGE_NAMES
from jasna.gui.font_backend import (
    GuiFontBackendError,
    font_backend_error,
    font_backend_problem,
    font_backend_status_json,
    inspect_font_backend,
)
from jasna._frozen import is_frozen

logger = logging.getLogger(__name__)

_DEFAULT_WINDOW_SIZE = (1320, 960)
_MIN_WINDOW_SIZE = (900, 580)


def _warm_up_cuda() -> None:
    """Import torch and create the CUDA context. Run off the UI thread after the window
    paints, so the window shows first and the first job skips cold torch/CUDA init."""
    import torch

    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")
        if sys.platform == "win32" and getattr(getattr(torch, "version", None), "hip", None):
            # Finish the warmup on its owner thread before native Windows HIP
            # teardown; pending work can otherwise hang even os._exit().
            torch.cuda.synchronize()


class JasnaApp(ctk.CTk, TkinterDnD.DnDWrapper):
    """Main application window for Jasna GUI."""
    
    def __init__(self, skip_wizard: bool = False):
        super().__init__()
        font_status = inspect_font_backend(self)
        font_problem = font_backend_problem(font_status)
        if font_problem is not None:
            from tkinter import messagebox

            message = font_backend_error(font_status, frozen=is_frozen())
            self.withdraw()
            messagebox.showerror("Jasna GUI font error", message, parent=self)
            self.destroy()
            raise GuiFontBackendError(message)
        self._t_init_start = startup_timing.elapsed_ms()
        try:
            self.TkdndVersion = TkinterDnD._require(self)
        except RuntimeError:
            # tkdnd native lib may fail to load (e.g. uv-managed python bundles
            # Tcl/Tk 9 while tkinterdnd2 ships a Tcl 8.x binary) - run without drag&drop
            self.TkdndVersion = None
        
        self.title("Jasna GUI")
        self._window_icon = install_window_icon(self)
        self.configure(fg_color=Colors.BG_MAIN)

        self._size_and_center()

        # Set appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self._logs_visible = False
        self._processor: Processor | None = None
        self._job_start_times: dict[int, float] = {}
        self._processing_start_time: float = 0.0
        # Processor callbacks arrive off the UI thread.  Keep one UI-owned
        # generation so callbacks already queued for an older run cannot alter
        # a later run's controls or jobs.
        self._ui_run_epoch = 0
        self._ui_run_state = "idle"
        self._preview_gpu_busy = False
        self._video_player_dialog = None
        self._closing_after_player = False
        self._preset_manager = PresetManager()
        self._run_log: AsyncRunLog | None = None
        self._run_telemetry_sampler: RunTelemetrySampler | None = None
        self._closing_run_logs: list[AsyncRunLog] = []

        self._system_stats_stop = threading.Event()
        self._system_stats_thread: threading.Thread | None = None
        
        self._build_ui()
        self._t_ui_built = startup_timing.elapsed_ms()
        self._setup_processor()
        self._start_system_stats_poller()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.after_idle(self._log_startup_timing)
        self.after_idle(self._start_cuda_warmup)

        if not skip_wizard:
            if self._preset_manager.get_system_check_passed_version() != __version__:
                self.after(100, self._show_wizard)
            
    def _size_and_center(self):
        self.update_idletasks()
        rect = scaling.screen_rect(self)
        width, height = scaling.fit_size(
            scaling.to_physical(self, *_DEFAULT_WINDOW_SIZE),
            rect[2:],
            scaling.to_physical(self, *scaling.SCREEN_MARGIN),
        )
        x = rect[0] + (rect[2] - width) // 2
        y = rect[1] + max(0, (rect[3] - height) // 2 - int(rect[3] * 0.15 / 2))
        scaling.apply_geometry(self, width, height, x, y)
        scaling.apply_minsize(self, *_MIN_WINDOW_SIZE)

    def _build_ui(self):
        # Footer before the body: the packer starves its last slaves when the window is
        # shorter than the requested layout, and the control bar must never be the casualty.
        self._build_header()
        self._build_footer()
        self._build_main_body()
        
    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=Colors.BG_PANEL, height=Sizing.HEADER_HEIGHT, corner_radius=0)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        # Left: Logo and title
        left = ctk.CTkFrame(header, fg_color="transparent")
        left.pack(side="left", padx=Sizing.PADDING_MEDIUM)
        
        self._header_logo = create_header_logo()
        logo = ctk.CTkLabel(
            left,
            text="",
            image=self._header_logo,
            fg_color="transparent",
            width=HEADER_LOGO_SIZE[0],
            height=HEADER_LOGO_SIZE[1],
        )
        logo.pack(side="left")
        
        title = ctk.CTkLabel(
            left,
            text=t("app_title"),
            font=(Fonts.FAMILY, Fonts.SIZE_TITLE, "bold"),
            text_color=Colors.TEXT_PRIMARY,
        )
        title.pack(side="left", padx=(8, 4))
        
        version = ctk.CTkLabel(
            left,
            text=f"v{__version__}",
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.TEXT_PRIMARY,
        )
        version.pack(side="left", pady=(4, 0))

        # Status pill, left-aligned next to the version (the right side holds the
        # license/support chips, which used to overlap a centered pill).
        self._status_pill = StatusPill(left)
        self._status_pill.pack(side="left", padx=(12, 0))

        # Right: Language, Help and About
        right = ctk.CTkFrame(header, fg_color="transparent")
        right.pack(side="right", padx=Sizing.PADDING_MEDIUM)
        
        # Language selector
        self._language_icon = create_native_icon_image(
            right,
            "globe",
            16,
            Colors.TEXT_PRIMARY,
        )
        lang_label = tk.Label(
            right,
            image=self._language_icon,
            background=Colors.BG_PANEL,
            width=scaling.raw_tk_size(right, 18),
            height=scaling.raw_tk_size(right, 18),
            borderwidth=0,
            highlightthickness=0,
        )
        lang_label.pack(side="left", padx=(0, 4))
        
        locale = get_locale()
        lang_values = [LANGUAGE_NAMES[code] for code in locale.available_languages]
        current_lang_name = LANGUAGE_NAMES.get(locale.current_language, "English")
        
        self._lang_dropdown = ctk.CTkOptionMenu(
            right,
            values=lang_values,
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            fg_color=Colors.BG_CARD,
            button_color=Colors.BG_CARD,
            button_hover_color=Colors.BORDER_LIGHT,
            dropdown_fg_color=Colors.BG_CARD,
            dropdown_hover_color=Colors.PRIMARY,
            text_color=Colors.TEXT_PRIMARY,
            width=100,
            height=28,
            command=self._on_language_changed,
        )
        self._lang_dropdown.pack(side="left", padx=(0, 12))
        self._lang_dropdown.set(current_lang_name)

        self._video_player_icon = create_icon("play", 16, Colors.PLAYER_TEXT)
        self._video_player_btn = ctk.CTkButton(
            right,
            text=t("btn_video_player"),
            image=self._video_player_icon,
            compound="left",
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL, "bold"),
            fg_color=Colors.PLAYER,
            hover_color=Colors.PLAYER_HOVER,
            border_color=Colors.PLAYER_BORDER,
            border_width=1,
            border_spacing=6,
            text_color=Colors.PLAYER_TEXT,
            text_color_disabled=Colors.STATUS_PENDING,
            corner_radius=8,
            width=145,
            height=34,
            command=self._open_video_player,
        )
        self._video_player_btn.pack(side="left", padx=(0, 12))
        
        # Support buttons — back the project on Buy Me a Coffee or Unifans
        self._bmc_btn = BuyMeCoffeeButton(right, compact=False)
        self._bmc_btn.pack(side="left", padx=(0, 8))

        self._unifans_btn = UnifansButton(right, compact=False)
        self._unifans_btn.pack(side="left", padx=(0, 12))

        # Supporter license chip — only shown when the gated (encrypted) model ships.
        if UNET4X_ONNX_ENC_PATH.exists():
            self._license_chip = ctk.CTkButton(
                right, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
                fg_color="transparent", hover_color=Colors.BG_CARD,
                width=130, command=self._open_license_dialog,
            )
            self._license_chip.pack(side="left", padx=(0, 12))
            self._refresh_license_chip()

        self._system_check_btn = ctk.CTkButton(
            right,
            text=t("btn_system_check"),
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            fg_color="transparent",
            hover_color=Colors.BG_CARD,
            text_color=Colors.TEXT_PRIMARY,
            width=80,
            command=self._show_system_check,
        )
        self._system_check_btn.pack(side="left", padx=(0, 4))
        
        self._help_btn = ctk.CTkButton(
            right,
            text=t("btn_help"),
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            fg_color="transparent",
            hover_color=Colors.BG_CARD,
            text_color=Colors.TEXT_PRIMARY,
            width=50,
            command=self._show_help,
        )
        self._help_btn.pack(side="left")
        
        self._about_btn = ctk.CTkButton(
            right,
            text=t("btn_about"),
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            fg_color="transparent",
            hover_color=Colors.BG_CARD,
            text_color=Colors.TEXT_PRIMARY,
            width=50,
            command=self._show_about,
        )
        self._about_btn.pack(side="left")
        
    def _build_main_body(self):
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True)

        # Not sashcursor: Tk applies it to the whole panedwindow window, and the
        # cursorless panes inherit it, showing resize arrows everywhere. The
        # widget cursor covers only the exposed sash strip, and the panes mask
        # inheritance with an explicit "arrow".
        self._workspace = tk.PanedWindow(
            body,
            orient=tk.HORIZONTAL,
            background=Colors.BORDER,
            borderwidth=0,
            cursor="sb_h_double_arrow",
            opaqueresize=True,
            sashpad=0,
            sashrelief=tk.FLAT,
            sashwidth=scaling.raw_tk_size(body, 4),
        )
        self._workspace.pack(fill="both", expand=True)

        self._queue_panel = QueuePanel(self._workspace)
        self._queue_panel.configure(cursor="arrow")
        self._queue_panel.set_on_jobs_changed(self._on_jobs_changed)

        self._settings_panel = SettingsPanel(self._workspace)
        self._settings_panel.configure(cursor="arrow")
        self._settings_panel.set_on_interactive_image_restore(self._open_interactive_image_restore)

        self._workspace.add(
            self._queue_panel,
            minsize=scaling.raw_tk_size(self._workspace, Sizing.QUEUE_PANEL_MIN_WIDTH),
            stretch="never",
            width=scaling.raw_tk_size(self._workspace, Sizing.QUEUE_PANEL_WIDTH),
        )
        self._workspace.add(
            self._settings_panel,
            minsize=scaling.raw_tk_size(self._workspace, Sizing.SETTINGS_PANEL_MIN_WIDTH),
            stretch="always",
        )
        
        self._queue_panel.set_segment_editor_context(
            self._settings_panel.get_settings,
            lambda: self._processor is not None and self._processor.is_running(),
            self._set_preview_gpu_busy,
        )
        self._queue_panel.set_initial_output(
            self._settings_panel.get_last_output_folder(),
            self._settings_panel.get_last_output_pattern(),
            self._settings_panel.get_last_preserve_input_structure(),
        )
        self._queue_panel.set_on_output_changed(self._on_output_changed)
        self._queue_panel.set_on_play(
            lambda path: JasnaApp._open_video_player(self, path)
        )
        if self.TkdndVersion is not None:
            self._queue_panel.enable_file_drop()
        
    def _build_footer(self):
        # Log panel (bottom, collapsible) - hidden by default
        self._log_panel = LogPanel(self)
        # Don't pack initially - logs are hidden by default
        
        # Separator
        sep = ctk.CTkFrame(self, fg_color=Colors.BORDER, height=1)
        sep.pack(fill="x", side="bottom")
        
        # Control bar
        self._control_bar = ControlBar(self)
        self._control_bar.pack(fill="x", side="bottom")
        self._control_bar.set_callbacks(
            on_start=self._on_start,
            on_stop=self._on_stop,
            on_toggle_logs=self._toggle_logs,
        )
        self.after(0, self._update_start_button_state)
        
    def _setup_processor(self):
        self._processor = Processor(
            on_progress=self._on_processor_progress,
            on_log=self._on_processor_log,
            on_complete=self._on_processor_complete,
            video_job_isolation="linux-amd",
        )

    def _log_startup_timing(self):
        logger.info(
            "startup: first paint %.0f ms (pre-window %.0f ms, ui build %.0f ms)",
            startup_timing.elapsed_ms(),
            self._t_init_start,
            self._t_ui_built - self._t_init_start,
        )

    def _start_cuda_warmup(self):
        # A HIP context in the long-lived GUI defeats per-video native-resource
        # isolation. Linux AMD initializes the GPU only inside each worker.
        if _is_linux_amd_runtime():
            return

        def _run():
            try:
                _warm_up_cuda()
            except Exception:
                logger.debug("CUDA warm-up failed", exc_info=True)

        threading.Thread(target=_run, daemon=True, name="cuda-warmup").start()

    def _start_system_stats_poller(self):
        if self._system_stats_thread and self._system_stats_thread.is_alive():
            return

        self._system_stats_stop.clear()

        def _loop():
            from jasna.gui.system_stats import read_system_stats
            while not self._system_stats_stop.is_set():
                stats = read_system_stats()
                try:
                    self.after(0, lambda s=stats: self._control_bar.set_system_stats(s))
                except Exception:
                    logger.debug("System stats poller stopping (widget gone)", exc_info=True)
                    return
                self._system_stats_stop.wait(1.5)

        self._system_stats_thread = threading.Thread(target=_loop, daemon=True)
        self._system_stats_thread.start()

    def _stop_system_stats_poller(self):
        self._system_stats_stop.set()
        if self._system_stats_thread:
            self._system_stats_thread.join(timeout=1.0)
            self._system_stats_thread = None

    def _start_run_log(
        self,
        settings,
        jobs,
        output_folder: str,
        output_pattern: str,
    ) -> Path | None:
        """Start best-effort diagnostics without file I/O on the GUI thread."""

        if not bool(getattr(settings, "save_run_log", False)):
            return None
        try:
            run_log = AsyncRunLog.start(
                output_folder,
                on_status=self._on_run_log_status,
            )
            self._run_log = run_log
            if run_log.failed:
                self._run_log = None
                return None
            run_log.enqueue(
                "INFO",
                "Run context: "
                f"gui_pid={os.getpid()} platform={platform.platform()} "
                f"python={sys.version.split()[0]}",
            )
            run_log.enqueue(
                "INFO",
                "Run context: "
                f"output_folder={output_folder or '<same as input>'} "
                f"output_pattern={output_pattern}",
            )
            run_log.enqueue(
                "INFO",
                "Run context: "
                f"detection_model={settings.detection_model} codec={settings.codec} "
                f"fp16_mode={bool(settings.fp16_mode)} vr_mode={settings.vr_mode} "
                f"secondary_restoration={settings.secondary_restoration}",
            )
            for index, job in enumerate(jobs, start=1):
                run_log.enqueue("INFO", f"Queued input {index}: {job.path}")
            run_log.enqueue(
                "INFO",
                (
                    "Kernel/driver crash capture is not started; inspect the Windows "
                    "Event Viewer System log after a reset for driver/kernel events."
                    if sys.platform == "win32" else
                    "Kernel journal capture is not started; inspect journalctl and pstore "
                    "after a reset for kernel-only events."
                ),
            )
            sampler = RunTelemetrySampler(
                run_log.enqueue,
                is_active=lambda: run_log.accepts_events,
                context_provider=self._run_telemetry_context,
            )
            self._run_telemetry_sampler = sampler
            sampler.start()
            return run_log.path
        except Exception as exc:
            self._close_run_log()
            self._on_run_log_status("WARNING", f"Run log unavailable: {exc}")
            return None

    def _enqueue_run_log(self, level: str, message: str) -> None:
        run_log = getattr(self, "_run_log", None)
        if run_log is None:
            return
        try:
            run_log.enqueue(level, message)
        except Exception:
            pass

    def _add_run_log_event(self, level: str, message: str) -> None:
        self._enqueue_run_log(level, message)
        self._log_panel.add_log(level, message)

    def _on_run_log_status(self, level: str, message: str) -> None:
        if level.upper() != "WARNING":
            return
        try:
            self.after(
                0,
                lambda: self._log_panel.add_log("WARNING", t("run_log_unavailable")),
            )
        except Exception:
            pass

    def _run_telemetry_context(self) -> dict[str, object]:
        context: dict[str, object] = {"gui_pid": os.getpid()}
        processor = getattr(self, "_processor", None)
        if processor is not None:
            try:
                job = processor.active_job()
                if job is not None:
                    context["job"] = str(job.path)
            except Exception:
                pass
            try:
                worker_pid = processor.isolated_worker_pid()
                if worker_pid is not None:
                    context["worker_pid"] = worker_pid
            except Exception:
                pass
        return context

    def _close_run_log(self) -> None:
        sampler = getattr(self, "_run_telemetry_sampler", None)
        self._run_telemetry_sampler = None
        if sampler is not None:
            try:
                sampler.stop(timeout=0.0)
            except Exception:
                pass
        run_log = getattr(self, "_run_log", None)
        self._run_log = None
        if run_log is None:
            return
        try:
            run_log.close(timeout=0.0)
            closing_logs = [
                log
                for log in getattr(self, "_closing_run_logs", [])
                if getattr(log, "is_running", False)
            ]
            if getattr(run_log, "is_running", False):
                closing_logs.append(run_log)
            self._closing_run_logs = closing_logs
        except Exception:
            pass

    def _wait_for_run_log_shutdown(self, timeout: float = 1.0) -> None:
        deadline = time.monotonic() + max(0.0, timeout)
        closing_logs = getattr(self, "_closing_run_logs", [])
        self._closing_run_logs = []
        for run_log in closing_logs:
            try:
                run_log.close(timeout=max(0.0, deadline - time.monotonic()))
            except Exception:
                pass

    def _on_close(self):
        if self._video_player_dialog is not None:
            self._closing_after_player = True
            self._video_player_dialog.request_close()
            return
        self._ui_run_epoch = getattr(self, "_ui_run_epoch", 0) + 1
        self._ui_run_state = "closed"
        try:
            if self._processor:
                self._processor.stop()
                self._processor.join(timeout=5.0)
        finally:
            self._close_run_log()
            self._stop_system_stats_poller()
            self.destroy()
        
    def _refresh_license_chip(self):
        from jasna.license_api import license_store
        licensed = license_store.is_licensed()
        self._license_chip.configure(
            text=t("license_chip_active") if licensed else t("license_chip_inactive"),
            text_color=Colors.STATUS_COMPLETED if licensed else Colors.STATUS_PAUSED,
        )

    def _open_license_dialog(self):
        LicenseDialog(self, on_activated=self._refresh_license_chip)

    def _show_wizard(self):
        from jasna.gui.wizard import FirstRunWizard
        FirstRunWizard(self, on_complete=self._on_wizard_complete)

    def _show_system_check(self):
        from jasna.gui.wizard import FirstRunWizard
        FirstRunWizard(self, on_complete=self._on_system_check_complete)

    def _on_wizard_complete(self, can_continue: bool, all_passed: bool = False):
        if not can_continue:
            self._log_panel.error(t("wizard_log_blocked"))
            self._on_close()
            return
        if all_passed:
            self._preset_manager.set_system_check_passed_version(__version__)
        self._log_panel.info(t("wizard_log_ready"))

    def _on_system_check_complete(self, can_continue: bool, all_passed: bool = False):
        """Dismissing a re-run system check only closes the dialog; it never quits the app."""
        if not can_continue:
            return
        if all_passed:
            self._preset_manager.set_system_check_passed_version(__version__)
        self._log_panel.info(t("wizard_log_ready"))
            
    def _on_jobs_changed(self):
        jobs = self._queue_panel.get_jobs()
        self._control_bar.update_progress(queue_total=len(jobs))
        self._update_start_button_state()

    def _on_output_changed(
        self,
        folder: str,
        pattern: str,
        preserve_input_structure: bool,
    ):
        self._settings_panel.set_last_output_folder(folder)
        self._settings_panel.set_last_output_pattern(pattern)
        self._settings_panel.set_last_preserve_input_structure(
            preserve_input_structure
        )

    def _open_interactive_image_restore(self):
        from tkinter import filedialog

        files = filedialog.askopenfilenames(
            title=t("interactive_select_images"),
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not files:
            return

        from jasna.gui.interactive_image_restore import InteractiveImageRestoreDialog

        InteractiveImageRestoreDialog(
            self,
            [Path(f) for f in files],
            self._settings_panel.get_settings(),
            self._queue_panel.get_output_folder(),
            self._queue_panel.get_output_pattern(),
            on_log=lambda level, message: self._log_panel.add_log(level, message),
        )

    def _open_video_player(self, path: Path | None = None):
        if self._preview_gpu_busy or (
            self._processor is not None and self._processor.is_running()
        ):
            return
        from jasna.gui.video_player import VideoPlayerDialog

        self._set_preview_gpu_busy(True)
        try:
            self._video_player_dialog = VideoPlayerDialog(
                self,
                self._settings_panel.get_settings(),
                initial_path=path,
                on_closed=self._video_player_closed,
            )
        except Exception:
            self._video_player_dialog = None
            self._set_preview_gpu_busy(False)
            raise

    def _video_player_closed(self):
        self._video_player_dialog = None
        self._set_preview_gpu_busy(False)
        if self._closing_after_player:
            self._closing_after_player = False
            self.after(0, self._on_close)
        
    def _update_start_button_state(self):
        jobs = self._queue_panel.get_jobs()
        can_start = bool(jobs) and not self._preview_gpu_busy
        if can_start:
            self._control_bar.set_start_enabled(True)
        elif self._preview_gpu_busy:
            self._control_bar.set_start_enabled(False, t("segments_restore_restoring"))
        else:
            self._control_bar.set_start_enabled(False, t("toast_no_files"))
        self._update_video_player_button_state()

    def _update_video_player_button_state(self) -> None:
        video_player_btn = self.__dict__.get("_video_player_btn")
        if video_player_btn is None:
            return
        processing = self._processor is not None and self._processor.is_running()
        disabled = self._preview_gpu_busy or processing
        video_player_btn.configure(
            state="disabled" if disabled else "normal",
            fg_color=Colors.BG_CARD if disabled else Colors.PLAYER,
            border_color=Colors.BORDER_LIGHT if disabled else Colors.PLAYER_BORDER,
        )

    def _set_preview_gpu_busy(self, busy: bool) -> None:
        self._preview_gpu_busy = bool(busy)
        try:
            self.after(0, self._update_start_button_state)
        except (tk.TclError, RuntimeError):
            pass
        
    def _show_toast(self, message: str, type_: str = "info"):
        """Show a toast notification."""
        toast = Toast(self, message, type_)
        toast.place(relx=0.5, rely=0.9, anchor="center")
        
    def _on_start(self):
        if self._preview_gpu_busy or (
            self._processor is not None and self._processor.is_running()
        ):
            return
        restart_reason_fn = getattr(self._processor, "restart_required_reason", None)
        restart_reason = restart_reason_fn() if callable(restart_reason_fn) else None
        if isinstance(restart_reason, str) and restart_reason:
            from tkinter import messagebox

            message = t("error_smart_render_restart_required")
            self._log_panel.error(message)
            messagebox.showerror(
                t("error_smart_render_restart_required_title"),
                message,
            )
            return
        jobs = self._queue_panel.get_jobs()
        if not jobs:
            self._log_panel.warning(t("toast_no_files"))
            return
            
        output_folder = self._queue_panel.get_output_folder()
        output_pattern = self._queue_panel.get_output_pattern()
        preserve_input_structure = self._queue_panel.get_preserve_input_structure()
        settings = self._settings_panel.get_settings()

        run_log_path = self._start_run_log(
            settings,
            jobs,
            output_folder,
            output_pattern,
        )
        if run_log_path is not None:
            self._add_run_log_event("INFO", t("run_log_path", path=str(run_log_path)))

        from jasna.gui.validation import validate_gui_start
        errors = validate_gui_start(settings)
        if errors:
            from tkinter import messagebox

            msg = t("error_cannot_start") + "\n\n" + "\n".join(f"- {e}" for e in errors)
            self._add_run_log_event("ERROR", msg)
            self._close_run_log()
            messagebox.showerror(t("error_invalid_tvai"), msg)
            return

        disable_basicvsrpp_tensorrt = False
        try:
            from jasna.gui.engine_preflight import run_engine_preflight

            preflight = run_engine_preflight(settings)
            missing_keys = [r.key for r in preflight.missing]

            def _engine_name(key: str) -> str:
                if key == "rfdetr":
                    return t("engine_name_rfdetr")
                if key == "yolo":
                    return t("engine_name_yolo")
                if key == "basicvsrpp":
                    return t("engine_name_basicvsrpp")
                return key

            if preflight.should_warn_first_run_slow:
                from tkinter import messagebox

                missing_lines = "\n".join(f"- {_engine_name(k)}" for k in missing_keys)
                msg = t("engine_first_run_body")
                if missing_lines:
                    msg += "\n\n" + t("engine_first_run_missing") + "\n" + missing_lines
                messagebox.showinfo(t("engine_first_run_title"), msg)
                self._add_run_log_event("WARNING", msg)
        except Exception as e:
            self._add_run_log_event("WARNING", f"Engine preflight warning failed: {e}")

        self._queue_panel.reset_jobs_for_run()
        
        self._status_pill.set_status("PROCESSING", Colors.STATUS_PROCESSING)
        self._control_bar.set_running(True)
        self._video_player_btn.configure(state="disabled")
        
        # Disable settings and output controls while running
        self._settings_panel.set_enabled(False)
        self._queue_panel.set_output_enabled(False)
        self._queue_panel.set_running(True)
        
        self._add_run_log_event("INFO", "Processing started by user")
        self._add_run_log_event("INFO", f"Output folder: {output_folder}")
        self._add_run_log_event("INFO", f"Output pattern: {output_pattern}")
        self._add_run_log_event(
            "INFO",
            f"Preserve input structure: {preserve_input_structure}",
        )
        self._add_run_log_event("INFO", f"Files queued: {len(jobs)}")

        self._processing_start_time = time.time()
        self._job_start_times.clear()
        jobs_ref = self._queue_panel.get_jobs_ref()
        self._ui_run_epoch = getattr(self, "_ui_run_epoch", 0) + 1
        run_epoch = self._ui_run_epoch
        self._ui_run_state = "running"
        try:
            self._processor.start(
                jobs_ref,
                settings,
                output_folder,
                output_pattern,
                disable_basicvsrpp_tensorrt=disable_basicvsrpp_tensorrt,
                preserve_input_structure=preserve_input_structure,
            )
        except Exception:
            if self._ui_run_epoch == run_epoch:
                self._ui_run_state = "idle"
            self._close_run_log()
            raise
        self._update_video_player_button_state()
                
    def _on_stop(self):
        if getattr(self, "_ui_run_state", None) == "running":
            # Keep this generation for its authoritative completion callback,
            # but reject any queued PROCESSING updates immediately.
            self._ui_run_state = "stopping"
        if self._processor:
            self._processor.stop()
            self._add_run_log_event("INFO", "Processing stopped by user")
            
        self._status_pill.set_status("IDLE", Colors.STATUS_PENDING)
        self._control_bar.reset()
        # Start stays disabled until the worker thread has finished unwinding;
        # _handle_complete re-enables it.
        self._control_bar.set_start_enabled(False)

        # Re-enable settings and output controls
        self._settings_panel.set_enabled(True)
        self._queue_panel.set_output_enabled(True)
        
    def _toggle_logs(self):
        self._logs_visible = not self._logs_visible
        if self._logs_visible:
            self._log_panel.pack(fill="x", side="bottom")
        else:
            self._log_panel.pack_forget()
            
    def _on_processor_progress(self, update: ProgressUpdate):
        """Queue a progress update only for the active UI run."""
        run_epoch = getattr(self, "_ui_run_epoch", 0)
        run_state = getattr(self, "_ui_run_state", None)
        if run_state not in {"running", "stopping", "finishing", "completed"}:
            return
        if update.status is JobStatus.PROCESSING and run_state != "running":
            return
        try:
            self.after(
                0,
                lambda update=update, run_epoch=run_epoch:
                    self._handle_progress(update, run_epoch),
            )
        except (tk.TclError, RuntimeError):
            logger.debug("Progress callback ignored because the UI is gone", exc_info=True)
        
    def _handle_progress(self, update: ProgressUpdate, run_epoch: int):
        if run_epoch != getattr(self, "_ui_run_epoch", None):
            return
        run_state = getattr(self, "_ui_run_state", None)
        if run_state not in {"running", "stopping", "finishing", "completed"}:
            return
        jobs = self._queue_panel.get_jobs()
        job_id = update.job_id
        job = next((item for item in jobs if item.id == job_id), None)
        # The processor owns the live JobItem state.  Do not let a delayed UI
        # notification overwrite a terminal state or revive a removed job.
        if job is None:
            return
        
        if update.status == JobStatus.PROCESSING:
            if run_state != "running" or job.status in {
                JobStatus.COMPLETED,
                JobStatus.SKIPPED,
                JobStatus.ERROR,
            }:
                return
            self._job_start_times.setdefault(job_id, time.time())
            # Resolve current queue position for control bar display
            self._control_bar.update_progress(
                filename=job.filename,
                percent=update.progress,
                fps=update.fps,
                eta_seconds=update.eta_seconds,
                queue_current=jobs.index(job) + 1,
                queue_total=len(jobs),
            )
            try:
                self._queue_panel.set_running(True, processing_job_id=job_id)
            except Exception:
                logger.warning("Failed to mark queue panel running", exc_info=True)
        elif job.status is not update.status:
            return
        if update.status == JobStatus.PENDING:
            self._job_start_times.pop(job_id, None)
        job_elapsed: float | None = None
        if update.status == JobStatus.COMPLETED:
            start = self._job_start_times.pop(job_id, None)
            job_elapsed = time.time() - start if start is not None else 0.0
        try:
            self._queue_panel.update_job_status(
                job_id,
                update.status,
                update.progress / 100.0,
                update.fps,
                update.eta_seconds,
                elapsed_seconds=job_elapsed,
                phase=update.phase,
            )
        except Exception:
            logger.warning("Failed to update job status in queue panel", exc_info=True)
            
    def _on_processor_log(self, level: str, message: str):
        self._enqueue_run_log(level, message)
        self.after(0, lambda: self._log_panel.add_log(level, message))
        
    def _on_processor_complete(self):
        run_epoch = getattr(self, "_ui_run_epoch", 0)
        if getattr(self, "_ui_run_state", None) not in {"running", "stopping"}:
            return
        # Once the worker has signalled completion, a queued PROCESSING update
        # from this generation is no longer authoritative.
        self._ui_run_state = "finishing"
        try:
            self.after(
                0,
                lambda run_epoch=run_epoch: self._handle_complete(run_epoch),
            )
        except (tk.TclError, RuntimeError):
            logger.debug("Completion callback ignored because the UI is gone", exc_info=True)
        
    def _handle_complete(self, run_epoch: int | None = None):
        if run_epoch is not None:
            if (
                run_epoch != getattr(self, "_ui_run_epoch", None)
                or getattr(self, "_ui_run_state", None) != "finishing"
            ):
                return
            self._ui_run_state = "completed"
        elif hasattr(self, "_ui_run_state"):
            # Keep direct test/support invocations compatible while ensuring a
            # real callback must first transition through ``finishing``.
            if self._ui_run_state != "finishing":
                return
            self._ui_run_state = "completed"
        try:
            restart_reason_fn = getattr(self._processor, "restart_required_reason", None)
            restart_reason = restart_reason_fn() if callable(restart_reason_fn) else None
            if isinstance(restart_reason, str) and restart_reason:
                from tkinter import messagebox

                message = t("error_smart_render_restart_required")
                self._status_pill.set_status("ERROR", Colors.STATUS_ERROR)
                self._control_bar.reset()
                self._control_bar.set_start_enabled(False, message)
                self._add_run_log_event("ERROR", message)
                self._settings_panel.set_enabled(True)
                self._queue_panel.set_output_enabled(True)
                try:
                    self._queue_panel.set_running(False)
                except Exception:
                    logger.warning("Failed to clear queue panel running state", exc_info=True)
                messagebox.showerror(
                    t("error_smart_render_restart_required_title"),
                    message,
                )
                return
            self._status_pill.set_status("IDLE", Colors.STATUS_PENDING)
            elapsed_seconds = time.time() - self._processing_start_time if self._processing_start_time else 0.0
            was_stopped_fn = getattr(self._processor, "was_stopped", None)
            was_stopped = bool(was_stopped_fn()) if callable(was_stopped_fn) else False
            jobs = self._queue_panel.get_jobs()
            completed_or_skipped = bool(jobs) and all(
                job.status in {JobStatus.COMPLETED, JobStatus.SKIPPED}
                for job in jobs
            )
            if completed_or_skipped and not was_stopped:
                self._control_bar.set_completed(elapsed_seconds)
                self._add_run_log_event("INFO", "All jobs completed")
            else:
                self._control_bar.reset()
                if was_stopped:
                    self._add_run_log_event("INFO", "Processing stopped by user")
                elif any(job.status is JobStatus.ERROR for job in jobs):
                    failed_jobs = sum(
                        job.status is JobStatus.ERROR
                        for job in jobs
                    )
                    self._add_run_log_event(
                        "ERROR",
                        f"Processing finished with {failed_jobs} failed job(s)",
                    )
                elif jobs:
                    self._add_run_log_event(
                        "WARNING",
                        "Processing ended before all jobs completed",
                    )
                else:
                    self._add_run_log_event(
                        "WARNING",
                        "Processing ended with no queued jobs",
                    )
            self._update_start_button_state()

            self._settings_panel.set_enabled(True)
            self._queue_panel.set_output_enabled(True)
            try:
                self._queue_panel.set_running(False)
            except Exception:
                logger.warning("Failed to clear queue panel running state", exc_info=True)
        finally:
            self._close_run_log()
        
    def _on_language_changed(self, lang_name: str):
        """Handle language selection change."""
        locale = get_locale()
        # Convert display name back to language code
        for code, name in LANGUAGE_NAMES.items():
            if name == lang_name:
                if code != locale.current_language:
                    locale.set_language(code)
                    self._refresh_ui_text()
                    from tkinter import messagebox
                    messagebox.showinfo(
                        t("dialog_language_changed"),
                        t("dialog_language_restart"),
                    )
                break
                
    def _refresh_ui_text(self):
        """Refresh UI text after language change."""
        # Update header buttons
        self._help_btn.configure(text=t("btn_help"))
        self._about_btn.configure(text=t("btn_about"))
        self._video_player_btn.configure(text=t("btn_video_player"))
        self._status_pill.refresh_text()
        # Note: Other panels would need their own refresh methods
        # For a full implementation, each panel should listen to locale changes
        
    def _show_help(self):
        import webbrowser
        webbrowser.open("https://github.com/Kruk2/jasna")
        
    def _show_about(self):
        dialog = ctk.CTkToplevel(self)
        dialog.title(t("dialog_about_title"))
        dialog.resizable(False, False)
        dialog.configure(fg_color=Colors.BG_MAIN)
        dialog.transient(self)
        dialog.wait_visibility()  # X11: window must be viewable before grab_set, else TclError
        dialog.grab_set()

        ctk.CTkLabel(
            dialog,
            text="Jasna",
            font=(Fonts.FAMILY, 24, "bold"),
            text_color=Colors.TEXT_PRIMARY,
        ).pack(pady=(30, 8))
        
        ctk.CTkLabel(
            dialog,
            text=t("dialog_about_version", version=__version__),
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            text_color=Colors.TEXT_PRIMARY,
        ).pack()
        
        ctk.CTkLabel(
            dialog,
            text=t("dialog_about_description"),
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            text_color=Colors.TEXT_PRIMARY,
        ).pack(pady=(16, 8))
        
        ctk.CTkLabel(
            dialog,
            text=t("dialog_about_credit"),
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_PRIMARY,
        ).pack()
        
        ctk.CTkButton(
            dialog,
            text=t("btn_close"),
            fg_color=Colors.BG_CARD,
            hover_color=Colors.BORDER_LIGHT,
            text_color=Colors.TEXT_PRIMARY,
            command=dialog.destroy,
        ).pack(pady=30)

        # Size to content so translated text never clips the close button
        dialog.update_idletasks()
        minimum_width, _ = scaling.to_physical(dialog, 400, 0)
        scaling.place_centered_on_parent(
            dialog,
            self,
            max(minimum_width, dialog.winfo_reqwidth()),
            dialog.winfo_reqheight(),
        )
        return dialog


class GUILogHandler(logging.Handler):
    """Custom logging handler that forwards logs to the GUI log panel."""
    
    def __init__(self, log_panel: LogPanel, on_log=None):
        super().__init__()
        self._log_panel = log_panel
        self._on_log = on_log
        
    def emit(self, record):
        try:
            msg = self.format(record)
            if self._on_log is not None:
                self._on_log(record.levelname, msg)
            # Use after_idle to thread-safely update GUI
            self._log_panel.after_idle(self._log_panel.add_log, record.levelname, msg)
        except Exception:
            pass  # Ignore errors in log handler


def run_gui():
    """Entry point to run the GUI application."""
    from jasna._frozen import patch_frozen_torch
    patch_frozen_torch()

    import logging
    # Set up basic logging - will be connected to GUI after app creation
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.StreamHandler()]  # Temporary console output
    )

    scaling.activate_static_dpi(_MIN_WINDOW_SIZE)

    if os.environ.get("JASNA_GUI_FONT_PROBE") == "1":
        root = ctk.CTk()
        try:
            status = inspect_font_backend(root)
            print(font_backend_status_json(status), flush=True)
            raise SystemExit(0 if font_backend_problem(status) is None else 1)
        finally:
            root.destroy()

    try:
        app = JasnaApp()
    except GuiFontBackendError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error
    
    # Replace console handler with GUI handler for all jasna loggers
    gui_handler = GUILogHandler(app._log_panel, on_log=app._enqueue_run_log)
    gui_handler.setFormatter(logging.Formatter('%(message)s'))
    
    # Set up root logger to capture all logs
    root_logger = logging.getLogger()
    root_logger.handlers = [gui_handler]
    
    # Also capture jasna-specific logger
    jasna_logger = logging.getLogger("jasna")
    jasna_logger.handlers = [gui_handler]
    jasna_logger.propagate = False

    def _apply_runtime_log_level(filter_level: str) -> None:
        runtime_level = runtime_log_level_for_filter(filter_level=filter_level)
        root_logger.setLevel(runtime_level)
        jasna_logger.setLevel(runtime_level)

    app._log_panel.set_filter_changed_callback(_apply_runtime_log_level)
    _apply_runtime_log_level(app._log_panel.get_filter_level())
    
    app.mainloop()

    app._wait_for_run_log_shutdown(timeout=1.0)

    # Force-exit the process. CUDA/TensorRT may leave non-daemon threads
    # or background subprocesses that prevent a clean interpreter shutdown.
    os._exit(0)
