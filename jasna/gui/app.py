"""Main Jasna GUI application window."""

import customtkinter as ctk
import logging
import os
from pathlib import Path
import sys
import threading
import time
import tkinter as tk

from tkinterdnd2 import TkinterDnD

from jasna import __version__
from jasna import license_status, os_utils, startup_timing, system_requirements, update_check
from jasna.accelerator import AcceleratorVendor
from jasna.gui.branding import install_window_icon
from jasna.gui import scaling, system_checks, system_stats
from jasna.gui import activation_worker
from jasna.gui.theme import Colors, Sizing, apply_ui_fonts
from jasna.gui.components import XianyuQrDialog, Toast, ShutdownCountdownDialog, CloseWhileProcessingDialog, OptionsDialog
from jasna.gui.header_bar import HeaderBar
from jasna.gui.queue_panel import QueuePanel
from jasna.gui.settings_panel import SettingsPanel
from jasna.gui.control_bar import ControlBar
from jasna.gui.log_panel import LogPanel
from jasna.gui.gui_logging import install_gui_logging
from jasna.gui.processor import Processor, ProgressUpdate
from jasna.user_settings import read_settings
from jasna.gui.queues import MainThreadCalls
from jasna.gui import saved_queue, tray
from jasna.gui.models import AppSettings, JobStatus, PresetManager
from jasna.gui.start_flow import StartBlock, StartBlocked, read_ltx_license, start_blocker
from jasna.gui.validation import any_job_uses_ltx, job_is_resumable
from jasna.gui.locales import get_locale, t
from jasna.media.media_files import IMAGE_EXTENSIONS, file_dialog_patterns
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
_SHUTDOWN_COUNTDOWN_SECONDS = 60
_WINDHAWK_MODULE = "windhawk.dll"


class JasnaApp(ctk.CTk, TkinterDnD.DnDWrapper):
    """Main application window for Jasna GUI."""
    
    def __init__(self):
        super().__init__()
        self._main_thread = MainThreadCalls(self, 50)
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
        apply_ui_fonts(sys.platform, get_locale().current_language)
        
        self._logs_visible = False
        self._processor: Processor | None = None
        self._job_start_times: dict[int, float] = {}
        self._job_resumable: dict[int, bool] = {}
        self._completed_job_elapsed: dict[int, float] = {}
        self._queue_elapsed_seconds: float = 0.0
        self._closing = False
        self._preview_gpu_owners: set[object] = set()
        self._video_player_dialog = None
        self._segment_editor = None
        self._closing_after_player = False
        self._queue_restored = False
        self._preset_manager = PresetManager()
        self._tray = tray.TrayIcon(
            on_open=lambda: self._main_thread.post(self._show_from_tray),
            on_quit=lambda: self._main_thread.post(self._quit_from_tray),
        )

        self._build_ui()
        self._t_ui_built = startup_timing.elapsed_ms()
        self._setup_processor()
        self._system_stats_stop = threading.Event()
        self._system_stats_thread = system_stats.start_poller(
            self._system_stats_stop,
            lambda stats: self._main_thread.post(lambda: self._control_bar.set_system_stats(stats)),
        )

        self.protocol("WM_DELETE_WINDOW", self._on_close_requested)
        self.bind("<Unmap>", self._on_unmap, add="+")

        self.after_idle(self._restore_saved_queue)
        self.after_idle(self._log_startup_timing)
        self.after_idle(self._start_cuda_warmup)
        self.after_idle(self._warn_if_windhawk_loaded)
        self.after_idle(self._warn_if_installed_on_hard_disk)
        self.after_idle(self._warn_if_older_version_files_left)
        self.after_idle(self._start_update_check)
        self.after_idle(license_status.start_pc_license_refresh)

        if self._preset_manager.get_system_check_passed_version() != __version__:
            self.after(100, lambda: self._show_system_check(quit_when_blocked=True))
            
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
        self._header_bar = HeaderBar(
            self,
            self._main_thread,
            on_options=self._show_options,
            on_open_player=self._open_video_player,
            on_system_check=lambda: self._show_system_check(quit_when_blocked=False),
        )
        self._header_bar.pack(fill="x")
        self._build_footer()
        self._build_main_body()
        
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

        from jasna.engine_paths import default_restoration_model_path
        from jasna.gui.dlss5_runtime import Dlss5Runtime
        from jasna.gui.ltx_models import LtxModels

        self._ltx_models = LtxModels(
            default_restoration_model_path("ltx"), self._main_thread, self._on_ltx_models_changed
        )
        self._dlss5_runtime = Dlss5Runtime(self._main_thread, self._update_start_button_state)
        self._settings_panel = SettingsPanel(self._workspace, self._preset_manager, ltx_models=self._ltx_models)
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
        
        self._queue_panel.set_on_edit_segments(self._edit_segments)
        self._queue_panel.set_initial_output(
            self._preset_manager.get_last_output_folder(),
            self._preset_manager.get_last_output_pattern(),
        )
        self._queue_panel.set_on_output_changed(self._on_output_changed)
        self._queue_panel.set_on_play(self._open_video_player)
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
            on_resume_conflict=self._confirm_resume_restart,
        )

    def _confirm_resume_restart(self, directory):
        from concurrent.futures import Future
        from tkinter import messagebox

        answer = Future()

        def ask():
            if not answer.set_running_or_notify_cancel():
                return
            self._show_from_tray()
            answer.set_result(messagebox.askyesno(
                t("resume_restart_title"), t("resume_restart_confirm"), parent=self,
            ))

        self._main_thread.post(ask)
        return answer

    def _log_startup_timing(self):
        logger.info(
            "startup: first paint %.0f ms (pre-window %.0f ms, ui build %.0f ms)",
            startup_timing.elapsed_ms(),
            self._t_init_start,
            self._t_ui_built - self._t_init_start,
        )

    def _warn_if_windhawk_loaded(self):
        if not os_utils.windows_module_loaded(_WINDHAWK_MODULE):
            return
        from tkinter import messagebox

        message = t("windhawk_warning_body")
        self._log_panel.warning(message)
        messagebox.showwarning(t("windhawk_warning_title"), message, parent=self)

    def _warn_if_installed_on_hard_disk(self):
        if system_requirements.install_on_hard_disk():
            self._log_panel.warning(t("hard_disk_warning"))

    def _warn_if_older_version_files_left(self):
        leftover = system_requirements.leftover_install_dll()
        if leftover is None:
            return
        from tkinter import messagebox

        message = t("old_files_warning_body", file=leftover)
        self._log_panel.warning(message)
        messagebox.showwarning(t("old_files_warning_title"), message, parent=self)

    def _start_cuda_warmup(self):
        system_checks.start_cuda_warmup(
            lambda vendor, blackwell, frame_generation: self._main_thread.post(
                lambda: self._set_gpu_support(vendor, blackwell, frame_generation)
            )
        )

    def _set_gpu_support(self, vendor: AcceleratorVendor, blackwell: bool, frame_generation: bool) -> None:
        self._settings_panel.set_gpu_support(vendor=vendor, blackwell=blackwell, frame_generation=frame_generation)

    def _stop_system_stats_poller(self):
        self._system_stats_stop.set()
        if self._system_stats_thread:
            self._system_stats_thread.join(timeout=1.0)
            self._system_stats_thread = None

    def _on_ltx_models_changed(self):
        self._settings_panel.refresh_ltx_models()
        self._update_start_button_state()

    def _on_close_requested(self):
        if read_settings().get("close_to_tray", False) and self._hide_to_tray(needs_menu=True):
            return
        self._request_quit()

    def _request_quit(self):
        if self._processor and self._processor.is_running():
            CloseWhileProcessingDialog(self, on_minimize=self.iconify, on_quit=self._on_close)
            return
        self._on_close()

    def _on_close(self):
        if self._ltx_models.downloading or self._dlss5_runtime.downloading:
            from tkinter import messagebox

            title = t("dlss5_download_title") if self._dlss5_runtime.downloading else t("ltx_download_title")
            if not messagebox.askyesno(title, t("ltx_download_close_confirm")):
                return
        if self._video_player_dialog is not None:
            self._closing_after_player = True
            self._video_player_dialog.request_close()
            return
        if not self._close_segment_editor():
            return
        try:
            self._closing = True
            if self._processor:
                self._processor.stop()
                self._processor.join(timeout=5.0)
            self._flush_active_jobs_for_close()
            self._save_queue()
        finally:
            self._tray.hide()
            self._stop_system_stats_poller()
            self._main_thread.close()
            self.destroy()

    def _pause_active_job(self, job, job_id: int, now: float) -> None:
        start = self._job_start_times.pop(job_id, None)
        resumable = self._job_resumable.pop(job_id, False)
        if job is None:
            return
        if resumable:
            if start is not None:
                job.elapsed_seconds = (job.elapsed_seconds or 0.0) + max(0.0, now - start)
        else:
            job.elapsed_seconds = None

    def _flush_active_jobs_for_close(self) -> None:
        drain = getattr(self._main_thread, "drain_pending", None)
        if drain is not None:
            drain(ignore_tcl_error=True)
        if not self._job_start_times:
            return
        now = time.monotonic()
        jobs_by_id = {job.id: job for job in self._queue_panel.get_jobs()}
        for job_id in list(self._job_start_times):
            self._pause_active_job(jobs_by_id.get(job_id), job_id, now)
        
    def _show_options(self):
        OptionsDialog(self, tray.tray_support(), self._on_update_check_changed)

    def _start_update_check(self):
        activation_worker.start_update_check(
            lambda version, error: self._main_thread.post(lambda: self._update_check_ready(version, error))
        )

    def _update_check_ready(self, version, error):
        if error is not None:
            logger.warning("Could not check for a new version: %s", error)
        self._header_bar.show_update(version if read_settings().get(update_check.ENABLED_SETTING, True) else None)

    def _on_update_check_changed(self, enabled: bool):
        if enabled:
            self._start_update_check()
        else:
            self._header_bar.show_update(None)

    def _on_unmap(self, event):
        if event.widget is self and self.state() == "iconic" and read_settings().get("minimize_to_tray", False):
            self.after_idle(self._hide_to_tray)

    def _hide_to_tray(self, needs_menu: bool = False) -> bool:
        """Hide the window behind a tray icon; False (window untouched) when there is no tray."""
        support = tray.tray_support()
        if not support.available or (needs_menu and not support.has_menu):
            return False
        try:
            self._tray.show()
        except Exception:
            logger.warning("Could not show the tray icon; keeping the window open", exc_info=True)
            return False
        self.withdraw()
        return True

    def _show_from_tray(self):
        if not self._tray.shown:
            return
        self._tray.hide()
        self.deiconify()
        self.lift()
        self.focus_force()

    def _quit_from_tray(self):
        self._show_from_tray()
        self._request_quit()

    def _show_system_check(self, quit_when_blocked: bool):
        from jasna.gui.wizard import FirstRunWizard
        FirstRunWizard(
            self,
            on_complete=lambda can_continue, all_passed: self._on_system_check_complete(
                can_continue, all_passed, quit_when_blocked
            ),
        )

    def _on_system_check_complete(self, can_continue: bool, all_passed: bool, quit_when_blocked: bool):
        """Dismissing the first-run check quits the app; dismissing a re-run only closes the dialog."""
        if not can_continue:
            if quit_when_blocked:
                self._log_panel.error(t("wizard_log_blocked"))
                self._on_close()
            return
        if all_passed:
            self._preset_manager.set_system_check_passed_version(__version__)
            self._log_panel.info(t("wizard_log_ready"))
        else:
            self._log_panel.warning(t("wizard_warnings_only"))
            
    def _on_jobs_changed(self):
        jobs = self._queue_panel.get_jobs()
        if self._completed_job_elapsed:
            completed_ids = {job.id for job in jobs if job.status is JobStatus.COMPLETED}
            for job_id in [job.id for job in jobs if job.id in self._completed_job_elapsed and job.id not in completed_ids]:
                del self._completed_job_elapsed[job_id]
            self._queue_elapsed_seconds = sum(self._completed_job_elapsed.values())
        self._control_bar.set_queue_total(len(jobs))
        self._update_start_button_state()
        self._save_queue()

    def _restore_saved_queue(self):
        """Runs once the log panel shows logs, before anything may save over the saved queue."""
        self._queue_restored = True
        self._queue_panel.add_jobs(saved_queue.load_queue())

    def _save_queue(self):
        if self._queue_restored:
            saved_queue.save_queue(self._queue_panel.get_jobs())

    def _on_output_changed(self, folder: str, pattern: str):
        self._preset_manager.set_last_output_folder(folder)
        self._preset_manager.set_last_output_pattern(pattern)

    def _open_interactive_image_restore(self):
        if self._preview_gpu_busy or (
            self._processor is not None and self._processor.is_running()
        ):
            return
        from tkinter import filedialog

        files = filedialog.askopenfilenames(
            title=t("interactive_select_images"),
            filetypes=[
                (t("file_type_image"), file_dialog_patterns(IMAGE_EXTENSIONS)),
                (t("file_type_all"), "*.*"),
            ],
        )
        if not files:
            return

        from jasna.gui.interactive_image_restore import InteractiveImageRestoreDialog

        self._set_preview_gpu_busy(InteractiveImageRestoreDialog, True)
        try:
            InteractiveImageRestoreDialog(
                self,
                [Path(f) for f in files],
                self._settings_panel.get_settings(),
                self._queue_panel.get_output_folder(),
                self._queue_panel.get_output_pattern(),
                on_log=lambda level, message: self._log_panel.add_log(level, message),
                on_stopped=lambda: self._set_preview_gpu_busy(InteractiveImageRestoreDialog, False),
            )
        except Exception:
            self._set_preview_gpu_busy(InteractiveImageRestoreDialog, False)
            raise

    def _open_video_player(self, path: Path | None = None):
        if self._preview_gpu_busy or (
            self._processor is not None and self._processor.is_running()
        ):
            return
        from jasna.gui.video_player import VideoPlayerDialog

        self._set_preview_gpu_busy(VideoPlayerDialog, True)
        try:
            self._video_player_dialog = VideoPlayerDialog(
                self,
                self._settings_panel.get_settings(),
                initial_path=path,
                on_closed=self._video_player_closed,
            )
        except Exception:
            self._video_player_dialog = None
            self._set_preview_gpu_busy(VideoPlayerDialog, False)
            raise

    def _edit_segments(self, job) -> None:
        if self._segment_editor is not None and self._segment_editor.winfo_exists():
            self._segment_editor.lift()
            self._segment_editor.focus_force()
            return
        self._segment_editor = None
        try:
            from jasna.gui.segment_editor import SegmentEditor

            self._segment_editor = SegmentEditor(
                self,
                job,
                self._settings_panel.get_settings,
                lambda: self._processor is not None and self._processor.is_running(),
                self._set_preview_gpu_busy,
                self._settings_panel.ltx_unavailable_reason(),
                lambda segments: self._queue_panel.on_segments_saved(job, segments),
                self._segment_editor_closed,
            )
        except Exception as exc:
            from tkinter import messagebox

            self._segment_editor = None
            logger.warning("Failed to open segment editor", exc_info=True)
            messagebox.showerror(t("segments_title"), str(exc), parent=self)

    def _segment_editor_closed(self) -> None:
        self._segment_editor = None

    def _close_segment_editor(self) -> bool:
        """Closes the open segment editor the way its own Close does and waits for its GPU
        workers to stop; False when the user keeps the editor open."""
        editor = self._segment_editor
        if editor is None:
            return True
        editor.request_close()
        if self._segment_editor is not None:
            return False
        editor.join_workers(timeout=10.0)
        return True

    def _video_player_closed(self):
        from jasna.gui.video_player import VideoPlayerDialog

        self._video_player_dialog = None
        self._set_preview_gpu_busy(VideoPlayerDialog, False)
        if self._closing_after_player:
            self._closing_after_player = False
            self.after(0, self._on_close)
        
    def _update_start_button_state(self):
        jobs = self._queue_panel.get_jobs()
        downloading = self._ltx_models.downloading or self._dlss5_runtime.downloading
        can_start = bool(jobs) and not self._preview_gpu_busy and not downloading
        if can_start:
            self._control_bar.set_start_enabled(True)
        elif self._dlss5_runtime.downloading:
            self._control_bar.set_start_enabled(False, t("dlss5_downloading", percent=self._dlss5_runtime.percent))
        elif downloading:
            self._control_bar.set_start_enabled(False, t("ltx_downloading_short"))
        elif self._preview_gpu_busy:
            self._control_bar.set_start_enabled(False, t("segments_restore_restoring"))
        else:
            self._control_bar.set_start_enabled(False, t("toast_no_files"))
        self._update_video_player_button_state()

    def _update_video_player_button_state(self) -> None:
        processing = self._processor is not None and self._processor.is_running()
        disabled = self._preview_gpu_busy or processing
        self._header_bar.set_player_enabled(not disabled)

    @property
    def _preview_gpu_busy(self) -> bool:
        return bool(self._preview_gpu_owners)

    def _set_preview_gpu_busy(self, owner: object, busy: bool) -> None:
        """Record that ``owner`` (a preview, scan or player, called from any thread) holds or gave
        back the GPU; it stays busy while any owner holds it."""
        if busy:
            self._preview_gpu_owners.add(owner)
        else:
            self._preview_gpu_owners.discard(owner)
        self._main_thread.post(self._update_start_button_state)
        
    def _on_start(self):
        if self._preview_gpu_busy or (
            self._processor is not None and self._processor.is_running()
        ):
            return
        jobs = self._queue_panel.get_jobs()
        continuing = self._processor is not None and self._processor.was_stopped and any(
            job.status is JobStatus.PENDING for job in jobs
        )
        if continuing:
            jobs = [job for job in jobs if job.status is JobStatus.PENDING]
        if not jobs:
            self._log_panel.warning(t("toast_no_files"))
            return
            
        output_folder = self._queue_panel.get_output_folder()
        output_pattern = self._queue_panel.get_output_pattern()
        settings = self._settings_panel.get_settings()

        blocked = start_blocker(
            settings,
            jobs,
            ltx_ready=self._settings_panel.ltx_unavailable_reason() is None,
            ltx_runs_on_this_pc=self._settings_panel.ltx_runs_on_this_pc(),
            ltx_downloadable=self._ltx_models.state.downloadable,
            dlss5_installed=self._dlss5_runtime.installed,
            ltx_license=lambda: read_ltx_license(self._ltx_models.directory, settings),
        )
        if blocked is not None:
            self._resolve_start_block(blocked, settings)
            return

        try:
            from jasna.gui.engine_preflight import first_run_warning, run_engine_preflight

            preflight = run_engine_preflight(settings)
            if preflight.missing:
                from tkinter import messagebox

                msg = first_run_warning([requirement.key for requirement in preflight.missing])
                messagebox.showinfo(t("engine_first_run_title"), msg)
                self._log_panel.warning(msg)
        except Exception as e:
            self._log_panel.warning(f"Engine preflight warning failed: {e}")

        trial = settings.ltx_trial and any_job_uses_ltx(jobs, settings)
        if trial:
            Toast.show(self, t("toast_ltx_trial"), "warning", rely=0.9)
            self._log_panel.warning(t("toast_ltx_trial"))
        self._control_bar.set_trial(trial)
        if not continuing:
            self._queue_panel.reset_jobs_for_run()
            self._completed_job_elapsed.clear()
            self._queue_elapsed_seconds = 0.0
        
        self._header_bar.set_status("PROCESSING", Colors.STATUS_PROCESSING)
        self._control_bar.set_resumable(any(job_is_resumable(job, settings) for job in jobs))
        self._control_bar.set_running(True)
        
        # Disable settings and output controls while running
        self._settings_panel.set_enabled(False)
        self._queue_panel.set_output_enabled(False)
        self._queue_panel.set_running(True)
        
        self._log_panel.info("Processing started by user")
        self._log_panel.info(f"Output folder: {output_folder}")
        self._log_panel.info(f"Output pattern: {output_pattern}")
        self._log_panel.info(f"Files queued: {len(jobs)}")

        self._job_start_times.clear()
        self._job_resumable.clear()
        self._processor.start(
            self._queue_panel.job_queue,
            settings,
            output_folder,
            output_pattern,
        )
        self._update_video_player_button_state()
                
    def _resolve_start_block(self, blocked: StartBlocked, settings: AppSettings) -> None:
        from tkinter import messagebox

        match blocked.block:
            case StartBlock.DOWNLOAD_LTX:
                self._ltx_models.ensure(settings.ltx_model, settings.ltx_fast, on_ready=self._on_start)
            case StartBlock.DOWNLOAD_DLSS5:
                self._dlss5_runtime.ensure(on_ready=self._on_start)
            case StartBlock.LTX_LOCKED:
                messagebox.showwarning(t("ltx_blocked_title"), t("ltx_blocked_text"))
            case StartBlock.OFFER_LTX_TRIAL:
                if messagebox.askyesno(t("ltx_license_trial_title"), t("ltx_license_trial_confirm")):
                    self._settings_panel.set_ltx_trial(True)
                    self._on_start()
            case StartBlock.INVALID:
                msg = t("error_cannot_start") + "\n\n" + "\n".join(f"- {e}" for e in blocked.errors)
                self._log_panel.error(msg)
                messagebox.showerror(t("error_start_title"), msg)

    def _on_stop(self):
        if self._processor is None or not self._processor.is_running():
            return
        self._processor.stop()
        self._log_panel.info("Processing stopped by user")

        self._header_bar.set_status("IDLE", Colors.STATUS_PENDING)
        self._control_bar.reset()
        # Start stays disabled until the worker thread has finished unwinding;
        # _handle_complete re-enables it.
        self._control_bar.set_start_enabled(False)
        self._enable_run_settings()

    def _enable_run_settings(self):
        self._settings_panel.set_enabled(True)
        self._queue_panel.set_output_enabled(True)

    def _toggle_logs(self):
        self._logs_visible = not self._logs_visible
        if self._logs_visible:
            self._log_panel.pack(fill="x", side="bottom")
        else:
            self._log_panel.pack_forget()
            
    def _on_processor_progress(self, update: ProgressUpdate):
        self._main_thread.post(lambda: self._handle_progress(update))
        
    def _handle_progress(self, update: ProgressUpdate):
        jobs = self._queue_panel.get_jobs()
        job_id = update.job_id
        job = next((item for item in jobs if item.id == job_id), None)
        now = time.monotonic()

        if update.status == JobStatus.PROCESSING:
            if update.resumable is not None:
                self._job_resumable[job_id] = update.resumable
            if update.reset_elapsed:
                if job is not None:
                    job.elapsed_seconds = None
                self._job_start_times[job_id] = now
            else:
                self._job_start_times.setdefault(job_id, now)
        elif update.status == JobStatus.PENDING:
            self._pause_active_job(job, job_id, now)
        elif update.status in (JobStatus.ERROR, JobStatus.SKIPPED):
            self._job_start_times.pop(job_id, None)
            resumable = self._job_resumable.pop(job_id, False)
            if job is not None and not resumable:
                job.elapsed_seconds = None

        job_elapsed: float | None = None
        if update.status == JobStatus.COMPLETED:
            start = self._job_start_times.pop(job_id, None)
            self._job_resumable.pop(job_id, None)
            interval = max(0.0, now - start) if start is not None else 0.0
            prior = job.elapsed_seconds if job is not None and job.elapsed_seconds is not None else 0.0
            job_elapsed = prior + interval
            self._completed_job_elapsed[job_id] = job_elapsed
            self._queue_elapsed_seconds = sum(self._completed_job_elapsed.values())

        if self._closing:
            if job is not None:
                job.status = update.status
                job.progress = update.progress / 100.0
            return

        if update.license_problem:
            self._show_from_tray()
        if update.license_problem == "unofficial_key":
            XianyuQrDialog(self, unofficial_notice=True)
        elif update.license_problem:
            from tkinter import messagebox

            messagebox.showerror(t("license_problem_title"), update.message, parent=self)

        if update.status == JobStatus.PROCESSING:
            if update.resumable is not None:
                self._control_bar.set_resumable(update.resumable)
            # Resolve current queue position for control bar display
            filename = ""
            queue_current = 0
            for i, j in enumerate(jobs):
                if j.id == job_id:
                    filename = j.filename
                    queue_current = i + 1
                    break
            self._control_bar.update_progress(
                filename=filename,
                percent=update.progress,
                fps=update.fps,
                eta_seconds=update.eta_seconds,
                queue_current=queue_current,
                queue_total=len(jobs),
                stage=update.stage,
            )
            self._tray.set_title(f"Jasna - {update.progress:.0f}% ({queue_current}/{len(jobs)})")
            self._queue_panel.set_running(True, processing_job_id=job_id)
        self._queue_panel.update_job_status(
            job_id,
            update.status,
            update.progress / 100.0,
            update.fps,
            update.eta_seconds,
            elapsed_seconds=job_elapsed,
            stage=update.stage,
        )
        if update.status is not JobStatus.PROCESSING:
            self._save_queue()
            
    def _on_processor_log(self, level: str, message: str):
        self._main_thread.post(lambda: None if self._closing else self._log_panel.add_log(level, message))
        
    def _on_processor_complete(self, queue_finished: bool):
        self._main_thread.post(lambda: self._handle_complete(queue_finished))

    def _handle_complete(self, queue_finished: bool):
        if self._closing:
            return
        self._header_bar.set_status("IDLE", Colors.STATUS_PENDING)
        self._tray.set_title("Jasna")
        if queue_finished:
            self._control_bar.set_completed(self._queue_elapsed_seconds)
            self._completed_job_elapsed.clear()
            self._queue_elapsed_seconds = 0.0
            self._log_panel.info("All jobs completed")
        else:
            self._control_bar.reset()
        self._update_start_button_state()
        self._enable_run_settings()
        self._queue_panel.set_running(False)
        if queue_finished:
            self._run_post_export_action()

    def _run_post_export_action(self):
        from jasna.post_export_action import (
            POST_EXPORT_ACTION_NONE,
            POST_EXPORT_ACTION_SHUTDOWN,
            run_post_export_action_safely,
        )

        settings = self._settings_panel.get_settings()
        action = settings.post_export_action
        if action == POST_EXPORT_ACTION_NONE:
            return
        if action == POST_EXPORT_ACTION_SHUTDOWN:
            self._log_panel.info(f"Shutting down in {_SHUTDOWN_COUNTDOWN_SECONDS} seconds")
            self._show_from_tray()
            ShutdownCountdownDialog(
                self,
                _SHUTDOWN_COUNTDOWN_SECONDS,
                on_expired=lambda: run_post_export_action_safely(action, "", self._log_panel.error),
                on_cancelled=lambda: self._log_panel.info("Shutdown cancelled by user"),
            )
            return
        self._log_panel.info(f"Running post-export action: {action}")
        run_post_export_action_safely(action, settings.post_export_command, self._log_panel.error)


def run_gui():
    """Entry point to run the GUI application."""
    from jasna.bootstrap import configure_process
    configure_process()

    # Set up basic logging - will be connected to GUI after app creation
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.StreamHandler()]  # Temporary console output
    )
    from jasna.media.ffmpeg_log import log_ffmpeg_errors
    log_ffmpeg_errors()

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
        return
    
    install_gui_logging(app._main_thread, app._log_panel)

    app.mainloop()

    os_utils.exit_without_joining_threads()
