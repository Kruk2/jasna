"""Main Jasna GUI application window."""

import customtkinter as ctk
import logging
from pathlib import Path
import threading

from jasna import __version__
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.gui.components import StatusPill, BuyMeCoffeeButton, Toast
from jasna.gui.queue_panel import QueuePanel
from jasna.gui.settings_panel import SettingsPanel
from jasna.gui.control_bar import ControlBar
from jasna.gui.log_panel import LogPanel
from jasna.gui.wizard import FirstRunWizard
from jasna.gui.processor import Processor, ProgressUpdate
from jasna.gui.models import JobStatus
from jasna.gui.validation import validate_gui_start
from jasna.gui.locales import get_locale, t, LANGUAGE_NAMES
from jasna.gui.system_stats import read_system_stats


class JasnaApp(ctk.CTk):
    """Main application window for Jasna GUI."""
    
    def __init__(self, skip_wizard: bool = False):
        super().__init__()
        
        self.title("Jasna GUI")
        self.geometry("1200x880")
        self.minsize(1000, 660)
        self.configure(fg_color=Colors.BG_MAIN)
        
        # Center on screen
        self.update_idletasks()
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        x = (screen_w - 1200) // 2
        y = (screen_h - 880) // 2
        self.geometry(f"1200x880+{x}+{y}")
        
        # Set appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self._logs_visible = False
        self._processor: Processor | None = None

        self._system_stats_stop = threading.Event()
        self._system_stats_thread: threading.Thread | None = None
        
        self._build_ui()
        self._setup_processor()
        self._start_system_stats_poller()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        
        if not skip_wizard:
            self.after(100, self._show_wizard)
            
    def _build_ui(self):
        self._build_header()
        self._build_main_body()
        self._build_footer()
        
    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=Colors.BG_PANEL, height=Sizing.HEADER_HEIGHT, corner_radius=0)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        # Left: Logo and title
        left = ctk.CTkFrame(header, fg_color="transparent")
        left.pack(side="left", padx=Sizing.PADDING_MEDIUM)
        
        logo = ctk.CTkLabel(
            left,
            text="J",
            font=(Fonts.FAMILY, 18, "bold"),
            text_color=Colors.PRIMARY,
            fg_color=Colors.PRIMARY_DARK,
            corner_radius=4,
            width=28,
            height=28,
        )
        logo.pack(side="left")
        
        title = ctk.CTkLabel(
            left,
            text="JASNA GUI",
            font=(Fonts.FAMILY, Fonts.SIZE_TITLE, "bold"),
            text_color=Colors.TEXT_PRIMARY,
        )
        title.pack(side="left", padx=(8, 4))
        
        version = ctk.CTkLabel(
            left,
            text=f"v{__version__}",
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.TEXT_MUTED,
        )
        version.pack(side="left", pady=(4, 0))
        
        # Center: Status pill
        self._status_pill = StatusPill(header)
        self._status_pill.place(relx=0.5, rely=0.5, anchor="center")
        
        # Right: Language, Help and About
        right = ctk.CTkFrame(header, fg_color="transparent")
        right.pack(side="right", padx=Sizing.PADDING_MEDIUM)
        
        # Language selector
        lang_label = ctk.CTkLabel(
            right,
            text="🌐",
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            text_color=Colors.TEXT_SECONDARY,
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
        
        # Buy Me a Coffee button
        self._bmc_btn = BuyMeCoffeeButton(right, compact=False)
        self._bmc_btn.pack(side="left", padx=(0, 12))
        
        self._help_btn = ctk.CTkButton(
            right,
            text=t("btn_help"),
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            fg_color="transparent",
            hover_color=Colors.BG_CARD,
            text_color=Colors.TEXT_SECONDARY,
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
            text_color=Colors.TEXT_SECONDARY,
            width=50,
            command=self._show_about,
        )
        self._about_btn.pack(side="left")
        
    def _build_main_body(self):
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True)
        
        # Left: Queue panel
        self._queue_panel = QueuePanel(body)
        self._queue_panel.pack(side="left", fill="y")
        self._queue_panel.set_on_jobs_changed(self._on_jobs_changed)
        
        # Separator
        sep = ctk.CTkFrame(body, fg_color=Colors.BORDER, width=1)
        sep.pack(side="left", fill="y")
        
        # Right: Settings panel
        self._settings_panel = SettingsPanel(body)
        self._settings_panel.pack(side="right", fill="both", expand=True)
        
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
        
    def _setup_processor(self):
        self._processor = Processor(
            on_progress=self._on_processor_progress,
            on_log=self._on_processor_log,
            on_complete=self._on_processor_complete,
        )

    def _start_system_stats_poller(self):
        if self._system_stats_thread and self._system_stats_thread.is_alive():
            return

        self._system_stats_stop.clear()

        def _loop():
            while not self._system_stats_stop.is_set():
                stats = read_system_stats()
                try:
                    self.after(0, lambda s=stats: self._control_bar.set_system_stats(s))
                except Exception:
                    return
                self._system_stats_stop.wait(1.5)

        self._system_stats_thread = threading.Thread(target=_loop, daemon=True)
        self._system_stats_thread.start()

    def _stop_system_stats_poller(self):
        self._system_stats_stop.set()
        if self._system_stats_thread:
            self._system_stats_thread.join(timeout=1.0)
            self._system_stats_thread = None

    def _on_close(self):
        try:
            if self._processor:
                self._processor.stop()
        finally:
            self._stop_system_stats_poller()
            self.destroy()
        
    def _show_wizard(self):
        FirstRunWizard(self, on_complete=self._on_wizard_complete)
        
    def _on_wizard_complete(self, all_passed: bool):
        if all_passed:
            self._log_panel.info("System check passed - ready to process")
        else:
            self._log_panel.warning("Some dependencies missing - check setup")
            
    def _on_jobs_changed(self):
        jobs = self._queue_panel.get_jobs()
        self._control_bar.update_progress(queue_total=len(jobs))
        
    def _show_toast(self, message: str, type_: str = "info"):
        """Show a toast notification."""
        toast = Toast(self, message, type_)
        toast.place(relx=0.5, rely=0.9, anchor="center")
        
    def _on_start(self):
        jobs = self._queue_panel.get_jobs()
        if not jobs:
            self._log_panel.warning("No files in queue")
            return
            
        output_folder = self._queue_panel.get_output_folder()
        if not output_folder:
            self._show_toast("Please select an output folder before starting", "warning")
            return
            
        settings = self._settings_panel.get_settings()
        output_pattern = self._queue_panel.get_output_pattern()

        errors = validate_gui_start(settings)
        if errors:
            from tkinter import messagebox

            msg = "Cannot start processing:\n\n" + "\n".join(f"- {e}" for e in errors)
            self._log_panel.error(msg)
            messagebox.showerror("Invalid TVAI configuration", msg)
            return
        
        self._status_pill.set_status("PROCESSING", Colors.STATUS_PROCESSING)
        self._control_bar.set_running(True)
        
        # Disable settings and output controls while running
        self._settings_panel.set_enabled(False)
        self._queue_panel.set_output_enabled(False)
        
        self._log_panel.info("Processing started by user")
        self._log_panel.info(f"Output folder: {output_folder}")
        self._log_panel.info(f"Output pattern: {output_pattern}")
        self._log_panel.info(f"Files queued: {len(jobs)}")

        # Start processor with a live reference to the queue so new items
        # added while processing will be picked up.
        jobs_ref = self._queue_panel.get_jobs_ref()
        self._processor.start(jobs_ref, settings, output_folder, output_pattern)
                
    def _on_stop(self):
        if self._processor:
            self._processor.stop()
            self._log_panel.info("Processing stopped by user")
            
        self._status_pill.set_status("IDLE", Colors.TEXT_SECONDARY)
        self._control_bar.reset()
        
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
        # Schedule UI update on main thread
        self.after(0, lambda: self._handle_progress(update))
        
    def _handle_progress(self, update: ProgressUpdate):
        jobs = self._queue_panel.get_jobs()
        
        self._queue_panel.update_job_status(
            update.job_index,
            update.status,
            update.progress / 100.0,
        )
        
        if update.status == JobStatus.PROCESSING:
            filename = jobs[update.job_index].filename if update.job_index < len(jobs) else ""
            self._control_bar.update_progress(
                filename=filename,
                percent=update.progress,
                fps=update.fps,
                eta_seconds=update.eta_seconds,
                queue_current=update.job_index + 1,
                queue_total=len(jobs),
            )
            # Mark queue as running and protect the processing job from removal
            try:
                self._queue_panel.set_running(True, processing_index=update.job_index)
            except Exception:
                pass
        # Update per-item status (include fps and eta)
        try:
            self._queue_panel.update_job_status(
                update.job_index,
                update.status,
                update.progress / 100.0,
                update.fps,
                update.eta_seconds,
            )
        except Exception:
            pass
            
    def _on_processor_log(self, level: str, message: str):
        self.after(0, lambda: self._log_panel.add_log(level, message))
        
    def _on_processor_complete(self):
        self.after(0, self._handle_complete)
        
    def _handle_complete(self):
        self._status_pill.set_status("IDLE", Colors.TEXT_SECONDARY)
        self._control_bar.reset()
        self._log_panel.info("All jobs completed")
        
        # Re-enable settings and output controls
        self._settings_panel.set_enabled(True)
        self._queue_panel.set_output_enabled(True)
        # Clear running mode
        try:
            self._queue_panel.set_running(False)
        except Exception:
            pass
        
    def _on_language_changed(self, lang_name: str):
        """Handle language selection change."""
        locale = get_locale()
        # Convert display name back to language code
        for code, name in LANGUAGE_NAMES.items():
            if name == lang_name:
                if code != locale.current_language:
                    locale.set_language(code)
                    self._refresh_ui_text()
                    # Show restart notice
                    from tkinter import messagebox
                    messagebox.showinfo(
                        "Language Changed",
                        "Please restart the application for full language change."
                    )
                break
                
    def _refresh_ui_text(self):
        """Refresh UI text after language change."""
        # Update header buttons
        self._help_btn.configure(text=t("btn_help"))
        self._about_btn.configure(text=t("btn_about"))
        # Note: Other panels would need their own refresh methods
        # For a full implementation, each panel should listen to locale changes
        
    def _show_help(self):
        import webbrowser
        webbrowser.open("https://github.com/Kruk2/jasna")
        
    def _show_about(self):
        dialog = ctk.CTkToplevel(self)
        dialog.title("About Jasna")
        dialog.geometry("400x250")
        dialog.resizable(False, False)
        dialog.configure(fg_color=Colors.BG_MAIN)
        dialog.transient(self)
        dialog.grab_set()
        
        # Center on parent
        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - 400) // 2
        y = self.winfo_y() + (self.winfo_height() - 250) // 2
        dialog.geometry(f"+{x}+{y}")
        
        ctk.CTkLabel(
            dialog,
            text="Jasna",
            font=(Fonts.FAMILY, 24, "bold"),
            text_color=Colors.TEXT_PRIMARY,
        ).pack(pady=(30, 8))
        
        ctk.CTkLabel(
            dialog,
            text=f"Version {__version__}",
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            text_color=Colors.TEXT_SECONDARY,
        ).pack()
        
        ctk.CTkLabel(
            dialog,
            text="JAV mosaic restoration tool",
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            text_color=Colors.TEXT_MUTED,
        ).pack(pady=(16, 8))
        
        ctk.CTkLabel(
            dialog,
            text="Inspired by Lada",
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_MUTED,
        ).pack()
        
        ctk.CTkButton(
            dialog,
            text="Close",
            fg_color=Colors.BG_CARD,
            hover_color=Colors.BORDER_LIGHT,
            text_color=Colors.TEXT_PRIMARY,
            command=dialog.destroy,
        ).pack(pady=30)


class GUILogHandler(logging.Handler):
    """Custom logging handler that forwards logs to the GUI log panel."""
    
    def __init__(self, log_panel: LogPanel):
        super().__init__()
        self._log_panel = log_panel
        
    def emit(self, record):
        try:
            msg = self.format(record)
            # Use after_idle to thread-safely update GUI
            self._log_panel.after_idle(self._log_panel.add_log, record.levelname, msg)
        except Exception:
            pass  # Ignore errors in log handler


def run_gui():
    """Entry point to run the GUI application."""
    import logging
    # Set up basic logging - will be connected to GUI after app creation
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(message)s',
        handlers=[logging.StreamHandler()]  # Temporary console output
    )
    
    app = JasnaApp()
    
    # Replace console handler with GUI handler for all jasna loggers
    gui_handler = GUILogHandler(app._log_panel)
    gui_handler.setFormatter(logging.Formatter('%(message)s'))
    
    # Set up root logger to capture all logs
    root_logger = logging.getLogger()
    root_logger.handlers = [gui_handler]
    root_logger.setLevel(logging.DEBUG)
    
    # Also capture jasna-specific logger
    jasna_logger = logging.getLogger("jasna")
    jasna_logger.handlers = [gui_handler]
    jasna_logger.setLevel(logging.DEBUG)
    jasna_logger.propagate = False
    
    app.mainloop()
