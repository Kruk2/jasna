"""Queue panel - left side job list management."""

import customtkinter as ctk
from pathlib import Path
from tkinter import filedialog

from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.gui.models import JobItem, JobStatus
from jasna.gui.components import JobListItem
from jasna.gui.locales import t


class QueuePanel(ctk.CTkFrame):
    """Left panel containing the job queue and output settings."""
    
    def __init__(self, master, **kwargs):
        super().__init__(
            master,
            fg_color=Colors.BG_PANEL,
            corner_radius=0,
            width=Sizing.QUEUE_PANEL_WIDTH,
            **kwargs
        )
        self.pack_propagate(False)
        
        self._jobs: list[JobItem] = []
        self._job_widgets: list[JobListItem] = []
        self._on_jobs_changed: callable = None
        
        self._build_toolbar()
        self._build_list_area()
        self._build_footer()
        
    def _build_toolbar(self):
        toolbar = ctk.CTkFrame(self, fg_color="transparent", height=48)
        toolbar.pack(fill="x", padx=Sizing.PADDING_MEDIUM, pady=(Sizing.PADDING_MEDIUM, 0))
        toolbar.pack_propagate(False)
        
        self._add_files_btn = ctk.CTkButton(
            toolbar,
            text=t("btn_add_files"),
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
            height=Sizing.BUTTON_HEIGHT,
            command=self._on_add_files,
        )
        self._add_files_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))
        
        self._add_folder_btn = ctk.CTkButton(
            toolbar,
            text="📂",
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            fg_color=Colors.BG_CARD,
            hover_color=Colors.BORDER_LIGHT,
            text_color=Colors.TEXT_PRIMARY,
            width=40,
            height=Sizing.BUTTON_HEIGHT,
            command=self._on_add_folder,
        )
        self._add_folder_btn.pack(side="right")
        
    def _build_list_area(self):
        self._list_frame = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            scrollbar_button_color=Colors.BORDER_LIGHT,
            scrollbar_button_hover_color=Colors.BORDER_LIGHT,
        )
        self._list_frame.pack(fill="both", expand=True, padx=Sizing.PADDING_MEDIUM, pady=Sizing.PADDING_SMALL)
        
        self._empty_state = ctk.CTkFrame(
            self._list_frame,
            fg_color="transparent",
            border_color=Colors.BORDER_LIGHT,
            border_width=2,
            corner_radius=Sizing.BORDER_RADIUS,
        )
        self._empty_label = ctk.CTkLabel(
            self._empty_state,
            text=t("queue_empty"),
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            text_color=Colors.TEXT_PRIMARY,
        )
        self._empty_label.pack(padx=40, pady=60)
        self._empty_state.pack(fill="x", pady=20)
        
    def _build_footer(self):
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill="x", side="bottom", padx=Sizing.PADDING_MEDIUM, pady=Sizing.PADDING_MEDIUM)
        
        # Queue count and clear button
        count_row = ctk.CTkFrame(footer, fg_color="transparent")
        count_row.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        
        self._queue_count = ctk.CTkLabel(
            count_row,
            text=t("items_queued", count=0),
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_PRIMARY,
        )
        self._queue_count.pack(side="left")
        
        self._clear_btn = ctk.CTkButton(
            count_row,
            text=t("btn_clear"),
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            fg_color=Colors.BG_CARD,
            hover_color=Colors.BORDER_LIGHT,
            text_color=Colors.TEXT_PRIMARY,
            height=28,
            width=70,
            command=self._on_clear_queue,
        )
        self._clear_btn.pack(side="right")
        
        self._clear_completed_btn = ctk.CTkButton(
            count_row,
            text=t("btn_clear_completed"),
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            fg_color=Colors.BG_CARD,
            hover_color=Colors.BORDER_LIGHT,
            text_color=Colors.TEXT_PRIMARY,
            height=28,
            width=100,
            command=self._on_clear_completed,
        )
        self._clear_completed_btn.pack(side="right", padx=(0, 6))
        
        # Output location section
        output_label = ctk.CTkLabel(
            footer,
            text=t("output_location"),
            font=(Fonts.FAMILY, Fonts.SIZE_TINY, "bold"),
            text_color=Colors.TEXT_PRIMARY,
            anchor="w",
        )
        output_label.pack(fill="x", pady=(Sizing.PADDING_SMALL, 4))
        
        output_row = ctk.CTkFrame(footer, fg_color="transparent")
        output_row.pack(fill="x")
        
        self._output_entry = ctk.CTkEntry(
            output_row,
            placeholder_text=t("same_as_input"),
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            fg_color=Colors.BG_CARD,
            border_color=Colors.BORDER,
            text_color=Colors.TEXT_PRIMARY,
            height=Sizing.INPUT_HEIGHT,
            state="disabled",
        )
        self._output_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        
        self._output_browse_btn = ctk.CTkButton(
            output_row,
            text="📂",
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            fg_color=Colors.BG_CARD,
            hover_color=Colors.BORDER_LIGHT,
            text_color=Colors.TEXT_PRIMARY,
            width=32,
            height=Sizing.INPUT_HEIGHT,
            command=self._on_browse_output,
        )
        self._output_browse_btn.pack(side="right")
        
        # Output pattern
        self._pattern_entry = ctk.CTkEntry(
            footer,
            placeholder_text="{original}_restored.mp4",
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            fg_color=Colors.BG_CARD,
            border_color=Colors.BORDER,
            text_color=Colors.TEXT_PRIMARY,
            height=Sizing.INPUT_HEIGHT,
        )
        self._pattern_entry.pack(fill="x", pady=(4, 0))
        self._pattern_entry.insert(0, "{original}_restored.mp4")
        self._pattern_entry.bind("<KeyRelease>", lambda e: self._refresh_conflicts())
        
    def _on_add_files(self):
        files = filedialog.askopenfilenames(
            title=t("select_video_files"),
            filetypes=[
                ("Video files", "*.mp4 *.mkv *.avi *.mov *.wmv *.flv *.webm"),
                ("All files", "*.*"),
            ]
        )
        for f in files:
            self.add_job(Path(f))
            
    def _on_add_folder(self):
        folder = filedialog.askdirectory(title=t("select_folder"))
        if folder:
            folder_path = Path(folder)
            video_exts = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"}
            for f in folder_path.rglob("*"):
                if f.suffix.lower() in video_exts:
                    self.add_job(f)
                    
    def _on_browse_output(self):
        folder = filedialog.askdirectory(title=t("select_output_folder"))
        if folder:
            self._output_entry.configure(state="normal")
            self._output_entry.delete(0, "end")
            self._output_entry.insert(0, folder)
            self._refresh_conflicts()
            
    def _on_clear_queue(self):
        self._jobs.clear()
        for w in self._job_widgets:
            w.destroy()
        self._job_widgets.clear()
        self._update_empty_state()
        self._update_count()
        if self._on_jobs_changed:
            self._on_jobs_changed()
            
    def _on_clear_completed(self):
        """Clear completed, errored, and skipped jobs from the queue."""
        completed_statuses = {JobStatus.COMPLETED, JobStatus.ERROR, JobStatus.SKIPPED}
        indices_to_remove = [
            i for i, job in enumerate(self._jobs)
            if job.status in completed_statuses
        ]
        # Remove in reverse order to preserve indices
        for i in reversed(indices_to_remove):
            self._jobs.pop(i)
            widget = self._job_widgets.pop(i)
            widget.destroy()
        self._update_empty_state()
        self._update_count()
        if self._on_jobs_changed:
            self._on_jobs_changed()
            
    def _update_empty_state(self):
        if self._jobs:
            self._empty_state.pack_forget()
        else:
            self._empty_state.pack(fill="x", pady=20)
            
    def _update_count(self):
        count = len(self._jobs)
        self._queue_count.configure(text=t("items_queued", count=count))
        
    def add_job(self, path: Path):
        if any(j.path == path for j in self._jobs):
            return
            
        job = JobItem(path=path)
        self._jobs.append(job)
        
        # Check for output file conflict
        output_path = self._get_output_path(path)
        job.has_conflict = output_path.exists() if output_path else False
        
        widget = JobListItem(
            self._list_frame,
            filename=job.filename,
            duration=job.duration_str or "",
            status=t("job_pending"),
            on_remove=lambda j=job: self._remove_job(j),
            on_drag_start=self._on_widget_drag_start,
            on_drag_move=self._on_widget_drag_move,
            on_drag_end=self._on_widget_drag_end,
        )
        widget.pack(fill="x", pady=(0, 4))
        self._job_widgets.append(widget)
        
        # Show conflict indicator if needed
        if job.has_conflict:
            widget.set_conflict(True, t("conflict_tooltip"))
        
        self._update_empty_state()
        self._update_count()
        if self._on_jobs_changed:
            self._on_jobs_changed()
            
    def _get_output_path(self, input_path: Path) -> Path | None:
        """Get the output path for a given input file based on current settings."""
        output_folder = self._output_entry.get()
        pattern = self._pattern_entry.get() or "{original}_restored.mkv"
        
        if not output_folder:
            # Use same folder as input
            output_folder = str(input_path.parent)
            
        original_stem = input_path.stem
        output_name = pattern.replace("{original}", original_stem)
        return Path(output_folder) / output_name
        
        self._update_empty_state()
        self._update_count()
        if self._on_jobs_changed:
            self._on_jobs_changed()
            
    def _remove_job(self, job: JobItem):
        if job in self._jobs:
            idx = self._jobs.index(job)
            self._jobs.remove(job)
            self._job_widgets[idx].destroy()
            self._job_widgets.pop(idx)
            self._update_empty_state()
            self._update_count()
            if self._on_jobs_changed:
                self._on_jobs_changed()
        
                
    def get_jobs(self) -> list[JobItem]:
        # Return a shallow copy for callers that should not modify the queue
        return self._jobs.copy()

    def get_jobs_ref(self) -> list[JobItem]:
        """Return the live jobs list reference so callers (like the Processor)
        can observe additions/removals while processing is running.
        Use with care: this returns the internal list, not a defensive copy."""
        return self._jobs
        
    def get_output_folder(self) -> str:
        return self._output_entry.get() or ""
        
    def get_output_pattern(self) -> str:
        return self._pattern_entry.get() or "{original}_restored.mkv"
        
    def set_on_jobs_changed(self, callback: callable):
        self._on_jobs_changed = callback
        
    def update_job_status(self, index: int, status: JobStatus, progress: float = 0.0, fps: float = 0.0, eta_seconds: float = 0.0):
        if 0 <= index < len(self._job_widgets):
            widget = self._job_widgets[index]
            job = self._jobs[index]
            job.status = status
            job.progress = progress
            
            status_map = {
                JobStatus.PENDING: (t("job_pending"), "", Colors.STATUS_PENDING),
                JobStatus.PROCESSING: (t("job_processing"), "○", Colors.STATUS_PROCESSING),
                JobStatus.COMPLETED: (t("job_completed"), "✓", Colors.STATUS_COMPLETED),
                JobStatus.ERROR: (t("job_error"), "✕", Colors.STATUS_ERROR),
                JobStatus.PAUSED: (t("job_paused"), "⏸", Colors.STATUS_PAUSED),
                JobStatus.SKIPPED: (t("job_skipped"), "⊘", Colors.STATUS_CONFLICT),
            }
            text, icon, color = status_map.get(status, ("", "", Colors.STATUS_PENDING))
            widget.set_status(text, icon, color)
            
            if status == JobStatus.PROCESSING:
                widget.set_progress(progress)
                widget.set_fps_eta(fps=fps, eta_seconds=eta_seconds)
            else:
                widget.hide_progress()
                
            # Hide conflict indicator once processing starts
            if status != JobStatus.PENDING:
                widget.set_conflict(False)
                
    def _refresh_conflicts(self):
        """Re-check all jobs for output file conflicts."""
        for job, widget in zip(self._jobs, self._job_widgets):
            if job.status == JobStatus.PENDING:
                output_path = self._get_output_path(job.path)
                job.has_conflict = output_path.exists() if output_path else False
                widget.set_conflict(job.has_conflict, t("conflict_tooltip") if job.has_conflict else "")
    
    def set_output_enabled(self, enabled: bool):
        """Enable or disable output location controls (but not queue add/remove)."""
        state = "normal" if enabled else "disabled"
        self._output_browse_btn.configure(state=state)
        self._pattern_entry.configure(state=state)
        self._clear_btn.configure(state=state)
        self._clear_completed_btn.configure(state=state)

    # --- Drag & reorder support ---
    def _on_widget_drag_start(self, widget: 'JobListItem', event):
        try:
            widget.lift()
        except Exception:
            pass
        try:
            widget.configure(cursor="hand2")
        except Exception:
            pass

    def _on_widget_drag_move(self, widget: 'JobListItem', event):
        # Compute pointer y relative to list frame and determine new index
        lf = self._list_frame
        try:
            y = event.y_root - lf.winfo_rooty()
        except Exception:
            return

        # Compute new index among widgets based on center positions
        new_index = 0
        for w in self._job_widgets:
            if w is widget:
                continue
            center = (w.winfo_rooty() + w.winfo_height() / 2) - lf.winfo_rooty()
            if y > center:
                new_index += 1

        try:
            current_index = self._job_widgets.index(widget)
        except ValueError:
            return

        if new_index != current_index:
            # Remove and reinsert data + widget and repack
            job = self._jobs.pop(current_index)
            self._job_widgets.pop(current_index)
            self._jobs.insert(new_index, job)
            self._job_widgets.insert(new_index, widget)
            # Repack in order
            for w in self._job_widgets:
                w.pack_forget()
            for w in self._job_widgets:
                w.pack(fill="x", pady=(0, 4))
            if self._on_jobs_changed:
                self._on_jobs_changed()

    def _on_widget_drag_end(self, widget: 'JobListItem', event):
        try:
            widget.configure(cursor="")
        except Exception:
            pass
        for w in self._job_widgets:
            w.pack_forget()
        for w in self._job_widgets:
            w.pack(fill="x", pady=(0, 4))
        if self._on_jobs_changed:
            self._on_jobs_changed()

    def set_running(self, running: bool, processing_index: int | None = None):
        """Set queue running state.

        When running=True the panel is visually dimmed and controls are
        disabled except the add buttons and removing jobs (except the
        currently processing item which is protected).
        """
        if running:
            # Disable controls we don't want interactive while running
            self._clear_btn.configure(state="disabled")
            self._clear_completed_btn.configure(state="disabled")
            self._output_browse_btn.configure(state="disabled")
            self._pattern_entry.configure(state="disabled")
            # Allow adding files/folders
            self._add_files_btn.configure(state="normal")
            self._add_folder_btn.configure(state="normal")
            # Per-item removability: protect processing index
            for i, widget in enumerate(self._job_widgets):
                widget.set_removable(i != processing_index)
        else:
            # Restore normal appearance and enable controls
            self._clear_btn.configure(state="normal")
            self._clear_completed_btn.configure(state="normal")
            self._output_browse_btn.configure(state="normal")
            self._pattern_entry.configure(state="normal")
            self._add_files_btn.configure(state="normal")
            self._add_folder_btn.configure(state="normal")
            for widget in self._job_widgets:
                widget.set_removable(True)
