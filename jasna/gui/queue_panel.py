"""Queue panel - left side job list management."""

import logging

import customtkinter as ctk
from pathlib import Path
from tkinter import filedialog
from tkinter import messagebox

from tkinterdnd2 import DND_FILES

from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.gui.models import DEFAULT_OUTPUT_PATTERN, JobItem, JobStatus
from jasna.gui.components import AutoHidingScrollableFrame, Tooltip
from jasna.gui.job_list_item import JobListItem
from jasna.gui.icons import create_icon
from jasna.gui.file_actions import open_containing_folder
from jasna.gui.file_actions import open_file
from jasna.gui.locales import t
from jasna.gui.output_paths import OutputPathError, job_output_path

from jasna.media.media_files import (
    IMAGE_EXTENSIONS,
    MEDIA_EXTENSIONS,
    VIDEO_EXTENSIONS,
    file_dialog_patterns,
    folder_media_in_processing_order,
    folder_output_path,
    is_image,
)

logger = logging.getLogger(__name__)
from jasna.segments import SegmentRange

_PAGE_SIZE = 20


_PROCESSING_PHASE_KEYS = {
    "preparing": "job_phase_preparing",
    "coarse_scan": "job_phase_coarse_scan",
    "fine_scan": "job_phase_fine_scan",
    "source_copy": "job_phase_source_copy",
    "restoring": "job_phase_restoring",
    "finalizing": "job_phase_finalizing",
}


def processing_phase_text(phase: str) -> str:
    key = _PROCESSING_PHASE_KEYS.get(str(phase).strip().lower())
    return t(key) if key is not None else t("job_processing")


def _direct_file_input_root(path: Path) -> Path | None:
    """Return a useful root when a file is added without a folder import.

    Folder imports carry their selected root explicitly.  For an individual
    file, retain the immediate containing folder below the output root by
    using the parent of that folder.  Files directly under a filesystem root
    remain flat because there is no meaningful parent to recreate.
    """

    containing_directory = path.parent
    input_root = containing_directory.parent
    return None if input_root == containing_directory else input_root


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
        self._job_widgets: list[JobListItem | None] = []
        self._page = 0
        self._on_jobs_changed: callable = None
        self._on_output_changed: callable = None
        self._on_play: callable = None
        self._running = False
        self._processing_job_id: int | None = None
        self._segment_editor = None
        self._get_settings: callable = None
        self._is_gpu_busy: callable = None
        self._set_preview_gpu_busy: callable = None
        
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
            image=create_icon("folder", 18, Colors.TEXT_PRIMARY),
            compound="left",
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
            height=Sizing.BUTTON_HEIGHT,
            command=self._on_add_files,
        )
        self._add_files_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))
        
        self._add_folder_btn = ctk.CTkButton(
            toolbar,
            text="",
            image=create_icon("folder", 18, Colors.TEXT_PRIMARY),
            fg_color=Colors.BG_CARD,
            hover_color=Colors.BORDER_LIGHT,
            text_color=Colors.TEXT_PRIMARY,
            width=40,
            height=Sizing.BUTTON_HEIGHT,
            command=self._on_add_folder,
        )
        self._add_folder_btn.pack(side="right")
        
    def _build_list_area(self):
        self._list_frame = AutoHidingScrollableFrame(
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
            wraplength=230,
            justify="center",
        )
        self._empty_label.pack(padx=20, pady=60)
        self._empty_state.pack(fill="x", pady=20)
        
    def _build_footer(self):
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill="x", side="bottom", padx=Sizing.PADDING_MEDIUM, pady=Sizing.PADDING_MEDIUM)
        
        count_row = ctk.CTkFrame(footer, fg_color="transparent")
        count_row.pack(fill="x")
        
        self._queue_count = ctk.CTkLabel(
            count_row,
            text=t("items_queued", count=0),
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_PRIMARY,
        )
        self._queue_count.pack(side="left")

        self._page_row = ctk.CTkFrame(footer, fg_color="transparent")
        self._page_row.pack(fill="x", pady=(4, 0))
        self._prev_page_btn = ctk.CTkButton(
            self._page_row, text="◀", width=32, height=24,
            command=lambda: self._change_page(-1),
        )
        self._prev_page_btn.pack(side="left")
        Tooltip(self._prev_page_btn, t("queue_previous_page"))
        self._page_label = ctk.CTkLabel(self._page_row, text="1 / 1")
        self._page_label.pack(side="left", expand=True)
        self._next_page_btn = ctk.CTkButton(
            self._page_row, text="▶", width=32, height=24,
            command=lambda: self._change_page(1),
        )
        self._next_page_btn.pack(side="right")
        Tooltip(self._next_page_btn, t("queue_next_page"))
        self._page_row.pack_forget()

        actions_row = ctk.CTkFrame(footer, fg_color="transparent")
        actions_row.pack(fill="x", pady=(4, Sizing.PADDING_SMALL))
        
        self._clear_btn = ctk.CTkButton(
            actions_row,
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
            actions_row,
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
        output_label_row = ctk.CTkFrame(footer, fg_color="transparent")
        output_label_row.pack(fill="x", pady=(Sizing.PADDING_SMALL, 4))
        output_label = ctk.CTkLabel(
            output_label_row,
            text=t("output_location"),
            font=(Fonts.FAMILY, Fonts.SIZE_TINY, "bold"),
            text_color=Colors.TEXT_PRIMARY,
            anchor="w",
        )
        output_label.pack(side="left")
        self._same_as_input_btn = ctk.CTkButton(
            output_label_row,
            text=t("same_as_input"),
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
            text_color=Colors.TEXT_PRIMARY,
            height=22,
            width=94,
            command=self._on_same_as_input,
        )
        self._same_as_input_btn.pack(side="right")
        output_tip = ctk.CTkLabel(output_label_row, text="\u24d8", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        output_tip.pack(side="left", padx=4)
        Tooltip(output_tip, t("tip_output_location"))

        self._preserve_structure_var = ctk.BooleanVar(value=False)
        self._preserve_structure_checkbox = ctk.CTkCheckBox(
            footer,
            text=t("preserve_input_structure"),
            variable=self._preserve_structure_var,
            command=self._on_preserve_structure_changed,
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_PRIMARY,
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
            border_color=Colors.BORDER_LIGHT,
            checkbox_width=18,
            checkbox_height=18,
        )
        self._preserve_structure_checkbox.pack(fill="x", pady=(0, 4))
        Tooltip(
            self._preserve_structure_checkbox,
            t("tip_preserve_input_structure"),
        )
        
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
        )
        self._output_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self._output_entry.bind("<KeyRelease>", self._on_output_entry_changed)
        
        self._output_browse_btn = ctk.CTkButton(
            output_row,
            text="",
            image=create_icon("folder", 16, Colors.TEXT_PRIMARY),
            fg_color=Colors.BG_CARD,
            hover_color=Colors.BORDER_LIGHT,
            text_color=Colors.TEXT_PRIMARY,
            width=32,
            height=Sizing.INPUT_HEIGHT,
            command=self._on_browse_output,
        )
        self._output_browse_btn.pack(side="right")
        
        # Output pattern
        pattern_label_row = ctk.CTkFrame(footer, fg_color="transparent")
        pattern_label_row.pack(fill="x", pady=(4, 2))
        pattern_label = ctk.CTkLabel(
            pattern_label_row,
            text=t("output_pattern"),
            font=(Fonts.FAMILY, Fonts.SIZE_TINY, "bold"),
            text_color=Colors.TEXT_PRIMARY,
            anchor="w",
        )
        pattern_label.pack(side="left")
        pattern_tip = ctk.CTkLabel(pattern_label_row, text="\u24d8", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        pattern_tip.pack(side="left", padx=4)
        Tooltip(pattern_tip, t("tip_output_pattern"))
        self._pattern_entry = ctk.CTkEntry(
            footer,
            placeholder_text=DEFAULT_OUTPUT_PATTERN,
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            fg_color=Colors.BG_CARD,
            border_color=Colors.BORDER,
            text_color=Colors.TEXT_PRIMARY,
            height=Sizing.INPUT_HEIGHT,
        )
        self._pattern_entry.pack(fill="x", pady=(4, 0))
        self._pattern_entry.insert(0, DEFAULT_OUTPUT_PATTERN)
        self._pattern_entry.bind("<KeyRelease>", self._on_pattern_or_conflicts)

    def _on_pattern_or_conflicts(self, event=None):
        self._refresh_conflicts()
        self._notify_output_changed()

    def _on_preserve_structure_changed(self):
        self._refresh_conflicts()
        self._notify_output_changed()

    def _notify_output_changed(self):
        if self._on_output_changed:
            self._on_output_changed(
                self.get_output_folder(),
                self.get_output_pattern(),
                self.get_preserve_input_structure(),
            )

    def _on_output_entry_changed(self, event=None) -> None:
        self._update_same_as_input_style()
        self._refresh_conflicts()
        self._notify_output_changed()
        
    def _on_add_files(self):
        files = filedialog.askopenfilenames(
            title=t("select_video_files"),
            filetypes=[
                (t("file_type_media"), file_dialog_patterns(MEDIA_EXTENSIONS)),
                (t("file_type_video"), file_dialog_patterns(VIDEO_EXTENSIONS)),
                (t("file_type_image"), file_dialog_patterns(IMAGE_EXTENSIONS)),
                (t("file_type_all"), "*.*"),
            ]
        )
        for f in files:
            path = Path(f)
            self.add_job(path, input_root=_direct_file_input_root(path))

    def _on_add_folder(self):
        folder = filedialog.askdirectory(title=t("select_folder"))
        if folder:
            folder_path = Path(folder)
            for f in folder_media_in_processing_order(folder_path):
                self.add_job(f, input_root=folder_path)
                    
    def _on_browse_output(self):
        folder = filedialog.askdirectory(title=t("select_output_folder"))
        if folder:
            self._set_output_folder(folder)

    def _on_same_as_input(self) -> None:
        self._set_output_folder("")

    def _set_output_folder(self, folder: str) -> None:
        self._output_entry.configure(state="normal")
        self._output_entry.delete(0, "end")
        if folder:
            self._output_entry.insert(0, folder)
        self._update_same_as_input_style()
        self._refresh_conflicts()
        if self._on_jobs_changed:
            self._on_jobs_changed()
        self._notify_output_changed()

    def _update_same_as_input_style(self) -> None:
        same_as_input = not self.get_output_folder()
        self._same_as_input_btn.configure(
            fg_color=Colors.PRIMARY if same_as_input else Colors.BG_CARD,
            hover_color=Colors.PRIMARY_HOVER if same_as_input else Colors.BORDER_LIGHT,
        )

    def _on_clear_queue(self):
        self._jobs.clear()
        for w in self._job_widgets:
            if w is not None:
                w.destroy()
        self._job_widgets.clear()
        self._page = 0
        self._update_empty_state()
        self._update_count()
        if self._on_jobs_changed:
            self._on_jobs_changed()
            
    def _on_clear_completed(self):
        """Clear completed, errored, and skipped jobs from the queue."""
        completed_statuses = {JobStatus.COMPLETED, JobStatus.ERROR, JobStatus.SKIPPED}
        self._jobs[:] = [job for job in self._jobs if job.status not in completed_statuses]
        self._render_page()
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
        pages = max(1, (count + _PAGE_SIZE - 1) // _PAGE_SIZE)
        self._page_label.configure(text=f"{self._page + 1} / {pages}")
        self._prev_page_btn.configure(state="normal" if self._page else "disabled")
        self._next_page_btn.configure(state="normal" if self._page + 1 < pages else "disabled")
        if pages > 1:
            self._page_row.pack(fill="x", pady=(4, 0))
        else:
            self._page_row.pack_forget()

    def _change_page(self, step: int) -> None:
        pages = max(1, (len(self._jobs) + _PAGE_SIZE - 1) // _PAGE_SIZE)
        page = min(max(self._page + step, 0), pages - 1)
        if page == self._page:
            return
        self._page = page
        self._render_page()
        self._list_frame._parent_canvas.yview_moveto(0)
        self._update_count()

    def _render_page(self) -> None:
        for widget in self._job_widgets:
            if widget is not None:
                widget.destroy()
        self._page = min(self._page, max(0, (len(self._jobs) - 1) // _PAGE_SIZE))
        self._job_widgets = [None] * len(self._jobs)
        for index in range(self._page * _PAGE_SIZE, min(len(self._jobs), (self._page + 1) * _PAGE_SIZE)):
            self._create_job_widget(index)

    def _create_job_widget(self, index: int) -> None:
        job = self._jobs[index]
        path = job.path
        widget = JobListItem(
            self._list_frame,
            filename=job.filename,
            duration=job.duration_str or "",
            status=t("job_pending"),
            on_remove=lambda j=job: self._remove_job(j),
            on_drag_start=self._on_widget_drag_start,
            on_drag_move=self._on_widget_drag_move,
            on_drag_end=self._on_widget_drag_end,
            on_edit_segments=(None if is_image(path) else lambda j=job: self._edit_segments(j)),
            on_play=(None if is_image(path) else lambda j=job: self._play_job(j)),
            on_open_containing_folder=lambda j=job: self._open_containing_folder(j),
            on_copy_path=lambda j=job: self._copy_job_path(j),
            on_open_restored_output=lambda j=job: self._open_restored_output(j),
            on_requeue=lambda j=job: self._requeue_job(j),
        )
        widget.pack(fill="x", pady=(0, 4))
        self._job_widgets[index] = widget
        widget.set_player_enabled(not self._running)
        widget.set_removable(not self._running or job.id != self._processing_job_id)
        widget.set_segment_summary(self._segment_summary(job), selected=bool(job.segments))
        self.update_job_status(job.id, job.status, job.progress, elapsed_seconds=job.elapsed_seconds)
        if job.has_conflict and job.status is JobStatus.PENDING:
            widget.set_conflict(True)

    def add_job(self, path: Path, *, input_root: Path | None = None):
        duplicate = next((job for job in self._jobs if job.path == path), None)
        if duplicate is not None:
            if duplicate.status is JobStatus.PENDING and duplicate.input_root is None and input_root is not None:
                duplicate.input_root = input_root
                self._refresh_conflicts()
                if self._on_jobs_changed:
                    self._on_jobs_changed()
            return

        job = JobItem(path=path, input_root=input_root)
        self._jobs.append(job)

        # Check for output file conflict
        output_path = self._get_output_path(job)
        job.has_conflict = output_path.exists() if output_path else False
        self._job_widgets.append(None)
        if len(self._jobs) <= (self._page + 1) * _PAGE_SIZE and len(self._jobs) > self._page * _PAGE_SIZE:
            self._create_job_widget(len(self._jobs) - 1)

        self._update_empty_state()
        self._update_count()
        if self._on_jobs_changed:
            self._on_jobs_changed()

    def _play_job(self, job: JobItem) -> None:
        if not self._running and self._on_play:
            self._on_play(job.path)

    def _open_containing_folder(self, job: JobItem) -> None:
        completed = job.status is JobStatus.COMPLETED
        path = job.output_path if completed and job.output_path is not None else job.path
        open_containing_folder(
            path,
            parent=self.winfo_toplevel(),
            select_file=completed,
        )

    def _copy_job_path(self, job: JobItem) -> None:
        path = self._action_path(job)
        window = self.winfo_toplevel()
        window.clipboard_clear()
        window.clipboard_append(str(path))
        window.update_idletasks()

    def _open_restored_output(self, job: JobItem) -> None:
        if job.status is JobStatus.COMPLETED and job.output_path is not None:
            open_file(job.output_path, parent=self.winfo_toplevel())

    def _requeue_job(self, job: JobItem) -> None:
        terminal_statuses = {JobStatus.COMPLETED, JobStatus.ERROR, JobStatus.SKIPPED}
        if job.status not in terminal_statuses:
            return
        index = self._find_job_index_by_id(job.id)
        if index is None:
            return
        self._jobs.append(self._jobs.pop(index))
        widget = self._job_widgets.pop(index)
        if widget is not None:
            widget.destroy()
        self._job_widgets.append(None)
        job.error_message = ""
        job.output_path = None
        job.status = JobStatus.PENDING
        job.progress = 0.0
        job.elapsed_seconds = None
        self._show_last_page_row()
        self._refresh_conflicts()
        if self._on_jobs_changed:
            self._on_jobs_changed()

    def _action_path(self, job: JobItem) -> Path:
        if job.status is JobStatus.COMPLETED and job.output_path is not None:
            return job.output_path
        return job.path

    def _set_widget_action_options(self, job: JobItem, widget: JobListItem) -> None:
        widget.set_action_options(
            has_restored_output=(
                job.status is JobStatus.COMPLETED and job.output_path is not None
            ),
            requeueable=(
                job.status in {JobStatus.COMPLETED, JobStatus.ERROR, JobStatus.SKIPPED}
            ),
        )

    def _edit_segments(self, job: JobItem) -> None:
        if job.status is not JobStatus.PENDING:
            return
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
                self._get_settings,
                self._is_gpu_busy,
                self._set_preview_gpu_busy,
                self._ltx_unavailable_reason(),
                lambda segments, j=job: self._segments_saved(j, segments),
                self._segment_editor_closed,
            )
        except Exception as exc:
            self._segment_editor = None
            logger.warning("Failed to open segment editor", exc_info=True)
            messagebox.showerror(t("segments_title"), str(exc), parent=self.winfo_toplevel())

    def _segment_editor_closed(self) -> None:
        self._segment_editor = None

    def _segments_saved(self, job: JobItem, segments: tuple[SegmentRange, ...]) -> None:
        idx = self._find_job_index_by_id(job.id)
        if idx is None:
            return
        widget = self._job_widgets[idx]
        if widget is not None:
            widget.set_segment_summary(self._segment_summary(job), selected=bool(segments))

    def _segment_summary(self, job: JobItem) -> str:
        if job.segments:
            duration = sum(segment.duration for segment in job.segments)
            if job.duration_seconds:
                return t(
                    "segments_summary_percent",
                    count=len(job.segments),
                    seconds=duration,
                    percent=duration / job.duration_seconds * 100,
                )
            return t("segments_summary", count=len(job.segments), seconds=duration)
        return t("segments_full_video")
            
    def _get_output_path(self, job: JobItem) -> Path | None:
        """Get the output path for a given input file based on current settings."""
        output_folder = self._output_entry.get()
        pattern = self._pattern_entry.get() or DEFAULT_OUTPUT_PATTERN
        input_path = job.path
        preserve_structure = bool(output_folder) and self.get_preserve_input_structure()

        if not output_folder:
            # Use same folder as input
            output_folder = str(input_path.parent)

        try:
            return job_output_path(
                output_folder,
                input_path,
                pattern,
                input_root=job.input_root,
                preserve_structure=preserve_structure,
            )
        except OutputPathError:
            # Conflict refreshes run while the user is typing.  Invalid
            # templates are surfaced by Processor when Start is pressed, but
            # must not make the queue panel callback itself fail.
            return None


    def _remove_job(self, job: JobItem):
        if job in self._jobs:
            index = self._jobs.index(job)
            self._jobs.pop(index)
            widget = self._job_widgets.pop(index)
            if widget is not None:
                widget.destroy()
            if self._page and self._page * _PAGE_SIZE >= len(self._jobs):
                self._page -= 1
                self._render_page()
            else:
                self._show_last_page_row()
            self._update_empty_state()
            self._update_count()
            if self._on_jobs_changed:
                self._on_jobs_changed()

    def _show_last_page_row(self) -> None:
        last_index = min(len(self._jobs), (self._page + 1) * _PAGE_SIZE) - 1
        if last_index >= 0 and self._job_widgets[last_index] is None:
            self._create_job_widget(last_index)

    def get_jobs(self) -> list[JobItem]:
        # Return a shallow copy for callers that should not modify the queue
        return self._jobs.copy()

    def get_jobs_ref(self) -> list[JobItem]:
        """Return the live jobs list reference so callers (like the Processor)
        can observe additions/removals while processing is running.
        Use with care: this returns the internal list, not a defensive copy."""
        return self._jobs

    def reset_jobs_for_run(self) -> None:
        for job in self._jobs:
            job.error_message = ""
            job.output_path = None
            job.elapsed_seconds = None
            self.update_job_status(job.id, JobStatus.PENDING)
        self._refresh_conflicts()
        
    def get_output_folder(self) -> str:
        return self._output_entry.get() or ""
        
    def get_output_pattern(self) -> str:
        return self._pattern_entry.get() or DEFAULT_OUTPUT_PATTERN

    def get_preserve_input_structure(self) -> bool:
        return bool(self._preserve_structure_var.get())
        
    def set_on_jobs_changed(self, callback: callable):
        self._on_jobs_changed = callback

    def set_segment_editor_context(
        self,
        get_settings: callable,
        is_gpu_busy: callable,
        set_preview_gpu_busy: callable,
        ltx_unavailable_reason: callable,
    ):
        self._get_settings = get_settings
        self._is_gpu_busy = is_gpu_busy
        self._set_preview_gpu_busy = set_preview_gpu_busy
        self._ltx_unavailable_reason = ltx_unavailable_reason

    def set_on_output_changed(self, callback: callable):
        self._on_output_changed = callback

    def set_on_play(self, callback: callable) -> None:
        self._on_play = callback

    def set_initial_output(
        self,
        folder: str = "",
        pattern: str = "",
        preserve_input_structure: bool = False,
    ):
        self._set_output_folder(folder)
        if pattern:
            self._pattern_entry.delete(0, "end")
            self._pattern_entry.insert(0, pattern)
        self._preserve_structure_var.set(bool(preserve_input_structure))
        self._refresh_conflicts()
        
    def _find_job_index_by_id(self, job_id: int) -> int | None:
        for i, job in enumerate(self._jobs):
            if job.id == job_id:
                return i
        return None

    def update_job_status(
        self,
        job_id: int,
        status: JobStatus,
        progress: float = 0.0,
        fps: float = 0.0,
        eta_seconds: float = 0.0,
        elapsed_seconds: float | None = None,
        phase: str = "",
    ):
        idx = self._find_job_index_by_id(job_id)
        if idx is None:
            return
        job = self._jobs[idx]
        job.status = status
        job.progress = progress
        if status is JobStatus.COMPLETED and elapsed_seconds is not None:
            job.elapsed_seconds = elapsed_seconds
        widget = self._job_widgets[idx]
        if widget is None:
            return
        
        status_map = {
            JobStatus.PENDING: (t("job_pending"), "", Colors.STATUS_PENDING),
            JobStatus.PROCESSING: (t("job_processing"), "○", Colors.STATUS_PROCESSING),
            JobStatus.COMPLETED: (t("job_completed"), "✓", Colors.STATUS_COMPLETED),
            JobStatus.ERROR: (t("job_error"), "✕", Colors.STATUS_ERROR),
            JobStatus.SKIPPED: (t("job_skipped"), "⊘", Colors.STATUS_CONFLICT),
        }
        text, icon, color = status_map.get(status, ("", "", Colors.STATUS_PENDING))
        if status is JobStatus.PROCESSING:
            text = processing_phase_text(phase)
        widget.set_status(text, icon, color)
        
        if status == JobStatus.PROCESSING:
            widget.set_progress(progress)
            widget.set_fps_eta(
                fps=fps,
                eta_seconds=eta_seconds,
                stage_eta=bool(phase),
            )
        elif status == JobStatus.COMPLETED and elapsed_seconds is not None:
            widget.hide_progress()
            widget.set_completed(elapsed_seconds)
        else:
            widget.hide_progress()
            widget.set_fps_eta(0.0, 0.0)
            
        # Hide conflict indicator once processing starts
        if status != JobStatus.PENDING:
            widget.set_conflict(False)
        widget.set_segments_editable(status is JobStatus.PENDING)
        widget.set_action_menu_visible(status is not JobStatus.PROCESSING)
        self._set_widget_action_options(job, widget)
                
    def _refresh_conflicts(self):
        """Re-check all jobs for output file conflicts."""
        for job, widget in zip(self._jobs, self._job_widgets):
            if job.status == JobStatus.PENDING:
                output_path = self._get_output_path(job)
                job.has_conflict = output_path.exists() if output_path else False
                if widget is not None:
                    widget.set_conflict(job.has_conflict)
    
    def set_output_enabled(self, enabled: bool):
        """Enable or disable output location controls (but not queue add/remove)."""
        state = "normal" if enabled else "disabled"
        self._output_entry.configure(state=state)
        self._output_browse_btn.configure(state=state)
        self._same_as_input_btn.configure(state=state)
        self._pattern_entry.configure(state=state)
        self._preserve_structure_checkbox.configure(state=state)
        self._clear_btn.configure(state=state)
        self._clear_completed_btn.configure(state=state)

    # --- Drag & reorder support ---
    def _is_processing_widget(self, widget: 'JobListItem') -> bool:
        if self._processing_job_id is None:
            return False
        try:
            idx = self._job_widgets.index(widget)
        except ValueError:
            return False
        return idx < len(self._jobs) and self._jobs[idx].id == self._processing_job_id

    def _on_widget_drag_start(self, widget: 'JobListItem', event):
        if self._is_processing_widget(widget):
            return
        try:
            widget.lift()
        except Exception:
            logger.debug("widget.lift failed on drag start", exc_info=True)
        try:
            widget.configure(cursor="hand2")
        except Exception:
            logger.debug("cursor configure failed on drag start", exc_info=True)

    def _on_widget_drag_move(self, widget: 'JobListItem', event):
        if self._is_processing_widget(widget):
            return
        # Compute pointer y relative to list frame and determine new index
        lf = self._list_frame
        try:
            y = event.y_root - lf.winfo_rooty()
        except Exception:
            logger.debug("drag move aborted (widget geometry unavailable)", exc_info=True)
            return

        # Compute new index among widgets based on center positions
        new_index = 0
        for w in self._job_widgets:
            if w is None:
                continue
            if w is widget:
                continue
            center = (w.winfo_rooty() + w.winfo_height() / 2) - lf.winfo_rooty()
            if y > center:
                new_index += 1

        try:
            current_index = self._job_widgets.index(widget)
        except ValueError:
            return

        new_index += self._page * _PAGE_SIZE
        if new_index != current_index:
            # Remove and reinsert data + widget and repack
            job = self._jobs.pop(current_index)
            self._job_widgets.pop(current_index)
            self._jobs.insert(new_index, job)
            self._job_widgets.insert(new_index, widget)
            # Repack in order
            for w in self._job_widgets:
                if w is None:
                    continue
                w.pack_forget()
            for w in self._job_widgets:
                if w is None:
                    continue
                w.pack(fill="x", pady=(0, 4))
            if self._on_jobs_changed:
                self._on_jobs_changed()

    def _on_widget_drag_end(self, widget: 'JobListItem', event):
        try:
            widget.configure(cursor="")
        except Exception:
            logger.debug("cursor reset failed on drag end", exc_info=True)
        for w in self._job_widgets:
            if w is None:
                continue
            w.pack_forget()
        for w in self._job_widgets:
            if w is None:
                continue
            w.pack(fill="x", pady=(0, 4))
        if self._on_jobs_changed:
            self._on_jobs_changed()

    def set_running(self, running: bool, processing_job_id: int | None = None):
        """Set queue running state.

        When running=True the panel is visually dimmed and controls are
        disabled except the add buttons and removing jobs (except the
        currently processing item which is protected).
        """
        if self._running == running and (
            not running or self._processing_job_id == processing_job_id
        ):
            return
        self._running = running
        self._processing_job_id = processing_job_id if running else None
        if running:
            # Disable controls we don't want interactive while running
            self._clear_btn.configure(state="disabled")
            self._clear_completed_btn.configure(state="disabled")
            self._output_browse_btn.configure(state="disabled")
            self._output_entry.configure(state="disabled")
            self._same_as_input_btn.configure(state="disabled")
            self._pattern_entry.configure(state="disabled")
            self._preserve_structure_checkbox.configure(state="disabled")
            # Allow adding files/folders
            self._add_files_btn.configure(state="normal")
            self._add_folder_btn.configure(state="normal")
            processing_idx = self._find_job_index_by_id(processing_job_id) if processing_job_id is not None else None
            for i, widget in enumerate(self._job_widgets):
                if widget is None:
                    continue
                widget.set_removable(i != processing_idx)
                widget.set_segments_editable(self._jobs[i].status is JobStatus.PENDING)
                widget.set_player_enabled(False)
                widget.set_action_menu_visible(
                    i != processing_idx
                    and self._jobs[i].status is not JobStatus.PROCESSING
                )
                self._set_widget_action_options(self._jobs[i], widget)
        else:
            # Restore normal appearance and enable controls
            self._clear_btn.configure(state="normal")
            self._clear_completed_btn.configure(state="normal")
            self._output_browse_btn.configure(state="normal")
            self._output_entry.configure(state="normal")
            self._same_as_input_btn.configure(state="normal")
            self._pattern_entry.configure(state="normal")
            self._preserve_structure_checkbox.configure(state="normal")
            self._add_files_btn.configure(state="normal")
            self._add_folder_btn.configure(state="normal")
            for job, widget in zip(self._jobs, self._job_widgets):
                if widget is None:
                    continue
                widget.set_removable(True)
                widget.set_segments_editable(job.status is JobStatus.PENDING)
                widget.set_player_enabled(True)
                widget.set_action_menu_visible(job.status is not JobStatus.PROCESSING)
                self._set_widget_action_options(job, widget)

    def enable_file_drop(self):
        """Register the list area as an OS file drop target."""
        inner = self._list_frame._parent_frame
        inner.drop_target_register(DND_FILES)
        inner.dnd_bind("<<Drop>>", self._on_file_drop)

    def _parse_drop_data(self, data: str) -> list[Path]:
        paths = []
        i = 0
        while i < len(data):
            if data[i] == "{":
                end = data.index("}", i)
                paths.append(Path(data[i + 1 : end]))
                i = end + 2
            elif data[i] == " ":
                i += 1
            else:
                end = data.find(" ", i)
                if end == -1:
                    end = len(data)
                paths.append(Path(data[i:end]))
                i = end + 1
        return paths

    def _on_file_drop(self, event):
        paths = self._parse_drop_data(event.data)
        for p in paths:
            if p.is_dir():
                for f in folder_media_in_processing_order(p):
                    self.add_job(f, input_root=p)
            elif p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS:
                self.add_job(p, input_root=_direct_file_input_root(p))
