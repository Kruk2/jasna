"""Queue panel - left side job list management."""

import logging
from typing import Iterable

import customtkinter as ctk
from pathlib import Path
from tkinter import filedialog

from tkinterdnd2 import DND_FILES

from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.gui.models import JobItem, JobStatus
from jasna.gui.components import AutoHidingScrollableFrame, Tooltip
from jasna.gui.job_list_item import JobListItem
from jasna.gui.job_queue import FINISHED_STATUSES, JobQueue
from jasna.gui.output_location_bar import OutputLocationBar
from jasna.gui.icons import create_icon
from jasna.gui.file_actions import open_containing_folder
from jasna.gui.file_actions import open_file
from jasna.gui.locales import t
from jasna.progressbar import StageProgress

from jasna.media.media_files import (
    IMAGE_EXTENSIONS,
    MEDIA_EXTENSIONS,
    VIDEO_EXTENSIONS,
    file_dialog_patterns,
    folder_media_in_processing_order,
    folder_output_path,
    is_image,
)
from jasna.segments import SegmentRange

logger = logging.getLogger(__name__)

_PAGE_SIZE = 20


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
        
        self._queue = JobQueue()
        self._job_widgets: list[JobListItem | None] = []
        self._page = 0
        self._on_jobs_changed: callable = None
        self._on_output_changed: callable = None
        self._on_play: callable = None
        self._running = False
        self._processing_job_id: int | None = None
        self._on_edit_segments: callable = None
        
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
        
        self._output_bar = OutputLocationBar(
            footer, on_edited=self._on_output_edited, on_folder_set=self._on_output_folder_set
        )
        self._output_bar.pack(fill="x")

    def _notify_output_changed(self) -> None:
        if self._on_output_changed:
            self._on_output_changed(self.get_output_folder(), self.get_output_pattern())

    def _on_output_edited(self) -> None:
        self._refresh_conflicts()
        self._notify_output_changed()

    def _on_output_folder_set(self) -> None:
        self._refresh_conflicts()
        if self._on_jobs_changed:
            self._on_jobs_changed()
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
        self.add_jobs(JobItem(path=Path(f)) for f in files)

    def _on_add_folder(self):
        folder = filedialog.askdirectory(title=t("select_folder"))
        if folder:
            self.add_jobs(JobItem(path=f) for f in folder_media_in_processing_order(Path(folder)))
                    
    def _on_clear_queue(self):
        self._queue.clear()
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
        self._queue.clear_finished()
        self._render_page()
        self._update_empty_state()
        self._update_count()
        if self._on_jobs_changed:
            self._on_jobs_changed()
            
    def _update_empty_state(self):
        if len(self._queue):
            self._empty_state.pack_forget()
        else:
            self._empty_state.pack(fill="x", pady=20)
            
    def _update_count(self):
        count = len(self._queue)
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
        pages = max(1, (len(self._queue) + _PAGE_SIZE - 1) // _PAGE_SIZE)
        page = min(max(self._page + step, 0), pages - 1)
        if page == self._page:
            return
        self._page = page
        self._render_page()
        self._list_frame.scroll_to_top()
        self._update_count()

    def _render_page(self) -> None:
        for widget in self._job_widgets:
            if widget is not None:
                widget.destroy()
        self._page = min(self._page, max(0, (len(self._queue) - 1) // _PAGE_SIZE))
        self._job_widgets = [None] * len(self._queue)
        for index in range(self._page * _PAGE_SIZE, min(len(self._queue), (self._page + 1) * _PAGE_SIZE)):
            self._create_job_widget(index)

    def _create_job_widget(self, index: int) -> None:
        job = self._queue[index]
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

    def add_jobs(self, jobs: Iterable[JobItem]) -> None:
        """Append the jobs whose input is not queued yet, keeping their own options."""
        added = self._queue.add_unique(jobs)
        if not added:
            return
        first_index = len(self._queue) - len(added)
        for index, job in enumerate(added, first_index):
            output_path = self._get_output_path(job.path)
            job.has_conflict = output_path.exists() if output_path else False
            self._job_widgets.append(None)
            if self._page * _PAGE_SIZE <= index < (self._page + 1) * _PAGE_SIZE:
                self._create_job_widget(index)

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
        index = self._queue.requeue(job)
        if index is None:
            return
        self._job_widgets.pop(index).destroy()
        self._job_widgets.append(None)
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
            requeueable=job.status in FINISHED_STATUSES,
        )

    def _edit_segments(self, job: JobItem) -> None:
        if job.status is JobStatus.PENDING:
            self._on_edit_segments(job)

    def on_segments_saved(self, job: JobItem, segments: tuple[SegmentRange, ...]) -> None:
        idx = self._queue.index_of(job.id)
        if idx is None:
            return
        widget = self._job_widgets[idx]
        if widget is not None:
            widget.set_segment_summary(self._segment_summary(job), selected=bool(segments))
        if self._on_jobs_changed:
            self._on_jobs_changed()

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
            
    def _get_output_path(self, input_path: Path) -> Path | None:
        """Get the output path for a given input file based on current settings."""
        return folder_output_path(self.get_output_folder() or input_path.parent, input_path, self.get_output_pattern())
            
    def _remove_job(self, job: JobItem):
        index = self._queue.remove(job)
        if index is not None:
            self._job_widgets.pop(index).destroy()
            if self._page and self._page * _PAGE_SIZE >= len(self._queue):
                self._page -= 1
                self._render_page()
            else:
                self._show_last_page_row()
            self._update_empty_state()
            self._update_count()
            if self._on_jobs_changed:
                self._on_jobs_changed()

    def _show_last_page_row(self) -> None:
        last_index = min(len(self._queue), (self._page + 1) * _PAGE_SIZE) - 1
        if last_index >= 0 and self._job_widgets[last_index] is None:
            self._create_job_widget(last_index)

    def get_jobs(self) -> list[JobItem]:
        return list(self._queue)

    @property
    def job_queue(self) -> JobQueue:
        """The live queue, which the processor works through while the user edits it."""
        return self._queue

    def reset_jobs_for_run(self) -> None:
        for job in self._queue:
            job.error_message = ""
            job.output_path = None
            job.elapsed_seconds = None
            self.update_job_status(job.id, JobStatus.PENDING)
        self._refresh_conflicts()
        
    def get_output_folder(self) -> str:
        return self._output_bar.folder

    def get_output_pattern(self) -> str:
        return self._output_bar.pattern
        
    def set_on_jobs_changed(self, callback: callable):
        self._on_jobs_changed = callback

    def set_on_edit_segments(self, callback: callable) -> None:
        self._on_edit_segments = callback

    def set_on_output_changed(self, callback: callable):
        self._on_output_changed = callback

    def set_on_play(self, callback: callable) -> None:
        self._on_play = callback

    def set_initial_output(self, folder: str = "", pattern: str = ""):
        self._output_bar.set_folder(folder)
        if pattern:
            self._output_bar.set_pattern(pattern)
        
    def update_job_status(self, job_id: int, status: JobStatus, progress: float = 0.0, fps: float = 0.0, eta_seconds: float = 0.0, elapsed_seconds: float | None = None, stage: StageProgress | None = None):
        idx = self._queue.index_of(job_id)
        if idx is None:
            return
        job = self._queue[idx]
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
        widget.set_status(text, icon, color)
        
        if status == JobStatus.PROCESSING:
            widget.set_progress(progress)
            widget.set_fps_eta(fps, eta_seconds, stage)
        elif status == JobStatus.COMPLETED and elapsed_seconds is not None:
            widget.hide_progress()
            widget.set_completed(elapsed_seconds)
        else:
            widget.hide_progress()
            widget.set_fps_eta(0.0, 0.0, None)
            
        # Hide conflict indicator once processing starts
        if status != JobStatus.PENDING:
            widget.set_conflict(False)
        widget.set_segments_editable(status is JobStatus.PENDING)
        widget.set_action_menu_visible(status is not JobStatus.PROCESSING)
        self._set_widget_action_options(job, widget)
                
    def _refresh_conflicts(self):
        """Re-check all jobs for output file conflicts."""
        for job, widget in zip(self._queue, self._job_widgets):
            if job.status == JobStatus.PENDING:
                output_path = self._get_output_path(job.path)
                job.has_conflict = output_path.exists() if output_path else False
                if widget is not None:
                    widget.set_conflict(job.has_conflict)
    
    def set_output_enabled(self, enabled: bool):
        """Enable or disable output location controls (but not queue add/remove)."""
        state = "normal" if enabled else "disabled"
        self._output_bar.set_enabled(enabled)
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
        return idx < len(self._queue) and self._queue[idx].id == self._processing_job_id

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
            self._queue.move(current_index, new_index)
            self._job_widgets.pop(current_index)
            self._job_widgets.insert(new_index, widget)
            self._repack()
            if self._on_jobs_changed:
                self._on_jobs_changed()

    def _on_widget_drag_end(self, widget: 'JobListItem', event):
        try:
            widget.configure(cursor="")
        except Exception:
            logger.debug("cursor reset failed on drag end", exc_info=True)
        self._repack()
        if self._on_jobs_changed:
            self._on_jobs_changed()

    def _repack(self) -> None:
        shown = [widget for widget in self._job_widgets if widget is not None]
        for widget in shown:
            widget.pack_forget()
        for widget in shown:
            widget.pack(fill="x", pady=(0, 4))

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
        self.set_output_enabled(not running)
        processing_idx = self._queue.index_of(self._processing_job_id) if self._processing_job_id is not None else None
        for index, (job, widget) in enumerate(zip(self._queue, self._job_widgets)):
            if widget is None:
                continue
            widget.set_removable(index != processing_idx)
            widget.set_segments_editable(job.status is JobStatus.PENDING)
            widget.set_player_enabled(not running)
            widget.set_action_menu_visible(index != processing_idx and job.status is not JobStatus.PROCESSING)
            self._set_widget_action_options(job, widget)

    def enable_file_drop(self):
        """Register the list area as an OS file drop target."""
        inner = self._list_frame._parent_frame
        inner.drop_target_register(DND_FILES)
        inner.dnd_bind("<<Drop>>", self._on_file_drop)

    def _on_file_drop(self, event):
        files = []
        for p in map(Path, self.tk.splitlist(event.data)):
            if p.is_dir():
                files.extend(folder_media_in_processing_order(p))
            elif p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS:
                files.append(p)
        self.add_jobs(JobItem(path=f) for f in files)
