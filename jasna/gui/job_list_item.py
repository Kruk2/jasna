import tkinter
from collections.abc import Callable

import customtkinter as ctk

from jasna.gui.components import Tooltip, format_duration
from jasna.gui.locales import t
from jasna.gui.theme import Colors, Fonts, Sizing


class JobListItem(ctk.CTkFrame):
    """Individual job item in the queue list."""
    
    def __init__(
        self,
        master,
        filename: str,
        duration: str,
        status: str,
        on_remove: Callable[[], None],
        on_drag_start: Callable[["JobListItem", tkinter.Event], None],
        on_drag_move: Callable[["JobListItem", tkinter.Event], None],
        on_drag_end: Callable[["JobListItem", tkinter.Event], None],
        on_edit_segments: Callable[[], None] | None,
        on_play: Callable[[], None] | None,
        on_open_containing_folder: Callable[[], None],
        on_copy_path: Callable[[], None],
        on_open_restored_output: Callable[[], None],
        on_requeue: Callable[[], None],
    ):
        super().__init__(
            master,
            fg_color=Colors.BG_CARD,
            corner_radius=Sizing.BORDER_RADIUS,
            height=72,
        )
        self.pack_propagate(False)
        
        self._on_edit_segments = on_edit_segments
        self._on_play = on_play
        self._on_open_containing_folder = on_open_containing_folder
        self._on_copy_path = on_copy_path
        self._on_open_restored_output = on_open_restored_output
        self._on_requeue = on_requeue
        self._on_drag_start = on_drag_start
        self._on_drag_move = on_drag_move
        self._on_drag_end = on_drag_end
        self._progress_visible = False
        self._conflict_visible = False
        self._segments_editable = True
        self._player_enabled = True
        self._action_menu_visible = True
        self._has_restored_output = False
        self._requeueable = False
        self._segment_tooltips: list[Tooltip] = []
        self._removable = True
        self._hide_after_id = None
        self._action_menu = None
        
        # Main content container
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=8, pady=6)
        
        # Top row: handle + filename + duration
        top_row = ctk.CTkFrame(content, fg_color="transparent")
        top_row.pack(fill="x")
        
        # Conflict indicator (amber dot)
        self._conflict_dot = ctk.CTkLabel(
            top_row,
            text="●",
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            text_color=Colors.STATUS_CONFLICT,
            width=16,
        )
        Tooltip(self._conflict_dot, t("conflict_tooltip"))

        # Drag handle
        self._handle = ctk.CTkLabel(
            top_row,
            text="⋮⋮",
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            text_color=Colors.TEXT_PRIMARY,
            width=20,
            cursor="hand2",
        )
        self._handle.pack(side="left")
        # Bind drag events to handle
        self._handle.bind("<ButtonPress-1>", lambda event: self._on_drag_start(self, event))
        self._handle.bind("<B1-Motion>", lambda event: self._on_drag_move(self, event))
        self._handle.bind("<ButtonRelease-1>", lambda event: self._on_drag_end(self, event))
        
        # Info area (filename + duration inline)
        self._info = ctk.CTkFrame(top_row, fg_color="transparent")
        self._info.pack(side="left", fill="x", expand=True, padx=4)
        
        self._filename = ctk.CTkLabel(
            self._info,
            text=filename,
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
            text_color=Colors.TEXT_PRIMARY,
            anchor="w",
        )
        self._filename.pack(side="left")
        
        self._duration = ctk.CTkLabel(
            self._info,
            text=f"  •  {duration}" if duration else "",
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_PRIMARY,
            anchor="w",
        )
        self._duration.pack(side="left")
        
        # Bottom row: status
        bottom_row = ctk.CTkFrame(content, fg_color="transparent")
        bottom_row.pack(fill="x", pady=(4, 0))
        
        # Status area
        self._status_frame = ctk.CTkFrame(bottom_row, fg_color="transparent")
        self._status_frame.pack(side="left")
        
        self._status_icon = ctk.CTkLabel(
            self._status_frame,
            text="",
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            width=16,
        )
        
        self._status_label = ctk.CTkLabel(
            self._status_frame,
            text=status,
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_PRIMARY,
        )
        self._status_label.pack(side="left")

        self._segment_summary = ctk.CTkLabel(
            bottom_row,
            text="",
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.STATUS_PENDING,
            cursor="hand2" if self._on_edit_segments else "arrow",
        )
        self._segment_summary.pack(side="left", padx=(8, 0))
        if self._on_edit_segments:
            self._segment_summary.bind("<Button-1>", lambda _event: self._handle_edit_segments())
        
        # FPS / ETA small labels on the right of bottom row
        self._stats_frame = ctk.CTkFrame(bottom_row, fg_color="transparent")
        self._stats_frame.pack(side="right")

        self._segments_btn = ctk.CTkButton(
            bottom_row,
            text="✂",
            width=30,
            height=22,
            fg_color=Colors.BG_PANEL,
            hover_color=Colors.BORDER_LIGHT,
            text_color=Colors.TEXT_PRIMARY,
            cursor="hand2",
            command=self._handle_edit_segments,
        )
        if self._on_edit_segments:
            self._segments_btn.pack(side="right", padx=(0, 6))
            self._segment_tooltips = [
                Tooltip(self._segments_btn, t("segments_edit_tooltip")),
                Tooltip(self._segment_summary, t("segments_edit_tooltip")),
            ]

        self._play_btn = ctk.CTkButton(
            bottom_row,
            text="▶",
            width=30,
            height=22,
            fg_color=Colors.BG_PANEL,
            hover_color=Colors.BORDER_LIGHT,
            text_color=Colors.TEXT_PRIMARY,
            cursor="hand2",
            command=self._handle_play,
        )
        if self._on_play:
            self._play_btn.pack(side="right", padx=(0, 6))
            Tooltip(self._play_btn, t("queue_play_tooltip"))

        self._overflow_btn = ctk.CTkButton(
            bottom_row,
            text="⋯",
            width=24,
            height=22,
            fg_color=Colors.BG_PANEL,
            hover_color=Colors.BORDER_LIGHT,
            text_color=Colors.TEXT_PRIMARY,
            cursor="hand2",
            command=self._show_action_menu,
        )
        self._overflow_btn.pack(side="right")

        self._fps_label = ctk.CTkLabel(
            self._stats_frame,
            text="",
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.TEXT_PRIMARY,
        )
        self._fps_label.pack(side="left", padx=(0, 8))

        self._eta_label = ctk.CTkLabel(
            self._stats_frame,
            text="",
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.TEXT_PRIMARY,
        )
        self._eta_label.pack(side="left")
        
        # Progress bar (hidden by default)
        self._progress = ctk.CTkProgressBar(
            self,
            height=3,
            fg_color=Colors.BG_PANEL,
            progress_color=Colors.PRIMARY,
        )
        
        # Store references for hover binding
        self._top_row = top_row
        self._bottom_row = bottom_row
        
        self._remove_btn = ctk.CTkButton(
            self,
            text="✕",
            width=24,
            height=24,
            fg_color="transparent",
            hover_color=Colors.STATUS_ERROR,
            text_color=Colors.TEXT_PRIMARY,
            command=on_remove,
        )

        # Children get the hover bindings too, so moving between them doesn't flicker the remove button.
        hover_widgets = [
            self, self._remove_btn,
            content, self._handle, self._info, self._filename, self._duration,
            self._conflict_dot, self._status_frame, self._status_icon,
            self._status_label, self._stats_frame, self._fps_label, self._eta_label,
            self._top_row, self._bottom_row, self._segment_summary, self._segments_btn,
            self._play_btn, self._overflow_btn, self._progress,
        ]
        for widget in hover_widgets:
            widget.bind("<Enter>", self._on_enter)
            widget.bind("<Leave>", self._on_leave)
            widget.bind("<Button-3>", self._show_action_menu)

    def _pointer_inside(self) -> bool:
        x, y = self.winfo_pointerxy()
        left, top = self.winfo_rootx(), self.winfo_rooty()
        return left <= x <= left + self.winfo_width() and top <= y <= top + self.winfo_height()

    def _on_enter(self, event=None):
        if not self._removable:
            return
        if self._hide_after_id:
            self.after_cancel(self._hide_after_id)
            self._hide_after_id = None
        if self._pointer_inside():
            self._remove_btn.place(relx=1.0, rely=0, anchor="ne", x=-4, y=4)

    def _on_leave(self, event=None):
        if self._hide_after_id:
            self.after_cancel(self._hide_after_id)
        self._hide_after_id = self.after(80, self._hide_remove_if_outside)

    def _hide_remove_if_outside(self) -> None:
        self._hide_after_id = None
        if not self._pointer_inside():
            self._remove_btn.place_forget()

    def _handle_edit_segments(self):
        if self._on_edit_segments and self._segments_editable:
            for tooltip in self._segment_tooltips:
                tooltip.hide()
            self._on_edit_segments()

    def _handle_play(self) -> None:
        if self._on_play and self._player_enabled:
            self._on_play()

    def _show_action_menu(self, event=None):
        if not self._action_menu_visible:
            return "break"
        menu = self._action_menu
        if menu is None:
            menu = tkinter.Menu(self, tearoff=False)
            menu.bind("<Unmap>", lambda _event: menu.grab_release())
            self._action_menu = menu
        else:
            menu.delete(0, "end")
        menu.add_command(
            label=t("open_containing_folder"),
            command=self._on_open_containing_folder,
        )
        menu.add_command(label=t("copy_path"), command=self._on_copy_path)
        if self._has_restored_output:
            menu.add_command(
                label=t("open_restored_output"),
                command=self._on_open_restored_output,
            )
        if self._requeueable:
            menu.add_separator()
            menu.add_command(label=t("requeue"), command=self._on_requeue)
        if event is None:
            x = self._overflow_btn.winfo_rootx()
            y = self._overflow_btn.winfo_rooty() + self._overflow_btn.winfo_height()
        else:
            x, y = event.x_root, event.y_root
        menu.tk_popup(x, y)
        return "break"

    def set_segment_summary(self, text: str, *, selected: bool = False) -> None:
        self._segment_summary.configure(
            text=text,
            text_color=Colors.PRIMARY if selected else Colors.STATUS_PENDING,
        )

    def set_segments_editable(self, editable: bool) -> None:
        if self._on_edit_segments:
            self._segments_editable = bool(editable)
            if editable:
                self._segments_btn.configure(state="normal")
                if not self._segments_btn.winfo_manager():
                    self._segments_btn.pack(side="right", padx=(0, 6))
            else:
                for tooltip in self._segment_tooltips:
                    tooltip.hide()
                self._segments_btn.pack_forget()
            self._segment_summary.configure(cursor="hand2" if editable else "arrow")

    def set_player_enabled(self, enabled: bool) -> None:
        if self._on_play:
            self._player_enabled = bool(enabled)
            if enabled:
                self._play_btn.configure(state="normal")
                if not self._play_btn.winfo_manager():
                    pack_options = {"side": "right", "padx": (0, 6)}
                    if self._overflow_btn.winfo_manager():
                        pack_options["before"] = self._overflow_btn
                    self._play_btn.pack(**pack_options)
            else:
                self._play_btn.pack_forget()

    def set_action_menu_visible(self, visible: bool) -> None:
        self._action_menu_visible = bool(visible)
        if visible:
            if not self._overflow_btn.winfo_manager():
                self._overflow_btn.pack(side="right")
        else:
            self._overflow_btn.pack_forget()

    def set_action_options(
        self, *, has_restored_output: bool, requeueable: bool
    ) -> None:
        self._has_restored_output = bool(has_restored_output)
        self._requeueable = bool(requeueable)

    def set_status(self, status: str, icon: str = "", color: str = Colors.STATUS_PENDING):
        self._status_label.configure(text=status, text_color=color)
        if icon:
            self._status_icon.configure(text=icon, text_color=color)
            self._status_icon.pack(side="left", padx=(0, 4), before=self._status_label)
        else:
            self._status_icon.pack_forget()

    def set_removable(self, removable: bool):
        self._removable = removable
        if not removable:
            self._remove_btn.place_forget()

    def set_progress(self, value: float):
        if not self._progress_visible:
            self._progress.place(relx=0, rely=1.0, anchor="sw", relwidth=1.0)
            self._progress_visible = True
        self._progress.set(value)

    def set_fps_eta(self, fps: float = 0.0, eta_seconds: float = 0.0):
        """Update small FPS and ETA labels shown on the tile."""
        if fps and fps > 0:
            self._fps_label.configure(text=f"{fps:.1f}fps")
        else:
            self._fps_label.configure(text="")

        if eta_seconds and eta_seconds > 0:
            self._eta_label.configure(text=f"ETA: {format_duration(eta_seconds)}")
        else:
            self._eta_label.configure(text="")

    def set_completed(self, elapsed_seconds: float):
        self._status_label.configure(text=f"{t('completed_in')} {format_duration(elapsed_seconds)}")
        self._fps_label.configure(text="")
        self._eta_label.configure(text="")
        
    def hide_progress(self):
        self._progress.place_forget()
        self._progress_visible = False

    def set_conflict(self, has_conflict: bool) -> None:
        if has_conflict and not self._conflict_visible:
            self._conflict_dot.pack(side="left", before=self._handle)
        elif not has_conflict and self._conflict_visible:
            self._conflict_dot.pack_forget()
        self._conflict_visible = has_conflict
