"""Range panel of the segment editor: the range list with undo/redo, new and clear, and the
start/end editor; the restoration inspector grids itself into ``controls``."""

from __future__ import annotations

from collections.abc import Callable
from tkinter import messagebox

import customtkinter as ctk

from jasna.gui import scaling
from jasna.gui.components import AutoHidingScrollableFrame, ScrollableFrame, Tooltip
from jasna.gui.locales import t
from jasna.gui.segment_editor_state import RangeOutOfBoundsError, SegmentEditorState
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.segments import SegmentRange, format_timestamp, parse_timestamp

RANGE_LIST_MIN_HEIGHT = 3 * 44 + 24
RANGE_TIMES_ONE_ROW_MIN_WIDTH = 520


def range_controls_height(shared: int, content: int, list_min: int) -> int:
    """Height of the range controls under the range list, which share ``shared`` pixels:
    all of ``content`` while the list keeps ``list_min``, else the rest (never under half)."""
    return max(0, min(content, max(shared - list_min, shared // 2)))


def merged_notice(merged_count: int) -> str | None:
    return t("segments_merged", count=merged_count) if merged_count else None


class RangePanel(ctk.CTkFrame):
    """Edits the ranges of ``state``. Reports the editor notice through ``on_notice(text, warning)``
    and every change of the ranges or the selection through ``on_changed``."""

    def __init__(
        self,
        master,
        *,
        state: SegmentEditorState,
        ltx_label: str,
        current_seconds: Callable[[], float],
        scanning: Callable[[], bool],
        on_seek: Callable[[float], None],
        on_show: Callable[[float], None],
        on_notice: Callable[[str | None, bool], None],
        on_changed: Callable[[], None],
    ) -> None:
        super().__init__(master, fg_color=Colors.BG_CARD, corner_radius=Sizing.BORDER_RADIUS)
        self._state = state
        self._ltx_label = ltx_label
        self._current_seconds = current_seconds
        self._scanning = scanning
        self._on_seek = on_seek
        self._on_show = on_show
        self._on_notice = on_notice
        self._on_changed = on_changed
        self._segment_action_widgets: list = []
        self._segment_rows: list = []
        self._segment_rows_key: tuple | None = None
        self._highlighted_row: int | None = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        range_header = ctk.CTkFrame(self, fg_color="transparent")
        range_header.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))
        ctk.CTkLabel(
            range_header,
            text=t("segments_ranges"),
            font=(Fonts.FAMILY, Fonts.SIZE_HEADING, "bold"),
            text_color=Colors.TEXT_PRIMARY,
        ).pack(side="left")
        self._undo_btn = ctk.CTkButton(
            range_header,
            text="↶",
            width=30,
            height=26,
            fg_color=Colors.BG_PANEL,
            hover_color=Colors.BORDER_LIGHT,
            command=self.undo,
        )
        Tooltip(self._undo_btn, t("segments_undo"))
        self._redo_btn = ctk.CTkButton(
            range_header,
            text="↷",
            width=30,
            height=26,
            fg_color=Colors.BG_PANEL,
            hover_color=Colors.BORDER_LIGHT,
            command=self.redo,
        )
        self._redo_btn.pack(side="right")
        self._undo_btn.pack(side="right", padx=4)
        Tooltip(self._redo_btn, t("segments_redo"))

        self._segment_list = ScrollableFrame(
            self,
            fg_color=Colors.BG_PANEL,
            corner_radius=Sizing.BORDER_RADIUS,
        )
        self._segment_list.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)

        list_actions = ctk.CTkFrame(self, fg_color="transparent")
        list_actions.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        self._new_btn = ctk.CTkButton(
            list_actions,
            text=t("segments_new_range"),
            height=28,
            command=self._new_range,
        )
        self._new_btn.pack(side="left")
        self._clear_btn = ctk.CTkButton(
            list_actions,
            text=t("segments_clear_all"),
            height=28,
            fg_color=Colors.BG_PANEL,
            hover_color=Colors.STATUS_ERROR,
            command=self._clear_ranges,
        )
        self._clear_btn.pack(side="right")

        self.controls = AutoHidingScrollableFrame(self, fg_color="transparent", height=1)
        self.controls.grid(row=3, column=0, sticky="nsew", padx=4, pady=(0, 4))
        self.controls.grid_columnconfigure(0, weight=1)
        self.bind("<Configure>", self._fit_range_controls, add="+")
        self.controls.bind("<Configure>", self._fit_range_controls, add="+")

        editor = ctk.CTkFrame(self.controls, fg_color="transparent")
        editor.grid(row=0, column=0, sticky="ew", padx=4, pady=(2, 4))
        editor.bind("<Configure>", self._layout_range_times, add="+")
        self._range_times: list[tuple[ctk.CTkLabel, ctk.CTkEntry, ctk.CTkButton]] = []
        for column, (label, entry_name, mark_name, mark_text, mark_hint, mark_command) in enumerate(
            (
                ("segments_start", "_start_entry", "_mark_in_btn", "segments_mark_in_short",
                 "segments_mark_in_hint", self.set_mark_in),
                ("segments_end", "_end_entry", "_mark_out_btn", "segments_mark_out_short",
                 "segments_mark_out_hint", self.set_mark_out),
            )
        ):
            text = ctk.CTkLabel(editor, text=t(label), text_color=Colors.TEXT_PRIMARY)
            entry = ctk.CTkEntry(editor, width=40, font=(Fonts.FAMILY_MONO, Fonts.SIZE_SMALL))
            setattr(self, entry_name, entry)
            mark = ctk.CTkButton(editor, text=t(mark_text), width=30, command=mark_command)
            Tooltip(mark, t(mark_hint))
            setattr(self, mark_name, mark)
            self._range_times.append((text, entry, mark))
        self._range_action = ctk.CTkButton(
            editor,
            text=t("segments_add_range"),
            command=self.add_or_update,
        )
        self._range_times_one_row: bool | None = None
        self._layout_range_times()

    def _layout_range_times(self, event=None) -> None:
        """Start, End and the add/update button share one row when the panel is wide enough, else stack."""
        width = RANGE_TIMES_ONE_ROW_MIN_WIDTH if event is None else event.width
        margin = 24 if self._range_times_one_row is False else 0
        one_row = width >= scaling.raw_tk_size(self, RANGE_TIMES_ONE_ROW_MIN_WIDTH + margin)
        if one_row == self._range_times_one_row:
            return
        self._range_times_one_row = one_row
        editor = self._range_action.master
        for column in range(7):
            editor.grid_columnconfigure(column, weight=0)
        editor.grid_columnconfigure((1, 4) if one_row else 1, weight=1)
        for index, (text, entry, mark) in enumerate(self._range_times):
            row, first = (0, index * 3) if one_row else (index, 0)
            text.grid(row=row, column=first, sticky="w", padx=(0 if first == 0 else 8, 4), pady=3)
            entry.grid(row=row, column=first + 1, sticky="ew", pady=3)
            mark.grid(row=row, column=first + 2, padx=(3, 0), pady=3)
        if one_row:
            self._range_action.grid(row=0, column=6, columnspan=1, sticky="ew", padx=(8, 0), pady=3)
        else:
            self._range_action.grid(row=2, column=0, columnspan=3, sticky="ew", padx=0, pady=(5, 0))

    def _fit_range_controls(self, _event=None) -> None:
        """Size the range controls to their content, but keep room for about three list rows."""
        fixed = self.grid_bbox(0, 0)[3] + self.grid_bbox(0, 2)[3] + scaling.raw_tk_size(self, 4)
        height = range_controls_height(
            self.winfo_height() - fixed,
            self.controls.winfo_reqheight(),
            scaling.raw_tk_size(self, RANGE_LIST_MIN_HEIGHT),
        )
        logical = max(1, round(height / scaling.widget_scaling(self)))
        if logical != self.controls.cget("height"):
            self.controls.configure(height=logical)

    def lockable_widgets(self) -> tuple:
        return (
            self._new_btn,
            self._clear_btn,
            self._undo_btn,
            self._redo_btn,
            self._range_action,
            self._mark_in_btn,
            self._mark_out_btn,
            self._start_entry,
            self._end_entry,
            *self._segment_action_widgets,
        )

    def refresh(self) -> None:
        state = self._state
        if state.selected_segment is not None:
            _set_entry(self._start_entry, state.selected_segment.start)
            _set_entry(self._end_entry, state.selected_segment.end)
        self._range_action.configure(
            text=(
                t("segments_update_range")
                if state.selected_segment is not None
                else t("segments_add_range")
            )
        )
        self._undo_btn.configure(
            state="normal" if state.can_undo and not self._scanning() else "disabled"
        )
        self._redo_btn.configure(
            state="normal" if state.can_redo and not self._scanning() else "disabled"
        )
        self._render_segment_list()

    def _edited(self, notice: str | None = None) -> None:
        self._on_notice(notice, notice is not None)
        self._on_changed()

    def _new_range(self) -> None:
        state = self._state
        state.select(None)
        state.clear_marks()
        start, end = state.new_range_at(self._current_seconds())
        _set_entry(self._start_entry, start)
        _set_entry(self._end_entry, end)
        self._edited()
        self._start_entry.focus_set()

    def set_mark_in(self) -> None:
        _set_entry(self._start_entry, self._state.set_mark_in(self._current_seconds()))
        self._on_notice(None, False)

    def set_mark_out(self) -> None:
        _set_entry(self._end_entry, self._state.set_mark_out(self._current_seconds()))
        self._on_notice(None, False)

    def add_or_update(self) -> None:
        state = self._state
        try:
            result = state.replace_selected(
                parse_timestamp(self._start_entry.get()), parse_timestamp(self._end_entry.get())
            )
        except RangeOutOfBoundsError:
            self._on_notice(t("segments_time_out_of_bounds"), False)
            return
        except ValueError:
            self._on_notice(t("segments_invalid_range"), False)
            return
        self._on_notice(merged_notice(result.merged_count), bool(result.merged_count))
        self._on_show(state.selected_segment.start)
        self._on_changed()

    def create(self, start: float, end: float) -> None:
        result = self._state.add(start, end)
        self._edited(merged_notice(result.merged_count))

    def adjust(self, index: int, start: float, end: float) -> None:
        self._state.select(index)
        result = self._state.replace_selected(start, end)
        self._edited(merged_notice(result.merged_count))

    def select(self, index: int) -> None:
        state = self._state
        state.select(index)
        segment = state.selected_segment
        if segment is not None:
            _set_entry(self._start_entry, segment.start)
            _set_entry(self._end_entry, segment.end)
        self._on_changed()

    def _select_from_list(self, index: int) -> None:
        self.select(index)
        selected = self._state.selected_segment
        if selected is not None:
            self._on_seek(selected.start)

    def _delete_range(self, index: int) -> None:
        self._state.select(index)
        self._state.delete_selected()
        self._edited()

    def delete_selected(self) -> None:
        if self._state.delete_selected():
            self._edited()

    def _clear_ranges(self) -> None:
        state = self._state
        if not state.segments:
            return
        if not messagebox.askyesno(
            t("segments_clear_all"),
            t("segments_clear_confirm"),
            parent=self,
        ):
            return
        state.clear()
        self._edited()

    def undo(self) -> None:
        if self._state.undo():
            self._edited()

    def redo(self) -> None:
        if self._state.redo():
            self._edited()

    def _render_segment_list(self) -> None:
        """Rebuild the rows only when the ranges change; a new selection recolors its
        row and scrolls the list to it."""
        state = self._state
        key = (state.segments, self._scanning())
        if key != self._segment_rows_key:
            self._segment_rows_key = key
            self._highlighted_row = None
            self._build_segment_rows(state)
        selected = state.selected_index if state.selected_segment is not None else None
        if selected == self._highlighted_row:
            return
        if self._highlighted_row is not None:
            self._segment_rows[self._highlighted_row].configure(fg_color=Colors.BG_CARD)
        self._highlighted_row = selected
        if selected is not None:
            self._segment_rows[selected].configure(fg_color=Colors.PRIMARY_DARK)
            self._reveal_segment_row(selected)

    def _reveal_segment_row(self, index: int) -> None:
        self._segment_list.update_idletasks()
        content_height = self._segment_list.winfo_height()
        if content_height <= 1:
            return
        row = self._segment_rows[index]
        canvas = self._segment_list._parent_canvas
        top, bottom = canvas.yview()
        row_top = row.winfo_y() / content_height
        row_bottom = (row.winfo_y() + row.winfo_height()) / content_height
        if row_top < top or row_bottom > bottom:
            canvas.yview_moveto(row_top)

    def _model_text(self, segment: SegmentRange) -> str:
        restoration = segment.restoration
        if restoration.model == "ltx":
            return f"{self._ltx_label} {restoration.ltx_seed}"
        return t(f"model_{restoration.model}")

    def _build_segment_rows(self, state: SegmentEditorState) -> None:
        self._segment_action_widgets = []
        self._segment_rows = []
        for child in self._segment_list.winfo_children():
            child.destroy()
        if not state.segments:
            ctk.CTkLabel(
                self._segment_list,
                text=t("segments_drag_hint"),
                text_color=Colors.STATUS_PENDING,
                font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
                wraplength=230,
            ).pack(fill="x", padx=8, pady=14)
            return
        for index, segment in enumerate(state.segments):
            row = ctk.CTkFrame(
                self._segment_list,
                fg_color=Colors.BG_CARD,
                corner_radius=Sizing.BORDER_RADIUS,
            )
            row.pack(fill="x", pady=2)
            self._segment_rows.append(row)
            label = ctk.CTkButton(
                row,
                text=t(
                    "segments_range_row",
                    index=index + 1,
                    start=format_timestamp(segment.start),
                    end=format_timestamp(segment.end),
                    duration=segment.duration,
                    model=self._model_text(segment),
                ),
                anchor="w",
                fg_color="transparent",
                hover_color=Colors.BORDER_LIGHT,
                text_color=Colors.TEXT_PRIMARY,
                font=(Fonts.FAMILY_MONO, Fonts.SIZE_TINY),
                command=lambda i=index: self._select_from_list(i),
            )
            label.pack(side="left", fill="x", expand=True)
            label.configure(state="disabled" if self._scanning() else "normal")
            self._segment_action_widgets.append(label)
            delete = ctk.CTkButton(
                row,
                text="✕",
                width=28,
                height=28,
                fg_color="transparent",
                hover_color=Colors.STATUS_ERROR,
                command=lambda i=index: self._delete_range(i),
            )
            delete.pack(side="right", padx=3)
            delete.configure(state="disabled" if self._scanning() else "normal")
            self._segment_action_widgets.append(delete)
            Tooltip(delete, t("segments_delete_range"))


def _set_entry(entry: ctk.CTkEntry, seconds: float) -> None:
    entry.delete(0, "end")
    entry.insert(0, format_timestamp(seconds))
