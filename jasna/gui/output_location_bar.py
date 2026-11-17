"""Output folder and file name pattern fields under the job queue."""

from tkinter import filedialog
from typing import Callable

import customtkinter as ctk

from jasna.gui.components import Tooltip
from jasna.gui.icons import create_icon
from jasna.gui.locales import t
from jasna.gui.models import DEFAULT_OUTPUT_PATTERN
from jasna.gui.theme import Colors, Fonts, Sizing


class OutputLocationBar(ctk.CTkFrame):
    """``on_edited`` runs after the user types in either field, ``on_folder_set`` after the
    folder is chosen with a button or set by the program."""

    def __init__(self, master, *, on_edited: Callable[[], None], on_folder_set: Callable[[], None]):
        super().__init__(master, fg_color="transparent")
        self._on_edited = on_edited
        self._on_folder_set = on_folder_set

        output_label_row = ctk.CTkFrame(self, fg_color="transparent")
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
            command=lambda: self.set_folder(""),
        )
        self._same_as_input_btn.pack(side="right")
        output_tip = ctk.CTkLabel(output_label_row, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        output_tip.pack(side="left", padx=4)
        Tooltip(output_tip, t("tip_output_location"))

        output_row = ctk.CTkFrame(self, fg_color="transparent")
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
        self._output_entry.bind("<KeyRelease>", self._on_folder_typed)

        self._output_browse_btn = ctk.CTkButton(
            output_row,
            text="",
            image=create_icon("folder", 16, Colors.TEXT_PRIMARY),
            fg_color=Colors.BG_CARD,
            hover_color=Colors.BORDER_LIGHT,
            text_color=Colors.TEXT_PRIMARY,
            width=32,
            height=Sizing.INPUT_HEIGHT,
            command=self._on_browse,
        )
        self._output_browse_btn.pack(side="right")

        pattern_label_row = ctk.CTkFrame(self, fg_color="transparent")
        pattern_label_row.pack(fill="x", pady=(4, 2))
        pattern_label = ctk.CTkLabel(
            pattern_label_row,
            text=t("output_pattern"),
            font=(Fonts.FAMILY, Fonts.SIZE_TINY, "bold"),
            text_color=Colors.TEXT_PRIMARY,
            anchor="w",
        )
        pattern_label.pack(side="left")
        pattern_tip = ctk.CTkLabel(pattern_label_row, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        pattern_tip.pack(side="left", padx=4)
        Tooltip(pattern_tip, t("tip_output_pattern"))
        self._pattern_entry = ctk.CTkEntry(
            self,
            placeholder_text=DEFAULT_OUTPUT_PATTERN,
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            fg_color=Colors.BG_CARD,
            border_color=Colors.BORDER,
            text_color=Colors.TEXT_PRIMARY,
            height=Sizing.INPUT_HEIGHT,
        )
        self._pattern_entry.pack(fill="x", pady=(4, 0))
        self._pattern_entry.insert(0, DEFAULT_OUTPUT_PATTERN)
        self._pattern_entry.bind("<KeyRelease>", lambda _event: self._on_edited())

    @property
    def folder(self) -> str:
        return self._output_entry.get() or ""

    @property
    def pattern(self) -> str:
        return self._pattern_entry.get() or DEFAULT_OUTPUT_PATTERN

    def set_folder(self, folder: str) -> None:
        self._output_entry.configure(state="normal")
        self._output_entry.delete(0, "end")
        if folder:
            self._output_entry.insert(0, folder)
        self._update_same_as_input_style()
        self._on_folder_set()

    def set_pattern(self, pattern: str) -> None:
        self._pattern_entry.delete(0, "end")
        self._pattern_entry.insert(0, pattern)

    def set_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self._output_entry.configure(state=state)
        self._output_browse_btn.configure(state=state)
        self._same_as_input_btn.configure(state=state)
        self._pattern_entry.configure(state=state)

    def _on_browse(self) -> None:
        folder = filedialog.askdirectory(title=t("select_output_folder"))
        if folder:
            self.set_folder(folder)

    def _on_folder_typed(self, event=None) -> None:
        self._update_same_as_input_style()
        self._on_edited()

    def _update_same_as_input_style(self) -> None:
        same_as_input = not self.folder
        self._same_as_input_btn.configure(
            fg_color=Colors.PRIMARY if same_as_input else Colors.BG_CARD,
            hover_color=Colors.PRIMARY_HOVER if same_as_input else Colors.BORDER_LIGHT,
        )
