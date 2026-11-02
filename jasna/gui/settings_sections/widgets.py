"""Shared widget helpers for the settings sections."""

import tkinter as tk
from dataclasses import dataclass

import customtkinter as ctk

from jasna.gui import scaling
from jasna.gui.components import Tooltip
from jasna.gui.locales import t
from jasna.gui.theme import Colors, Fonts


def get_tooltip(key: str) -> str:
    """Get localized tooltip for a setting key."""
    return t(f"tip_{key}")


def add_setting_label(row, label_key: str, tooltip_key: str | None = None, *, in_card: bool = False) -> None:
    """Pack a setting's name and its ⓘ tooltip icon at the left of ``row``."""
    label = ctk.CTkLabel(row, text=t(label_key), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
    tip = ctk.CTkLabel(row, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
    if in_card:
        label.pack(side="left", padx=12, pady=8)
        tip.pack(side="left")
    else:
        label.pack(side="left")
        tip.pack(side="left", padx=4)
    Tooltip(tip, get_tooltip(tooltip_key or label_key))


def create_slider_value_label(
    master,
    text: str,
    width: int,
    background: str,
) -> tk.Label:
    return tk.Label(
        master,
        text=text,
        foreground=Colors.TEXT_PRIMARY,
        background=background,
        font=(Fonts.FAMILY, scaling.raw_tk_font_size(master, Fonts.SIZE_NORMAL)),
        width=width,
        borderwidth=0,
        highlightthickness=0,
    )


class ValueOptionMenu(ctk.CTkOptionMenu):
    """Option menu keyed by internal values; translated labels are display-only.

    ``options`` maps internal value -> display label at construction time, so
    reading the selection never requires a reverse lookup through translations.
    """

    def __init__(self, master, *, options: dict[str, str], command=None, **kwargs):
        self._value_to_label = dict(options)
        self._label_to_value = {label: value for value, label in options.items()}
        self._value_command = command
        super().__init__(
            master,
            values=list(self._value_to_label.values()),
            command=self._on_label_selected,
            **kwargs,
        )

    def _on_label_selected(self, label: str):
        if self._value_command is not None:
            self._value_command(self._label_to_value[label])

    def get_value(self) -> str:
        return self._label_to_value[self.get()]

    def offer(self, values) -> None:
        """Limit the menu to ``values``; the current selection stays as it is."""
        self.configure(values=[self._value_to_label[value] for value in values])

    def set_value(self, value: str):
        label = self._value_to_label.get(value)
        if label is None:
            label = next(iter(self._value_to_label.values()))
        self.set(label)


def pack_rows(rows: list[tuple[tk.Misc, dict]], hidden: tuple[tk.Misc, ...]) -> None:
    """Pack ``rows`` (widget, pack options) in order, leaving out ``hidden``."""
    for row, _options in rows:
        row.pack_forget()
    for row, options in rows:
        if row not in hidden:
            row.pack(**options)


LEVELS = ("simple", "advanced", "expert")


@dataclass(frozen=True)
class Row:
    """One packed settings row: the lowest view level that shows it and, when it only
    matters for one restoration model, that model ("standard" or "ltx")."""

    widget: tk.Misc
    pack: dict
    level: str
    model: str | None = None


def layout_rows(rows: list[Row], level: str, *, standard: bool, ltx: bool, hidden: tuple = ()) -> bool:
    """Pack the rows that belong to ``level`` and the job's models; True when any is shown."""
    used = {None: True, "standard": standard, "ltx": ltx}
    shown = [
        row for row in rows
        if LEVELS.index(row.level) <= LEVELS.index(level) and used[row.model] and row.widget not in hidden
    ]
    pack_rows([(row.widget, row.pack) for row in rows], tuple(row.widget for row in rows if row not in shown))
    return bool(shown)
