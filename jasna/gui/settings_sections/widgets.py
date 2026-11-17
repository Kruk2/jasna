"""Shared widget helpers for the settings sections."""

import tkinter as tk
from collections.abc import Iterable
from dataclasses import dataclass

import customtkinter as ctk

from jasna.gui import scaling
from jasna.gui.components import Tooltip
from jasna.gui.icons import create_icon
from jasna.gui.locales import t
from jasna.gui.theme import Colors, Fonts


def get_tooltip(key: str) -> str:
    """Get localized tooltip for a setting key."""
    return t(f"tip_{key}")


def add_info_icon(master, tooltip: str) -> ctk.CTkLabel:
    """An ⓘ label showing ``tooltip`` on hover; the caller packs it."""
    icon = ctk.CTkLabel(master, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
    Tooltip(icon, tooltip)
    return icon


def add_setting_label(row, label_key: str, tooltip_key: str | None = None, *, in_card: bool = False) -> None:
    """Pack a setting's name and its ⓘ tooltip icon at the left of ``row``."""
    label = ctk.CTkLabel(row, text=t(label_key), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
    tip = add_info_icon(row, get_tooltip(tooltip_key or label_key))
    if in_card:
        label.pack(side="left", padx=12, pady=8)
        tip.pack(side="left")
    else:
        label.pack(side="left")
        tip.pack(side="left", padx=4)


def folder_button(master, command, *, background: str, height: int) -> ctk.CTkButton:
    """The folder-icon button beside a path entry; the caller packs it."""
    return ctk.CTkButton(
        master, text="", image=create_icon("folder", 16, Colors.TEXT_PRIMARY), width=32, height=height,
        fg_color=background, hover_color=Colors.BORDER_LIGHT, text_color=Colors.TEXT_PRIMARY, command=command,
    )


# One slider length and one right-aligned value column keep the rows lined up.
SLIDER_WIDTH = 200


def create_slider_value_label(master, background: str) -> tk.Label:
    """The value shown right of a slider; whoever sets the slider also sets this text."""
    return tk.Label(
        master,
        foreground=Colors.TEXT_PRIMARY,
        background=background,
        font=(Fonts.FAMILY, scaling.raw_tk_font_size(master, Fonts.SIZE_NORMAL)),
        width=4,
        anchor="e",
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


def option_menu(
    master, options: dict[str, str], command, *, width: int, background: str = Colors.BG_CARD, **style
) -> ValueOptionMenu:
    """A ``ValueOptionMenu`` in the settings look: ``background`` is the card colour on panel rows
    and the panel colour inside cards."""
    return ValueOptionMenu(
        master,
        options=options,
        command=command,
        fg_color=background,
        button_color=background,
        button_hover_color=Colors.BORDER_LIGHT,
        dropdown_fg_color=background,
        dropdown_hover_color=Colors.PRIMARY,
        text_color=Colors.TEXT_PRIMARY,
        width=width,
        **style,
    )


def set_widgets_enabled(widgets: Iterable, enabled: bool) -> None:
    state = "normal" if enabled else "disabled"
    for widget in widgets:
        widget.configure(state=state)


def bind_edits(entry: ctk.CTkEntry, on_edit) -> None:
    """Call ``on_edit()`` after each key the user types into ``entry``."""
    entry.bind("<KeyRelease>", lambda _event: on_edit())


def set_entry_text(entry: ctk.CTkEntry, text: str) -> None:
    entry.delete(0, "end")
    entry.insert(0, text)


def parse_int(text: str, fallback: int) -> int:
    try:
        return int(text.strip())
    except ValueError:
        return fallback


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
