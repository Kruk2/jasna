"""Shared widget helpers for the settings sections."""

import tkinter as tk

import customtkinter as ctk

from jasna.gui.locales import t
from jasna.gui.theme import Colors, Fonts


def get_tooltip(key: str) -> str:
    """Get localized tooltip for a setting key."""
    return t(f"tip_{key}")


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
        font=(Fonts.FAMILY, -Fonts.SIZE_NORMAL),
        width=width,
        borderwidth=0,
        highlightthickness=0,
    )


class ValueOptionMenu(ctk.CTkOptionMenu):
    """Option menu keyed by internal values; translated labels are display-only.

    ``options`` maps internal value -> display label at construction time, so
    reading the selection never requires a reverse lookup through translations.
    """

    def __init__(self, master, *, options: dict[str, str], command, **kwargs):
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
        self._value_command(self._label_to_value[label])

    def get_value(self) -> str:
        return self._label_to_value[self.get()]

    def set_value(self, value: str):
        label = self._value_to_label.get(value)
        if label is None:
            label = next(iter(self._value_to_label.values()))
        self.set(label)
