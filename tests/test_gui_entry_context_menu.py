"""Right-click clipboard menu on entries (license dialog paste support)."""

from __future__ import annotations

from tkinter import TclError

import customtkinter as ctk
import pytest

from jasna.gui.components import attach_entry_context_menu
from jasna.gui.locales import t


@pytest.fixture
def root():
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    yield root
    root.destroy()


def test_menu_has_clipboard_actions(root):
    entry = ctk.CTkEntry(root)
    menu = attach_entry_context_menu(entry)

    labels = [menu.entrycget(i, "label") for i in (0, 1, 2, 4)]
    assert labels == [t("ctx_cut"), t("ctx_copy"), t("ctx_paste"), t("ctx_select_all")]
    assert menu.type(3) == "separator"


def test_right_click_is_bound(root):
    entry = ctk.CTkEntry(root)
    attach_entry_context_menu(entry)

    assert entry._entry.bind("<Button-3>")


def test_paste_inserts_clipboard_text(root):
    entry = ctk.CTkEntry(root)
    entry.pack()
    menu = attach_entry_context_menu(entry)

    entry.clipboard_clear()
    entry.clipboard_append("user@example.com")
    root.update()
    entry._entry.focus_force()
    root.update()
    menu.invoke(2)
    root.update()

    assert entry.get() == "user@example.com"


def test_select_all_selects_entry_text(root):
    entry = ctk.CTkEntry(root)
    menu = attach_entry_context_menu(entry)

    entry.insert(0, "some-key")
    menu.invoke(4)

    assert entry._entry.selection_get() == "some-key"
