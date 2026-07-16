from __future__ import annotations

from tkinter import TclError

import customtkinter as ctk
import pytest

from jasna.gui.queue_panel import QueuePanel


def test_queue_footer_stacks_count_above_action_buttons() -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    try:
        root.geometry("320x800")
        panel = QueuePanel(root)
        panel.pack(fill="both", expand=True)
        root.update_idletasks()

        count_bottom = panel._queue_count.winfo_rooty() + panel._queue_count.winfo_height()
        actions_top = min(
            panel._clear_completed_btn.winfo_rooty(),
            panel._clear_btn.winfo_rooty(),
        )
        assert count_bottom <= actions_top

        empty_content_width = panel._empty_state.winfo_width() - 40
        assert panel._empty_label.winfo_reqwidth() <= empty_content_width
    finally:
        root.destroy()
