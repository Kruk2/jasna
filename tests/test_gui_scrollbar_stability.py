from __future__ import annotations

from tkinter import TclError

import customtkinter as ctk
import pytest

from jasna.gui.components import AutoHidingScrollableFrame
from jasna.gui.queue_panel import QueuePanel

TOGGLE_STORM_LIMIT = 25


@pytest.mark.parametrize("hidpi", [2.0], indirect=True)
def test_scrollbar_height_request_stays_out_of_layout_negotiation(hidpi) -> None:
    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    try:
        frame = AutoHidingScrollableFrame(root)
        frame.pack(fill="both", expand=True)
        root.update()

        assert frame._scrollbar.winfo_reqheight() < frame._parent_canvas.winfo_reqheight()
    finally:
        root.destroy()


@pytest.mark.parametrize("hidpi", [2.0], indirect=True)
def test_queue_panel_settles_under_height_shortage(hidpi) -> None:
    toggles = []
    orig_update = AutoHidingScrollableFrame._update_scrollbar

    def disarming_update(self, first, last):
        if len(toggles) > TOGGLE_STORM_LIMIT:
            return
        before = self._scrollbar_visible
        orig_update(self, first, last)
        if self._scrollbar_visible != before:
            toggles.append((first, last))

    AutoHidingScrollableFrame._update_scrollbar = disarming_update
    try:
        try:
            root = ctk.CTk()
        except TclError as exc:
            pytest.skip(f"Tk display unavailable: {exc}")

        try:
            root.geometry("420x300+50+50")
            panel = QueuePanel(root)
            panel.pack(fill="both", expand=True)

            for _ in range(30):
                root.update()
            settled = len(toggles)
            assert settled <= TOGGLE_STORM_LIMIT

            for _ in range(30):
                root.update()

            assert len(toggles) == settled
        finally:
            root.destroy()
    finally:
        AutoHidingScrollableFrame._update_scrollbar = orig_update
