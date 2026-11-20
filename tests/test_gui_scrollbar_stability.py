from __future__ import annotations


import sys

import customtkinter as ctk
import pytest

from jasna.gui.components import AutoHidingScrollableFrame, ScrollableFrame
from jasna.gui.queue_panel import QueuePanel

TOGGLE_STORM_LIMIT = 25


@pytest.mark.parametrize("hidpi", [2.0], indirect=True)
def test_scrollbar_height_request_stays_out_of_layout_negotiation(hidpi, tk_root) -> None:
    frame = AutoHidingScrollableFrame(tk_root)
    frame.pack(fill="both", expand=True)
    tk_root.update()

    assert frame._scrollbar.winfo_reqheight() < frame._parent_canvas.winfo_reqheight()


@pytest.mark.parametrize("hidpi", [2.0], indirect=True)
def test_queue_panel_settles_under_height_shortage(hidpi, tk_root, monkeypatch) -> None:
    toggles = []
    orig_update = AutoHidingScrollableFrame._update_scrollbar

    def disarming_update(self, first, last):
        if len(toggles) > TOGGLE_STORM_LIMIT:
            return
        before = self._scrollbar_visible
        orig_update(self, first, last)
        if self._scrollbar_visible != before:
            toggles.append((first, last))

    monkeypatch.setattr(AutoHidingScrollableFrame, "_update_scrollbar", disarming_update)
    tk_root.geometry("420x300+50+50")
    panel = QueuePanel(tk_root)
    panel.pack(fill="both", expand=True)

    for _ in range(30):
        tk_root.update()
    settled = len(toggles)
    assert settled <= TOGGLE_STORM_LIMIT

    for _ in range(30):
        tk_root.update()

    assert len(toggles) == settled


def _wheel_down(root, widget_path: str) -> None:
    wheel = ("<Button-5>",) if "linux" in sys.platform else ("<MouseWheel>", "-delta", -120)
    root.tk.call("event", "generate", widget_path, *wheel, "-when", "now")


@pytest.mark.parametrize("frame_class", [ScrollableFrame, AutoHidingScrollableFrame])
def test_wheel_over_a_widget_tkinter_does_not_know_raises_nothing(tk_root, frame_class) -> None:
    errors = []
    tk_root.report_callback_exception = lambda _type, exc, _tb: errors.append(exc)
    tk_root.geometry("300x200")
    frame = frame_class(tk_root)
    frame.pack(fill="both", expand=True)
    rows = [ctk.CTkLabel(frame, text=f"row {i}") for i in range(40)]
    for row in rows:
        row.pack()
    # Tk-made widgets such as file dialogs or menu clones reach Tkinter handlers as path strings.
    tk_root.tk.call("frame", ".tk_only")
    tk_root.tk.call("place", ".tk_only", "-x", 0, "-y", 0)
    tk_root.update()

    _wheel_down(tk_root, ".tk_only")
    _wheel_down(tk_root, str(rows[0]))
    tk_root.update()

    assert errors == []
    assert frame._parent_canvas.yview()[0] > 0.0
