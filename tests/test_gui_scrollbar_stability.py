from __future__ import annotations


import pytest

from jasna.gui.components import AutoHidingScrollableFrame
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
