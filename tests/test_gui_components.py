from __future__ import annotations

from unittest.mock import MagicMock

from jasna.gui.components import JobListItem


def test_segment_tooltips_hide_before_editor_opens() -> None:
    item = object.__new__(JobListItem)
    item._segments_editable = True
    item._on_edit_segments = MagicMock()
    item._segment_tooltips = [MagicMock(), MagicMock()]

    JobListItem._handle_edit_segments(item)

    for tooltip in item._segment_tooltips:
        tooltip.hide.assert_called_once_with()
    item._on_edit_segments.assert_called_once_with()
