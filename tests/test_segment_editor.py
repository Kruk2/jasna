from __future__ import annotations

from unittest.mock import MagicMock

from jasna.gui.locales import t
from jasna.gui.segment_editor import SegmentEditor
from jasna.gui.segment_editor_state import SegmentEditorState


def test_out_of_bounds_range_uses_specific_message() -> None:
    editor = object.__new__(SegmentEditor)
    editor._state = SegmentEditorState(duration=10, fps=30)
    editor._start_entry = MagicMock()
    editor._start_entry.get.return_value = "9"
    editor._end_entry = MagicMock()
    editor._end_entry.get.return_value = "11"
    editor._refresh_notice = MagicMock()

    SegmentEditor._add_or_update(editor)

    assert editor._edit_notice == t("segments_time_out_of_bounds")
    editor._refresh_notice.assert_called_once_with()
