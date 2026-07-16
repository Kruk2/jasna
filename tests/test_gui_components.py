from __future__ import annotations

from unittest.mock import MagicMock

from jasna.gui.components import JobListItem
from jasna.gui.control_bar import ControlBar


def test_segment_tooltips_hide_before_editor_opens() -> None:
    item = object.__new__(JobListItem)
    item._segments_editable = True
    item._on_edit_segments = MagicMock()
    item._segment_tooltips = [MagicMock(), MagicMock()]

    JobListItem._handle_edit_segments(item)

    for tooltip in item._segment_tooltips:
        tooltip.hide.assert_called_once_with()
    item._on_edit_segments.assert_called_once_with()


def test_enabling_start_button_hides_disabled_tooltip() -> None:
    control_bar = object.__new__(ControlBar)
    tooltip = MagicMock()
    control_bar._start_disabled_tooltip = tooltip
    control_bar._start_btn = MagicMock()
    control_bar._start_btn_normal_fg = "normal"
    control_bar._start_btn_normal_hover = "hover"

    ControlBar.set_start_enabled(control_bar, True)

    tooltip.hide.assert_called_once_with()
    control_bar._start_btn.configure.assert_called_once_with(
        state="normal",
        fg_color="normal",
        hover_color="hover",
    )


def test_updating_disabled_start_button_hides_previous_tooltip() -> None:
    control_bar = object.__new__(ControlBar)
    tooltip = MagicMock()
    control_bar._start_disabled_tooltip = tooltip
    control_bar._start_btn = MagicMock()

    ControlBar.set_start_enabled(control_bar, False)

    tooltip.hide.assert_called_once_with()
