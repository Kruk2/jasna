from __future__ import annotations

import pytest

from jasna.gui.segment_editor_state import SegmentEditorState
from jasna.segments import SegmentRange


def test_empty_editor_means_full_video() -> None:
    state = SegmentEditorState(duration=10, fps=30)

    assert state.output_segments == ()
    assert not state.dirty


def test_add_snaps_to_frames_and_selects_new_range() -> None:
    state = SegmentEditorState(duration=10, fps=10)

    state.add(1.04, 2.06)

    assert state.output_segments == (SegmentRange(1.0, 2.1),)
    assert state.selected_segment == SegmentRange(1.0, 2.1)
    assert state.selected_duration == pytest.approx(1.1)


def test_overlapping_ranges_merge_without_losing_selection() -> None:
    state = SegmentEditorState(
        duration=10,
        fps=10,
        segments=(SegmentRange(1, 2), SegmentRange(4, 5)),
    )
    state.select(None)

    result = state.add(1.5, 4.5)

    assert result.merged_count == 2
    assert state.segments == (SegmentRange(1, 5),)
    assert state.selected_index == 0


def test_replace_selected_can_merge_with_neighbor() -> None:
    state = SegmentEditorState(
        duration=10,
        fps=10,
        segments=(SegmentRange(1, 2), SegmentRange(4, 5)),
    )

    result = state.replace_selected(1, 4.2)

    assert result.merged_count == 1
    assert state.segments == (SegmentRange(1, 5),)


def test_delete_undo_and_redo_restore_range_state() -> None:
    state = SegmentEditorState(
        duration=10,
        fps=30,
        segments=(SegmentRange(1, 2), SegmentRange(3, 4)),
    )
    state.select(1)

    assert state.delete_selected()
    assert state.segments == (SegmentRange(1, 2),)
    assert state.undo()
    assert state.segments == (SegmentRange(1, 2), SegmentRange(3, 4))
    assert state.selected_index == 1
    assert state.redo()
    assert state.segments == (SegmentRange(1, 2),)


def test_clearing_all_ranges_means_full_video() -> None:
    original = (SegmentRange(1, 2),)
    state = SegmentEditorState(duration=10, fps=30, segments=original)

    assert state.clear()

    assert state.output_segments == ()
    assert state.dirty
    assert state.undo()

    assert state.output_segments == original
    assert not state.dirty


def test_invalid_or_subframe_range_is_rejected() -> None:
    state = SegmentEditorState(duration=10, fps=30)

    with pytest.raises(ValueError, match="greater than start"):
        state.add(1.001, 1.002)
