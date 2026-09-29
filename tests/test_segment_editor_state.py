from __future__ import annotations

from fractions import Fraction

import pytest

from jasna.gui.segment_editor_state import SegmentEditorState, smart_render_error_key
from jasna.media.splice import KeyframeIndex
from jasna.segments import SegmentRange, SegmentRestoration

STANDARD = SegmentRestoration("basicvsrpp", None)
LTX_7 = SegmentRestoration("ltx", 7)


def _r(start: float, end: float, restoration: SegmentRestoration = STANDARD) -> SegmentRange:
    return SegmentRange(start, end, restoration)


def test_empty_editor_means_full_video() -> None:
    state = SegmentEditorState(duration=10, fps=30, segments=(), default_restoration=STANDARD)

    assert state.output_segments == ()
    assert not state.dirty


def test_add_snaps_to_frames_and_selects_new_range() -> None:
    state = SegmentEditorState(duration=10, fps=10, segments=(), default_restoration=STANDARD)

    state.add(1.04, 2.06)

    assert state.output_segments == (_r(1.0, 2.1),)
    assert state.selected_segment == _r(1.0, 2.1)
    assert state.selected_duration == pytest.approx(1.1)


def test_overlapping_ranges_merge_without_losing_selection() -> None:
    state = SegmentEditorState(
        duration=10,
        fps=10,
        segments=(_r(1, 2), _r(4, 5)),
        default_restoration=STANDARD,
    )
    state.select(None)

    result = state.add(1.5, 4.5)

    assert result.merged_count == 2
    assert state.segments == (_r(1, 5),)
    assert state.selected_index == 0


def test_replace_selected_can_merge_with_neighbor() -> None:
    state = SegmentEditorState(
        duration=10,
        fps=10,
        segments=(_r(1, 2), _r(4, 5)),
        default_restoration=STANDARD,
    )

    result = state.replace_selected(1, 4.2)

    assert result.merged_count == 1
    assert state.segments == (_r(1, 5),)


def test_delete_undo_and_redo_restore_range_state() -> None:
    state = SegmentEditorState(
        duration=10,
        fps=30,
        segments=(_r(1, 2), _r(3, 4)),
        default_restoration=STANDARD,
    )
    state.select(1)

    assert state.delete_selected()
    assert state.segments == (_r(1, 2),)
    assert state.undo()
    assert state.segments == (_r(1, 2), _r(3, 4))
    assert state.selected_index == 1
    assert state.redo()
    assert state.segments == (_r(1, 2),)


def test_clearing_all_ranges_means_full_video() -> None:
    original = (_r(1, 2),)
    state = SegmentEditorState(duration=10, fps=30, segments=original, default_restoration=STANDARD)

    assert state.clear()

    assert state.output_segments == ()
    assert state.dirty
    assert state.undo()

    assert state.output_segments == original
    assert not state.dirty


def test_invalid_or_subframe_range_is_rejected() -> None:
    state = SegmentEditorState(duration=10, fps=30, segments=(), default_restoration=STANDARD)

    with pytest.raises(ValueError, match="greater than start"):
        state.add(1.001, 1.002)


def test_smart_render_error_key_explains_whole_video_reencode() -> None:
    index = KeyframeIndex(pts=(0,), time_base=Fraction(1, 1), start_pts=0, end_pts=60)

    assert smart_render_error_key((_r(0.0, 58.0),), index, 60.0) == "segments_smart_render_whole_video"


def test_new_and_scanned_ranges_get_the_default_restoration() -> None:
    state = SegmentEditorState(duration=10, fps=10, segments=(), default_restoration=LTX_7)

    state.add(1, 2)
    assert state.add_many((SegmentRange(4, 5), SegmentRange(6, 7))) == 2

    assert {segment.restoration for segment in state.segments} == {LTX_7}


def test_ranges_without_a_restoration_get_the_default_when_the_editor_opens() -> None:
    state = SegmentEditorState(duration=10, fps=10, segments=(SegmentRange(1, 2),), default_restoration=LTX_7)

    assert state.segments == (_r(1, 2, LTX_7),)
    assert not state.dirty


def test_moving_a_range_keeps_its_restoration() -> None:
    state = SegmentEditorState(
        duration=10, fps=10, segments=(_r(1, 2, LTX_7), _r(5, 6)), default_restoration=STANDARD
    )
    state.select(0)

    state.adjust_selected(1.5, 3)
    state.select(0)
    state.replace_selected(0.5, 3)

    assert state.segments == (_r(0.5, 3, LTX_7), _r(5, 6))


def test_setting_the_selected_restoration_is_one_undo_step() -> None:
    state = SegmentEditorState(duration=10, fps=10, segments=(_r(1, 2), _r(5, 6)), default_restoration=STANDARD)
    state.select(1)

    assert state.set_selected_restoration(LTX_7)
    assert not state.set_selected_restoration(LTX_7)
    assert state.segments == (_r(1, 2), _r(5, 6, LTX_7))
    assert state.selected_index == 1
    assert state.dirty

    assert state.undo()
    assert state.segments == (_r(1, 2), _r(5, 6))
    assert state.redo()
    assert state.segments[1].restoration == LTX_7


def test_touching_ranges_merge_once_they_share_a_restoration() -> None:
    state = SegmentEditorState(
        duration=10, fps=10, segments=(_r(1, 2), _r(2, 3, LTX_7)), default_restoration=STANDARD
    )
    assert len(state.segments) == 2
    state.select(1)

    state.set_selected_restoration(STANDARD)

    assert state.segments == (_r(1, 3),)
    assert state.selected_index == 0


def test_set_all_restoration_gives_every_range_one_choice() -> None:
    state = SegmentEditorState(duration=10, fps=10, segments=(_r(1, 2), _r(5, 6)), default_restoration=STANDARD)

    assert state.set_all_restoration(LTX_7)
    assert not state.set_all_restoration(LTX_7)

    assert state.segments == (_r(1, 2, LTX_7), _r(5, 6, LTX_7))
    assert state.undo()
    assert state.segments == (_r(1, 2), _r(5, 6))


def test_restoration_at_follows_the_range_under_the_playhead() -> None:
    state = SegmentEditorState(duration=10, fps=10, segments=(_r(1, 2, LTX_7),), default_restoration=STANDARD)

    assert state.restoration_at(1.5) == LTX_7
    assert state.restoration_at(2.0) == STANDARD
    assert state.restoration_at(0.5) == STANDARD


def test_smart_render_error_key_explains_mixed_models_too_close() -> None:
    index = KeyframeIndex(pts=(0, 10_000), time_base=Fraction(1, 1000), start_pts=0, end_pts=20_000)

    segments = (_r(1.0, 2.0), _r(3.0, 4.0, LTX_7))

    assert smart_render_error_key(segments, index, 20.0) == "segments_mixed_models_too_close"
