from __future__ import annotations

import pytest

from jasna.gui.mosaic_scan import (
    MosaicScanResult,
    scan_sample_stride,
    segments_from_scores,
)
from jasna.gui.segment_editor_state import SegmentEditorState
from jasna.segments import SegmentRange


def test_stride_follows_fps():
    assert scan_sample_stride(29.97) == 30
    assert scan_sample_stride(60.0, seconds=0.5) == 30
    assert scan_sample_stride(1.0) == 1
    assert scan_sample_stride(0.1) == 1


def test_consecutive_hits_merge_into_one_range():
    times = (0.0, 1.0, 2.0, 3.0, 4.0)
    scores = (0.0, 0.8, 0.9, 0.7, 0.0)
    segments = segments_from_scores(
        times, scores, threshold=0.5, stride=1.0, duration=10.0
    )
    assert segments == (SegmentRange(0.5, 4.5),)


def test_isolated_hits_stay_separate():
    times = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    scores = (0.9, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0)
    segments = segments_from_scores(
        times, scores, threshold=0.5, stride=1.0, duration=10.0, pad=0.25
    )
    assert segments == (SegmentRange(0.0, 1.25), SegmentRange(2.75, 4.25))


def test_threshold_filters_hits():
    times = (0.0, 1.0)
    scores = (0.3, 0.6)
    assert segments_from_scores(times, scores, threshold=0.7, stride=1.0, duration=5.0) == ()
    low = segments_from_scores(times, scores, threshold=0.2, stride=1.0, duration=5.0)
    assert low == (SegmentRange(0.0, 2.5),)


def test_ranges_clamped_to_video():
    segments = segments_from_scores(
        (0.0, 9.0), (0.9, 0.9), threshold=0.5, stride=1.0, duration=9.8
    )
    assert segments[0].start == 0.0
    assert segments[-1].end == pytest.approx(9.8)


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        segments_from_scores((0.0,), (0.5, 0.6), threshold=0.5, stride=1.0, duration=5.0)


def test_mask_lookup_picks_nearest_sample():
    result = MosaicScanResult(
        times=(0.0, 1.0, 2.0),
        scores=(0.1, 0.9, 0.2),
        masks=["m0", "m1", "m2"],
        stride=1.0,
        duration=10.0,
        completed_until=2.0,
    )
    assert result.mask_at(1.2) == "m1"
    assert result.mask_at(2.4) == "m2"
    assert result.mask_at(5.0) is None


def test_add_many_is_single_undo_step():
    state = SegmentEditorState(duration=100.0, fps=25.0)
    added = state.add_many((SegmentRange(1.0, 2.0), SegmentRange(5.0, 7.0)))
    assert added == 2
    assert len(state.segments) == 2
    assert state.undo()
    assert state.segments == ()


def test_add_many_skips_already_covered_ranges():
    state = SegmentEditorState(duration=100.0, fps=25.0)
    state.add(0.0, 10.0)
    assert state.add_many((SegmentRange(2.0, 3.0),)) == 0
    added = state.add_many((SegmentRange(2.0, 3.0), SegmentRange(20.0, 21.0)))
    assert added == 1
    assert state.segments == (SegmentRange(0.0, 10.0), SegmentRange(20.0, 21.0))
