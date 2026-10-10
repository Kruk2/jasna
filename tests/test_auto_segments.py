"""Automatic segmentation: scores in, render ranges out. No GPU required."""

from __future__ import annotations

import pytest

from jasna.mosaic.auto_segments import (
    AutoSegmentPlan,
    VideoScoreScan,
    plan_auto_segments,
)
from jasna.mosaic.scan import covered_seconds


def make_scan(scores, *, stride=1.0, duration=None) -> VideoScoreScan:
    times = tuple(index * stride for index in range(len(scores)))
    return VideoScoreScan(
        times=times,
        scores=tuple(float(score) for score in scores),
        stride=stride,
        duration=float(duration if duration is not None else len(scores) * stride),
    )


def test_no_hits_means_nothing_to_render():
    plan = plan_auto_segments(make_scan([0.0, 0.04, 0.0, 0.02]), threshold=0.05)

    assert plan.segments == ()
    assert plan.coverage == 0.0
    assert plan.worth_segmenting is True
    assert "no mosaic" in plan.describe()


def test_hits_are_padded_by_half_a_stride_and_merged():
    # One hit at t=3s of a 10s video, stride 1s -> [2.5, 4.5]
    plan = plan_auto_segments(make_scan([0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), threshold=0.05)

    assert len(plan.segments) == 1
    assert plan.segments[0].start == pytest.approx(2.5)
    assert plan.segments[0].end == pytest.approx(4.5)


def test_neighbouring_hits_merge_into_one_range():
    plan = plan_auto_segments(make_scan([0.7, 0.8, 0.9, 0.6]), threshold=0.05)

    assert len(plan.segments) == 1
    assert plan.segments[0].start == pytest.approx(0.0, abs=1e-9)
    assert plan.segments[0].end == pytest.approx(4.0)


def test_threshold_controls_the_hit_set():
    scan = make_scan([0.2, 0.5, 0.2, 0.2, 0.5, 0.2, 0.2])

    low = plan_auto_segments(scan, threshold=0.1)
    high = plan_auto_segments(scan, threshold=0.4)

    assert covered_seconds(low.segments) > covered_seconds(high.segments)
    assert low.scan.samples == 7


def test_coverage_decides_whether_segmenting_is_worth_it():
    dense = plan_auto_segments(make_scan([0.9] * 20), threshold=0.05, coverage_limit=0.98)
    sparse = plan_auto_segments(make_scan([0.9] + [0.0] * 19), threshold=0.05, coverage_limit=0.98)

    assert dense.coverage >= 0.98
    assert dense.worth_segmenting is False
    assert sparse.coverage < 0.98
    assert sparse.worth_segmenting is True


def test_segments_never_exceed_the_video_duration():
    plan = plan_auto_segments(make_scan([0.9, 0.9, 0.9], duration=3.0), threshold=0.05)

    assert plan.segments
    for segment in plan.segments:
        assert 0.0 <= segment.start < segment.end <= 3.0


def test_plan_carries_the_threshold_and_stride_for_reporting():
    scan = make_scan([0.9, 0.0, 0.0, 0.0], stride=2.5)
    plan = plan_auto_segments(scan, threshold=0.3)

    assert plan.threshold == 0.3
    assert plan.scan.stride == 2.5
    assert isinstance(plan, AutoSegmentPlan)
    assert "region(s)" in plan.describe()
