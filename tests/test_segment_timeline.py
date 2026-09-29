from __future__ import annotations

import pytest

from jasna.gui.segment_timeline import segment_fill, timeline_seconds_to_x, timeline_x_to_seconds
from jasna.gui.theme import Colors
from jasna.segments import SegmentRange, SegmentRestoration


def test_timeline_coordinate_mapping_round_trips() -> None:
    x = timeline_seconds_to_x(
        35,
        view_start=10,
        view_end=60,
        width=520,
        padding=10,
    )

    assert x == pytest.approx(260)
    assert timeline_x_to_seconds(
        x,
        view_start=10,
        view_end=60,
        width=520,
        padding=10,
    ) == pytest.approx(35)


def test_timeline_x_mapping_clamps_to_visible_range() -> None:
    assert timeline_x_to_seconds(
        -100,
        view_start=20,
        view_end=40,
        width=300,
    ) == 20
    assert timeline_x_to_seconds(
        1000,
        view_start=20,
        view_end=40,
        width=300,
    ) == 40


def test_ranges_are_coloured_by_their_model() -> None:
    assert segment_fill(SegmentRange(0, 1, SegmentRestoration("ltx", 3))) == Colors.MODEL_LTX
    assert segment_fill(SegmentRange(0, 1, SegmentRestoration("basicvsrpp", None))) == Colors.PRIMARY
    assert segment_fill(SegmentRange(0, 1)) == Colors.PRIMARY
