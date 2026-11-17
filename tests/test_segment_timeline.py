from __future__ import annotations

import pytest

from jasna.gui.segment_timeline import SegmentTimeline, segment_fill, timeline_seconds_to_x, timeline_x_to_seconds
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


def _timeline(tk_root) -> SegmentTimeline:
    timeline = SegmentTimeline(
        tk_root,
        duration=10.0,
        fps=30.0,
        on_seek=lambda _seconds: None,
        on_create=lambda _start, _end: None,
        on_select=lambda _index: None,
        on_adjust=lambda _index, _start, _end: None,
    )
    timeline.pack(fill="x")
    tk_root.update()
    timeline.set_data(segments=(SegmentRange(1, 2), SegmentRange(4, 6)), selected_index=0, playhead=0.0)
    return timeline


def _playhead_x(timeline: SegmentTimeline) -> float:
    canvas = timeline._canvas
    (line,) = (item for item in canvas.find_all() if canvas.type(item) == "line")
    return canvas.coords(line)[0]


def test_moving_the_playhead_keeps_the_rest_of_the_timeline_drawn(tk_root) -> None:
    timeline = _timeline(tk_root)
    canvas = timeline._canvas
    lines = {item for item in canvas.find_all() if canvas.type(item) in ("line", "polygon")}
    background = set(canvas.find_all()) - lines

    timeline.set_playhead(5.0)

    assert background <= set(canvas.find_all())
    assert _playhead_x(timeline) == pytest.approx(
        timeline_seconds_to_x(5.0, view_start=0.0, view_end=10.0, width=canvas.winfo_width())
    )


def test_playhead_outside_the_zoomed_view_scrolls_the_view_to_it(tk_root) -> None:
    timeline = _timeline(tk_root)
    timeline.zoom_in()
    timeline.zoom_in()

    timeline.set_playhead(9.0)

    width = timeline._canvas.winfo_width()
    assert 0 < _playhead_x(timeline) < width


def test_ranges_are_coloured_by_their_model() -> None:
    assert segment_fill(SegmentRange(0, 1, SegmentRestoration("ltx", 3))) == Colors.MODEL_LTX
    assert segment_fill(SegmentRange(0, 1, SegmentRestoration("basicvsrpp", None))) == Colors.PRIMARY
    assert segment_fill(SegmentRange(0, 1)) == Colors.PRIMARY
