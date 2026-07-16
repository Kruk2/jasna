from __future__ import annotations

import pytest

from jasna.gui.segment_timeline import timeline_seconds_to_x, timeline_x_to_seconds


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
