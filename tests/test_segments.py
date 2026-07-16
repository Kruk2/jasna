from __future__ import annotations

import pytest

from jasna.segments import (
    SegmentRange,
    format_segments,
    format_timestamp,
    normalize_segments,
    parse_segments,
    parse_timestamp,
)


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("12.5", 12.5), ("01:02.5", 62.5), ("1:02:03.25", 3723.25)],
)
def test_parse_timestamp(text: str, seconds: float) -> None:
    assert parse_timestamp(text) == seconds


@pytest.mark.parametrize("text", ["", "1:60", "1:60:00", "a", "1:2:3:4", "-1"])
def test_parse_timestamp_rejects_invalid_values(text: str) -> None:
    with pytest.raises(ValueError):
        parse_timestamp(text)


def test_parse_segments_sorts_and_merges_ranges() -> None:
    assert parse_segments("10-20,00:05-00:12.5,30-31", duration=40) == (
        SegmentRange(5, 20),
        SegmentRange(30, 31),
    )


def test_normalize_segments_merges_adjacent_ranges() -> None:
    assert normalize_segments([SegmentRange(1, 2), SegmentRange(2, 4)]) == (
        SegmentRange(1, 4),
    )


def test_parse_segments_rejects_range_after_duration() -> None:
    with pytest.raises(ValueError, match="exceeds video duration"):
        parse_segments("9-11", duration=10)


def test_segment_formatting_is_round_trippable() -> None:
    segments = (SegmentRange(1.25, 62.5), SegmentRange(3600, 3601))
    assert parse_segments(format_segments(segments)) == segments
    assert format_timestamp(62.5) == "00:01:02.500"
