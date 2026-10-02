from __future__ import annotations

import random
import time
from dataclasses import replace

import pytest

from jasna.segments import (
    SegmentRange,
    SegmentRestoration,
    format_segments,
    format_timestamp,
    job_restoration,
    normalize_segments,
    parse_segments,
    parse_timestamp,
    resolve_restorations,
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


LTX_A = SegmentRestoration("ltx", 1)
LTX_B = SegmentRestoration("ltx", 2)
STANDARD = SegmentRestoration("basicvsrpp", None)


def test_segment_restoration_seed_belongs_to_ltx_only() -> None:
    assert job_restoration("basicvsrpp", 7) == STANDARD
    assert job_restoration("ltx", 7) == SegmentRestoration("ltx", 7)
    for model, seed in (("ltx", None), ("basicvsrpp", 3)):
        with pytest.raises(ValueError):
            SegmentRestoration(model, seed)


def test_normalize_merges_only_neighbours_with_the_same_restoration() -> None:
    assert normalize_segments(
        (SegmentRange(0, 5, LTX_A), SegmentRange(5, 8, LTX_A), SegmentRange(8, 9, LTX_B))
    ) == (SegmentRange(0, 8, LTX_A), SegmentRange(8, 9, LTX_B))


def test_normalize_lets_a_later_range_cut_an_earlier_one() -> None:
    assert normalize_segments((SegmentRange(0, 10, LTX_A), SegmentRange(3, 5, STANDARD))) == (
        SegmentRange(0, 3, LTX_A),
        SegmentRange(3, 5, STANDARD),
        SegmentRange(5, 10, LTX_A),
    )
    assert normalize_segments((SegmentRange(3, 5, STANDARD), SegmentRange(0, 10, LTX_A))) == (
        SegmentRange(0, 10, LTX_A),
    )


def test_resolve_restorations_fills_in_only_the_job_default() -> None:
    assert resolve_restorations((SegmentRange(0, 1), SegmentRange(2, 3, LTX_B)), LTX_A) == (
        SegmentRange(0, 1, LTX_A),
        SegmentRange(2, 3, LTX_B),
    )


def _painted_reference(segments):
    """Range-by-range painting: each later range removes what it covers from the earlier ones."""
    painted = []
    for segment in segments:
        kept = []
        for old in painted:
            if segment.start > old.start:
                kept.append(replace(old, end=min(old.end, segment.start)))
            if old.end > segment.end:
                kept.append(replace(old, start=max(old.start, segment.end)))
        painted = kept + [segment]
    merged = []
    for segment in sorted(painted):
        if merged and segment.restoration == merged[-1].restoration and segment.start <= merged[-1].end:
            merged[-1] = replace(merged[-1], end=max(merged[-1].end, segment.end))
        else:
            merged.append(segment)
    return tuple(merged)


def test_normalize_paints_later_ranges_over_earlier_ones_like_range_by_range_painting() -> None:
    rng = random.Random(3)
    choices = (None, SegmentRestoration("basicvsrpp", None), SegmentRestoration("ltx", 1), SegmentRestoration("ltx", 2))
    for _ in range(300):
        segments = []
        for _ in range(rng.randrange(1, 12)):
            start = rng.randrange(0, 40) / 2
            segments.append(SegmentRange(start, start + rng.randrange(1, 12) / 2, rng.choice(choices)))
        assert normalize_segments(segments) == _painted_reference(segments)


def test_normalize_handles_a_long_scan_quickly() -> None:
    hits = [SegmentRange(max(0.0, second - 0.5), second + 1.5) for second in range(5000)]
    started = time.perf_counter()
    assert normalize_segments(hits) == (SegmentRange(0.0, 5000.5),)
    assert time.perf_counter() - started < 1.0
