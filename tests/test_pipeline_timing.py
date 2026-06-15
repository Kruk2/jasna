from __future__ import annotations

import pytest

import jasna.pipeline_timing as pipeline_timing
from jasna.pipeline_timing import LoopTimer


def _fake_clock(monkeypatch) -> None:
    state = {"t": 0.0}

    def fake_perf_counter() -> float:
        state["t"] += 1.0
        return state["t"]

    monkeypatch.setattr(pipeline_timing.time, "perf_counter", fake_perf_counter)


def test_measure_accumulates_per_category(monkeypatch) -> None:
    _fake_clock(monkeypatch)
    timer = LoopTimer("decode-detect")

    with timer.measure("decode"):
        pass
    with timer.measure("decode"):
        pass
    with timer.measure("detect"):
        pass

    assert timer.totals["decode"] == 2.0
    assert timer.totals["detect"] == 1.0


def test_measure_accumulates_on_exception(monkeypatch) -> None:
    _fake_clock(monkeypatch)
    timer = LoopTimer("primary")

    with pytest.raises(ValueError):
        with timer.measure("restore"):
            raise ValueError("boom")

    assert timer.totals["restore"] == 1.0


def test_timed_iter_yields_all_items_and_times_next(monkeypatch) -> None:
    _fake_clock(monkeypatch)
    timer = LoopTimer("decode-detect")

    items = list(timer.timed_iter(iter([1, 2, 3]), "decode"))

    assert items == [1, 2, 3]
    assert timer.totals["decode"] == 4.0


def test_summary_contains_name_and_categories(monkeypatch) -> None:
    _fake_clock(monkeypatch)
    timer = LoopTimer("blend-encode")
    with timer.measure("blend"):
        pass

    summary = timer.summary()
    assert "blend-encode" in summary
    assert "blend" in summary
    assert "1.0" in summary


def test_summary_empty_timer() -> None:
    timer = LoopTimer("secondary")
    assert "secondary" in timer.summary()
