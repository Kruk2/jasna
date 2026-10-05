from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "probe_single_decode_capacity.py"
SPEC = importlib.util.spec_from_file_location("probe_single_decode_capacity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_capacity_audit_tracks_bounded_ordered_backlog() -> None:
    ticks = iter((0.0, 1.0, 2.0, 3.0))
    audit = probe.CapacityAudit(clock=lambda: next(ticks))
    audit.produce([10, 20, 30, 40], 100)
    audit.consume(10)
    audit.consume(20)
    audit.consume(30)
    report = audit.report()
    assert report["produced_frames"] == 4
    assert report["consumed_frames"] == 3
    assert report["pending_frames"] == 1
    assert report["max_outstanding_frames"] == 4
    assert report["max_outstanding_rgb_bytes"] == 400
    assert report["residency_seconds"] == {"median": 2.0, "p95": 3.0, "max": 3.0}


def test_capacity_audit_fails_closed_on_pts_reordering() -> None:
    audit = probe.CapacityAudit(clock=lambda: 0.0)
    audit.produce([10, 20], 100)
    with pytest.raises(RuntimeError, match="order mismatch"):
        audit.consume(20)


def test_capacity_audit_rejects_duplicate_pts_and_size_change() -> None:
    audit = probe.CapacityAudit(clock=lambda: 0.0)
    audit.produce([10], 100)
    with pytest.raises(RuntimeError, match="duplicate"):
        audit.produce([10], 100)
    with pytest.raises(RuntimeError, match="size changed"):
        audit.produce([20], 200)
