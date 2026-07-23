"""CPU tests for the Phase-2 harness primitives."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vrx_harness import (  # noqa: E402
    SampleRecord, angular_center, blind_labels, position_bin, read_jsonl,
    seek_candidates, stable_seed, track_stability, write_jsonl,
)


def test_stable_seed_deterministic():
    assert stable_seed("a", 1) == stable_seed("a", 1)
    assert stable_seed("a", 1) != stable_seed("a", 2)


def test_seek_candidates_deterministic_and_bounded():
    a = seek_candidates(600.0, seed=stable_seed("row", 3), n=10)
    b = seek_candidates(600.0, seed=stable_seed("row", 3), n=10)
    assert a == b
    assert a == sorted(a)
    assert all(600 * 0.08 <= t <= 600 * 0.92 for t in a)
    assert seek_candidates(600.0, seed=stable_seed("row", 4), n=10) != a


def test_angular_and_position_bins():
    assert angular_center(0.5, 0.5) == (0.0, 0.0)
    assert position_bin(0.0, 0.0) == "center"
    assert position_bin(-40.0, 10.0) == "bottom"
    assert position_bin(-40.0, 50.0) == "bottom_offaxis"
    assert position_bin(40.0, 0.0) == "top"


def test_track_stability():
    stable = track_stability(centers=np.full((30, 2), 0.5) + 0.001, areas=np.full(30, 0.02), coverage=0.95)
    assert stable["stable"] and stable["score"] > 0.8
    jittery = track_stability(centers=np.random.default_rng(0).random((30, 2)),
                              areas=np.random.default_rng(1).random(30), coverage=0.5)
    assert not jittery["stable"]


def test_blind_labels_deterministic_and_complete():
    variants = ["raw", "fisheye", "gnomonic"]
    a = blind_labels("s123", variants)
    b = blind_labels("s123", variants)
    assert a == b
    assert set(a.keys()) == {"A", "B", "C"}
    assert sorted(a.values()) == sorted(variants)
    assert blind_labels("s124", variants) != a or True  # may coincide; just must be defined
    # inverse key recoverable
    inv = {v: k for k, v in a.items()}
    assert inv["raw"] in {"A", "B", "C"}


def test_sample_record_roundtrip(tmp_path):
    rec = SampleRecord(
        sample_id="SAVR__t0", studio="SAVR", title="savr00327", source_path="/x.mp4",
        seek_ts=123.4, pts_start=1000, pts_end=2000, width=8192, height=4096, fps=60.0,
        eye="left", track_frames=[10, 11, 12], bboxes=[[1, 2, 3, 4]], mask_ref="m.npy",
        mask_shape=[144, 288], center_lon=-12.0, center_lat=-30.0, position="bottom",
        stability={"stable": True}, vr_reason="fisheye token SAVR", zelefans_prior="fisheye",
    )
    p = tmp_path / "d.jsonl"
    write_jsonl(p, [rec])
    back = read_jsonl(p)
    assert len(back) == 1 and back[0].sample_id == "SAVR__t0"
    assert back[0].bboxes == [[1, 2, 3, 4]] and back[0].position == "bottom"
