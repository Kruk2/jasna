from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import pytest

from jasna.gui.models import AppSettings
from jasna.gui.mosaic_scan import (
    MosaicScanResult,
    ScanCheckpoint,
    ScanCompleted,
    ScanFailed,
    ScanProgress,
)
from jasna.gui.pre_scan_routing import (
    PRE_SCAN_ALGORITHM_VERSION,
    PreScanFailed,
    PreScanCoordinator,
    PreScanStopped,
    _ScanCheckpointStore,
    _checkpoint_signature,
    coarse_execution_strategy,
    coarse_route,
    filter_short_scan_candidates,
    normalize_scan_segments,
    precise_scan_segments,
    resolve_timeline_pad_seconds,
    sampled_coverage,
    segment_coverage,
)
from jasna.segments import SegmentRange


def _result(scores, *, stride=2.0, duration=40.0):
    times = tuple(index * stride for index in range(len(scores)))
    return MosaicScanResult(
        times=times,
        scores=tuple(scores),
        masks=(),
        stride=stride,
        duration=duration,
        completed_until=times[-1] if times else 0.0,
    )


def test_coarse_route_uses_configured_85_percent_boundary():
    exact = _result([0.9] * 17 + [0.0] * 3)
    below = _result([0.9] * 16 + [0.0] * 4)

    assert sampled_coverage(exact, threshold=0.35) == pytest.approx(0.85)
    assert coarse_route(
        exact,
        detection_threshold=0.35,
        full_threshold=0.85,
    ) == "full"
    assert coarse_route(
        below,
        detection_threshold=0.35,
        full_threshold=0.85,
    ) == "fine"


def test_coarse_zero_hits_selects_source_copy():
    assert coarse_route(
        _result([0.0] * 20),
        detection_threshold=0.35,
        full_threshold=0.85,
    ) == "copy"


def test_sampled_coverage_duration_weights_irregular_samples():
    result = MosaicScanResult(
        times=(0.0, 1.0, 2.0, 10.0),
        scores=(0.9, 0.9, 0.0, 0.0),
        masks=(),
        stride=4.0,
        duration=20.0,
        completed_until=10.0,
    )

    # Midpoint cells are [0,.5), [.5,1.5), [1.5,6), [6,20): two
    # dense keyframe hits therefore account for 1.5 s, not 50% of samples.
    assert sampled_coverage(result, threshold=0.35) == pytest.approx(1.5 / 20.0)


def test_precise_scan_uses_sample_grid_before_safe_normalization():
    result = MosaicScanResult(
        times=(49.5, 50.0, 50.5, 51.0, 51.5, 52.0, 52.5),
        scores=(0.0, 0.95, 0.95, 0.95, 0.95, 0.0, 0.0),
        masks=(),
        stride=0.5,
        duration=120.0,
        completed_until=52.5,
    )

    # The two-second high-confidence core is retained and Auto adds one precise
    # sample interval (0.5s) on each side. There is no 30-second expansion.
    assert precise_scan_segments(result, threshold=0.35) == (
        SegmentRange(49.5, 52.5),
    )


def test_precise_scan_keeps_sustained_5994fps_hits_despite_pts_jitter():
    stride = 0.5005
    times = (
        291.292244444,
        291.792744444,
        292.294244444,
        292.793744444,
        293.295244444,
        293.794744444,
    )
    result = MosaicScanResult(
        times=times,
        scores=(0.92,) * len(times),
        masks=(),
        stride=stride,
        duration=400.0,
        completed_until=times[-1],
    )

    assert precise_scan_segments(
        result,
        threshold=0.35,
        pad_seconds=0.0,
    ) == (SegmentRange(times[0], times[-1] + stride),)


def test_short_precise_candidates_need_duration_and_high_confidence_core():
    one_second = _result([0.0, 0.99, 0.99, 0.0], stride=0.5, duration=2.0)
    weak_five_seconds = _result(
        [0.0] + [0.89] * 10 + [0.0], stride=0.5, duration=6.0
    )
    strong_five_seconds = _result(
        [0.0] + [0.91] * 10 + [0.0], stride=0.5, duration=6.0
    )
    interrupted_high_core = _result(
        [0.0, 0.95, 0.95, 0.95, 0.50, 0.95, 0.95, 0.95, 0.0],
        stride=0.5,
        duration=4.5,
    )

    assert precise_scan_segments(one_second, threshold=0.35, pad_seconds=0.0) == ()
    assert precise_scan_segments(
        weak_five_seconds, threshold=0.35, pad_seconds=0.0
    ) == ()
    assert precise_scan_segments(
        strong_five_seconds, threshold=0.35, pad_seconds=0.0
    ) == (SegmentRange(0.5, 5.5),)
    assert precise_scan_segments(
        interrupted_high_core, threshold=0.35, pad_seconds=0.0
    ) == ()


def test_short_high_confidence_tail_uses_actual_clipped_duration():
    result = MosaicScanResult(
        times=(0.0, 0.5, 1.0, 1.5),
        scores=(0.95, 0.95, 0.95, 0.95),
        masks=(),
        stride=0.5,
        duration=1.7,
        completed_until=1.5,
    )

    assert precise_scan_segments(result, threshold=0.35, pad_seconds=0.0) == ()


def test_retained_ranges_clamp_padding_at_first_and_last_frame_boundaries():
    result = MosaicScanResult(
        times=tuple(index * 0.5 for index in range(9)),
        scores=(0.95,) * 9,
        masks=(),
        stride=0.5,
        duration=4.25,
        completed_until=4.0,
    )

    assert precise_scan_segments(result, threshold=0.35, pad_seconds=0.5) == (
        SegmentRange(0.0, 4.25),
    )


def test_candidate_just_over_one_second_is_evaluated_as_short_not_ignored(monkeypatch):
    result = MosaicScanResult(
        times=(0.0, 0.5, 1.0),
        scores=(0.95, 0.95, 0.95),
        masks=(),
        stride=0.5,
        duration=1.003,
        completed_until=1.0,
    )

    monkeypatch.setattr(
        "jasna.gui.pre_scan_routing._has_sustained_high_confidence_core",
        lambda *_args, **_kwargs: True,
    )
    assert filter_short_scan_candidates(
        result,
        (SegmentRange(0.0, 1.003),),
        detection_threshold=0.35,
    ) == (SegmentRange(0.0, 1.003),)


def test_long_precise_candidate_keeps_normal_detection_threshold():
    result = _result([0.0] + [0.40] * 22 + [0.0], stride=0.5, duration=12.0)

    assert precise_scan_segments(result, threshold=0.35, pad_seconds=0.0) == (
        SegmentRange(0.5, 11.5),
    )


def test_candidate_just_over_ten_seconds_uses_normal_detection_threshold():
    result = MosaicScanResult(
        times=tuple(index * 0.5 for index in range(21)),
        scores=(0.40,) * 21,
        masks=(),
        stride=0.5,
        duration=10.003,
        completed_until=10.0,
    )

    assert precise_scan_segments(result, threshold=0.35, pad_seconds=0.0) == (
        SegmentRange(0.0, 10.003),
    )


def test_auto_precise_padding_tracks_interval_with_half_to_one_second_bounds():
    assert resolve_timeline_pad_seconds("auto", fine_interval=0.25) == 0.5
    assert resolve_timeline_pad_seconds("auto", fine_interval=0.5) == 0.5
    assert resolve_timeline_pad_seconds("auto", fine_interval=1.0) == 1.0
    assert resolve_timeline_pad_seconds("auto", fine_interval=2.0) == 1.0
    assert resolve_timeline_pad_seconds("2.0", fine_interval=0.5) == 2.0


def test_scan_policy_finishes_from_precise_grid_without_boundary_worker():
    result = MosaicScanResult(
        times=(49.5, 50.0, 50.5, 51.0, 51.5, 52.0, 52.5),
        scores=(0.0, 0.95, 0.95, 0.95, 0.95, 0.0, 0.0),
        masks=(),
        stride=0.5,
        duration=120.0,
        completed_until=52.5,
    )

    class FakeCheckpoint:
        outcome = None

        def completed_outcome(self):
            return None

        def samples(self, *_args):
            return {}

        def set_outcome(self, outcome):
            self.outcome = outcome

    coordinator = PreScanCoordinator.__new__(PreScanCoordinator)
    coordinator.metadata = SimpleNamespace(duration=120.0)
    coordinator.settings = AppSettings(pre_scan_policy="scan")
    coordinator.checkpoint = FakeCheckpoint()
    coordinator._log = lambda *_args: None
    coordinator._run_stage = lambda *_args, **_kwargs: (result, None)

    outcome = coordinator.run()

    assert outcome.processing_path == "smart"
    assert outcome.segments == (SegmentRange(49.5, 52.5),)
    assert coordinator.checkpoint.outcome == outcome


def test_precise_padding_does_not_force_minimum_duration_or_merge_30s_gaps():
    one = normalize_scan_segments(
        (SegmentRange(50.0, 51.0),),
        duration=120.0,
        pad_seconds=0.5,
    )
    assert one == (SegmentRange(49.5, 51.5),)

    separate = normalize_scan_segments(
        (SegmentRange(10.0, 20.0), SegmentRange(50.0, 60.0)),
        duration=120.0,
        pad_seconds=0.5,
    )
    assert separate == (
        SegmentRange(9.5, 20.5),
        SegmentRange(49.5, 60.5),
    )
    assert segment_coverage(separate, 120.0) == pytest.approx(22 / 120)


def test_precise_padding_merges_only_ranges_that_touch_after_expansion():
    touching = normalize_scan_segments(
        (SegmentRange(1.0, 2.0), SegmentRange(3.0, 4.0)),
        duration=10.0,
        pad_seconds=0.5,
    )
    separated = normalize_scan_segments(
        (SegmentRange(1.0, 2.0), SegmentRange(3.001, 4.0)),
        duration=10.0,
        pad_seconds=0.5,
    )

    assert touching == (SegmentRange(0.5, 4.5),)
    assert separated == (
        SegmentRange(0.5, 2.5),
        SegmentRange(2.501, 4.5),
    )


def test_scan_checkpoint_round_trip_uses_timebase_keys_and_completion(tmp_path):
    path = tmp_path / "scan" / "manifest.json"
    signature = {"source": "demo", "settings": {"fine": 0.5}}
    store = _ScanCheckpointStore(
        path,
        signature,
        fps=29.97,
        time_base=1 / 90_000,
    )
    store.add_stage_samples(
        "fine",
        0.5,
        (0.0, 0.5005, 1.001),
        (0.0, 0.8, 0.2),
    )
    store.mark_stage_complete("fine", 0.5)

    resumed = _ScanCheckpointStore(
        path,
        signature,
        fps=29.97,
        time_base=1 / 90_000,
    )
    samples = resumed.samples("fine", 0.5)
    assert sorted(samples) == [0, 45045, 90090]
    assert samples[45045][1] == pytest.approx(0.8)
    assert resumed.stage_complete("fine", 0.5)


def test_scan_checkpoint_prefers_exact_pts_keys(tmp_path):
    store = _ScanCheckpointStore(
        tmp_path / "scan" / "manifest.json",
        {"source": "demo"},
        fps=29.97,
        time_base=1 / 90_000,
    )

    store.add_stage_samples(
        "fine",
        0.5,
        (0.5005,),
        (0.8,),
        sample_keys=(45_046,),
    )

    assert store.samples("fine", 0.5) == {45_046: (0.5005, 0.8)}


def test_corrupt_checkpoint_is_reset_instead_of_marked_complete(tmp_path):
    path = tmp_path / "scan" / "manifest.json"
    signature = {"source": "demo"}
    store = _ScanCheckpointStore(
        path,
        signature,
        fps=30.0,
        time_base=1 / 90_000,
    )
    store.add_stage_samples("fine", 0.5, (0.0,), (0.8,))
    store.mark_stage_complete("fine", 0.5)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["stages"]["fine"]["samples"] = [["broken"]]
    path.write_text(json.dumps(raw), encoding="utf-8")

    resumed = _ScanCheckpointStore(
        path,
        signature,
        fps=30.0,
        time_base=1 / 90_000,
    )

    assert resumed.samples("fine", 0.5) == {}
    assert not resumed.stage_complete("fine", 0.5)


def test_incomplete_stage_decodes_from_start_and_reuses_exact_pts(tmp_path):
    checkpoint = _ScanCheckpointStore(
        tmp_path / "scan" / "manifest.json",
        {"source": "demo"},
        fps=30.0,
        time_base=1 / 90_000,
    )
    checkpoint.add_stage_samples(
        "fine",
        0.5,
        (0.5005,),
        (0.8,),
        sample_keys=(45_046,),
    )
    checkpoint.flush()
    created = {}

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            created.update(kwargs)
            self.events = queue.Queue()
            self.events.put(ScanProgress(0.5, 12.0, 8.0))
            self.events.put(
                ScanCompleted(
                    MosaicScanResult(
                        times=(0.0, 0.5005),
                        scores=(0.1, 0.8),
                        masks=(),
                        stride=0.5,
                        duration=2.0,
                        completed_until=0.5005,
                    ),
                    stopped=False,
                )
            )

        def start(self):
            pass

        def close(self):
            pass

        def join(self):
            pass

    coordinator = PreScanCoordinator.__new__(PreScanCoordinator)
    coordinator.source = tmp_path / "video.mp4"
    coordinator.output = tmp_path / "output.mp4"
    coordinator.metadata = SimpleNamespace(video_fps=30.0, duration=2.0)
    coordinator.settings = AppSettings()
    coordinator._stopped = lambda: False
    coordinator._log = lambda *_args: None
    progress = []
    coordinator._progress = lambda *values: progress.append(values)
    coordinator._worker_factory = FakeWorker
    coordinator._active_worker = None
    coordinator.checkpoint = checkpoint

    coordinator._run_stage(
        "fine",
        0.5,
        seed_samples={90_090: (1.001, 0.4)},
    )

    assert created["start_seconds"] == 0.0
    assert created["known_sample_scores"] == {45_046: 0.8, 90_090: 0.4}
    assert created["emit_checkpoints"] is True
    assert progress == [("fine", 0.5, 12.0, 8.0)]


def test_adaptive_coarse_checkpoint_keeps_exact_pts_and_reuses_completed_stage(tmp_path):
    checkpoint = _ScanCheckpointStore(
        tmp_path / "scan" / "manifest.json",
        {"source": "demo", "algorithm": "adaptive"},
        fps=59.94,
        time_base=0.01,
    )
    created = []
    plan = object()

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            created.append(kwargs)
            self.events = queue.Queue()
            self.events.put(ScanCheckpoint((0, 401), (0.0, 4.01), (0.1, 0.8)))
            self.events.put(
                ScanCompleted(
                    MosaicScanResult(
                        times=(0.0, 4.01),
                        scores=(0.1, 0.8),
                        masks=(),
                        stride=4.0,
                        duration=8.0,
                        completed_until=4.01,
                    ),
                    stopped=False,
                )
            )

        def start(self):
            pass

        def close(self):
            pass

        def join(self):
            pass

    coordinator = PreScanCoordinator.__new__(PreScanCoordinator)
    coordinator.source = tmp_path / "video.mp4"
    coordinator.output = tmp_path / "output.mp4"
    coordinator.metadata = SimpleNamespace(video_fps=59.94, duration=8.0, time_base=0.01)
    coordinator.settings = AppSettings()
    coordinator._stopped = lambda: False
    coordinator._log = lambda *_args: None
    coordinator._progress = None
    coordinator._worker_factory = FakeWorker
    coordinator._active_worker = None
    coordinator.checkpoint = checkpoint
    coordinator._adaptive_coarse_plan = lambda _interval: plan

    result, _worker = coordinator._run_stage("coarse", 4.0, adaptive_coarse=True)

    assert result.stride == pytest.approx(4.0)
    assert created[0]["adaptive_coarse_plan"] is plan
    assert checkpoint.samples("coarse", 4.0) == {0: (0.0, 0.1), 401: (4.01, 0.8)}
    assert checkpoint.stage_complete("coarse", 4.0)

    # A complete stage neither re-probes its GOP policy nor opens a worker.
    reused, _worker = coordinator._run_stage("coarse", 4.0, adaptive_coarse=True)
    assert reused.times == pytest.approx((0.0, 4.01))
    assert len(created) == 1


def test_adaptive_coarse_cancellation_persists_completed_batches(tmp_path):
    checkpoint = _ScanCheckpointStore(
        tmp_path / "scan" / "manifest.json",
        {"source": "demo", "algorithm": "adaptive"},
        fps=60.0,
        time_base=0.01,
    )

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            self.events = queue.Queue()
            self.events.put(ScanCheckpoint((0,), (0.0,), (0.8,)))
            self.events.put(
                ScanCompleted(
                    MosaicScanResult(
                        times=(0.0,),
                        scores=(0.8,),
                        masks=(),
                        stride=4.0,
                        duration=8.0,
                        completed_until=0.0,
                    ),
                    stopped=True,
                )
            )

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

        def join(self):
            pass

    coordinator = PreScanCoordinator.__new__(PreScanCoordinator)
    coordinator.source = tmp_path / "video.mp4"
    coordinator.output = tmp_path / "output.mp4"
    coordinator.metadata = SimpleNamespace(video_fps=60.0, duration=8.0, time_base=0.01)
    coordinator.settings = AppSettings()
    coordinator._stopped = lambda: False
    coordinator._log = lambda *_args: None
    coordinator._progress = None
    coordinator._worker_factory = FakeWorker
    coordinator._active_worker = None
    coordinator.checkpoint = checkpoint
    coordinator._adaptive_coarse_plan = lambda _interval: object()

    with pytest.raises(PreScanStopped):
        coordinator._run_stage("coarse", 4.0, adaptive_coarse=True)

    assert checkpoint.samples("coarse", 4.0) == {0: (0.0, 0.8)}
    assert not checkpoint.stage_complete("coarse", 4.0)


def test_adaptive_coarse_native_failure_is_reported_without_second_worker(tmp_path):
    checkpoint = _ScanCheckpointStore(
        tmp_path / "scan" / "manifest.json",
        {"source": "demo", "algorithm": "adaptive"},
        fps=60.0,
        time_base=0.01,
    )
    created = []

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            created.append(kwargs)
            self.events = queue.Queue()
            self.events.put(ScanFailed("native decoder error"))

        def start(self):
            pass

        def close(self):
            pass

        def join(self):
            pass

    coordinator = PreScanCoordinator.__new__(PreScanCoordinator)
    coordinator.source = tmp_path / "video.mp4"
    coordinator.output = tmp_path / "output.mp4"
    coordinator.metadata = SimpleNamespace(video_fps=60.0, duration=8.0, time_base=0.01)
    coordinator.settings = AppSettings()
    coordinator._stopped = lambda: False
    coordinator._log = lambda *_args: None
    coordinator._progress = None
    coordinator._worker_factory = FakeWorker
    coordinator._active_worker = None
    coordinator.checkpoint = checkpoint
    coordinator._adaptive_coarse_plan = lambda _interval: object()

    with pytest.raises(PreScanFailed, match="native decoder error"):
        coordinator._run_stage("coarse", 4.0, adaptive_coarse=True)

    assert len(created) == 1
    assert "adaptive_coarse_plan" in created[0]


def test_checkpoint_signature_versions_adaptive_coarse_policy(monkeypatch, tmp_path):
    import jasna.mosaic.detection_registry as registry

    source = tmp_path / "输入视频.mp4"
    source.write_bytes(b"source")
    weights = tmp_path / "detector.pt"
    weights.write_bytes(b"weights")
    monkeypatch.setattr(registry, "coerce_detection_model_name", lambda _name: "fake")
    monkeypatch.setattr(registry, "require_detection_model_weights", lambda _name: weights)

    signature = _checkpoint_signature(
        source,
        tmp_path / "output.mp4",
        AppSettings(pre_scan_coarse_interval=4.0),
        SimpleNamespace(codec_name="h264", is_10bit=False),
    )
    two_second_signature = _checkpoint_signature(
        source,
        tmp_path / "output.mp4",
        AppSettings(pre_scan_coarse_interval=2.0),
        SimpleNamespace(codec_name="h264", is_10bit=False),
    )

    assert signature["algorithm_version"] == PRE_SCAN_ALGORITHM_VERSION
    assert signature["scan"]["adaptive_coarse"] == {
        "policy_version": "keyframe-gop-v1",
        "tolerance_ratio": 0.25,
        "coverage_policy": "duration-weighted-midpoints-v1",
    }
    assert signature["scan"]["pad_seconds"] == {
        "configured": "auto",
        "resolved": 0.5,
    }
    assert signature["scan"]["merge_gap_seconds"] == 0.0
    assert signature["scan"]["precise_range_policy"] == (
        "confidence-filter-plus-sample-padding-v2"
    )
    assert signature["scan"]["coarse_interval"] == 4.0
    assert signature["scan"]["coarse_execution_strategy"] == "adaptive-direct-gop"
    assert two_second_signature["scan"]["coarse_interval"] == 2.0
    assert two_second_signature != signature


def test_windows_amd_hevc_main10_uses_fixed_grid_coarse(monkeypatch):
    from jasna.accelerator import AcceleratorVendor
    import jasna.gui.pre_scan_routing as routing

    monkeypatch.setattr(routing.sys, "platform", "win32")
    monkeypatch.setattr(
        routing,
        "vendor_for_device",
        lambda _device=None: AcceleratorVendor.AMD,
    )

    assert coarse_execution_strategy(
        SimpleNamespace(codec_name="hevc", is_10bit=True)
    ) == "fixed-grid"


@pytest.mark.parametrize(
    ("platform", "vendor_name", "codec", "is_10bit"),
    [
        ("win32", "amd", "h264", False),
        ("win32", "amd", "hevc", False),
        ("linux", "amd", "hevc", True),
        ("win32", "nvidia", "hevc", True),
    ],
)
def test_adaptive_gop_coarse_remains_enabled_for_other_paths(
    monkeypatch,
    platform,
    vendor_name,
    codec,
    is_10bit,
):
    from jasna.accelerator import AcceleratorVendor
    import jasna.gui.pre_scan_routing as routing

    monkeypatch.setattr(routing.sys, "platform", platform)
    monkeypatch.setattr(
        routing,
        "vendor_for_device",
        lambda _device=None: AcceleratorVendor(vendor_name),
    )

    assert coarse_execution_strategy(
        SimpleNamespace(codec_name=codec, is_10bit=is_10bit)
    ) == "adaptive-direct-gop"


def test_auto_worker_selection_uses_effective_coarse_strategy(monkeypatch):
    import jasna.gui.pre_scan_routing as routing

    coordinator = PreScanCoordinator.__new__(PreScanCoordinator)
    coordinator.metadata = SimpleNamespace(
        codec_name="hevc",
        is_10bit=True,
        duration=8.0,
    )
    coordinator.settings = AppSettings(pre_scan_policy="auto")
    coordinator._log = lambda *_args: None
    coordinator.checkpoint = SimpleNamespace(
        completed_outcome=lambda: None,
        set_outcome=lambda _outcome: None,
    )
    calls = []
    coordinator._run_stage = lambda *args, **kwargs: (
        calls.append((args, kwargs))
        or (_result([0.0, 0.0], stride=4.0, duration=8.0), None)
    )
    monkeypatch.setattr(routing, "coarse_execution_strategy", lambda _metadata: "fixed-grid")

    outcome = coordinator.run()

    assert outcome.processing_path == "copy"
    assert calls == [(("coarse", 4.0), {"adaptive_coarse": False})]


def test_checkpoint_signature_separates_effective_coarse_strategy(monkeypatch, tmp_path):
    import jasna.mosaic.detection_registry as registry
    import jasna.gui.pre_scan_routing as routing

    source = tmp_path / "source.mkv"
    source.write_bytes(b"source")
    weights = tmp_path / "detector.pt"
    weights.write_bytes(b"weights")
    monkeypatch.setattr(registry, "coerce_detection_model_name", lambda _name: "fake")
    monkeypatch.setattr(registry, "require_detection_model_weights", lambda _name: weights)
    metadata = SimpleNamespace(codec_name="hevc", is_10bit=True)

    monkeypatch.setattr(routing, "coarse_execution_strategy", lambda _metadata: "fixed-grid")
    fixed = _checkpoint_signature(source, tmp_path / "output.mp4", AppSettings(), metadata)
    monkeypatch.setattr(
        routing,
        "coarse_execution_strategy",
        lambda _metadata: "adaptive-direct-gop",
    )
    adaptive = _checkpoint_signature(source, tmp_path / "output.mp4", AppSettings(), metadata)

    assert fixed["scan"]["coarse_execution_strategy"] == "fixed-grid"
    assert adaptive["scan"]["coarse_execution_strategy"] == "adaptive-direct-gop"
    assert fixed != adaptive
