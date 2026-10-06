from __future__ import annotations

import ast
from fractions import Fraction
import inspect
from pathlib import Path
import textwrap
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jasna.gui.mosaic_scan import (
    AdaptiveCoarseDecodeGroup,
    AdaptiveCoarsePlan,
    MosaicScanResult,
    MosaicScanWorker,
    ScanCheckpoint,
    _ScanTensorCollector,
    _decode_adaptive_coarse_group,
    plan_adaptive_coarse_scan,
    scan_sample_stride,
    segments_from_scores,
    SCAN_SCORE_FLOOR,
)
from jasna.gui.models import AppSettings
from jasna.gui.segment_editor_state import SegmentEditorState
from jasna.segments import SegmentRange, SegmentRestoration

STANDARD = SegmentRestoration("basicvsrpp", None)
from jasna.media.splice import KeyframeIndex


def _keyframe_index(seconds, *, duration, time_base=Fraction(1, 90_000)):
    start_pts = 0
    return KeyframeIndex(
        tuple(round(float(value) / float(time_base)) for value in seconds),
        time_base,
        start_pts,
        round(float(duration) / float(time_base)),
    )


def test_stride_follows_fps():
    assert scan_sample_stride(29.97) == 30
    assert scan_sample_stride(60.0, seconds=0.5) == 30
    assert scan_sample_stride(30.0, seconds=0.0) == 1
    assert scan_sample_stride(1.0) == 1
    assert scan_sample_stride(0.1) == 1


def test_adaptive_coarse_regular_ntsc_jitter_uses_keyframes_directly():
    plan = plan_adaptive_coarse_scan(
        _keyframe_index((0.0, 5.005, 10.010), duration=15.015),
        duration=15.015,
        target_interval=4.0,
        fps=59.94,
    )

    assert plan.tolerance == pytest.approx(1.0)
    assert plan.classification_epsilon >= 0.005
    assert [group.mode for group in plan.groups] == ["regular", "regular", "regular"]
    assert [group.target_seconds for group in plan.groups] == pytest.approx(
        [(0.0,), (5.005,), (10.010,)]
    )
    assert all(group.frame_stride == 1 for group in plan.groups)

    outside_band = plan_adaptive_coarse_scan(
        _keyframe_index((0.0, 5.1), duration=10.2),
        duration=10.2,
        target_interval=4.0,
        fps=59.94,
    )
    assert [group.mode for group in outside_band.groups] == ["sparse", "sparse"]


def test_adaptive_coarse_dense_keyframes_choose_nearest_target_cadence():
    plan = plan_adaptive_coarse_scan(
        _keyframe_index(tuple(float(value) for value in range(10)), duration=10.0),
        duration=10.0,
        target_interval=4.0,
        fps=60.0,
    )

    assert [group.mode for group in plan.groups] == ["dense", "dense", "dense"]
    assert [group.start_seconds for group in plan.groups] == pytest.approx([0.0, 4.0, 8.0])
    assert plan.sample_count == 3


def test_adaptive_coarse_sparse_gops_keep_targets_in_one_group_each():
    plan = plan_adaptive_coarse_scan(
        _keyframe_index((0.0, 10.0), duration=20.0),
        duration=20.0,
        target_interval=4.0,
        fps=59.94,
    )

    assert [group.mode for group in plan.groups] == ["sparse", "sparse"]
    assert [group.target_seconds for group in plan.groups] == pytest.approx(
        [(0.0, 4.0, 8.0), (10.0, 14.0, 18.0)]
    )
    assert plan.sample_count == 6
    assert all(group.frame_stride == 240 for group in plan.groups)


def test_adaptive_coarse_mixed_gops_are_classified_locally():
    plan = plan_adaptive_coarse_scan(
        _keyframe_index((0.0, 5.005, 6.005, 14.005, 19.010), duration=24.015),
        duration=24.015,
        target_interval=4.0,
        fps=59.94,
    )

    assert [group.mode for group in plan.groups] == [
        "regular",
        "dense",
        "sparse",
        "regular",
        "regular",
    ]
    assert [group.start_seconds for group in plan.groups] == pytest.approx(
        [0.0, 5.005, 6.005, 14.005, 19.010]
    )
    assert plan.groups[2].target_seconds == pytest.approx((6.005, 10.005))


def test_adaptive_sparse_group_opens_one_reader_for_all_its_targets(monkeypatch):
    import torch
    import jasna.media.video_decoder as video_decoder

    group = AdaptiveCoarseDecodeGroup(
        mode="sparse",
        start_seconds=0.0,
        end_seconds=10.0,
        target_seconds=(0.0, 4.0, 8.0),
        target_pts=(0, 400, 800),
        frame_stride=1,
    )
    plan = AdaptiveCoarsePlan(
        target_interval=4.0,
        tolerance=1.0,
        classification_epsilon=0.01,
        start_pts=1_000,
        time_base=0.01,
        duration=10.0,
        groups=(group,),
    )
    opened = []

    class FakeReader:
        def __init__(self, *args, **kwargs):
            opened.append((args, kwargs))
            self.start_pts = 1_000

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def frames(self, *, seek_ts=None):
            assert seek_ts == 0.0
            yield torch.zeros((2, 3, 2, 2), dtype=torch.uint8), [1_000, 1_400]
            yield torch.zeros((2, 3, 2, 2), dtype=torch.uint8), [1_800, 1_900]

    monkeypatch.setattr(video_decoder, "VideoReader", FakeReader)
    batches = list(
        _decode_adaptive_coarse_group(
            Path("视频.mp4"),
            object(),
            object(),
            4,
            plan,
            group,
            stopped=lambda: False,
        )
    )

    assert len(opened) == 1
    assert opened[0][0][1] == 1
    assert opened[0][1]["frame_stride"] == 1
    assert [pts for _batch, pts_list in batches for pts in pts_list] == [1_000, 1_400, 1_800]


def test_adaptive_direct_gops_share_detector_batches_and_exact_checkpoint_pts(monkeypatch):
    import torch
    import jasna.gui.mosaic_scan as mosaic_scan
    import jasna.media.video_decoder as video_decoder

    batch_size = 3
    sample_keys = (7, 111, 222, 333, 444)
    groups = tuple(
        AdaptiveCoarseDecodeGroup(
            mode="regular",
            start_seconds=key * 0.01,
            end_seconds=key * 0.01 + 0.005,
            target_seconds=(key * 0.01,),
            target_pts=(key,),
            frame_stride=1 + index % 2,
        )
        for index, key in enumerate(sample_keys)
    )
    plan = AdaptiveCoarsePlan(
        target_interval=4.0,
        tolerance=1.0,
        classification_epsilon=0.001,
        start_pts=10_000,
        time_base=0.01,
        duration=5.0,
        groups=groups,
    )
    detector_batches = []

    class FakeDetector:
        def scan_scores_masks(self, batch, *, mask_hw):
            detector_batches.append(batch.shape[0])
            return (
                torch.arange(batch.shape[0], dtype=torch.float32),
                torch.zeros((batch.shape[0], *mask_hw), dtype=torch.uint8),
            )

    class FakeCollector:
        def __init__(self, _torch, **_kwargs):
            self.scores = []

        def add(self, scores, _masks, *, count):
            self.scores.extend(scores[:count].detach().cpu().tolist())

        def finish(self):
            return tuple(self.scores), torch.empty((len(self.scores), 1, 1), dtype=torch.uint8)

    opened_readers = []
    closed_readers = []
    reused_readers = []
    observed_strides = []

    class FakeReader:
        def __init__(self, *_args, frame_stride, **_kwargs):
            self.frame_stride = int(frame_stride)
            opened_readers.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            closed_readers.append(self)

    def fake_decode(
        _path,
        _metadata,
        _device,
        _reader_batch_size,
        active_plan,
        group,
        *,
        reader,
        **_kwargs,
    ):
        assert active_plan is plan
        reused_readers.append(reader)
        observed_strides.append(reader.frame_stride)
        key = group.target_pts[0]
        yield torch.full((1, 3, 2, 2), key % 255, dtype=torch.uint8), [
            active_plan.start_pts + key
        ]

    monkeypatch.setattr(mosaic_scan, "_ScanTensorCollector", FakeCollector)
    monkeypatch.setattr(mosaic_scan, "_decode_adaptive_coarse_group", fake_decode)
    monkeypatch.setattr(video_decoder, "VideoReader", FakeReader)
    worker = MosaicScanWorker(
        "video.mp4",
        SimpleNamespace(),
        AppSettings(batch_size=batch_size),
        stride_seconds=4.0,
        emit_checkpoints=True,
        adaptive_coarse_plan=plan,
    )
    worker._scan_adaptive_coarse(FakeDetector(), plan)

    checkpoint_events = []
    while not worker.events.empty():
        event = worker.events.get_nowait()
        if isinstance(event, ScanCheckpoint):
            checkpoint_events.append(event)
    assert detector_batches == [batch_size, batch_size]
    assert len(opened_readers) == 1
    assert closed_readers == opened_readers
    assert reused_readers == opened_readers * len(groups)
    assert observed_strides == [group.frame_stride for group in groups]
    assert [event.sample_keys for event in checkpoint_events] == [
        sample_keys[:batch_size],
        sample_keys[batch_size:],
    ]
    assert [event.times for event in checkpoint_events] == pytest.approx(
        [
            tuple(key * 0.01 for key in sample_keys[:batch_size]),
            tuple(key * 0.01 for key in sample_keys[batch_size:]),
        ]
    )


def test_editor_scan_does_not_emit_checkpoint_events_by_default():
    worker = MosaicScanWorker(
        "video.mp4",
        object(),
        AppSettings(),
        stride_seconds=1.0,
    )

    assert worker.emit_checkpoints is False
    assert worker.known_sample_scores == {}


def test_consecutive_hits_merge_into_one_range():
    times = (0.0, 1.0, 2.0, 3.0, 4.0)
    scores = (0.0, 0.8, 0.9, 0.7, 0.0)
    segments = segments_from_scores(
        times, scores, threshold=0.5, stride=1.0, duration=10.0
    )
    assert segments == (SegmentRange(0.5, 4.5),)


def test_pts_jitter_does_not_split_continuous_fine_scan_hits():
    stride = 0.5005
    times = (291.292244444, 291.792744444, 292.294244444, 292.793744444, 293.295244444)
    segments = segments_from_scores(
        times,
        (0.92,) * len(times),
        threshold=0.35,
        stride=stride,
        duration=400.0,
        pad=0.0,
    )

    assert segments == (SegmentRange(times[0], times[-1] + stride),)


def test_pts_jitter_tolerance_does_not_bridge_a_missing_sample():
    stride = 0.5005
    times = (10.0, 10.5015, 11.5025, 12.003)
    segments = segments_from_scores(
        times,
        (0.92,) * len(times),
        threshold=0.35,
        stride=stride,
        duration=20.0,
        pad=0.0,
    )

    assert segments == (
        SegmentRange(10.0, 11.002),
        SegmentRange(11.5025, 12.5035),
    )


def test_isolated_hits_stay_separate():
    times = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    scores = (0.9, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0)
    segments = segments_from_scores(
        times, scores, threshold=0.5, stride=1.0, duration=10.0, pad=0.25
    )
    assert segments == (SegmentRange(0.0, 1.25), SegmentRange(2.75, 4.25))


def test_threshold_filters_hits():
    times = (0.0, 1.0)
    scores = (0.3, 0.6)
    assert segments_from_scores(times, scores, threshold=0.7, stride=1.0, duration=5.0) == ()
    low = segments_from_scores(times, scores, threshold=0.2, stride=1.0, duration=5.0)
    assert low == (SegmentRange(0.0, 2.5),)


def test_ranges_clamped_to_video():
    segments = segments_from_scores(
        (0.0, 9.0), (0.9, 0.9), threshold=0.5, stride=1.0, duration=9.8
    )
    assert segments[0].start == 0.0
    assert segments[-1].end == pytest.approx(9.8)


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        segments_from_scores((0.0,), (0.5, 0.6), threshold=0.5, stride=1.0, duration=5.0)


def test_mask_lookup_only_returns_exact_sample():
    result = MosaicScanResult(
        times=(0.0, 1.0, 2.0),
        scores=(0.1, 0.9, 0.2),
        masks=["m0", "m1", "m2"],
        stride=1.0,
        duration=10.0,
        completed_until=2.0,
    )
    assert result.sample_at(1.01, tolerance=0.02) == (1.0, 0.9, "m1")
    assert result.sample_at(1.2, tolerance=0.02) is None
    assert result.sample_at(5.0, tolerance=0.02) is None


def test_scan_collector_recycles_gpu_chunk_when_free_memory_reaches_reserve(
    monkeypatch,
):
    import torch
    class FakeCuda:
        def __init__(self):
            self.free = [1_000, 100]
            self.empty_cache_calls = 0

        def mem_get_info(self):
            value = self.free.pop(0) if len(self.free) > 1 else self.free[0]
            return value, 2_000

        def empty_cache(self):
            self.empty_cache_calls += 1

    class FakeTorch:
        uint8 = torch.uint8
        float32 = torch.float32
        empty = staticmethod(torch.empty)
        cuda = FakeCuda()

    collector_globals = _ScanTensorCollector.__init__.__globals__
    monkeypatch.setitem(collector_globals, "SCAN_VRAM_RESERVE_BYTES", 100)
    monkeypatch.setitem(collector_globals, "SCAN_SPILL_CHUNK_BYTES", 16)
    spills = []
    collector = _ScanTensorCollector(
        FakeTorch,
        capacity=10,
        mask_hw=(2, 2),
        batch_size=2,
        device="cpu",
        on_spill=lambda: spills.append(True),
    )
    collector.add(
        torch.tensor([0.1, 0.2, 0.3, 0.4]),
        torch.arange(16, dtype=torch.uint8).reshape(4, 2, 2),
        count=4,
    )
    scores, masks = collector.finish()

    assert collector.spilling
    assert spills == [True]
    assert scores == pytest.approx((0.1, 0.2, 0.3, 0.4))
    assert torch.equal(masks, torch.arange(16, dtype=torch.uint8).reshape(4, 2, 2))
    assert FakeTorch.cuda.empty_cache_calls == 2


def test_add_many_is_single_undo_step():
    state = SegmentEditorState(duration=100.0, fps=25.0, segments=(), default_restoration=STANDARD)
    added = state.add_many((SegmentRange(1.0, 2.0), SegmentRange(5.0, 7.0)))
    assert added == 2
    assert len(state.segments) == 2
    assert state.undo()
    assert state.segments == ()


def test_add_many_skips_already_covered_ranges():
    state = SegmentEditorState(duration=100.0, fps=25.0, segments=(), default_restoration=STANDARD)
    state.add(0.0, 10.0)
    assert state.add_many((SegmentRange(2.0, 3.0),)) == 0
    added = state.add_many((SegmentRange(2.0, 3.0), SegmentRange(20.0, 21.0)))
    assert added == 1
    assert state.segments == (SegmentRange(0.0, 10.0, STANDARD), SegmentRange(20.0, 21.0, STANDARD))


def test_scan_decoder_count_keeps_nvidia_gate_and_enables_native_8k_linux_amd():
    from jasna.gui.mosaic_scan import scan_decoder_count

    assert scan_decoder_count(3840, 2160, 120.0, amd=False) == 2
    assert scan_decoder_count(8192, 4096, 15.0, amd=False) == 2
    assert scan_decoder_count(1920, 1080, 120.0, amd=False) == 1
    assert scan_decoder_count(3840, 2160, 5.0, amd=False) == 1
    assert scan_decoder_count(3840, 2160, 120.0, amd=True) == 1
    assert (
        scan_decoder_count(
            8192, 4096, 120.0, amd=True, linux_amd_native=True
        )
        == 2
    )
    assert (
        scan_decoder_count(
            3840, 2160, 120.0, amd=True, linux_amd_native=True
        )
        == 1
    )
    assert (
        scan_decoder_count(
            8192, 4096, 5.0, amd=True, linux_amd_native=True
        )
        == 1
    )


def test_segment_sample_indices_ownership():
    from jasna.gui.mosaic_scan import segment_sample_indices

    times = [9.8, 9.9, 10.0, 10.1]
    assert segment_sample_indices(times, 0.0, 10.0, is_last=False) == [0, 1]
    assert segment_sample_indices(times, 10.0, 20.0, is_last=True) == [2, 3]
    assert segment_sample_indices([5.0], 10.0, 20.0, is_last=True) == []


def test_scan_and_preview_inherit_shared_auto_decoder_without_backend_branch():
    for method in (MosaicScanWorker._scan, MosaicScanWorker._detect_mask):
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        reader_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "VideoReader"
        ]
        assert reader_calls
        assert all(
            keyword.arg != "decode_backend"
            for call in reader_calls
            for keyword in call.keywords
        )
def test_scan_detector_delegates_product_selection_to_shared_registry(monkeypatch):
    import jasna.session_factory as factory
    import jasna.mosaic.detection_registry as registry
    import jasna.vr180 as vr180
    weights = Path("models/rfdetr-v6.pt")
    detector = MagicMock()
    build = MagicMock(return_value=detector)
    monkeypatch.setattr(registry, "resolve_detection_model", lambda *args: ("rfdetr-v6", weights, None))
    monkeypatch.setattr(factory, "build_compiled_detection_model", build)
    monkeypatch.setattr(vr180, "resolve_vr_mode", lambda *args, **kwargs: SimpleNamespace(is_sbs=False))
    worker = MosaicScanWorker("input.mp4", SimpleNamespace(), SimpleNamespace(
        detection_model="rfdetr-v6", batch_size=4, fp16_mode=True, vr_mode="auto"), stride_seconds=1.0)
    assert worker._build_detector() is detector
    assert build.call_args.args == ("rfdetr-v6", weights)
    kwargs = build.call_args.kwargs
    assert kwargs["batch_size"] == 4
    assert kwargs["fp16"] is True
    assert kwargs["score_threshold"] == SCAN_SCORE_FLOOR
    assert callable(kwargs["log_callback"])
