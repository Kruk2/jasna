from __future__ import annotations

import pytest

from fractions import Fraction

import numpy as np

from jasna.gui.mosaic_scan import MosaicScanResult, _ScanTensorCollector
from jasna.media.splice import KeyframeIndex, build_splice_plan
from jasna.mosaic_scan import ltx_auto_segments, mosaic_boxes, mosaic_sizes, scan_sample_stride, segments_from_scores
from jasna.gui.segment_editor_state import SegmentEditorState
from jasna.segments import SegmentRange, SegmentRestoration

STANDARD = SegmentRestoration("basicvsrpp", None)


def test_stride_follows_fps():
    assert scan_sample_stride(29.97) == 30
    assert scan_sample_stride(60.0, seconds=0.5) == 30
    assert scan_sample_stride(30.0, seconds=0.0) == 1
    assert scan_sample_stride(1.0) == 1
    assert scan_sample_stride(0.1) == 1


def test_consecutive_hits_merge_into_one_range():
    times = (0.0, 1.0, 2.0, 3.0, 4.0)
    scores = (0.0, 0.8, 0.9, 0.7, 0.0)
    segments = segments_from_scores(
        times, scores, threshold=0.5, stride=1.0, duration=10.0
    )
    assert segments == (SegmentRange(0.5, 4.5),)


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
        sizes=(0.0,) * 3,
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


def test_scan_decoder_count_parallel_only_for_4k_on_nvidia():
    from jasna.gui.mosaic_scan import scan_decoder_count

    assert scan_decoder_count(3840, 2160, 120.0, amd=False) == 2
    assert scan_decoder_count(8192, 4096, 15.0, amd=False) == 2
    assert scan_decoder_count(1920, 1080, 120.0, amd=False) == 1
    assert scan_decoder_count(3840, 2160, 5.0, amd=False) == 1
    assert scan_decoder_count(3840, 2160, 120.0, amd=True) == 1


def test_segment_sample_indices_ownership():
    from jasna.gui.mosaic_scan import segment_sample_indices

    times = [9.8, 9.9, 10.0, 10.1]
    assert segment_sample_indices(times, 0.0, 10.0, is_last=False) == [0, 1]
    assert segment_sample_indices(times, 10.0, 20.0, is_last=True) == [2, 3]
    assert segment_sample_indices([5.0], 10.0, 20.0, is_last=True) == []


def test_mosaic_size_is_the_longest_side_of_the_biggest_mosaic_in_frame_heights():
    masks = np.zeros((3, 90, 160), dtype=np.uint8)
    masks[1, 10:19, 10:15] = 1
    masks[1, 50:56, 100:150] = 1
    masks[2, 0:45, 0:2] = 1
    sizes = mosaic_sizes(masks, frame_width=1920, frame_height=1080)
    assert sizes[0] == 0.0
    assert sizes[1] == pytest.approx(50 / 160 * 16 / 9)
    assert sizes[2] == pytest.approx(0.5)


LTX = SegmentRestoration("ltx", 7)


def _auto(sizes, *, keyframes=(0.0, 2.0, 4.0, 6.0, 8.0), min_size=0.12):
    times = tuple(float(t) for t in range(len(sizes)))
    scores = tuple(0.9 if size else 0.0 for size in sizes)
    return ltx_auto_segments(
        times,
        scores,
        sizes,
        threshold=0.5,
        ltx_min_size=min_size,
        stride=1.0,
        duration=10.0,
        keyframes=keyframes,
        ltx_seed=7,
    )


def test_auto_split_gives_big_mosaics_ltx_across_their_keyframe_interval():
    segments = _auto((0.0, 0.05, 0.05, 0.3, 0.05, 0.05, 0.05, 0.0, 0.0, 0.0))
    assert segments == (
        SegmentRange(0.5, 2.0, STANDARD),
        SegmentRange(2.0, 6.0, LTX),
        SegmentRange(6.0, 7.5, STANDARD),
    )


def test_auto_split_without_big_mosaics_keeps_one_standard_range():
    assert _auto((0.0, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)) == (SegmentRange(0.5, 3.5, STANDARD),)


def test_auto_split_ltx_stays_inside_detected_ranges():
    segments = _auto((0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), keyframes=(0.0, 8.0))
    assert segments == (SegmentRange(2.5, 4.5, LTX),)


def test_auto_split_ranges_smart_render_with_one_model_per_span():
    keyframes = (0, 60, 120, 180, 240)
    segments = _auto((0.0, 0.05, 0.3, 0.05, 0.05, 0.05, 0.3, 0.05, 0.0, 0.0))
    plan = build_splice_plan(segments, KeyframeIndex(keyframes, Fraction(1, 30), 0, 300), duration=10.0)
    models = [{segment.restoration.model for segment in owners} for owners in plan.render_span_segments()]
    assert all(len(span_models) == 1 for span_models in models)
    assert {"ltx", "basicvsrpp"} == set().union(*models)


def test_add_many_keeps_a_scanned_range_model():
    state = SegmentEditorState(duration=10.0, fps=30.0, segments=(), default_restoration=STANDARD)
    state.add_many((SegmentRange(1.0, 2.0, LTX), SegmentRange(3.0, 4.0)))
    assert [segment.restoration for segment in state.segments] == [LTX, STANDARD]


def test_mosaic_boxes_give_each_mosaic_its_box_and_size():
    mask = np.zeros((90, 160), dtype=np.uint8)
    mask[9:18, 16:24] = 1
    mask[45:90, 80:84] = 1
    boxes = sorted(mosaic_boxes(mask, aspect=16 / 9), key=lambda box: box.left)
    assert [(box.left, box.top) for box in boxes] == [(0.1, 0.1), (0.5, 0.5)]
    assert boxes[0].size == pytest.approx(9 / 90)
    assert boxes[1].size == pytest.approx(0.5)
