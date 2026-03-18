import numpy as np
import torch

from queue import Queue

from jasna.mosaic.detections import Detections
from jasna.pipeline_items import ClipRestoreItem, _SECONDARY_FLUSH, _SENTINEL
from jasna.restorer.restored_clip import RestoredClip
from jasna.tracking.clip_tracker import ClipTracker, TrackedClip
from jasna.tracking.frame_buffer import FrameBuffer
from jasna.pipeline_processing import process_frame_batch, finalize_processing


class _FakeRestorationPipeline:
    def restore_clip(
        self,
        clip: TrackedClip,
        frames: list[torch.Tensor],
        *,
        keep_start: int,
        keep_end: int,
    ) -> RestoredClip:
        del keep_start, keep_end
        restored_frames: list[torch.Tensor] = []
        enlarged_bboxes: list[tuple[int, int, int, int]] = []
        crop_shapes: list[tuple[int, int]] = []
        pad_offsets: list[tuple[int, int]] = []
        resize_shapes: list[tuple[int, int]] = []

        frame_h = int(frames[0].shape[1])
        frame_w = int(frames[0].shape[2])

        for bbox in clip.bboxes:
            x1 = int(np.floor(float(bbox[0])))
            y1 = int(np.floor(float(bbox[1])))
            x2 = int(np.ceil(float(bbox[2])))
            y2 = int(np.ceil(float(bbox[3])))
            enlarged_bboxes.append((x1, y1, x2, y2))
            crop_h = y2 - y1
            crop_w = x2 - x1
            crop_shapes.append((crop_h, crop_w))
            resize_shapes.append((crop_h, crop_w))
            pad_offsets.append((0, 0))
            restored_frames.append(torch.full((3, crop_h, crop_w), 200, dtype=torch.uint8))

        return RestoredClip(
            restored_frames=restored_frames,
            masks=[torch.ones((frame_h, frame_w), dtype=torch.bool) for _ in clip.masks],
            frame_shape=(frame_h, frame_w),
            enlarged_bboxes=enlarged_bboxes,
            crop_shapes=crop_shapes,
            pad_offsets=pad_offsets,
            resize_shapes=resize_shapes,
        )

    def restore_and_blend_clip(
        self,
        clip: TrackedClip,
        frames: list[torch.Tensor],
        *,
        keep_start: int,
        keep_end: int,
        frame_buffer: FrameBuffer,
        crossfade_weights: dict[int, float] | None = None,
    ) -> None:
        restored = self.restore_clip(clip, frames, keep_start=int(keep_start), keep_end=int(keep_end))
        frame_buffer.blend_clip(clip, restored, keep_start=int(keep_start), keep_end=int(keep_end))


def _drain_queue(clip_queue: Queue, frame_buffer: FrameBuffer, pipeline) -> None:
    while not clip_queue.empty():
        item = clip_queue.get_nowait()
        if item is _SENTINEL:
            break
        if item is _SECONDARY_FLUSH:
            continue
        ci: ClipRestoreItem = item  # type: ignore[assignment]
        pipeline.restore_and_blend_clip(
            ci.clip, ci.frames,
            keep_start=ci.keep_start, keep_end=ci.keep_end,
            frame_buffer=frame_buffer, crossfade_weights=ci.crossfade_weights,
        )


def _make_empty_det_batch(*, batch_size: int) -> Detections:
    return Detections(
        boxes_xyxy=[np.zeros((0, 4), dtype=np.float32) for _ in range(batch_size)],
        masks=[torch.zeros((0, 8, 8), dtype=torch.bool) for _ in range(batch_size)],
    )


def test_process_batch_triggers_secondary_flush_after_gap() -> None:
    batch_size = 1
    tracker = ClipTracker(max_clip_size=10, temporal_overlap=0, iou_threshold=0.0)
    fb = FrameBuffer(device=torch.device("cpu"))
    clip_queue: Queue[ClipRestoreItem | object] = Queue()
    raw_frame_context: dict[int, dict[int, torch.Tensor]] = {}
    frames = torch.zeros((1, 3, 8, 8), dtype=torch.uint8)

    streak = 0
    for i in range(5):
        res = process_frame_batch(
            frames=frames,
            pts_list=[i],
            start_frame_idx=i,
            batch_size=batch_size,
            target_hw=(8, 8),
            detections_fn=lambda _x, *, target_hw: _make_empty_det_batch(batch_size=batch_size),
            tracker=tracker,
            frame_buffer=fb,
            clip_queue=clip_queue,
            discard_margin=0,
            blend_frames=0,
            raw_frame_context=raw_frame_context,
            no_track_streak=streak,
            secondary_flush_gap_frames=5,
        )
        streak = res.no_track_streak
        if i < 4:
            assert res.should_flush_secondary is False
        else:
            assert res.should_flush_secondary is True


def test_process_batch_gap_flush_resets_after_detection() -> None:
    batch_size = 1
    tracker = ClipTracker(max_clip_size=10, temporal_overlap=0, iou_threshold=0.0)
    fb = FrameBuffer(device=torch.device("cpu"))
    clip_queue: Queue[ClipRestoreItem | object] = Queue()
    raw_frame_context: dict[int, dict[int, torch.Tensor]] = {}
    frames = torch.zeros((1, 3, 8, 8), dtype=torch.uint8)

    empty_fn = lambda _x, *, target_hw: _make_empty_det_batch(batch_size=batch_size)
    det_fn = lambda _x, *, target_hw: _make_single_det_batch(effective_bs=batch_size, batch_size=batch_size)

    streak = 0
    for i in range(3):
        res = process_frame_batch(
            frames=frames,
            pts_list=[i],
            start_frame_idx=i,
            batch_size=batch_size,
            target_hw=(8, 8),
            detections_fn=empty_fn,
            tracker=tracker,
            frame_buffer=fb,
            clip_queue=clip_queue,
            discard_margin=0,
            blend_frames=0,
            raw_frame_context=raw_frame_context,
            no_track_streak=streak,
            secondary_flush_gap_frames=5,
        )
        streak = res.no_track_streak
        assert res.should_flush_secondary is False
    assert streak == 3

    res = process_frame_batch(
        frames=frames,
        pts_list=[3],
        start_frame_idx=3,
        batch_size=batch_size,
        target_hw=(8, 8),
        detections_fn=det_fn,
        tracker=tracker,
        frame_buffer=fb,
        clip_queue=clip_queue,
        discard_margin=0,
        blend_frames=0,
        raw_frame_context=raw_frame_context,
        no_track_streak=streak,
        secondary_flush_gap_frames=5,
    )
    assert res.no_track_streak == 0
    assert res.should_flush_secondary is False


def _make_single_det_batch(*, effective_bs: int, batch_size: int, box=(2.0, 2.0, 6.0, 6.0)) -> Detections:
    boxes_xyxy = []
    masks = []
    for _ in range(batch_size):
        boxes_xyxy.append(np.array([box], dtype=np.float32))
        m = torch.zeros((1, 8, 8), dtype=torch.bool)
        m[0, 0, 0] = True
        masks.append(m)
    return Detections(boxes_xyxy=boxes_xyxy[:batch_size], masks=masks[:batch_size])


def _run_batches(
    *,
    pts_lists: list[list[int]],
    batch_size: int,
    max_clip_size: int,
    discard_margin: int,
    blend_frames: int = 0,
    detections_fn,
    pipeline,
) -> list[tuple[int, torch.Tensor, int]]:
    tracker = ClipTracker(max_clip_size=max_clip_size, temporal_overlap=discard_margin, iou_threshold=0.0)
    fb = FrameBuffer(device=torch.device("cpu"))
    clip_queue: Queue[ClipRestoreItem | object] = Queue()
    raw_frame_context: dict[int, dict[int, torch.Tensor]] = {}

    frames = torch.zeros((batch_size, 3, 8, 8), dtype=torch.uint8)
    frame_idx = 0

    for pts_list in pts_lists:
        res = process_frame_batch(
            frames=frames, pts_list=list(pts_list), start_frame_idx=frame_idx,
            batch_size=batch_size, target_hw=(8, 8), detections_fn=detections_fn,
            tracker=tracker, frame_buffer=fb, clip_queue=clip_queue,
            discard_margin=discard_margin, blend_frames=blend_frames,
            raw_frame_context=raw_frame_context,
        )
        _drain_queue(clip_queue, fb, pipeline)
        frame_idx = res.next_frame_idx

    finalize_processing(
        tracker=tracker, frame_buffer=fb, clip_queue=clip_queue,
        discard_margin=discard_margin, blend_frames=blend_frames,
        raw_frame_context=raw_frame_context,
    )
    _drain_queue(clip_queue, fb, pipeline)

    ready = list(fb.get_ready_frames())
    ready.extend(fb.flush())
    return ready


def test_process_batch_and_finalize_overlap_discard_delays_tail_until_continuation() -> None:
    batch_size = 2
    discard_margin = 1
    rest = _FakeRestorationPipeline()

    def detections_fn(_: torch.Tensor, *, target_hw: tuple[int, int]) -> Detections:
        return _make_single_det_batch(effective_bs=batch_size, batch_size=batch_size)

    out = _run_batches(
        pts_lists=[[0, 1], [2, 3], [4, 5]],
        batch_size=batch_size, max_clip_size=6, discard_margin=discard_margin,
        detections_fn=detections_fn, pipeline=rest,
    )
    assert [x[0] for x in out] == [0, 1, 2, 3, 4, 5]


def test_process_batch_with_crossfade_outputs_all_frames_in_order() -> None:
    batch_size = 2
    discard_margin = 2
    blend_frames = 1
    rest = _FakeRestorationPipeline()

    def detections_fn(_: torch.Tensor, *, target_hw: tuple[int, int]) -> Detections:
        return _make_single_det_batch(effective_bs=batch_size, batch_size=batch_size)

    out = _run_batches(
        pts_lists=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
        batch_size=batch_size, max_clip_size=6, discard_margin=discard_margin,
        blend_frames=blend_frames, detections_fn=detections_fn, pipeline=rest,
    )
    out_indices = [x[0] for x in out]
    assert out_indices == list(range(10)), f"expected frames 0-9, got {out_indices}"


def test_process_batch_without_discard_encodes_all_frames() -> None:
    batch_size = 2
    discard_margin = 0
    rest = _FakeRestorationPipeline()

    def detections_fn(_: torch.Tensor, *, target_hw: tuple[int, int]) -> Detections:
        return _make_single_det_batch(effective_bs=batch_size, batch_size=batch_size)

    out = _run_batches(
        pts_lists=[[0, 1], [2, 3]],
        batch_size=batch_size, max_clip_size=4, discard_margin=discard_margin,
        detections_fn=detections_fn, pipeline=rest,
    )
    assert [x[0] for x in out] == [0, 1, 2, 3]


def test_zero_overlap_split_blends_all_frames_including_boundary() -> None:
    """With temporal_overlap=0 and a clip split, the split-boundary frame must be blended, not raw."""
    batch_size = 1
    discard_margin = 0
    max_clip_size = 3
    rest = _FakeRestorationPipeline()

    def detections_fn(_: torch.Tensor, *, target_hw: tuple[int, int]) -> Detections:
        return _make_single_det_batch(effective_bs=batch_size, batch_size=batch_size)

    out = _run_batches(
        pts_lists=[[pts] for pts in range(5)],
        batch_size=batch_size, max_clip_size=max_clip_size, discard_margin=discard_margin,
        detections_fn=detections_fn, pipeline=rest,
    )
    out_indices = [x[0] for x in out]
    assert out_indices == list(range(5)), f"expected frames 0-4, got {out_indices}"

    for idx, blended, _ in out:
        region = blended[:, 2:6, 2:6]
        assert torch.all(region == 200), f"frame {idx} mosaic region was not blended (has raw pixels)"


def _run_real_pipeline_batches(
    monkeypatch,
    *,
    max_clip_size: int,
    temporal_overlap: int,
    blend_frames: int,
    num_frames: int,
    original_value: int,
    restored_float: float,
) -> list[tuple[int, torch.Tensor, int]]:
    import jasna.restorer.restoration_pipeline as rp
    monkeypatch.setattr(rp, "BORDER_RATIO", 0.0)
    monkeypatch.setattr(rp, "MIN_BORDER", 0)
    monkeypatch.setattr(rp, "MAX_EXPANSION_FACTOR", 0.0)

    class _ConstantRestorer:
        dtype = torch.float32
        def raw_process(self, crops: list[torch.Tensor]) -> torch.Tensor:
            stacked = []
            for f in crops:
                stacked.append(torch.full(f.permute(2, 0, 1).shape, restored_float, dtype=torch.float32))
            return torch.stack(stacked, dim=0)

    def _ones_blend_mask(crop: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(crop.squeeze(), dtype=torch.float32)

    from jasna.restorer.restoration_pipeline import RestorationPipeline
    pipeline = RestorationPipeline(restorer=_ConstantRestorer())  # type: ignore[arg-type]
    fb = FrameBuffer(device=torch.device("cpu"), blend_mask_fn=_ones_blend_mask)
    tracker = ClipTracker(max_clip_size=max_clip_size, temporal_overlap=temporal_overlap, iou_threshold=0.0)
    clip_queue: Queue[ClipRestoreItem | object] = Queue()

    bbox = np.array([2.0, 2.0, 6.0, 6.0], dtype=np.float32)

    def detections_fn(frames_in: torch.Tensor, *, target_hw: tuple[int, int]) -> Detections:
        bs = frames_in.shape[0]
        return Detections(
            boxes_xyxy=[np.array([bbox], dtype=np.float32) for _ in range(bs)],
            masks=[torch.ones((1, 8, 8), dtype=torch.bool) for _ in range(bs)],
        )

    raw_frame_context: dict[int, dict[int, torch.Tensor]] = {}
    frame_idx = 0

    for pts in range(num_frames):
        frame_batch = torch.full((1, 3, 8, 8), original_value, dtype=torch.uint8)
        res = process_frame_batch(
            frames=frame_batch, pts_list=[pts], start_frame_idx=frame_idx,
            batch_size=1, target_hw=(8, 8), detections_fn=detections_fn,
            tracker=tracker, frame_buffer=fb, clip_queue=clip_queue,
            discard_margin=temporal_overlap, blend_frames=blend_frames,
            raw_frame_context=raw_frame_context,
        )
        _drain_queue(clip_queue, fb, pipeline)
        frame_idx = res.next_frame_idx

    finalize_processing(
        tracker=tracker, frame_buffer=fb, clip_queue=clip_queue,
        discard_margin=temporal_overlap, blend_frames=blend_frames,
        raw_frame_context=raw_frame_context,
    )
    _drain_queue(clip_queue, fb, pipeline)

    ready = list(fb.get_ready_frames())
    ready.extend(fb.flush())
    return ready


def test_overlapping_crossfade_no_black_pixels(monkeypatch) -> None:
    """When 2*(temporal_overlap + blend_frames) > max_clip_size, the child and parent
    crossfade weight regions overlap within a middle clip (both continuation and split).
    The .update() in _process_ended_clips lets parent weights overwrite child weights,
    making combined weights sum > 1.0 at some frames, which produces black pixels via
    negative blending: original*(1 - sum) goes negative → clamped to 0."""
    max_clip_size = 5
    temporal_overlap = 2
    blend_frames = 2
    original_value = 200
    restored_float = 0.2
    restored_u8 = int(round(restored_float * 255))

    all_output = _run_real_pipeline_batches(
        monkeypatch, max_clip_size=max_clip_size, temporal_overlap=temporal_overlap,
        blend_frames=blend_frames, num_frames=15, original_value=original_value,
        restored_float=restored_float,
    )
    assert len(all_output) == 15

    for idx, blended, _ in all_output:
        region = blended[:, 2:6, 2:6]
        assert region.min().item() > 0, (
            f"frame {idx}: black pixel in mosaic region (min={region.min().item()}). "
            f"Crossfade weight overlap causes negative blending with "
            f"original={original_value}, restored={restored_u8}"
        )


def test_merged_crossfade_weights_sum_to_one_across_clip_boundaries() -> None:
    """Adjacent clips' crossfade weights must sum to 1.0 at each overlapping frame."""
    import pytest

    batch_size = 1
    discard_margin = 2
    blend_frames = 1
    max_clip_size = 10
    tracker = ClipTracker(max_clip_size=max_clip_size, temporal_overlap=discard_margin, iou_threshold=0.0)
    fb = FrameBuffer(device=torch.device("cpu"))
    clip_queue: Queue[ClipRestoreItem | object] = Queue()

    captured: list[tuple[int, int, dict[int, float] | None]] = []

    class _CapturePipeline(_FakeRestorationPipeline):
        def restore_and_blend_clip(
            self,
            clip: TrackedClip,
            frames: list[torch.Tensor],
            *,
            keep_start: int,
            keep_end: int,
            frame_buffer: FrameBuffer,
            crossfade_weights: dict[int, float] | None = None,
        ) -> None:
            captured.append((clip.start_frame, clip.frame_count, crossfade_weights))
            restored = self.restore_clip(clip, frames, keep_start=int(keep_start), keep_end=int(keep_end))
            frame_buffer.blend_clip(clip, restored, keep_start=int(keep_start), keep_end=int(keep_end))

    rest = _CapturePipeline()

    def detections_fn(_: torch.Tensor, *, target_hw: tuple[int, int]) -> Detections:
        return _make_single_det_batch(effective_bs=batch_size, batch_size=batch_size)

    frames = torch.zeros((batch_size, 3, 8, 8), dtype=torch.uint8)
    frame_idx = 0
    raw_frame_context: dict[int, dict[int, torch.Tensor]] = {}
    for pts in range(25):
        res = process_frame_batch(
            frames=frames, pts_list=[pts], start_frame_idx=frame_idx,
            batch_size=batch_size, target_hw=(8, 8), detections_fn=detections_fn,
            tracker=tracker, frame_buffer=fb, clip_queue=clip_queue,
            discard_margin=discard_margin, blend_frames=blend_frames,
            raw_frame_context=raw_frame_context,
        )
        _drain_queue(clip_queue, fb, rest)
        frame_idx = res.next_frame_idx

    finalize_processing(
        tracker=tracker, frame_buffer=fb, clip_queue=clip_queue,
        discard_margin=discard_margin, blend_frames=blend_frames,
        raw_frame_context=raw_frame_context,
    )
    _drain_queue(clip_queue, fb, rest)

    assert len(captured) >= 3

    for i in range(len(captured) - 1):
        start_a, fc_a, weights_a = captured[i]
        start_b, fc_b, weights_b = captured[i + 1]
        if weights_a is None or weights_b is None:
            continue

        for local_b, w_b in sorted(weights_b.items()):
            abs_frame = start_b + local_b
            local_a = abs_frame - start_a
            w_a = weights_a.get(local_a)
            if w_a is not None:
                assert w_a + w_b == pytest.approx(1.0), (
                    f"clips {i} and {i+1}: at absolute frame {abs_frame}, "
                    f"weights {w_a:.3f} + {w_b:.3f} = {w_a + w_b:.3f} != 1.0"
                )


def test_crossfade_produces_correct_pixel_values(monkeypatch) -> None:
    """With valid crossfade params, every mosaic pixel should be close to the restored value."""
    restored_float = 0.5
    restored_u8 = int(round(restored_float * 255))

    all_output = _run_real_pipeline_batches(
        monkeypatch, max_clip_size=10, temporal_overlap=2, blend_frames=1,
        num_frames=25, original_value=200, restored_float=restored_float,
    )

    assert len(all_output) == 25
    for idx, blended, _ in all_output:
        region = blended[:, 2:6, 2:6].float()
        assert (region - restored_u8).abs().max().item() <= 2, (
            f"frame {idx}: mosaic pixel deviates from expected {restored_u8} "
            f"(actual range [{region.min().item()}, {region.max().item()}])"
        )


def test_crossfade_at_exact_boundary_params(monkeypatch) -> None:
    """When 2*(d+bf) == max_clip_size exactly, crossfade regions are adjacent with no gap."""
    restored_float = 0.5
    restored_u8 = int(round(restored_float * 255))

    all_output = _run_real_pipeline_batches(
        monkeypatch, max_clip_size=8, temporal_overlap=3, blend_frames=1,
        num_frames=20, original_value=200, restored_float=restored_float,
    )

    assert len(all_output) == 20
    for idx, blended, _ in all_output:
        region = blended[:, 2:6, 2:6].float()
        assert (region - restored_u8).abs().max().item() <= 2, (
            f"frame {idx}: mosaic pixel deviates from expected {restored_u8} "
            f"(actual range [{region.min().item()}, {region.max().item()}])"
        )


def test_crossfade_weights_applied_in_blending(monkeypatch) -> None:
    """Verify that crossfade_weights are actually applied during blending, not ignored."""
    import jasna.restorer.restoration_pipeline as rp
    monkeypatch.setattr(rp, "BORDER_RATIO", 0.0)
    monkeypatch.setattr(rp, "MIN_BORDER", 0)
    monkeypatch.setattr(rp, "MAX_EXPANSION_FACTOR", 0.0)

    original_value = 200

    class _AlternatingRestorer:
        dtype = torch.float32
        def __init__(self) -> None:
            self._call_count = 0
        def raw_process(self, crops: list[torch.Tensor]) -> torch.Tensor:
            self._call_count += 1
            val = 0.3 if self._call_count % 2 == 1 else 0.7
            stacked = []
            for f in crops:
                stacked.append(torch.full(f.permute(2, 0, 1).shape, val, dtype=torch.float32))
            return torch.stack(stacked, dim=0)

    def _ones_blend_mask(crop: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(crop.squeeze(), dtype=torch.float32)

    from jasna.restorer.restoration_pipeline import RestorationPipeline

    bbox = np.array([2.0, 2.0, 6.0, 6.0], dtype=np.float32)

    def detections_fn(frames_in: torch.Tensor, *, target_hw: tuple[int, int]) -> Detections:
        bs = frames_in.shape[0]
        return Detections(
            boxes_xyxy=[np.array([bbox], dtype=np.float32) for _ in range(bs)],
            masks=[torch.ones((1, 8, 8), dtype=torch.bool) for _ in range(bs)],
        )

    discard_margin = 2
    max_clip_size = 10

    def _run_with_blend(bf: int) -> dict[int, torch.Tensor]:
        p = RestorationPipeline(restorer=_AlternatingRestorer())  # type: ignore[arg-type]
        fb = FrameBuffer(device=torch.device("cpu"), blend_mask_fn=_ones_blend_mask)
        t = ClipTracker(max_clip_size=max_clip_size, temporal_overlap=discard_margin, iou_threshold=0.0)
        q: Queue[ClipRestoreItem | object] = Queue()
        ctx: dict[int, dict[int, torch.Tensor]] = {}
        fi = 0
        for pts in range(15):
            res = process_frame_batch(
                frames=torch.full((1, 3, 8, 8), original_value, dtype=torch.uint8),
                pts_list=[pts], start_frame_idx=fi, batch_size=1, target_hw=(8, 8),
                detections_fn=detections_fn, tracker=t, frame_buffer=fb,
                clip_queue=q, discard_margin=discard_margin, blend_frames=bf,
                raw_frame_context=ctx,
            )
            _drain_queue(q, fb, p)
            fi = res.next_frame_idx
        finalize_processing(
            tracker=t, frame_buffer=fb, clip_queue=q,
            discard_margin=discard_margin, blend_frames=bf, raw_frame_context=ctx,
        )
        _drain_queue(q, fb, p)
        ready = list(fb.get_ready_frames())
        ready.extend(fb.flush())
        return {idx: blended for idx, blended, _ in ready}

    cf_by_idx = _run_with_blend(1)
    no_by_idx = _run_with_blend(0)

    differs = False
    for idx in cf_by_idx:
        if idx in no_by_idx:
            cf_region = cf_by_idx[idx][:, 2:6, 2:6]
            no_region = no_by_idx[idx][:, 2:6, 2:6]
            if not torch.equal(cf_region, no_region):
                differs = True
                break

    assert differs, (
        "crossfade had no effect on any frame — crossfade_weights are not being applied"
    )


def test_long_chain_of_splits_all_frames_correct(monkeypatch) -> None:
    """5+ consecutive clip splits must produce correct output for every frame."""
    restored_float = 0.6
    restored_u8 = int(round(restored_float * 255))

    all_output = _run_real_pipeline_batches(
        monkeypatch, max_clip_size=10, temporal_overlap=2, blend_frames=1,
        num_frames=50, original_value=180, restored_float=restored_float,
    )

    assert len(all_output) == 50
    out_indices = [idx for idx, _, _ in all_output]
    assert out_indices == list(range(50))

    for idx, blended, _ in all_output:
        region = blended[:, 2:6, 2:6].float()
        assert region.min().item() > 0, f"frame {idx}: black pixel in mosaic region"
        assert (region - restored_u8).abs().max().item() <= 2, (
            f"frame {idx}: mosaic pixel deviates from expected {restored_u8} "
            f"(actual range [{region.min().item()}, {region.max().item()}])"
        )


def test_bf_clamping_tight_params_no_artifacts(monkeypatch) -> None:
    """With params that trigger bf clamping (2*(d+bf) > max_clip_size), the runtime
    clamp should produce no black pixels and all frames should be output."""
    all_output = _run_real_pipeline_batches(
        monkeypatch, max_clip_size=8, temporal_overlap=3, blend_frames=3,
        num_frames=20, original_value=200, restored_float=0.3,
    )

    assert len(all_output) == 20
    out_indices = [idx for idx, _, _ in all_output]
    assert out_indices == list(range(20))

    for idx, blended, _ in all_output:
        region = blended[:, 2:6, 2:6]
        assert region.min().item() > 0, (
            f"frame {idx}: black pixel in mosaic region (min={region.min().item()})"
        )


def test_crossfade_with_split_assigns_parent_weights() -> None:
    """Verify that split clips get parent crossfade weights (not just continuations)."""
    batch_size = 1
    discard_margin = 2
    blend_frames = 1
    max_clip_size = 6
    tracker = ClipTracker(max_clip_size=max_clip_size, temporal_overlap=discard_margin, iou_threshold=0.0)
    fb = FrameBuffer(device=torch.device("cpu"))
    clip_queue: Queue[ClipRestoreItem | object] = Queue()

    captured_weights: list[dict[int, float] | None] = []

    class _CapturePipeline(_FakeRestorationPipeline):
        def restore_and_blend_clip(
            self,
            clip: TrackedClip,
            frames: list[torch.Tensor],
            *,
            keep_start: int,
            keep_end: int,
            frame_buffer: FrameBuffer,
            crossfade_weights: dict[int, float] | None = None,
        ) -> None:
            captured_weights.append(crossfade_weights)
            restored = self.restore_clip(clip, frames, keep_start=int(keep_start), keep_end=int(keep_end))
            frame_buffer.blend_clip(clip, restored, keep_start=int(keep_start), keep_end=int(keep_end))

    rest = _CapturePipeline()

    def detections_fn(_: torch.Tensor, *, target_hw: tuple[int, int]) -> Detections:
        return _make_single_det_batch(effective_bs=batch_size, batch_size=batch_size)

    frames = torch.zeros((batch_size, 3, 8, 8), dtype=torch.uint8)
    frame_idx = 0
    raw_frame_context: dict[int, dict[int, torch.Tensor]] = {}
    for pts in range(10):
        res = process_frame_batch(
            frames=frames, pts_list=[pts], start_frame_idx=frame_idx,
            batch_size=batch_size, target_hw=(8, 8), detections_fn=detections_fn,
            tracker=tracker, frame_buffer=fb, clip_queue=clip_queue,
            discard_margin=discard_margin, blend_frames=blend_frames,
            raw_frame_context=raw_frame_context,
        )
        _drain_queue(clip_queue, fb, rest)
        frame_idx = res.next_frame_idx

    finalize_processing(
        tracker=tracker, frame_buffer=fb, clip_queue=clip_queue,
        discard_margin=discard_margin, blend_frames=blend_frames,
        raw_frame_context=raw_frame_context,
    )
    _drain_queue(clip_queue, fb, rest)

    first_weights = captured_weights[0]
    assert first_weights is not None, "split clip should have parent crossfade weights"
    vals = [first_weights[k] for k in sorted(first_weights.keys())]
    for i in range(1, len(vals)):
        assert vals[i] < vals[i - 1], "parent crossfade weights should be descending"

    second_weights = captured_weights[1]
    assert second_weights is not None, "continuation clip should have child crossfade weights"
