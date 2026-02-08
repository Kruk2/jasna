import numpy as np
import torch

from jasna.mosaic.detections import Detections
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

    def flush_secondary(self, *, frame_buffer: FrameBuffer) -> None:
        del frame_buffer

    def poll_secondary(self, *, frame_buffer: FrameBuffer, limit: int | None = None) -> None:
        del frame_buffer, limit

    def flush_track(self, track_id: int) -> None:
        del track_id

    def transfer_track(self, old_track_id: int, new_track_id: int) -> None:
        del old_track_id, new_track_id


def _make_single_det_batch(*, effective_bs: int, batch_size: int, box=(2.0, 2.0, 6.0, 6.0)) -> Detections:
    boxes_xyxy = []
    masks = []
    for _ in range(batch_size):
        boxes_xyxy.append(np.array([box], dtype=np.float32))
        m = torch.zeros((1, 8, 8), dtype=torch.bool)
        m[0, 0, 0] = True
        masks.append(m)
    return Detections(boxes_xyxy=boxes_xyxy[:batch_size], masks=masks[:batch_size])


def test_process_batch_and_finalize_overlap_discard_delays_tail_until_continuation() -> None:
    batch_size = 2
    discard_margin = 1
    tracker = ClipTracker(max_clip_size=6, temporal_overlap=discard_margin, iou_threshold=0.0)
    fb = FrameBuffer(device=torch.device("cpu"))
    rest = _FakeRestorationPipeline()

    det_calls = {"n": 0}

    def detections_fn(_: torch.Tensor, *, target_hw: tuple[int, int]) -> Detections:
        det_calls["n"] += 1
        return _make_single_det_batch(effective_bs=batch_size, batch_size=batch_size)

    frames = torch.zeros((batch_size, 3, 8, 8), dtype=torch.uint8)

    ready_all: list[tuple[int, torch.Tensor, int]] = []
    frame_idx = 0
    raw_frame_context: dict[int, dict[int, torch.Tensor]] = {}
    for pts_list in ([0, 1], [2, 3], [4, 5]):
        res = process_frame_batch(
            frames=frames,
            pts_list=list(pts_list),
            start_frame_idx=frame_idx,
            batch_size=batch_size,
            target_hw=(8, 8),
            detections_fn=detections_fn,
            tracker=tracker,
            frame_buffer=fb,
            restoration_pipeline=rest,  # type: ignore[arg-type]
            discard_margin=discard_margin,
            raw_frame_context=raw_frame_context,
        )
        ready_all.extend(res.ready_frames)
        frame_idx = res.next_frame_idx

    assert [x[0] for x in ready_all] == [0, 1, 2, 3, 4]

    remaining = finalize_processing(
        tracker=tracker,
        frame_buffer=fb,
        restoration_pipeline=rest,
        discard_margin=discard_margin,
        raw_frame_context=raw_frame_context,
    )  # type: ignore[arg-type]
    assert [x[0] for x in remaining] == [5]


def test_process_batch_with_crossfade_outputs_all_frames_in_order() -> None:
    batch_size = 2
    discard_margin = 2
    blend_frames = 1
    tracker = ClipTracker(max_clip_size=6, temporal_overlap=discard_margin, iou_threshold=0.0)
    fb = FrameBuffer(device=torch.device("cpu"))
    rest = _FakeRestorationPipeline()

    def detections_fn(_: torch.Tensor, *, target_hw: tuple[int, int]) -> Detections:
        return _make_single_det_batch(effective_bs=batch_size, batch_size=batch_size)

    frames = torch.zeros((batch_size, 3, 8, 8), dtype=torch.uint8)

    ready_all: list[tuple[int, torch.Tensor, int]] = []
    frame_idx = 0
    raw_frame_context: dict[int, dict[int, torch.Tensor]] = {}
    for pts_list in ([0, 1], [2, 3], [4, 5], [6, 7], [8, 9]):
        res = process_frame_batch(
            frames=frames,
            pts_list=list(pts_list),
            start_frame_idx=frame_idx,
            batch_size=batch_size,
            target_hw=(8, 8),
            detections_fn=detections_fn,
            tracker=tracker,
            frame_buffer=fb,
            restoration_pipeline=rest,  # type: ignore[arg-type]
            discard_margin=discard_margin,
            blend_frames=blend_frames,
            raw_frame_context=raw_frame_context,
        )
        ready_all.extend(res.ready_frames)
        frame_idx = res.next_frame_idx

    remaining = finalize_processing(
        tracker=tracker,
        frame_buffer=fb,
        restoration_pipeline=rest,
        discard_margin=discard_margin,
        blend_frames=blend_frames,
        raw_frame_context=raw_frame_context,
    )  # type: ignore[arg-type]
    out = ready_all + remaining
    out_indices = [x[0] for x in out]
    assert out_indices == list(range(10)), f"expected frames 0-9, got {out_indices}"


def test_process_batch_without_discard_encodes_all_frames() -> None:
    batch_size = 2
    discard_margin = 0
    tracker = ClipTracker(max_clip_size=4, temporal_overlap=discard_margin, iou_threshold=0.0)
    fb = FrameBuffer(device=torch.device("cpu"))
    rest = _FakeRestorationPipeline()

    def detections_fn(_: torch.Tensor, *, target_hw: tuple[int, int]) -> Detections:
        return _make_single_det_batch(effective_bs=batch_size, batch_size=batch_size)

    frames = torch.zeros((batch_size, 3, 8, 8), dtype=torch.uint8)
    ready_all: list[tuple[int, torch.Tensor, int]] = []
    frame_idx = 0
    raw_frame_context: dict[int, dict[int, torch.Tensor]] = {}
    for pts_list in ([0, 1], [2, 3]):
        res = process_frame_batch(
            frames=frames,
            pts_list=list(pts_list),
            start_frame_idx=frame_idx,
            batch_size=batch_size,
            target_hw=(8, 8),
            detections_fn=detections_fn,
            tracker=tracker,
            frame_buffer=fb,
            restoration_pipeline=rest,  # type: ignore[arg-type]
            discard_margin=discard_margin,
            raw_frame_context=raw_frame_context,
        )
        ready_all.extend(res.ready_frames)
        frame_idx = res.next_frame_idx

    remaining = finalize_processing(
        tracker=tracker,
        frame_buffer=fb,
        restoration_pipeline=rest,
        discard_margin=discard_margin,
        raw_frame_context=raw_frame_context,
    )  # type: ignore[arg-type]
    out = ready_all + remaining
    assert [x[0] for x in out] == [0, 1, 2, 3]


def test_zero_overlap_split_blends_all_frames_including_boundary() -> None:
    """With temporal_overlap=0 and a clip split, the split-boundary frame must be blended, not raw."""
    batch_size = 1
    discard_margin = 0
    max_clip_size = 3
    tracker = ClipTracker(max_clip_size=max_clip_size, temporal_overlap=discard_margin, iou_threshold=0.0)
    fb = FrameBuffer(device=torch.device("cpu"))
    rest = _FakeRestorationPipeline()

    def detections_fn(_: torch.Tensor, *, target_hw: tuple[int, int]) -> Detections:
        return _make_single_det_batch(effective_bs=batch_size, batch_size=batch_size)

    frames = torch.zeros((batch_size, 3, 8, 8), dtype=torch.uint8)
    ready_all: list[tuple[int, torch.Tensor, int]] = []
    frame_idx = 0
    raw_frame_context: dict[int, dict[int, torch.Tensor]] = {}
    # Process 5 frames one at a time: clip splits at frame 2, new clip for frames 3-4
    for pts in range(5):
        res = process_frame_batch(
            frames=frames,
            pts_list=[pts],
            start_frame_idx=frame_idx,
            batch_size=batch_size,
            target_hw=(8, 8),
            detections_fn=detections_fn,
            tracker=tracker,
            frame_buffer=fb,
            restoration_pipeline=rest,  # type: ignore[arg-type]
            discard_margin=discard_margin,
            raw_frame_context=raw_frame_context,
        )
        ready_all.extend(res.ready_frames)
        frame_idx = res.next_frame_idx

    remaining = finalize_processing(
        tracker=tracker,
        frame_buffer=fb,
        restoration_pipeline=rest,
        discard_margin=discard_margin,
        raw_frame_context=raw_frame_context,
    )  # type: ignore[arg-type]
    out = ready_all + remaining
    out_indices = [x[0] for x in out]
    assert out_indices == list(range(5)), f"expected frames 0-4, got {out_indices}"

    # Every frame in the mosaic region (2:6, 2:6) must be blended to 200 (not raw 0).
    # Frame 2 is the split boundary -- the bug was that it stayed raw.
    for idx, blended, _ in out:
        region = blended[:, 2:6, 2:6]
        assert torch.all(region == 200), f"frame {idx} mosaic region was not blended (has raw pixels)"


def test_overlapping_crossfade_no_black_pixels(monkeypatch) -> None:
    """When 2*(temporal_overlap + blend_frames) > max_clip_size, the child and parent
    crossfade weight regions overlap within a middle clip (both continuation and split).
    The .update() in _process_ended_clips lets parent weights overwrite child weights,
    making combined weights sum > 1.0 at some frames, which produces black pixels via
    negative blending: original*(1 - sum) goes negative → clamped to 0."""
    import jasna.restorer.restoration_pipeline as rp
    monkeypatch.setattr(rp, "BORDER_RATIO", 0.0)
    monkeypatch.setattr(rp, "MIN_BORDER", 0)
    monkeypatch.setattr(rp, "MAX_EXPANSION_FACTOR", 0.0)

    max_clip_size = 5
    temporal_overlap = 2
    blend_frames = 2
    original_value = 200
    restored_float = 0.2
    restored_u8 = int(round(restored_float * 255))

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

    bbox = np.array([2.0, 2.0, 6.0, 6.0], dtype=np.float32)

    def detections_fn(frames_in: torch.Tensor, *, target_hw: tuple[int, int]) -> Detections:
        bs = frames_in.shape[0]
        return Detections(
            boxes_xyxy=[np.array([bbox], dtype=np.float32) for _ in range(bs)],
            masks=[torch.ones((1, 8, 8), dtype=torch.bool) for _ in range(bs)],
        )

    num_frames = 15
    raw_frame_context: dict[int, dict[int, torch.Tensor]] = {}
    ready_all: list[tuple[int, torch.Tensor, int]] = []
    frame_idx = 0

    for pts in range(num_frames):
        frame_batch = torch.full((1, 3, 8, 8), original_value, dtype=torch.uint8)
        res = process_frame_batch(
            frames=frame_batch,
            pts_list=[pts],
            start_frame_idx=frame_idx,
            batch_size=1,
            target_hw=(8, 8),
            detections_fn=detections_fn,
            tracker=tracker,
            frame_buffer=fb,
            restoration_pipeline=pipeline,
            discard_margin=temporal_overlap,
            blend_frames=blend_frames,
            raw_frame_context=raw_frame_context,
        )
        ready_all.extend(res.ready_frames)
        frame_idx = res.next_frame_idx

    remaining = finalize_processing(
        tracker=tracker,
        frame_buffer=fb,
        restoration_pipeline=pipeline,
        discard_margin=temporal_overlap,
        blend_frames=blend_frames,
        raw_frame_context=raw_frame_context,
    )
    all_output = ready_all + remaining
    assert len(all_output) == num_frames

    for idx, blended, _ in all_output:
        region = blended[:, 2:6, 2:6]
        assert region.min().item() > 0, (
            f"frame {idx}: black pixel in mosaic region (min={region.min().item()}). "
            f"Crossfade weight overlap causes negative blending with "
            f"original={original_value}, restored={restored_u8}"
        )


def _run_real_pipeline(
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

    bbox = np.array([2.0, 2.0, 6.0, 6.0], dtype=np.float32)

    def detections_fn(frames_in: torch.Tensor, *, target_hw: tuple[int, int]) -> Detections:
        bs = frames_in.shape[0]
        return Detections(
            boxes_xyxy=[np.array([bbox], dtype=np.float32) for _ in range(bs)],
            masks=[torch.ones((1, 8, 8), dtype=torch.bool) for _ in range(bs)],
        )

    raw_frame_context: dict[int, dict[int, torch.Tensor]] = {}
    ready_all: list[tuple[int, torch.Tensor, int]] = []
    frame_idx = 0

    for pts in range(num_frames):
        frame_batch = torch.full((1, 3, 8, 8), original_value, dtype=torch.uint8)
        res = process_frame_batch(
            frames=frame_batch,
            pts_list=[pts],
            start_frame_idx=frame_idx,
            batch_size=1,
            target_hw=(8, 8),
            detections_fn=detections_fn,
            tracker=tracker,
            frame_buffer=fb,
            restoration_pipeline=pipeline,
            discard_margin=temporal_overlap,
            blend_frames=blend_frames,
            raw_frame_context=raw_frame_context,
        )
        ready_all.extend(res.ready_frames)
        frame_idx = res.next_frame_idx

    remaining = finalize_processing(
        tracker=tracker,
        frame_buffer=fb,
        restoration_pipeline=pipeline,
        discard_margin=temporal_overlap,
        blend_frames=blend_frames,
        raw_frame_context=raw_frame_context,
    )
    return ready_all + remaining


def test_merged_crossfade_weights_sum_to_one_across_clip_boundaries() -> None:
    """Adjacent clips' crossfade weights must sum to 1.0 at each overlapping frame."""
    import pytest

    batch_size = 1
    discard_margin = 2
    blend_frames = 1
    max_clip_size = 10
    tracker = ClipTracker(max_clip_size=max_clip_size, temporal_overlap=discard_margin, iou_threshold=0.0)
    fb = FrameBuffer(device=torch.device("cpu"))

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
            frames=frames,
            pts_list=[pts],
            start_frame_idx=frame_idx,
            batch_size=batch_size,
            target_hw=(8, 8),
            detections_fn=detections_fn,
            tracker=tracker,
            frame_buffer=fb,
            restoration_pipeline=rest,  # type: ignore[arg-type]
            discard_margin=discard_margin,
            blend_frames=blend_frames,
            raw_frame_context=raw_frame_context,
        )
        frame_idx = res.next_frame_idx

    finalize_processing(
        tracker=tracker,
        frame_buffer=fb,
        restoration_pipeline=rest,
        discard_margin=discard_margin,
        blend_frames=blend_frames,
        raw_frame_context=raw_frame_context,
    )  # type: ignore[arg-type]

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

    all_output = _run_real_pipeline(
        monkeypatch,
        max_clip_size=10,
        temporal_overlap=2,
        blend_frames=1,
        num_frames=25,
        original_value=200,
        restored_float=restored_float,
    )

    assert len(all_output) == 25
    for idx, blended, _ in all_output:
        region = blended[:, 2:6, 2:6].float()
        assert (region - restored_u8).abs().max().item() <= 2, (
            f"frame {idx}: mosaic pixel deviates from expected {restored_u8} "
            f"(actual range [{region.min().item()}, {region.max().item()}])"
        )


def test_crossfade_at_exact_boundary_params(monkeypatch) -> None:
    """When 2*(d+bf) == max_clip_size exactly, crossfade regions are adjacent with no gap.
    This must still produce correct pixel values without artifacts.
    Uses d=3, bf=1 (2*(3+1)=8=max_clip_size) with bf<d to avoid the split-frame
    pending_clips edge case (the last frame of a split clip is only pending for the
    continuation, not the parent)."""
    restored_float = 0.5
    restored_u8 = int(round(restored_float * 255))

    all_output = _run_real_pipeline(
        monkeypatch,
        max_clip_size=8,
        temporal_overlap=3,
        blend_frames=1,
        num_frames=20,
        original_value=200,
        restored_float=restored_float,
    )

    assert len(all_output) == 20
    for idx, blended, _ in all_output:
        region = blended[:, 2:6, 2:6].float()
        assert (region - restored_u8).abs().max().item() <= 2, (
            f"frame {idx}: mosaic pixel deviates from expected {restored_u8} "
            f"(actual range [{region.min().item()}, {region.max().item()}])"
        )


def test_crossfade_weights_applied_in_blending(monkeypatch) -> None:
    """Verify that crossfade_weights are actually applied during blending, not ignored.
    Uses an alternating restorer (0.3 / 0.7 per clip) so adjacent clips produce different
    restoration values. With crossfade, seam frames blend both clips' values, producing
    pixels that differ from the no-crossfade run (where each frame is restored by one clip)."""
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

    # Run WITH crossfade (blend_frames=1)
    pipeline_cf = RestorationPipeline(restorer=_AlternatingRestorer())  # type: ignore[arg-type]
    fb_cf = FrameBuffer(device=torch.device("cpu"), blend_mask_fn=_ones_blend_mask)
    tracker_cf = ClipTracker(max_clip_size=max_clip_size, temporal_overlap=discard_margin, iou_threshold=0.0)
    raw_ctx_cf: dict[int, dict[int, torch.Tensor]] = {}
    ready_cf: list[tuple[int, torch.Tensor, int]] = []
    fi = 0
    for pts in range(15):
        res = process_frame_batch(
            frames=torch.full((1, 3, 8, 8), original_value, dtype=torch.uint8),
            pts_list=[pts], start_frame_idx=fi, batch_size=1, target_hw=(8, 8),
            detections_fn=detections_fn, tracker=tracker_cf, frame_buffer=fb_cf,
            restoration_pipeline=pipeline_cf, discard_margin=discard_margin,
            blend_frames=1, raw_frame_context=raw_ctx_cf,
        )
        ready_cf.extend(res.ready_frames)
        fi = res.next_frame_idx
    ready_cf.extend(finalize_processing(
        tracker=tracker_cf, frame_buffer=fb_cf, restoration_pipeline=pipeline_cf,
        discard_margin=discard_margin, blend_frames=1, raw_frame_context=raw_ctx_cf,
    ))

    # Run WITHOUT crossfade (blend_frames=0)
    pipeline_no = RestorationPipeline(restorer=_AlternatingRestorer())  # type: ignore[arg-type]
    fb_no = FrameBuffer(device=torch.device("cpu"), blend_mask_fn=_ones_blend_mask)
    tracker_no = ClipTracker(max_clip_size=max_clip_size, temporal_overlap=discard_margin, iou_threshold=0.0)
    raw_ctx_no: dict[int, dict[int, torch.Tensor]] = {}
    ready_no: list[tuple[int, torch.Tensor, int]] = []
    fi = 0
    for pts in range(15):
        res = process_frame_batch(
            frames=torch.full((1, 3, 8, 8), original_value, dtype=torch.uint8),
            pts_list=[pts], start_frame_idx=fi, batch_size=1, target_hw=(8, 8),
            detections_fn=detections_fn, tracker=tracker_no, frame_buffer=fb_no,
            restoration_pipeline=pipeline_no, discard_margin=discard_margin,
            blend_frames=0, raw_frame_context=raw_ctx_no,
        )
        ready_no.extend(res.ready_frames)
        fi = res.next_frame_idx
    ready_no.extend(finalize_processing(
        tracker=tracker_no, frame_buffer=fb_no, restoration_pipeline=pipeline_no,
        discard_margin=discard_margin, blend_frames=0, raw_frame_context=raw_ctx_no,
    ))

    cf_by_idx = {idx: blended for idx, blended, _ in ready_cf}
    no_by_idx = {idx: blended for idx, blended, _ in ready_no}

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

    all_output = _run_real_pipeline(
        monkeypatch,
        max_clip_size=10,
        temporal_overlap=2,
        blend_frames=1,
        num_frames=50,
        original_value=180,
        restored_float=restored_float,
    )

    assert len(all_output) == 50
    out_indices = [idx for idx, _, _ in all_output]
    assert out_indices == list(range(50))

    for idx, blended, _ in all_output:
        region = blended[:, 2:6, 2:6].float()
        assert region.min().item() > 0, (
            f"frame {idx}: black pixel in mosaic region"
        )
        assert (region - restored_u8).abs().max().item() <= 2, (
            f"frame {idx}: mosaic pixel deviates from expected {restored_u8} "
            f"(actual range [{region.min().item()}, {region.max().item()}])"
        )


def test_bf_clamping_tight_params_no_artifacts(monkeypatch) -> None:
    """With params that trigger bf clamping (2*(d+bf) > max_clip_size), the runtime
    clamp should produce no black pixels and all frames should be output."""
    all_output = _run_real_pipeline(
        monkeypatch,
        max_clip_size=8,
        temporal_overlap=3,
        blend_frames=3,
        num_frames=20,
        original_value=200,
        restored_float=0.3,
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
            frames=frames,
            pts_list=[pts],
            start_frame_idx=frame_idx,
            batch_size=batch_size,
            target_hw=(8, 8),
            detections_fn=detections_fn,
            tracker=tracker,
            frame_buffer=fb,
            restoration_pipeline=rest,  # type: ignore[arg-type]
            discard_margin=discard_margin,
            blend_frames=blend_frames,
            raw_frame_context=raw_frame_context,
        )
        frame_idx = res.next_frame_idx

    finalize_processing(
        tracker=tracker,
        frame_buffer=fb,
        restoration_pipeline=rest,
        discard_margin=discard_margin,
        blend_frames=blend_frames,
        raw_frame_context=raw_frame_context,
    )  # type: ignore[arg-type]

    # First clip (split, not continuation): should have parent crossfade weights
    first_weights = captured_weights[0]
    assert first_weights is not None, "split clip should have parent crossfade weights"
    # Parent weights are descending (near 1 → near 0)
    vals = [first_weights[k] for k in sorted(first_weights.keys())]
    for i in range(1, len(vals)):
        assert vals[i] < vals[i - 1], "parent crossfade weights should be descending"

    # Second clip (continuation): should have child crossfade weights
    second_weights = captured_weights[1]
    assert second_weights is not None, "continuation clip should have child crossfade weights"

