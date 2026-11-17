from queue import Queue

import numpy as np
import pytest
import torch

from factories import restore_synchronously
from jasna.vr180 import StereoLayout
from jasna.blend_buffer import BlendBuffer
from jasna.crop_buffer import CropBuffer, prepare_crops_for_restoration
from jasna.frame_queue import FrameQueue
from jasna.mosaic.detections import Detections
from jasna.pipeline_items import ClipRestoreItem, FrameMeta, SecondaryRestoreResult, SENTINEL
from jasna.pipeline_processing import process_frame_batch, finalize_processing
from jasna.restorer.restoration_pipeline import RestorationPipeline
from jasna.tracking.clip_tracker import ClipTracker


def _restore_to_200(ci: ClipRestoreItem, blend_buffer: BlendBuffer) -> None:
    """Restore a clip to a flat 200 fill under a full-frame mask, skipping the models."""
    resized_crops, pad_offsets, resize_shapes = prepare_crops_for_restoration(ci.raw_crops, device=torch.device("cpu"), dtype=torch.float32)
    enlarged_bboxes = [c.enlarged_bbox for c in ci.raw_crops]
    crop_shapes = [c.crop_shape for c in ci.raw_crops]

    frame_h, frame_w = ci.frame_shape
    n = len(ci.raw_crops)
    restored_frames = [torch.full((3, rc.shape[1], rc.shape[2]), 200, dtype=torch.uint8) for rc in resized_crops]
    masks = [torch.ones((frame_h, frame_w), dtype=torch.bool) for _ in range(n)]

    ks = max(0, ci.keep_start)
    ke = min(n, ci.keep_end)

    blend_buffer.add_result(SecondaryRestoreResult(
        track_id=ci.clip.track_id,
        start_frame=ci.clip.start_frame,
        frame_count=n,
        frame_shape=ci.frame_shape,
        frame_device=torch.device("cpu"),
        masks=masks[ks:ke],
        restored_frames=restored_frames[ks:ke],
        keep_start=0,
        keep_end=ke - ks,
        crossfade_weights=ci.crossfade_weights,
        enlarged_bboxes=enlarged_bboxes[ks:ke],
        crop_shapes=crop_shapes[ks:ke],
        pad_offsets=pad_offsets[ks:ke],
        resize_shapes=resize_shapes[ks:ke],
        clip_keep_offset=ks,
    ))


def _recording(captured: list[ClipRestoreItem], restore=_restore_to_200):
    def process(ci: ClipRestoreItem, blend_buffer: BlendBuffer) -> None:
        captured.append(ci)
        restore(ci, blend_buffer)
    return process


class _ConstantRestorer:
    """Primary restorer whose output is ``values[n % len(values)]`` for its n-th clip."""

    input_dtype = torch.float32
    device = torch.device("cpu")

    def __init__(self, *values: float) -> None:
        self.values = values
        self.calls = 0

    def raw_process(self, crops: list[torch.Tensor]) -> torch.Tensor:
        value = self.values[self.calls % len(self.values)]
        self.calls += 1
        return torch.stack([torch.full(crop.shape, value, dtype=torch.float32) for crop in crops])


def _restore_with(restorer):
    pipeline = RestorationPipeline(restorer=restorer)

    def process(ci: ClipRestoreItem, blend_buffer: BlendBuffer) -> None:
        blend_buffer.add_result(restore_synchronously(
            pipeline, ci.clip, ci.raw_crops, ci.frame_shape, ci.keep_start, ci.keep_end, ci.crossfade_weights
        ))

    return process


@pytest.fixture
def exact_blend_buffer(exact_crops, whole_box_blend_mask):
    """Crop exactly the detected box and blend all of it, so restored pixels are predictable."""
    return BlendBuffer(vr_projector=None)


def _drain_queue(clip_queue: FrameQueue, blend_buffer: BlendBuffer, process_fn) -> None:
    while clip_queue.qsize():
        item = clip_queue.get(timeout=0)
        if item is SENTINEL:
            break
        process_fn(item, blend_buffer)


def _make_empty_det_batch(*, batch_size: int) -> Detections:
    return Detections(
        boxes_xyxy=[np.zeros((0, 4), dtype=np.float32) for _ in range(batch_size)],
        masks=[torch.zeros((0, 8, 8), dtype=torch.bool) for _ in range(batch_size)],
    )


def _make_single_det_batch(*, batch_size: int, box=(2.0, 2.0, 6.0, 6.0)) -> Detections:
    boxes_xyxy = []
    masks = []
    for _ in range(batch_size):
        boxes_xyxy.append(np.array([box], dtype=np.float32))
        m = torch.zeros((1, 8, 8), dtype=torch.bool)
        m[0, 0, 0] = True
        masks.append(m)
    return Detections(boxes_xyxy=boxes_xyxy[:batch_size], masks=masks[:batch_size])


def _one_box_per_frame(frames: torch.Tensor, *, target_hw: tuple[int, int]) -> Detections:
    return _make_single_det_batch(batch_size=len(frames))


def _run_batches(
    *,
    pts_lists: list[list[int]],
    max_clip_size: int,
    discard_margin: int,
    blend_frames: int = 0,
    process_fn=_restore_to_200,
    detections_fn=_one_box_per_frame,
    blend_buffer: BlendBuffer | None = None,
    original_value: int = 0,
) -> list[tuple[int, torch.Tensor, int]]:
    """Detect, track, restore and blend ``pts_lists`` batches of 8x8 frames; returns
    ``(frame_idx, blended, pts)`` for every frame in output order."""
    batch_size = max(len(pts_list) for pts_list in pts_lists)
    tracker = ClipTracker(max_clip_size=max_clip_size, temporal_overlap=discard_margin, iou_threshold=0.0)
    blend_buffer = blend_buffer or BlendBuffer(vr_projector=None)
    crop_buffers: dict[int, CropBuffer] = {}
    clip_queue = FrameQueue(max_frames=9999)
    metadata_queue: Queue[FrameMeta | object] = Queue()
    frames = torch.full((batch_size, 3, 8, 8), original_value, dtype=torch.uint8)
    frame_idx = 0

    for pts_list in pts_lists:
        frame_idx = process_frame_batch(
            frames=frames[:len(pts_list)], pts_list=list(pts_list), start_frame_idx=frame_idx,
            target_hw=(8, 8), detections_fn=detections_fn,
            tracker=tracker, blend_buffer=blend_buffer, crop_buffers=crop_buffers,
            clip_queue=clip_queue, metadata_queue=metadata_queue,
            discard_margin=discard_margin, blend_frames=blend_frames,
            stereo=None, min_detection_duration=0, scene_detector=None, vr_projector=None,
        )
        _drain_queue(clip_queue, blend_buffer, process_fn)

    finalize_processing(
        tracker=tracker, blend_buffer=blend_buffer, crop_buffers=crop_buffers,
        clip_queue=clip_queue, frame_shape=(8, 8),
        discard_margin=discard_margin, blend_frames=blend_frames, min_detection_duration=0,
    )
    _drain_queue(clip_queue, blend_buffer, process_fn)

    ready = []
    while not metadata_queue.empty() and (meta := metadata_queue.get()) is not SENTINEL:
        ready.append((meta.frame_idx, blend_buffer.blend_frame(meta.frame_idx, frames[0].clone()), meta.pts))
    return ready


@pytest.mark.parametrize(
    ("pts_lists", "max_clip_size", "discard_margin", "blend_frames"),
    [
        ([[0, 1], [2, 3], [4, 5]], 6, 1, 0),
        ([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], 6, 2, 1),
        ([[0, 1], [2, 3]], 4, 0, 0),
    ],
    ids=["overlap-discard", "crossfade", "no-discard"],
)
def test_every_frame_is_output_once_in_order(pts_lists, max_clip_size, discard_margin, blend_frames) -> None:
    out = _run_batches(
        pts_lists=pts_lists, max_clip_size=max_clip_size, discard_margin=discard_margin, blend_frames=blend_frames,
    )
    assert [idx for idx, _, _ in out] == [pts for pts_list in pts_lists for pts in pts_list]


def test_zero_overlap_split_blends_all_frames_including_boundary() -> None:
    """With temporal_overlap=0 and a clip split, the split-boundary frame must be blended, not raw."""
    out = _run_batches(pts_lists=[[pts] for pts in range(5)], max_clip_size=3, discard_margin=0)

    assert [idx for idx, _, _ in out] == list(range(5))
    for idx, blended, _ in out:
        assert torch.all(blended[:, 2:6, 2:6] == 200), f"frame {idx} mosaic region was not blended (has raw pixels)"


def _full_mask_box(frames: torch.Tensor, *, target_hw: tuple[int, int]) -> Detections:
    return Detections(
        boxes_xyxy=[np.array([[2.0, 2.0, 6.0, 6.0]], dtype=np.float32) for _ in range(len(frames))],
        masks=[torch.ones((1, 8, 8), dtype=torch.bool) for _ in range(len(frames))],
    )


@pytest.mark.parametrize(
    ("max_clip_size", "temporal_overlap", "blend_frames", "num_frames", "original_value", "restored", "exact"),
    [
        # 2*(overlap + blend) > max_clip_size: child and parent crossfade regions overlap in a
        # middle clip; weights summing past 1 would blend negative, clamped to black.
        (5, 2, 2, 15, 200, 0.2, False),
        (10, 2, 1, 25, 200, 0.5, True),
        # 2*(overlap + blend) == max_clip_size: crossfade regions touch with no gap.
        (8, 3, 1, 20, 200, 0.5, True),
        (10, 2, 1, 50, 180, 0.6, True),
        # Tight parameters where blend_frames is clamped at runtime.
        (8, 3, 3, 20, 200, 0.3, False),
    ],
    ids=["overlapping-crossfade", "valid-crossfade", "exact-boundary", "long-split-chain", "clamped-blend"],
)
def test_restored_mosaic_survives_clip_splits_and_crossfades(
    exact_blend_buffer, max_clip_size, temporal_overlap, blend_frames, num_frames, original_value, restored, exact
) -> None:
    out = _run_batches(
        pts_lists=[[pts] for pts in range(num_frames)],
        max_clip_size=max_clip_size, discard_margin=temporal_overlap, blend_frames=blend_frames,
        process_fn=_restore_with(_ConstantRestorer(restored)), detections_fn=_full_mask_box,
        blend_buffer=exact_blend_buffer, original_value=original_value,
    )

    assert [idx for idx, _, _ in out] == list(range(num_frames))
    for idx, blended, _ in out:
        region = blended[:, 2:6, 2:6].float()
        assert region.min().item() > 0, f"frame {idx}: black pixel in mosaic region"
        if exact:
            assert (region - round(restored * 255)).abs().max().item() <= 2, f"frame {idx}: mosaic pixel off"


def test_merged_crossfade_weights_sum_to_one_across_clip_boundaries() -> None:
    captured: list[ClipRestoreItem] = []
    _run_batches(
        pts_lists=[[pts] for pts in range(25)], max_clip_size=10, discard_margin=2, blend_frames=1,
        process_fn=_recording(captured),
    )

    assert len(captured) >= 3
    sums = [
        w_a + w_b
        for a, b in zip(captured, captured[1:])
        if a.crossfade_weights is not None and b.crossfade_weights is not None
        for local_b, w_b in b.crossfade_weights.items()
        if (w_a := a.crossfade_weights.get(b.clip.start_frame + local_b - a.clip.start_frame)) is not None
    ]
    assert sums and sums == pytest.approx([1.0] * len(sums))


@pytest.mark.parametrize(
    ("blend_frames", "mosaic_values"),
    [
        (0, [76] * 8 + [178] * 7),
        # Weights 0.75/0.25 then 0.25/0.75 of the parent (0.3 -> 76) and child (0.7 -> 178) clip.
        (1, [76] * 7 + [102, 152] + [178] * 6),
    ],
    ids=["hard-cut", "crossfade"],
)
def test_crossfade_weights_mix_the_split_clips_in_blending(exact_blend_buffer, blend_frames, mosaic_values) -> None:
    out = _run_batches(
        pts_lists=[[pts] for pts in range(15)], max_clip_size=10, discard_margin=2, blend_frames=blend_frames,
        process_fn=_restore_with(_ConstantRestorer(0.3, 0.7)), detections_fn=_full_mask_box,
        blend_buffer=exact_blend_buffer, original_value=200,
    )

    regions = [blended[:, 2:6, 2:6] for _, blended, _ in out]
    assert all(region.unique().numel() == 1 for region in regions)
    assert [int(region[0, 0, 0]) for region in regions] == mosaic_values


def test_crossfade_with_split_assigns_parent_weights() -> None:
    captured: list[ClipRestoreItem] = []
    _run_batches(
        pts_lists=[[pts] for pts in range(10)], max_clip_size=6, discard_margin=2, blend_frames=1,
        process_fn=_recording(captured),
    )

    parent = captured[0].crossfade_weights
    assert parent is not None, "split clip should have parent crossfade weights"
    weights = [parent[k] for k in sorted(parent)]
    assert weights == sorted(weights, reverse=True) and len(set(weights)) == len(weights)
    assert captured[1].crossfade_weights is not None, "continuation clip should have child crossfade weights"


def test_clip_split_child_crop_count_matches_frame_count() -> None:
    """When a clip splits, the child CropBuffer must have exactly frame_count crops."""
    captured: list[ClipRestoreItem] = []
    _run_batches(pts_lists=[[pts] for pts in range(6)], max_clip_size=3, discard_margin=1, process_fn=_recording(captured))

    assert len(captured) >= 2
    assert [len(ci.raw_crops) for ci in captured] == [ci.clip.frame_count for ci in captured]


def _scripted_detections_fn(script: list[bool]):
    frames_iter = iter(script)

    def fn(_: torch.Tensor, *, target_hw: tuple[int, int]) -> Detections:
        if next(frames_iter):
            boxes = np.array([[2.0, 2.0, 6.0, 6.0]], dtype=np.float32)
            mask = torch.zeros((1, 8, 8), dtype=torch.bool)
            mask[0, 0, 0] = True
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            mask = torch.zeros((0, 8, 8), dtype=torch.bool)
        return Detections(boxes_xyxy=[boxes], masks=[mask])

    return fn


def _run_scripted(
    script: list[bool],
    *,
    max_clip_size: int = 10,
    temporal_overlap: int = 0,
    max_detection_gap: int = 0,
    min_detection_duration: int = 0,
) -> tuple[list[ClipRestoreItem], BlendBuffer, dict[int, CropBuffer]]:
    tracker = ClipTracker(
        max_clip_size=max_clip_size,
        temporal_overlap=temporal_overlap,
        iou_threshold=0.0,
        max_detection_gap=max_detection_gap,
    )
    blend_buffer = BlendBuffer(vr_projector=None)
    crop_buffers: dict[int, CropBuffer] = {}
    clip_queue = FrameQueue(max_frames=9999)
    metadata_queue: Queue[FrameMeta | object] = Queue()
    frames = torch.zeros((1, 3, 8, 8), dtype=torch.uint8)
    detections_fn = _scripted_detections_fn(script)

    items: list[ClipRestoreItem] = []

    def collect(ci: ClipRestoreItem, _: BlendBuffer) -> None:
        items.append(ci)

    frame_idx = 0
    for pts in range(len(script)):
        frame_idx = process_frame_batch(
            frames=frames, pts_list=[pts], start_frame_idx=frame_idx,
            target_hw=(8, 8), detections_fn=detections_fn,
            tracker=tracker, blend_buffer=blend_buffer, crop_buffers=crop_buffers,
            clip_queue=clip_queue, metadata_queue=metadata_queue,
            discard_margin=temporal_overlap, blend_frames=0, stereo=None,
            min_detection_duration=min_detection_duration, scene_detector=None, vr_projector=None,
        )
        _drain_queue(clip_queue, blend_buffer, collect)

    finalize_processing(
        tracker=tracker, blend_buffer=blend_buffer, crop_buffers=crop_buffers,
        clip_queue=clip_queue, frame_shape=(8, 8),
        discard_margin=temporal_overlap, blend_frames=0,
        min_detection_duration=min_detection_duration,
    )
    _drain_queue(clip_queue, blend_buffer, collect)
    return items, blend_buffer, crop_buffers


def test_short_clip_dropped_by_min_detection_duration() -> None:
    items, blend_buffer, crop_buffers = _run_scripted(
        [True, False, False], min_detection_duration=2,
    )
    assert items == []
    assert crop_buffers == {}
    assert blend_buffer.is_frame_ready(0)


def test_gap_bridged_produces_single_clip_with_crops_for_gap_frames() -> None:
    items, blend_buffer, _ = _run_scripted(
        [True, True, False, True, True], max_detection_gap=2,
    )
    assert len(items) == 1
    ci = items[0]
    assert ci.clip.frame_count == 5
    assert len(ci.raw_crops) == 5
    assert not blend_buffer.is_frame_ready(2)


def test_coast_death_trims_crops_and_clears_pending() -> None:
    items, blend_buffer, _ = _run_scripted(
        [True, True, True, False, False], max_detection_gap=1,
    )
    assert len(items) == 1
    ci = items[0]
    assert ci.clip.frame_count == 3
    assert len(ci.raw_crops) == 3
    assert blend_buffer.is_frame_ready(3)
    assert not blend_buffer.is_frame_ready(2)


def test_split_and_continuation_clips_not_dropped_by_min_duration() -> None:
    items, _, _ = _run_scripted(
        [True, True, True, True],
        max_clip_size=4, temporal_overlap=1, min_detection_duration=3,
    )
    assert len(items) == 2
    assert items[0].clip.frame_count == 4
    assert items[1].clip.is_continuation is True
    assert items[1].clip.frame_count == 2


class _StubSceneDetector:
    def __init__(self, cuts: set[int]):
        self._cuts = set(cuts)

    def find_cuts(self, frames_uint8_bchw: torch.Tensor) -> set[int]:
        return self._cuts


def test_scene_cut_flushes_tracker_and_splits_clip() -> None:
    tracker = ClipTracker(max_clip_size=60, temporal_overlap=0, iou_threshold=0.0)
    blend_buffer = BlendBuffer(vr_projector=None)
    crop_buffers: dict[int, CropBuffer] = {}
    clip_queue = FrameQueue(max_frames=9999)
    metadata_queue: Queue[FrameMeta | object] = Queue()
    frames = torch.zeros((8, 3, 8, 8), dtype=torch.uint8)

    def detections_fn(frames_eff, target_hw):
        return _make_single_det_batch(batch_size=len(frames_eff))

    process_frame_batch(
        frames=frames, pts_list=list(range(8)), start_frame_idx=0,
        target_hw=(8, 8), detections_fn=detections_fn,
        tracker=tracker, blend_buffer=blend_buffer, crop_buffers=crop_buffers,
        clip_queue=clip_queue, metadata_queue=metadata_queue,
        discard_margin=0, blend_frames=0, stereo=None, min_detection_duration=0,
        scene_detector=_StubSceneDetector({4}), vr_projector=None,
    )

    assert clip_queue.qsize() == 1
    first = clip_queue.get(timeout=0)
    assert first.clip.start_frame == 0
    assert first.clip.frame_count == 4
    assert len(first.raw_crops) == 4

    finalize_processing(
        tracker=tracker, blend_buffer=blend_buffer, crop_buffers=crop_buffers,
        clip_queue=clip_queue, frame_shape=(8, 8), discard_margin=0, blend_frames=0, min_detection_duration=0,
    )
    second = clip_queue.get(timeout=0)
    assert second.clip.start_frame == 4
    assert second.clip.frame_count == 4
    assert second.clip.track_id != first.clip.track_id
    assert second.clip.is_continuation is False


def test_scene_cut_ends_coasting_track_and_trims_synthetic_tail() -> None:
    tracker = ClipTracker(
        max_clip_size=60, temporal_overlap=0, iou_threshold=0.0, max_detection_gap=5,
    )
    blend_buffer = BlendBuffer(vr_projector=None)
    crop_buffers: dict[int, CropBuffer] = {}
    clip_queue = FrameQueue(max_frames=9999)
    metadata_queue: Queue[FrameMeta | object] = Queue()
    frames = torch.zeros((8, 3, 8, 8), dtype=torch.uint8)
    detected = [True, True, True, False, False, False, False, False]

    def detections_fn(frames_eff, target_hw):
        dets = _make_single_det_batch(batch_size=len(frames_eff))
        empty = _make_empty_det_batch(batch_size=1)
        boxes = [b if detected[i] else empty.boxes_xyxy[0] for i, b in enumerate(dets.boxes_xyxy)]
        masks = [m if detected[i] else empty.masks[0] for i, m in enumerate(dets.masks)]
        return Detections(boxes_xyxy=boxes, masks=masks)

    process_frame_batch(
        frames=frames, pts_list=list(range(8)), start_frame_idx=0,
        target_hw=(8, 8), detections_fn=detections_fn,
        tracker=tracker, blend_buffer=blend_buffer, crop_buffers=crop_buffers,
        clip_queue=clip_queue, metadata_queue=metadata_queue,
        discard_margin=0, blend_frames=0, stereo=None, min_detection_duration=0,
        scene_detector=_StubSceneDetector({5}), vr_projector=None,
    )

    assert clip_queue.qsize() == 1
    ended = clip_queue.get(timeout=0)
    assert ended.clip.frame_count == 3
    assert len(ended.raw_crops) == 3
    assert not tracker.active_clips


@pytest.mark.parametrize("layout", ["sbs", "tb"])
def test_vr_gnomonic_crop_and_blend_stays_within_one_eye(layout, whole_box_blend_mask) -> None:
    from jasna.vr_projection import GnomonicProjector

    eye_w, eye_h = 512, 64
    stereo = StereoLayout(layout, eye_w * 2, eye_h) if layout == "sbs" else StereoLayout(layout, eye_w, eye_h * 2)
    frame_w, height = stereo.frame_w, stereo.frame_h
    projector = GnomonicProjector(stereo, device=torch.device("cpu"))

    frame = torch.full((1, 3, height, frame_w), 150, dtype=torch.uint8)
    frame[:, :, :eye_h, :eye_w] = 50

    def _detections_fn(frames: torch.Tensor, *, target_hw) -> Detections:
        b = int(frames.shape[0])
        boxes = [np.array([[180.0, 20.0, 300.0, 44.0]], dtype=np.float32) for _ in range(b)]
        masks = [torch.ones((1, height, frame_w), dtype=torch.bool) for _ in range(b)]
        return Detections(boxes_xyxy=boxes, masks=masks)

    tracker = ClipTracker(max_clip_size=8, temporal_overlap=0, iou_threshold=0.0)
    blend_buffer = BlendBuffer(vr_projector=projector)
    crop_buffers: dict[int, CropBuffer] = {}
    clip_queue = FrameQueue(max_frames=9999)
    metadata_queue: Queue[FrameMeta | object] = Queue()

    process_frame_batch(
        frames=frame, pts_list=[0], start_frame_idx=0, target_hw=(height, frame_w),
        detections_fn=_detections_fn, tracker=tracker, blend_buffer=blend_buffer,
        crop_buffers=crop_buffers, clip_queue=clip_queue, metadata_queue=metadata_queue,
        discard_margin=0, blend_frames=0, stereo=stereo, min_detection_duration=0, scene_detector=None,
        vr_projector=projector,
    )
    finalize_processing(
        tracker=tracker, blend_buffer=blend_buffer, crop_buffers=crop_buffers,
        clip_queue=clip_queue, frame_shape=(height, frame_w), discard_margin=0, blend_frames=0,
        min_detection_duration=0,
    )
    _drain_queue(clip_queue, blend_buffer, _restore_to_200)

    blended = blend_buffer.blend_frame(0, frame[0])

    # The restored fill (200) landed in the left eye; the right eye is untouched.
    dx, dy = stereo.right_offset
    assert torch.equal(blended[:, dy:, dx:], frame[0, :, dy:, dx:])
    assert (blended[:, :eye_h, :eye_w] == 200).any()
