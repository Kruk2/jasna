import numpy as np
import pytest
import torch

from jasna.tracking.clip_tracker import ClipTracker


def _det(
    *,
    box: tuple[float, float, float, float] = (0.0, 0.0, 10.0, 10.0),
    mask_hw: tuple[int, int] = (4, 4),
) -> tuple[np.ndarray, torch.Tensor]:
    bboxes = np.array([box], dtype=np.float32)
    masks = torch.zeros((1, mask_hw[0], mask_hw[1]), dtype=torch.bool)
    masks[0, 0, 0] = True
    return bboxes, masks


def _no_det(*, mask_hw: tuple[int, int] = (4, 4)) -> tuple[np.ndarray, torch.Tensor]:
    bboxes = np.zeros((0, 4), dtype=np.float32)
    masks = torch.zeros((0, mask_hw[0], mask_hw[1]), dtype=torch.bool)
    return bboxes, masks


# single track: accumulate frames, flush returns clip
def test_single_track_accumulates_frames_and_flush() -> None:
    tracker = ClipTracker(max_clip_size=10, temporal_overlap=0, iou_threshold=0.0)

    for frame_idx in range(4):
        bboxes, masks = _det()
        ended, active = tracker.update(frame_idx, bboxes, masks)
        assert ended == []
        assert active == {0}

    assert set(tracker.active_clips.keys()) == {0}
    assert tracker.active_clips[0].start_frame == 0
    assert tracker.active_clips[0].end_frame == 3
    assert tracker.active_clips[0].frame_count == 4

    ended = tracker.flush()
    assert len(ended) == 1
    assert ended[0].split_due_to_max_size is False
    assert ended[0].clip.track_id == 0
    assert ended[0].clip.frame_count == 4


# end track when there are no detections
def test_clip_ends_when_no_detections() -> None:
    tracker = ClipTracker(max_clip_size=10, temporal_overlap=0, iou_threshold=0.0)

    for frame_idx in range(3):
        bboxes, masks = _det()
        ended, active = tracker.update(frame_idx, bboxes, masks)
        assert ended == []
        assert active == {0}

    bboxes, masks = _no_det()
    ended, active = tracker.update(3, bboxes, masks)
    assert active == set()
    assert len(ended) == 1
    assert ended[0].split_due_to_max_size is False
    assert ended[0].clip.start_frame == 0
    assert ended[0].clip.end_frame == 2
    assert ended[0].clip.frame_count == 3


# split by max size: first clip ends, next frame starts a new track
def test_split_due_to_max_clip_size_starts_new_track_next_frame() -> None:
    tracker = ClipTracker(max_clip_size=3, temporal_overlap=0, iou_threshold=0.0)

    for frame_idx in range(2):
        bboxes, masks = _det()
        ended, _ = tracker.update(frame_idx, bboxes, masks)
        assert ended == []

    bboxes, masks = _det()
    ended, _ = tracker.update(2, bboxes, masks)
    assert len(ended) == 1
    assert ended[0].split_due_to_max_size is True
    assert ended[0].clip.track_id == 0
    assert ended[0].clip.frame_count == 3
    assert ended[0].clip.end_frame == 2

    bboxes, masks = _det()
    ended, active = tracker.update(3, bboxes, masks)
    assert ended == []
    assert active == {1}
    assert set(tracker.active_clips.keys()) == {1}


# temporal overlap: continuation clips are shorter so (overlap + normal) == max_clip_size
def test_temporal_overlap_split_creates_overlapping_continuation_clip() -> None:
    tracker = ClipTracker(max_clip_size=10, temporal_overlap=2, iou_threshold=0.0)

    for frame_idx in range(9):
        bboxes, masks = _det()
        ended, _ = tracker.update(frame_idx, bboxes, masks)
        assert ended == []

    bboxes, masks = _det()
    ended, active = tracker.update(9, bboxes, masks)
    assert len(ended) == 1
    assert ended[0].split_due_to_max_size is True
    assert ended[0].clip.track_id == 0
    assert ended[0].clip.frame_count == 10
    assert ended[0].continuation_track_id is not None

    child_id = int(ended[0].continuation_track_id)
    assert active == {child_id}
    assert child_id in tracker.active_clips
    child = tracker.active_clips[child_id]
    assert child.is_continuation is True
    assert child.start_frame == 6  # last 2*overlap=4 frames: 6,7,8,9
    assert child.end_frame == 9
    assert child.frame_count == 4

    bboxes, masks = _det()
    ended, active = tracker.update(10, bboxes, masks)
    assert ended == []
    assert active == {child_id}
    assert tracker.active_clips[child_id].frame_count == 5


# overlapping detections within a frame are merged into one track
def test_merge_overlapping_boxes_results_in_single_track() -> None:
    tracker = ClipTracker(max_clip_size=10, temporal_overlap=0, iou_threshold=0.3)

    bboxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    masks = torch.zeros((2, 4, 4), dtype=torch.bool)
    masks[0, 0, 0] = True
    masks[1, 1, 1] = True

    ended, active = tracker.update(0, bboxes, masks)
    assert ended == []
    assert len(active) == 1
    assert len(tracker.active_clips) == 1


# matching loop breaks when IoU below threshold: old track ends, new track starts
def test_low_iou_breaks_matching_loop_and_ends_previous_track() -> None:
    tracker = ClipTracker(max_clip_size=10, temporal_overlap=0, iou_threshold=0.9)

    bboxes, masks = _det(box=(0.0, 0.0, 10.0, 10.0))
    ended, active = tracker.update(0, bboxes, masks)
    assert ended == []
    assert active == {0}

    bboxes, masks = _det(box=(100.0, 100.0, 110.0, 110.0))
    ended, active = tracker.update(1, bboxes, masks)
    assert len(ended) == 1
    assert ended[0].split_due_to_max_size is False
    assert ended[0].clip.track_id == 0
    assert ended[0].clip.frame_count == 1
    assert active == {1}


# one track matches, the other doesn't: unmatched active track is ended
def test_unmatched_track_is_ended_when_other_track_matches() -> None:
    tracker = ClipTracker(max_clip_size=10, temporal_overlap=0, iou_threshold=0.3)

    bboxes = np.array([[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 110.0, 110.0]], dtype=np.float32)
    masks = torch.zeros((2, 4, 4), dtype=torch.bool)
    masks[0, 0, 0] = True
    masks[1, 1, 1] = True

    ended, active = tracker.update(0, bboxes, masks)
    assert ended == []
    assert active == {0, 1}

    bboxes, masks = _det(box=(0.0, 0.0, 10.0, 10.0))
    ended, active = tracker.update(1, bboxes, masks)
    assert len(ended) == 1
    assert ended[0].split_due_to_max_size is False
    assert ended[0].clip.track_id == 1
    assert active == {0}


def test_one_mosaic_splits_into_two_detections_continues_best_and_starts_new_track() -> None:
    tracker = ClipTracker(max_clip_size=10, temporal_overlap=0, iou_threshold=0.3)

    bboxes, masks = _det(box=(0.0, 0.0, 10.0, 10.0))
    ended, active = tracker.update(0, bboxes, masks)
    assert ended == []
    assert active == {0}

    bboxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],  # best continuation for track 0
            [20.0, 0.0, 30.0, 10.0],  # new mosaic
        ],
        dtype=np.float32,
    )
    masks = torch.zeros((2, 4, 4), dtype=torch.bool)
    masks[0, 0, 0] = True
    masks[1, 1, 1] = True

    ended, active = tracker.update(1, bboxes, masks)
    assert ended == []
    assert active == {0, 1}
    assert tracker.active_clips[0].start_frame == 0
    assert tracker.active_clips[0].end_frame == 1
    assert tracker.active_clips[1].start_frame == 1


def test_two_mosaics_merge_into_one_detection_continues_best_and_ends_other_track() -> None:
    tracker = ClipTracker(max_clip_size=10, temporal_overlap=0, iou_threshold=0.3)

    bboxes = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 0.0, 30.0, 10.0]], dtype=np.float32)
    masks = torch.zeros((2, 4, 4), dtype=torch.bool)
    masks[0, 0, 0] = True
    masks[1, 1, 1] = True

    ended, active = tracker.update(0, bboxes, masks)
    assert ended == []
    assert active == {0, 1}

    # Single detection overlaps only track 1 (acts like "merged region" / blob).
    bboxes, masks = _det(box=(18.0, 0.0, 30.0, 10.0))
    ended, active = tracker.update(1, bboxes, masks)

    assert active == {1}
    assert len(ended) == 1
    assert ended[0].split_due_to_max_size is False
    assert ended[0].clip.track_id == 0
    assert ended[0].clip.frame_count == 1


def test_two_mosaics_blend_and_detector_outputs_two_overlapping_boxes_continues_one_track() -> None:
    tracker = ClipTracker(max_clip_size=10, temporal_overlap=0, iou_threshold=0.3)

    bboxes = np.array([[0.0, 0.0, 10.0, 10.0], [40.0, 0.0, 50.0, 10.0]], dtype=np.float32)
    masks = torch.zeros((2, 4, 4), dtype=torch.bool)
    masks[0, 0, 0] = True
    masks[1, 1, 1] = True

    ended, active = tracker.update(0, bboxes, masks)
    assert ended == []
    assert active == {0, 1}

    # Two detections overlap each other -> merged into one detection before matching.
    bboxes = np.array([[0.0, 0.0, 12.0, 10.0], [2.0, 0.0, 14.0, 10.0]], dtype=np.float32)
    masks = torch.zeros((2, 4, 4), dtype=torch.bool)
    masks[0, 0, 0] = True
    masks[1, 1, 1] = True

    ended, active = tracker.update(1, bboxes, masks)
    assert active == {0}
    assert len(ended) == 1
    assert ended[0].split_due_to_max_size is False
    assert ended[0].clip.track_id == 1
    assert ended[0].clip.frame_count == 1


# invalid overlap settings raise
@pytest.mark.parametrize(
    ("max_clip_size", "temporal_overlap"),
    [
        (5, 5),
        (5, 6),
        (5, 3),  # 2*overlap >= max_clip_size
        (1, 1),
    ],
)
def test_invalid_temporal_overlap_raises(max_clip_size: int, temporal_overlap: int) -> None:
    with pytest.raises(ValueError):
        ClipTracker(max_clip_size=max_clip_size, temporal_overlap=temporal_overlap)


def test_split_with_zero_overlap_keeps_track_id_in_active_for_split_frame() -> None:
    tracker = ClipTracker(max_clip_size=3, temporal_overlap=0, iou_threshold=0.0)

    for frame_idx in range(2):
        bboxes, masks = _det()
        ended, active = tracker.update(frame_idx, bboxes, masks)
        assert ended == []

    # Frame 2 is the split frame: clip reaches max_clip_size=3
    bboxes, masks = _det()
    ended, active = tracker.update(2, bboxes, masks)
    assert len(ended) == 1
    assert ended[0].split_due_to_max_size is True
    # The split frame's track_id must still be in active_track_ids so
    # frame_buffer.add_frame records it as pending for blending.
    assert 0 in active


# negative overlap raises
def test_negative_temporal_overlap_raises() -> None:
    with pytest.raises(ValueError):
        ClipTracker(max_clip_size=5, temporal_overlap=-1)

