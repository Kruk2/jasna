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
def test_temporal_overlap_counts_inside_max_clip_size_for_continuation() -> None:
    tracker = ClipTracker(max_clip_size=5, temporal_overlap=2, iou_threshold=0.0)

    for frame_idx in range(4):
        bboxes, masks = _det()
        ended, _ = tracker.update(frame_idx, bboxes, masks)
        assert ended == []

    bboxes, masks = _det()
    ended, _ = tracker.update(4, bboxes, masks)
    assert len(ended) == 1
    assert ended[0].split_due_to_max_size is True
    assert ended[0].clip.track_id == 0
    assert ended[0].clip.frame_count == 5

    bboxes, masks = _det()
    ended, active = tracker.update(5, bboxes, masks)
    assert ended == []
    assert active == {1}
    assert tracker.get_continuation_source(1) == 0

    bboxes, masks = _det()
    ended, active = tracker.update(6, bboxes, masks)
    assert ended == []
    assert active == {1}

    bboxes, masks = _det()
    ended, _ = tracker.update(7, bboxes, masks)
    assert len(ended) == 1
    assert ended[0].split_due_to_max_size is True
    assert ended[0].clip.track_id == 1
    assert ended[0].clip.frame_count == 3  # 5 - temporal_overlap(2)


# split + low IoU next frame: no continuation mapping, new track uses full max_clip_size
def test_split_then_low_iou_does_not_create_continuation_and_new_track_uses_full_max() -> None:
    tracker = ClipTracker(max_clip_size=5, temporal_overlap=2, iou_threshold=0.9)

    for frame_idx in range(4):
        bboxes, masks = _det(box=(0.0, 0.0, 10.0, 10.0))
        ended, _ = tracker.update(frame_idx, bboxes, masks)
        assert ended == []

    bboxes, masks = _det(box=(0.0, 0.0, 10.0, 10.0))
    ended, _ = tracker.update(4, bboxes, masks)
    assert len(ended) == 1
    assert ended[0].split_due_to_max_size is True
    assert ended[0].clip.track_id == 0

    bboxes, masks = _det(box=(100.0, 100.0, 110.0, 110.0))
    ended, active = tracker.update(5, bboxes, masks)
    assert ended == []
    assert active == {1}
    assert tracker.get_continuation_source(1) is None

    for frame_idx in (6, 7, 8):
        bboxes, masks = _det(box=(100.0, 100.0, 110.0, 110.0))
        ended, _ = tracker.update(frame_idx, bboxes, masks)
        assert ended == []

    bboxes, masks = _det(box=(100.0, 100.0, 110.0, 110.0))
    ended, _ = tracker.update(9, bboxes, masks)
    assert len(ended) == 1
    assert ended[0].split_due_to_max_size is True
    assert ended[0].clip.track_id == 1
    assert ended[0].clip.frame_count == 5


# split then gap: pending split expires, late detection is not linked as continuation
def test_pending_split_expires_after_gap_and_does_not_link_late_continuation() -> None:
    tracker = ClipTracker(max_clip_size=5, temporal_overlap=2, iou_threshold=0.0)

    for frame_idx in range(4):
        bboxes, masks = _det(box=(0.0, 0.0, 10.0, 10.0))
        ended, _ = tracker.update(frame_idx, bboxes, masks)
        assert ended == []

    bboxes, masks = _det(box=(0.0, 0.0, 10.0, 10.0))
    ended, _ = tracker.update(4, bboxes, masks)
    assert len(ended) == 1
    assert ended[0].split_due_to_max_size is True
    assert ended[0].clip.track_id == 0

    bboxes, masks = _no_det()
    ended, active = tracker.update(5, bboxes, masks)
    assert active == set()
    assert ended == []

    bboxes, masks = _det(box=(0.0, 0.0, 10.0, 10.0))
    ended, active = tracker.update(6, bboxes, masks)
    assert ended == []
    assert active == {1}
    assert tracker.get_continuation_source(1) is None


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


# invalid overlap settings raise
@pytest.mark.parametrize(
    ("max_clip_size", "temporal_overlap"),
    [
        (5, 5),
        (5, 6),
        (1, 1),
    ],
)
def test_invalid_temporal_overlap_raises(max_clip_size: int, temporal_overlap: int) -> None:
    with pytest.raises(ValueError):
        ClipTracker(max_clip_size=max_clip_size, temporal_overlap=temporal_overlap)


# negative overlap raises
def test_negative_temporal_overlap_raises() -> None:
    with pytest.raises(ValueError):
        ClipTracker(max_clip_size=5, temporal_overlap=-1)

