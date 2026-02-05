import torch

from jasna.tracking.frame_buffer import FrameBuffer


class _FakeClip:
    def __init__(self, track_id: int, frame_idxs: list[int]) -> None:
        self.track_id = track_id
        self._frame_idxs = frame_idxs

    def frame_indices(self) -> list[int]:
        return self._frame_idxs


class _FakeRestoredClip:
    def __init__(
        self,
        *,
        restored_frames: list[torch.Tensor],
        pad_offsets: list[tuple[int, int]],
        resize_shapes: list[tuple[int, int]],
        crop_shapes: list[tuple[int, int]],
        enlarged_bboxes: list[tuple[int, int, int, int]],
        masks: list[torch.Tensor],
        frame_shape: tuple[int, int],
    ) -> None:
        self.restored_frames = restored_frames
        self.pad_offsets = pad_offsets
        self.resize_shapes = resize_shapes
        self.crop_shapes = crop_shapes
        self.enlarged_bboxes = enlarged_bboxes
        self.masks = masks
        self.frame_shape = frame_shape


def test_frame_buffer_ready_when_no_pending_clips() -> None:
    fb = FrameBuffer(device=torch.device("cpu"))

    frame = torch.zeros((3, 8, 8), dtype=torch.uint8)
    fb.add_frame(frame_idx=0, pts=111, frame=frame, clip_track_ids=set())

    ready = fb.get_ready_frames()
    assert ready == [(0, frame, 111)]


def test_frame_buffer_waits_until_clips_blended_then_outputs_ready() -> None:
    captured: list[torch.Tensor] = []

    def blend_mask_fn(crop_mask: torch.Tensor) -> torch.Tensor:
        captured.append(crop_mask)
        return torch.ones_like(crop_mask.squeeze(), dtype=torch.float32)

    fb = FrameBuffer(device=torch.device("cpu"), blend_mask_fn=blend_mask_fn)

    frame = torch.zeros((3, 8, 8), dtype=torch.uint8)
    fb.add_frame(frame_idx=0, pts=123, frame=frame, clip_track_ids={7})
    assert fb.get_ready_frames() == []

    clip = _FakeClip(track_id=7, frame_idxs=[0])

    x1, y1, x2, y2 = (2, 2, 6, 6)
    crop_h, crop_w = (y2 - y1, x2 - x1)
    restored = torch.full((3, crop_h, crop_w), 200, dtype=torch.uint8)
    mask = torch.ones((8, 8), dtype=torch.bool)

    restored_clip = _FakeRestoredClip(
        restored_frames=[restored],
        pad_offsets=[(0, 0)],
        resize_shapes=[(crop_h, crop_w)],
        crop_shapes=[(crop_h, crop_w)],
        enlarged_bboxes=[(x1, y1, x2, y2)],
        masks=[mask],
        frame_shape=(8, 8),
    )

    fb.blend_clip(clip, restored_clip, keep_start=0, keep_end=1)
    ready = fb.get_ready_frames()

    assert len(captured) == 1
    assert captured[0].shape == (crop_h, crop_w)

    assert len(ready) == 1
    _, blended, pts = ready[0]
    assert pts == 123
    assert torch.all(blended[:, y1:y2, x1:x2] == 200)
    assert torch.all(blended[:, :y1, :] == 0)
    assert torch.all(frame == 0)


def test_frame_buffer_add_remove_pending_clip_ignores_missing_frames() -> None:
    fb = FrameBuffer(device=torch.device("cpu"))
    frame = torch.zeros((3, 4, 4), dtype=torch.uint8)
    fb.add_frame(frame_idx=1, pts=10, frame=frame, clip_track_ids=set())

    fb.add_pending_clip([0, 1, 2], track_id=7)
    assert 7 in fb.frames[1].pending_clips

    fb.remove_pending_clip([0, 1, 2], track_id=7)
    assert 7 not in fb.frames[1].pending_clips


def test_frame_buffer_blend_clip_unpads_from_larger_restored_frame_then_resizes_to_crop() -> None:
    fb = FrameBuffer(
        device=torch.device("cpu"),
        blend_mask_fn=lambda crop: torch.ones_like(crop.squeeze(), dtype=torch.float32),
    )

    frame = torch.zeros((3, 8, 8), dtype=torch.uint8)
    fb.add_frame(frame_idx=0, pts=123, frame=frame, clip_track_ids={7})

    clip = _FakeClip(track_id=7, frame_idxs=[0])

    x1, y1, x2, y2 = (2, 2, 6, 6)
    crop_h, crop_w = (y2 - y1, x2 - x1)

    out_h, out_w = (512, 512)
    pad_left, pad_top = (120, 100)
    resize_h, resize_w = (80, 60)

    restored = torch.zeros((3, out_h, out_w), dtype=torch.uint8)
    restored[:, pad_top : pad_top + resize_h, pad_left : pad_left + resize_w] = 200

    mask = torch.ones((8, 8), dtype=torch.bool)
    restored_clip = _FakeRestoredClip(
        restored_frames=[restored],
        pad_offsets=[(pad_left, pad_top)],
        resize_shapes=[(resize_h, resize_w)],
        crop_shapes=[(crop_h, crop_w)],
        enlarged_bboxes=[(x1, y1, x2, y2)],
        masks=[mask],
        frame_shape=(8, 8),
    )

    fb.blend_clip(clip, restored_clip, keep_start=0, keep_end=1)
    ready = fb.get_ready_frames()
    assert len(ready) == 1
    _, blended, _ = ready[0]
    assert torch.all(blended[:, y1:y2, x1:x2] == 200)
    assert torch.all(blended[:, :y1, :] == 0)


def test_frame_buffer_get_ready_frames_stops_at_first_pending() -> None:
    fb = FrameBuffer(
        device=torch.device("cpu"),
        blend_mask_fn=lambda crop: torch.ones_like(crop.squeeze(), dtype=torch.float32),
    )

    frame0 = torch.zeros((3, 8, 8), dtype=torch.uint8)
    frame1 = torch.zeros((3, 8, 8), dtype=torch.uint8)
    fb.add_frame(frame_idx=0, pts=10, frame=frame0, clip_track_ids=set())
    fb.add_frame(frame_idx=1, pts=11, frame=frame1, clip_track_ids={7})

    ready = fb.get_ready_frames()
    assert ready == [(0, frame0, 10)]

    clip = _FakeClip(track_id=7, frame_idxs=[1])
    x1, y1, x2, y2 = (2, 2, 6, 6)
    crop_h, crop_w = (y2 - y1, x2 - x1)
    restored = torch.full((3, crop_h, crop_w), 200, dtype=torch.uint8)
    mask = torch.ones((8, 8), dtype=torch.bool)

    restored_clip = _FakeRestoredClip(
        restored_frames=[restored],
        pad_offsets=[(0, 0)],
        resize_shapes=[(crop_h, crop_w)],
        crop_shapes=[(crop_h, crop_w)],
        enlarged_bboxes=[(x1, y1, x2, y2)],
        masks=[mask],
        frame_shape=(8, 8),
    )

    fb.blend_clip(clip, restored_clip, keep_start=0, keep_end=1)
    ready = fb.get_ready_frames()
    assert len(ready) == 1
    assert ready[0][0] == 1


def test_frame_buffer_flush_returns_remaining_in_order() -> None:
    fb = FrameBuffer(device=torch.device("cpu"))

    f0 = torch.zeros((3, 4, 4), dtype=torch.uint8)
    f2 = torch.ones((3, 4, 4), dtype=torch.uint8)
    fb.add_frame(frame_idx=0, pts=10, frame=f0, clip_track_ids={1})
    fb.add_frame(frame_idx=2, pts=30, frame=f2, clip_track_ids=set())

    remaining = fb.flush()
    assert [x[0] for x in remaining] == [0, 2]
    assert remaining[0][2] == 10
    assert remaining[1][2] == 30


def test_frame_buffer_blend_clip_ignores_missing_frames() -> None:
    fb = FrameBuffer(
        device=torch.device("cpu"),
        blend_mask_fn=lambda crop: torch.ones_like(crop.squeeze(), dtype=torch.float32),
    )
    clip = _FakeClip(track_id=7, frame_idxs=[0])

    restored_clip = _FakeRestoredClip(
        restored_frames=[torch.zeros((3, 4, 4), dtype=torch.uint8)],
        pad_offsets=[(0, 0)],
        resize_shapes=[(4, 4)],
        crop_shapes=[(4, 4)],
        enlarged_bboxes=[(0, 0, 4, 4)],
        masks=[torch.ones((4, 4), dtype=torch.bool)],
        frame_shape=(4, 4),
    )

    fb.blend_clip(clip, restored_clip, keep_start=0, keep_end=1)  # should not raise


def test_frame_buffer_uses_blend_mask_value() -> None:
    fb = FrameBuffer(
        device=torch.device("cpu"),
        blend_mask_fn=lambda crop: torch.zeros_like(crop.squeeze(), dtype=torch.float32),
    )

    frame = torch.zeros((3, 8, 8), dtype=torch.uint8)
    fb.add_frame(frame_idx=0, pts=123, frame=frame, clip_track_ids={7})

    clip = _FakeClip(track_id=7, frame_idxs=[0])
    x1, y1, x2, y2 = (2, 2, 6, 6)
    crop_h, crop_w = (y2 - y1, x2 - x1)
    restored = torch.full((3, crop_h, crop_w), 200, dtype=torch.uint8)
    mask = torch.ones((8, 8), dtype=torch.bool)

    restored_clip = _FakeRestoredClip(
        restored_frames=[restored],
        pad_offsets=[(0, 0)],
        resize_shapes=[(crop_h, crop_w)],
        crop_shapes=[(crop_h, crop_w)],
        enlarged_bboxes=[(x1, y1, x2, y2)],
        masks=[mask],
        frame_shape=(8, 8),
    )

    fb.blend_clip(clip, restored_clip, keep_start=0, keep_end=1)
    ready = fb.get_ready_frames()
    assert len(ready) == 1
    _, blended, _ = ready[0]
    assert torch.all(blended[:, y1:y2, x1:x2] == 0)


def test_frame_buffer_blend_clip_skips_frames_where_clip_is_not_pending() -> None:
    def blend_mask_fn(_: torch.Tensor) -> torch.Tensor:
        raise AssertionError("blend_mask_fn should not be called when clip is not pending")

    fb = FrameBuffer(device=torch.device("cpu"), blend_mask_fn=blend_mask_fn)

    frame = torch.zeros((3, 8, 8), dtype=torch.uint8)
    fb.add_frame(frame_idx=0, pts=123, frame=frame, clip_track_ids={8})

    clip = _FakeClip(track_id=7, frame_idxs=[0])

    x1, y1, x2, y2 = (2, 2, 6, 6)
    crop_h, crop_w = (y2 - y1, x2 - x1)
    restored = torch.full((3, crop_h, crop_w), 200, dtype=torch.uint8)
    mask = torch.ones((8, 8), dtype=torch.bool)

    restored_clip = _FakeRestoredClip(
        restored_frames=[restored],
        pad_offsets=[(0, 0)],
        resize_shapes=[(crop_h, crop_w)],
        crop_shapes=[(crop_h, crop_w)],
        enlarged_bboxes=[(x1, y1, x2, y2)],
        masks=[mask],
        frame_shape=(8, 8),
    )

    fb.blend_clip(clip, restored_clip, keep_start=0, keep_end=1)
    assert fb.get_ready_frames() == []


def test_blend_restored_frame_crossfade_weight_scales_blend() -> None:
    fb = FrameBuffer(
        device=torch.device("cpu"),
        blend_mask_fn=lambda crop: torch.ones_like(crop.squeeze(), dtype=torch.float32),
    )

    frame = torch.zeros((3, 8, 8), dtype=torch.uint8)
    fb.add_frame(frame_idx=0, pts=10, frame=frame, clip_track_ids={1, 2})

    x1, y1, x2, y2 = (2, 2, 6, 6)
    crop_h, crop_w = (y2 - y1, x2 - x1)
    mask = torch.ones((8, 8), dtype=torch.bool)

    # Parent blends with complementary weight 0.5 (value=200)
    # Additive delta against original(0): (200-0)*0.5 = 100
    fb.blend_restored_frame(
        frame_idx=0,
        track_id=1,
        restored=torch.full((3, crop_h, crop_w), 200, dtype=torch.uint8),
        mask_lr=mask,
        frame_shape=(8, 8),
        enlarged_bbox=(x1, y1, x2, y2),
        crop_shape=(crop_h, crop_w),
        pad_offset=(0, 0),
        resize_shape=(crop_h, crop_w),
        crossfade_weight=0.5,
    )
    blended = fb.frames[0].blended_frame
    assert torch.all(blended[:, y1:y2, x1:x2] == 100)

    # Child blends with crossfade_weight=0.5 (value=100)
    # Additive delta against original(0): (100-0)*0.5 = 50, accumulated: 100+50=150
    fb.blend_restored_frame(
        frame_idx=0,
        track_id=2,
        restored=torch.full((3, crop_h, crop_w), 100, dtype=torch.uint8),
        mask_lr=mask,
        frame_shape=(8, 8),
        enlarged_bbox=(x1, y1, x2, y2),
        crop_shape=(crop_h, crop_w),
        pad_offset=(0, 0),
        resize_shape=(crop_h, crop_w),
        crossfade_weight=0.5,
    )
    blended = fb.frames[0].blended_frame
    assert torch.all(blended[:, y1:y2, x1:x2] == 150)

    ready = fb.get_ready_frames()
    assert len(ready) == 1


def test_blend_restored_frame_crossfade_is_order_independent() -> None:
    """Verify that swapping the blend order of parent/child gives the same result."""
    x1, y1, x2, y2 = (2, 2, 6, 6)
    crop_h, crop_w = (y2 - y1, x2 - x1)
    mask = torch.ones((8, 8), dtype=torch.bool)

    def _blend_both(first_track: int, second_track: int) -> torch.Tensor:
        fb = FrameBuffer(
            device=torch.device("cpu"),
            blend_mask_fn=lambda crop: torch.ones_like(crop.squeeze(), dtype=torch.float32),
        )
        frame = torch.zeros((3, 8, 8), dtype=torch.uint8)
        fb.add_frame(frame_idx=0, pts=10, frame=frame, clip_track_ids={1, 2})
        values = {1: 200, 2: 100}
        weights = {1: 0.75, 2: 0.25}
        for tid in (first_track, second_track):
            fb.blend_restored_frame(
                frame_idx=0, track_id=tid,
                restored=torch.full((3, crop_h, crop_w), values[tid], dtype=torch.uint8),
                mask_lr=mask, frame_shape=(8, 8),
                enlarged_bbox=(x1, y1, x2, y2),
                crop_shape=(crop_h, crop_w),
                pad_offset=(0, 0), resize_shape=(crop_h, crop_w),
                crossfade_weight=weights[tid],
            )
        return fb.frames[0].blended_frame[:, y1:y2, x1:x2]

    result_ab = _blend_both(1, 2)
    result_ba = _blend_both(2, 1)
    assert torch.equal(result_ab, result_ba)
    # 200*0.75 + 100*0.25 = 175
    assert torch.all(result_ab == 175)


def test_blend_restored_frame_crossfade_weight_near_zero_keeps_parent() -> None:
    fb = FrameBuffer(
        device=torch.device("cpu"),
        blend_mask_fn=lambda crop: torch.ones_like(crop.squeeze(), dtype=torch.float32),
    )

    frame = torch.zeros((3, 8, 8), dtype=torch.uint8)
    fb.add_frame(frame_idx=0, pts=10, frame=frame, clip_track_ids={1, 2})

    x1, y1, x2, y2 = (2, 2, 6, 6)
    crop_h, crop_w = (y2 - y1, x2 - x1)
    mask = torch.ones((8, 8), dtype=torch.bool)

    # Parent blends with weight 0.95 (value=200)
    # Delta: (200-0)*0.95 = 190
    fb.blend_restored_frame(
        frame_idx=0, track_id=1,
        restored=torch.full((3, crop_h, crop_w), 200, dtype=torch.uint8),
        mask_lr=mask, frame_shape=(8, 8),
        enlarged_bbox=(x1, y1, x2, y2),
        crop_shape=(crop_h, crop_w),
        pad_offset=(0, 0), resize_shape=(crop_h, crop_w),
        crossfade_weight=0.95,
    )

    # Child blends with weight 0.05 (value=0)
    # Delta: (0-0)*0.05 = 0, result stays 190
    fb.blend_restored_frame(
        frame_idx=0, track_id=2,
        restored=torch.full((3, crop_h, crop_w), 0, dtype=torch.uint8),
        mask_lr=mask, frame_shape=(8, 8),
        enlarged_bbox=(x1, y1, x2, y2),
        crop_shape=(crop_h, crop_w),
        pad_offset=(0, 0), resize_shape=(crop_h, crop_w),
        crossfade_weight=0.05,
    )
    blended = fb.frames[0].blended_frame
    # 200*0.95 + 0*0.05 = 190
    assert torch.all(blended[:, y1:y2, x1:x2] == 190)


def test_blend_restored_frame_ignores_missing_frame() -> None:
    def blend_mask_fn(_: torch.Tensor) -> torch.Tensor:
        raise AssertionError("blend_mask_fn should not be called for missing frame")

    fb = FrameBuffer(device=torch.device("cpu"), blend_mask_fn=blend_mask_fn)

    x1, y1, x2, y2 = (2, 2, 6, 6)
    crop_h, crop_w = (y2 - y1, x2 - x1)

    fb.blend_restored_frame(
        frame_idx=99,
        track_id=1,
        restored=torch.full((3, crop_h, crop_w), 200, dtype=torch.uint8),
        mask_lr=torch.ones((8, 8), dtype=torch.bool),
        frame_shape=(8, 8),
        enlarged_bbox=(x1, y1, x2, y2),
        crop_shape=(crop_h, crop_w),
        pad_offset=(0, 0),
        resize_shape=(crop_h, crop_w),
    )  # should not raise


def test_blend_restored_frame_skips_when_track_not_pending() -> None:
    def blend_mask_fn(_: torch.Tensor) -> torch.Tensor:
        raise AssertionError("blend_mask_fn should not be called when track is not pending")

    fb = FrameBuffer(device=torch.device("cpu"), blend_mask_fn=blend_mask_fn)

    frame = torch.zeros((3, 8, 8), dtype=torch.uint8)
    fb.add_frame(frame_idx=0, pts=10, frame=frame, clip_track_ids={8})

    x1, y1, x2, y2 = (2, 2, 6, 6)
    crop_h, crop_w = (y2 - y1, x2 - x1)

    fb.blend_restored_frame(
        frame_idx=0,
        track_id=7,
        restored=torch.full((3, crop_h, crop_w), 200, dtype=torch.uint8),
        mask_lr=torch.ones((8, 8), dtype=torch.bool),
        frame_shape=(8, 8),
        enlarged_bbox=(x1, y1, x2, y2),
        crop_shape=(crop_h, crop_w),
        pad_offset=(0, 0),
        resize_shape=(crop_h, crop_w),
    )  # should not raise

    assert fb.frames[0].blended_frame is fb.frames[0].frame


def test_frame_buffer_blend_clip_discards_frames_outside_keep_range() -> None:
    fb = FrameBuffer(
        device=torch.device("cpu"),
        blend_mask_fn=lambda crop: torch.ones_like(crop.squeeze(), dtype=torch.float32),
    )

    frame0 = torch.zeros((3, 8, 8), dtype=torch.uint8)
    frame1 = torch.zeros((3, 8, 8), dtype=torch.uint8)
    fb.add_frame(frame_idx=0, pts=10, frame=frame0, clip_track_ids={7})
    fb.add_frame(frame_idx=1, pts=11, frame=frame1, clip_track_ids={7})

    clip = _FakeClip(track_id=7, frame_idxs=[0, 1])

    x1, y1, x2, y2 = (2, 2, 6, 6)
    crop_h, crop_w = (y2 - y1, x2 - x1)
    restored0 = torch.full((3, crop_h, crop_w), 123, dtype=torch.uint8)
    restored1 = torch.full((3, crop_h, crop_w), 200, dtype=torch.uint8)
    mask = torch.ones((8, 8), dtype=torch.bool)

    restored_clip = _FakeRestoredClip(
        restored_frames=[restored0, restored1],
        pad_offsets=[(0, 0), (0, 0)],
        resize_shapes=[(crop_h, crop_w), (crop_h, crop_w)],
        crop_shapes=[(crop_h, crop_w), (crop_h, crop_w)],
        enlarged_bboxes=[(x1, y1, x2, y2), (x1, y1, x2, y2)],
        masks=[mask, mask],
        frame_shape=(8, 8),
    )

    fb.blend_clip(clip, restored_clip, keep_start=1, keep_end=2)

    ready = fb.get_ready_frames()
    assert [r[0] for r in ready] == [0, 1]
    out0 = ready[0][1]
    out1 = ready[1][1]
    assert torch.all(out0[:, y1:y2, x1:x2] == 0)
    assert torch.all(out1[:, y1:y2, x1:x2] == 200)
