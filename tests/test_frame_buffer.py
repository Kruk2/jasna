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
        padded_masks: list[torch.Tensor],
        frame_shape: tuple[int, int],
    ) -> None:
        self.restored_frames = restored_frames
        self.pad_offsets = pad_offsets
        self.resize_shapes = resize_shapes
        self.crop_shapes = crop_shapes
        self.enlarged_bboxes = enlarged_bboxes
        self.padded_masks = padded_masks
        self.frame_shape = frame_shape


def test_frame_buffer_ready_when_no_pending_clips() -> None:
    fb = FrameBuffer(device=torch.device("cpu"), compute_dtype=torch.get_default_dtype())

    frame = torch.zeros((3, 8, 8), dtype=torch.uint8)
    fb.add_frame(frame_idx=0, pts=111, frame=frame, clip_track_ids=set())

    ready = fb.get_ready_frames()
    assert ready == [(0, frame, 111)]


def test_frame_buffer_waits_until_clips_blended_then_outputs_ready() -> None:
    captured: list[torch.Tensor] = []

    def blend_mask_fn(crop_mask: torch.Tensor) -> torch.Tensor:
        captured.append(crop_mask)
        return torch.ones_like(crop_mask.squeeze(), dtype=torch.float32)

    fb = FrameBuffer(device=torch.device("cpu"), compute_dtype=torch.get_default_dtype(), blend_mask_fn=blend_mask_fn)

    frame = torch.zeros((3, 8, 8), dtype=torch.uint8)
    fb.add_frame(frame_idx=0, pts=123, frame=frame, clip_track_ids={7})
    assert fb.get_ready_frames() == []

    clip = _FakeClip(track_id=7, frame_idxs=[0])

    x1, y1, x2, y2 = (2, 2, 6, 6)
    crop_h, crop_w = (y2 - y1, x2 - x1)
    restored = torch.full((3, crop_h, crop_w), 200, dtype=torch.uint8)
    padded_mask = torch.ones((crop_h, crop_w), dtype=torch.bool)

    restored_clip = _FakeRestoredClip(
        restored_frames=[restored],
        pad_offsets=[(0, 0)],
        resize_shapes=[(crop_h, crop_w)],
        crop_shapes=[(crop_h, crop_w)],
        enlarged_bboxes=[(x1, y1, x2, y2)],
        padded_masks=[padded_mask],
        frame_shape=(8, 8),
    )

    fb.blend_clip(clip, restored_clip)
    ready = fb.get_ready_frames()

    assert len(captured) == 1
    assert captured[0].shape == (crop_h, crop_w)

    assert len(ready) == 1
    _, blended, pts = ready[0]
    assert pts == 123
    assert torch.all(blended[:, y1:y2, x1:x2] == 200)
    assert torch.all(blended[:, :y1, :] == 0)
    assert torch.all(frame == 0)


def test_frame_buffer_get_ready_frames_stops_at_first_pending() -> None:
    fb = FrameBuffer(
        device=torch.device("cpu"),
        compute_dtype=torch.get_default_dtype(),
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
    padded_mask = torch.ones((crop_h, crop_w), dtype=torch.bool)

    restored_clip = _FakeRestoredClip(
        restored_frames=[restored],
        pad_offsets=[(0, 0)],
        resize_shapes=[(crop_h, crop_w)],
        crop_shapes=[(crop_h, crop_w)],
        enlarged_bboxes=[(x1, y1, x2, y2)],
        padded_masks=[padded_mask],
        frame_shape=(8, 8),
    )

    fb.blend_clip(clip, restored_clip)
    ready = fb.get_ready_frames()
    assert len(ready) == 1
    assert ready[0][0] == 1


def test_frame_buffer_flush_returns_remaining_in_order() -> None:
    fb = FrameBuffer(device=torch.device("cpu"), compute_dtype=torch.get_default_dtype())

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
        compute_dtype=torch.get_default_dtype(),
        blend_mask_fn=lambda crop: torch.ones_like(crop.squeeze(), dtype=torch.float32),
    )
    clip = _FakeClip(track_id=7, frame_idxs=[0])

    restored_clip = _FakeRestoredClip(
        restored_frames=[torch.zeros((3, 4, 4), dtype=torch.uint8)],
        pad_offsets=[(0, 0)],
        resize_shapes=[(4, 4)],
        crop_shapes=[(4, 4)],
        enlarged_bboxes=[(0, 0, 4, 4)],
        padded_masks=[torch.ones((4, 4), dtype=torch.bool)],
        frame_shape=(4, 4),
    )

    fb.blend_clip(clip, restored_clip)  # should not raise


def test_frame_buffer_uses_blend_mask_value() -> None:
    fb = FrameBuffer(
        device=torch.device("cpu"),
        compute_dtype=torch.get_default_dtype(),
        blend_mask_fn=lambda crop: torch.zeros_like(crop.squeeze(), dtype=torch.float32),
    )

    frame = torch.zeros((3, 8, 8), dtype=torch.uint8)
    fb.add_frame(frame_idx=0, pts=123, frame=frame, clip_track_ids={7})

    clip = _FakeClip(track_id=7, frame_idxs=[0])
    x1, y1, x2, y2 = (2, 2, 6, 6)
    crop_h, crop_w = (y2 - y1, x2 - x1)
    restored = torch.full((3, crop_h, crop_w), 200, dtype=torch.uint8)
    padded_mask = torch.ones((crop_h, crop_w), dtype=torch.bool)

    restored_clip = _FakeRestoredClip(
        restored_frames=[restored],
        pad_offsets=[(0, 0)],
        resize_shapes=[(crop_h, crop_w)],
        crop_shapes=[(crop_h, crop_w)],
        enlarged_bboxes=[(x1, y1, x2, y2)],
        padded_masks=[padded_mask],
        frame_shape=(8, 8),
    )

    fb.blend_clip(clip, restored_clip)
    ready = fb.get_ready_frames()
    assert len(ready) == 1
    _, blended, _ = ready[0]
    assert torch.all(blended[:, y1:y2, x1:x2] == 0)

