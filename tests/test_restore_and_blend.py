import numpy as np
import torch

from jasna.mosaic.detections import Detections
from jasna.pipeline_processing import process_frame_batch, finalize_processing
from jasna.restorer.restoration_pipeline import RestorationPipeline
from jasna.tracking.clip_tracker import ClipTracker, TrackedClip
from jasna.tracking.frame_buffer import FrameBuffer


class _ConstantRestorer:
    """Fills all pixels with a constant float value, ignoring input."""
    dtype = torch.float32

    def __init__(self, value: float) -> None:
        self._value = value

    def raw_process(self, crops: list[torch.Tensor]) -> torch.Tensor:
        stacked = []
        for f in crops:
            stacked.append(torch.full(f.permute(2, 0, 1).shape, self._value, dtype=torch.float32))
        return torch.stack(stacked, dim=0)


def _no_expansion(monkeypatch) -> None:
    import jasna.restorer.restoration_pipeline as rp
    monkeypatch.setattr(rp, "BORDER_RATIO", 0.0)
    monkeypatch.setattr(rp, "MIN_BORDER", 0)
    monkeypatch.setattr(rp, "MAX_EXPANSION_FACTOR", 0.0)


def _ones_blend_mask(crop: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(crop.squeeze(), dtype=torch.float32)


def test_restore_and_blend_clip_blends_single_frame(monkeypatch) -> None:
    _no_expansion(monkeypatch)

    pipeline = RestorationPipeline(restorer=_ConstantRestorer(1.0))  # type: ignore[arg-type]
    fb = FrameBuffer(device=torch.device("cpu"), blend_mask_fn=_ones_blend_mask)

    frame = torch.zeros((3, 8, 8), dtype=torch.uint8)
    bbox = np.array([2.0, 2.0, 6.0, 6.0], dtype=np.float32)
    mask = torch.ones((4, 4), dtype=torch.bool)
    clip = TrackedClip(track_id=1, start_frame=0, mask_resolution=(4, 4), bboxes=[bbox], masks=[mask])

    fb.add_frame(frame_idx=0, pts=10, frame=frame, clip_track_ids={1})
    assert fb.get_ready_frames() == []

    pipeline.restore_and_blend_clip(clip, [frame], keep_start=0, keep_end=1, frame_buffer=fb)

    ready = fb.get_ready_frames()
    assert len(ready) == 1
    _, blended, pts = ready[0]
    assert pts == 10
    assert torch.all(blended[:, 2:6, 2:6] == 255)
    assert torch.all(blended[:, :2, :] == 0)


def test_restore_and_blend_clip_discards_pending_outside_keep_range(monkeypatch) -> None:
    _no_expansion(monkeypatch)

    pipeline = RestorationPipeline(restorer=_ConstantRestorer(1.0))  # type: ignore[arg-type]
    fb = FrameBuffer(device=torch.device("cpu"), blend_mask_fn=_ones_blend_mask)

    bbox = np.array([2.0, 2.0, 6.0, 6.0], dtype=np.float32)
    mask = torch.ones((4, 4), dtype=torch.bool)
    frames = []
    for i in range(3):
        f = torch.zeros((3, 8, 8), dtype=torch.uint8)
        fb.add_frame(frame_idx=i, pts=i * 10, frame=f, clip_track_ids={1})
        frames.append(f)

    clip = TrackedClip(
        track_id=1, start_frame=0, mask_resolution=(4, 4),
        bboxes=[bbox] * 3, masks=[mask] * 3,
    )

    pipeline.restore_and_blend_clip(clip, frames, keep_start=1, keep_end=2, frame_buffer=fb)

    ready = fb.get_ready_frames()
    assert len(ready) == 3
    assert torch.all(ready[0][1] == 0)
    assert torch.all(ready[1][1][:, 2:6, 2:6] == 255)
    assert torch.all(ready[2][1] == 0)


def test_restore_and_blend_clip_noop_when_keep_range_empty(monkeypatch) -> None:
    _no_expansion(monkeypatch)

    class _NeverCalledRestorer:
        dtype = torch.float32

        def raw_process(self, crops: list[torch.Tensor]) -> torch.Tensor:
            raise AssertionError("raw_process should not be called")

    pipeline = RestorationPipeline(restorer=_NeverCalledRestorer())  # type: ignore[arg-type]
    fb = FrameBuffer(device=torch.device("cpu"))

    frame = torch.zeros((3, 8, 8), dtype=torch.uint8)
    bbox = np.array([2.0, 2.0, 6.0, 6.0], dtype=np.float32)
    mask = torch.ones((4, 4), dtype=torch.bool)
    clip = TrackedClip(track_id=1, start_frame=0, mask_resolution=(4, 4), bboxes=[bbox], masks=[mask])

    fb.add_frame(frame_idx=0, pts=10, frame=frame, clip_track_ids={1})
    pipeline.restore_and_blend_clip(clip, [frame], keep_start=5, keep_end=5, frame_buffer=fb)

    assert 1 not in fb.frames[0].pending_clips
    ready = fb.get_ready_frames()
    assert len(ready) == 1


def test_restore_and_blend_clip_passes_crossfade_weights(monkeypatch) -> None:
    _no_expansion(monkeypatch)

    pipeline = RestorationPipeline(restorer=_ConstantRestorer(1.0))  # type: ignore[arg-type]
    fb = FrameBuffer(device=torch.device("cpu"), blend_mask_fn=_ones_blend_mask)

    frame = torch.zeros((3, 8, 8), dtype=torch.uint8)
    bbox = np.array([2.0, 2.0, 6.0, 6.0], dtype=np.float32)
    mask = torch.ones((4, 4), dtype=torch.bool)
    clip = TrackedClip(track_id=1, start_frame=0, mask_resolution=(4, 4), bboxes=[bbox], masks=[mask])

    fb.add_frame(frame_idx=0, pts=10, frame=frame, clip_track_ids={1})

    pipeline.restore_and_blend_clip(
        clip, [frame], keep_start=0, keep_end=1, frame_buffer=fb,
        crossfade_weights={0: 0.5},
    )

    ready = fb.get_ready_frames()
    assert len(ready) == 1
    _, blended, _ = ready[0]
    # 0 + (255 - 0) * 0.5 = 127.5 -> 128
    assert torch.all(blended[:, 2:6, 2:6] == 128)
