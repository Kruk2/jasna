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

