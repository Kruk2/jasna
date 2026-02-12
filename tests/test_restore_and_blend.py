import collections

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


class _FakeCompleted:
    def __init__(self, meta: object, frame_u8: torch.Tensor) -> None:
        self.meta = meta
        self._frame_u8 = frame_u8

    def to_frame_u8(self, device: torch.device) -> torch.Tensor:
        return self._frame_u8 if self._frame_u8.device == device else self._frame_u8.to(device=device)


class _DeferredStreamingSecondary:
    """Buffers submitted items; only releases them on flush()."""
    name = "deferred"

    def __init__(self) -> None:
        self._pending: list[_FakeCompleted] = []
        self._completed: list[_FakeCompleted] = []

    def submit(self, frames_256: torch.Tensor, *, keep_start: int, keep_end: int, meta: list[object], track_id: int = 0) -> None:
        out_u8 = frames_256[keep_start:keep_end].clamp(0, 1).mul(255.0).round().clamp(0, 255).to(dtype=torch.uint8)
        for m, f in zip(meta, torch.unbind(out_u8, 0)):
            self._pending.append(_FakeCompleted(meta=m, frame_u8=f))

    def drain_completed(self, *, limit: int | None = None) -> list[_FakeCompleted]:
        if limit is None or limit >= len(self._completed):
            out = self._completed
            self._completed = []
            return out
        out = self._completed[:limit]
        self._completed = self._completed[limit:]
        return out

    def flush(self, *, timeout_s: float = 300.0) -> None:
        self._completed.extend(self._pending)
        self._pending.clear()

    def flush_track(self, track_id: int) -> None:
        pass

    def transfer_track(self, old_track_id: int, new_track_id: int) -> None:
        pass


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


def test_poll_secondary_with_limit(monkeypatch) -> None:
    _no_expansion(monkeypatch)

    deferred = _DeferredStreamingSecondary()
    pipeline = RestorationPipeline(restorer=_ConstantRestorer(1.0), secondary_restorer=deferred)  # type: ignore[arg-type]
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

    pipeline.restore_and_blend_clip(clip, frames, keep_start=0, keep_end=3, frame_buffer=fb)
    assert fb.get_ready_frames() == []

    deferred.flush()

    pipeline.poll_secondary(frame_buffer=fb, limit=1)
    ready = fb.get_ready_frames()
    assert len(ready) == 1
    assert ready[0][0] == 0

    pipeline.poll_secondary(frame_buffer=fb)
    ready = fb.get_ready_frames()
    assert len(ready) == 2
    assert [r[0] for r in ready] == [1, 2]


def test_flush_secondary_completes_deferred_items(monkeypatch) -> None:
    _no_expansion(monkeypatch)

    deferred = _DeferredStreamingSecondary()
    pipeline = RestorationPipeline(restorer=_ConstantRestorer(1.0), secondary_restorer=deferred)  # type: ignore[arg-type]
    fb = FrameBuffer(device=torch.device("cpu"), blend_mask_fn=_ones_blend_mask)

    frame = torch.zeros((3, 8, 8), dtype=torch.uint8)
    bbox = np.array([2.0, 2.0, 6.0, 6.0], dtype=np.float32)
    mask = torch.ones((4, 4), dtype=torch.bool)
    clip = TrackedClip(track_id=1, start_frame=0, mask_resolution=(4, 4), bboxes=[bbox], masks=[mask])

    fb.add_frame(frame_idx=0, pts=10, frame=frame, clip_track_ids={1})
    pipeline.restore_and_blend_clip(clip, [frame], keep_start=0, keep_end=1, frame_buffer=fb)
    assert fb.get_ready_frames() == []

    pipeline.flush_secondary(frame_buffer=fb)

    ready = fb.get_ready_frames()
    assert len(ready) == 1
    _, blended, _ = ready[0]
    assert torch.all(blended[:, 2:6, 2:6] == 255)


class _TvaiLikeStreamingSecondary:
    """Simulates TVAI's streaming behaviour with the priming fix:
    - Has temporal latency (configurable, default 3 frames for test speed).
    - The very first output frame from each worker is garbage (all-zeros / black)
      because the temporal model has no prior context.
    - FIX: On the first submit to a worker (or after flush_track), a priming frame
      is written first with None meta so the garbage output is discarded.
    - After the first frame, outputs are correct (pass-through of input as uint8).
    - Tracks are mapped to workers round-robin.
    - flush_track pushes padding to drain remaining real frames.
    """
    name = "tvai-fake"

    def __init__(self, *, latency: int = 3, num_workers: int = 1) -> None:
        self._latency = latency
        self._num_workers = num_workers
        self._worker_buffers: list[collections.deque[tuple[object | None, torch.Tensor]]] = [
            collections.deque() for _ in range(num_workers)
        ]
        self._worker_output_count: list[int] = [0] * num_workers
        self._worker_needs_prime: list[bool] = [True] * num_workers
        self._track_to_worker: dict[int, int] = {}
        self._worker_task_count: list[int] = [0] * num_workers
        self._completed: list[_FakeCompleted] = []

    def _get_worker(self, track_id: int) -> int:
        idx = self._track_to_worker.get(track_id)
        if idx is None:
            idx = min(range(self._num_workers), key=lambda i: self._worker_task_count[i])
            self._track_to_worker[track_id] = idx
        self._worker_task_count[idx] += 1
        return idx

    def submit(self, frames_256: torch.Tensor, *, keep_start: int, keep_end: int, meta: list[object], track_id: int = 0) -> None:
        worker_idx = self._get_worker(track_id)
        buf = self._worker_buffers[worker_idx]

        out_u8 = frames_256[keep_start:keep_end].clamp(0, 1).mul(255.0).round().clamp(0, 255).to(dtype=torch.uint8)

        if self._worker_needs_prime[worker_idx] and len(meta) > 0:
            self._worker_needs_prime[worker_idx] = False
            first_frame = out_u8[0]
            buf.append((None, first_frame.clone()))

        for m, f in zip(meta, torch.unbind(out_u8, 0)):
            buf.append((m, f))

        self._drain_worker(worker_idx)

    def _drain_worker(self, worker_idx: int) -> None:
        buf = self._worker_buffers[worker_idx]
        while len(buf) > self._latency:
            meta, frame_u8 = buf.popleft()
            if self._worker_output_count[worker_idx] == 0:
                frame_u8 = torch.zeros_like(frame_u8)
            self._worker_output_count[worker_idx] += 1
            if meta is not None:
                self._completed.append(_FakeCompleted(meta=meta, frame_u8=frame_u8))

    def drain_completed(self, *, limit: int | None = None) -> list[_FakeCompleted]:
        if limit is None or limit >= len(self._completed):
            out = self._completed
            self._completed = []
            return out
        out = self._completed[:limit]
        self._completed = self._completed[limit:]
        return out

    def flush_track(self, track_id: int) -> None:
        idx = self._track_to_worker.pop(track_id, None)
        if idx is None:
            return
        buf = self._worker_buffers[idx]
        for _ in range(self._latency + 5):
            buf.append((None, torch.zeros(3, 256, 256, dtype=torch.uint8)))
        self._drain_worker(idx)
        self._worker_needs_prime[idx] = True

    def transfer_track(self, old_track_id: int, new_track_id: int) -> None:
        idx = self._track_to_worker.pop(old_track_id, None)
        if idx is not None:
            self._track_to_worker[new_track_id] = idx

    def flush(self, *, timeout_s: float = 300.0) -> None:
        for idx in range(self._num_workers):
            buf = self._worker_buffers[idx]
            for _ in range(self._latency + 5):
                buf.append((None, torch.zeros(3, 256, 256, dtype=torch.uint8)))
            self._drain_worker(idx)


def test_tvai_first_frame_garbage_not_blended(monkeypatch) -> None:
    """TVAI's temporal filter produces a garbage (black) first output frame because
    it has no prior temporal context. This garbage frame must not be blended into
    the final video output.

    The bug: the first output from TVAI is black/green with artifacts. It gets
    matched 1:1 with the pending meta and blended into the video at the very
    first frame of the first clip.
    """
    import jasna.restorer.restoration_pipeline as rp
    monkeypatch.setattr(rp, "BORDER_RATIO", 0.0)
    monkeypatch.setattr(rp, "MIN_BORDER", 0)
    monkeypatch.setattr(rp, "MAX_EXPANSION_FACTOR", 0.0)

    restored_float = 0.5
    restored_u8 = int(round(restored_float * 255))
    original_value = 200

    class _ConstRestorer:
        dtype = torch.float32
        def raw_process(self, crops: list[torch.Tensor]) -> torch.Tensor:
            stacked = []
            for f in crops:
                stacked.append(torch.full(f.permute(2, 0, 1).shape, restored_float, dtype=torch.float32))
            return torch.stack(stacked, dim=0)

    tvai_fake = _TvaiLikeStreamingSecondary(latency=3, num_workers=1)
    pipeline = RestorationPipeline(
        restorer=_ConstRestorer(),  # type: ignore[arg-type]
        secondary_restorer=tvai_fake,
    )
    fb = FrameBuffer(device=torch.device("cpu"), blend_mask_fn=_ones_blend_mask)
    tracker = ClipTracker(max_clip_size=60, temporal_overlap=0, iou_threshold=0.0)

    bbox = np.array([2.0, 2.0, 6.0, 6.0], dtype=np.float32)

    def detections_fn(frames_in: torch.Tensor, *, target_hw: tuple[int, int]) -> Detections:
        bs = frames_in.shape[0]
        return Detections(
            boxes_xyxy=[np.array([bbox], dtype=np.float32) for _ in range(bs)],
            masks=[torch.ones((1, 8, 8), dtype=torch.bool) for _ in range(bs)],
        )

    num_frames = 60
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
            discard_margin=0,
            raw_frame_context=raw_frame_context,
        )
        ready_all.extend(res.ready_frames)
        frame_idx = res.next_frame_idx

    remaining = finalize_processing(
        tracker=tracker,
        frame_buffer=fb,
        restoration_pipeline=pipeline,
        discard_margin=0,
        raw_frame_context=raw_frame_context,
    )
    all_output = ready_all + remaining
    assert len(all_output) == num_frames

    for idx, blended, _ in all_output:
        region = blended[:, 2:6, 2:6]
        assert region.min().item() > 0, (
            f"frame {idx}: black pixel in mosaic region (min={region.min().item()}). "
            f"TVAI first-frame garbage was blended into the output."
        )
        assert (region.float() - restored_u8).abs().max().item() <= 2, (
            f"frame {idx}: mosaic pixel deviates from expected {restored_u8} "
            f"(actual range [{region.min().item()}, {region.max().item()}]). "
            f"TVAI first-frame garbage was blended into the output."
        )


def test_tvai_first_frame_garbage_not_blended_with_denoise(monkeypatch) -> None:
    """Same as above but with denoise_step=AFTER_PRIMARY. The user reported green
    artifacts in this mode at the first frame and also when mosaic begins mid-video.
    """
    import jasna.restorer.restoration_pipeline as rp
    from jasna.restorer.denoise import DenoiseStep, DenoiseStrength
    monkeypatch.setattr(rp, "BORDER_RATIO", 0.0)
    monkeypatch.setattr(rp, "MIN_BORDER", 0)
    monkeypatch.setattr(rp, "MAX_EXPANSION_FACTOR", 0.0)

    restored_float = 0.5
    restored_u8_approx = int(round(restored_float * 255))
    original_value = 200

    class _ConstRestorer:
        dtype = torch.float32
        def raw_process(self, crops: list[torch.Tensor]) -> torch.Tensor:
            stacked = []
            for f in crops:
                stacked.append(torch.full(f.permute(2, 0, 1).shape, restored_float, dtype=torch.float32))
            return torch.stack(stacked, dim=0)

    tvai_fake = _TvaiLikeStreamingSecondary(latency=3, num_workers=1)
    pipeline = RestorationPipeline(
        restorer=_ConstRestorer(),  # type: ignore[arg-type]
        secondary_restorer=tvai_fake,
        denoise_strength=DenoiseStrength.LOW,
        denoise_step=DenoiseStep.AFTER_PRIMARY,
    )
    fb = FrameBuffer(device=torch.device("cpu"), blend_mask_fn=_ones_blend_mask)
    tracker = ClipTracker(max_clip_size=60, temporal_overlap=0, iou_threshold=0.0)

    bbox = np.array([2.0, 2.0, 6.0, 6.0], dtype=np.float32)

    def detections_fn(frames_in: torch.Tensor, *, target_hw: tuple[int, int]) -> Detections:
        bs = frames_in.shape[0]
        return Detections(
            boxes_xyxy=[np.array([bbox], dtype=np.float32) for _ in range(bs)],
            masks=[torch.ones((1, 8, 8), dtype=torch.bool) for _ in range(bs)],
        )

    num_frames = 60
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
            discard_margin=0,
            raw_frame_context=raw_frame_context,
        )
        ready_all.extend(res.ready_frames)
        frame_idx = res.next_frame_idx

    remaining = finalize_processing(
        tracker=tracker,
        frame_buffer=fb,
        restoration_pipeline=pipeline,
        discard_margin=0,
        raw_frame_context=raw_frame_context,
    )
    all_output = ready_all + remaining
    assert len(all_output) == num_frames

    for idx, blended, _ in all_output:
        region = blended[:, 2:6, 2:6]
        assert region.min().item() > 0, (
            f"frame {idx}: black/green pixel in mosaic region (min={region.min().item()}). "
            f"TVAI first-frame garbage was blended into the output (denoise mode)."
        )


def test_tvai_garbage_frame_after_flush_track_new_mosaic(monkeypatch) -> None:
    """When mosaic disappears and reappears later (new track on same worker after
    flush_track), the first output frame of the new track is again garbage because
    TVAI's temporal context was reset by the padding frames.

    This reproduces the user report of artifacts at ~55s when mosaic begins mid-video.
    """
    import jasna.restorer.restoration_pipeline as rp
    monkeypatch.setattr(rp, "BORDER_RATIO", 0.0)
    monkeypatch.setattr(rp, "MIN_BORDER", 0)
    monkeypatch.setattr(rp, "MAX_EXPANSION_FACTOR", 0.0)

    restored_float = 0.5
    restored_u8 = int(round(restored_float * 255))
    original_value = 200

    class _ConstRestorer:
        dtype = torch.float32
        def raw_process(self, crops: list[torch.Tensor]) -> torch.Tensor:
            stacked = []
            for f in crops:
                stacked.append(torch.full(f.permute(2, 0, 1).shape, restored_float, dtype=torch.float32))
            return torch.stack(stacked, dim=0)

    class _TvaiFakeWithFlushGarbage(_TvaiLikeStreamingSecondary):
        """After flush_track resets temporal context, the next first frame is garbage again."""
        def flush_track(self, track_id: int) -> None:
            idx = self._track_to_worker.pop(track_id, None)
            if idx is None:
                return
            buf = self._worker_buffers[idx]
            for _ in range(self._latency + 5):
                buf.append((None, torch.zeros(3, 256, 256, dtype=torch.uint8)))
            self._drain_worker(idx)
            self._worker_output_count[idx] = 0
            self._worker_needs_prime[idx] = True

    tvai_fake = _TvaiFakeWithFlushGarbage(latency=3, num_workers=1)
    pipeline = RestorationPipeline(
        restorer=_ConstRestorer(),  # type: ignore[arg-type]
        secondary_restorer=tvai_fake,
    )
    fb = FrameBuffer(device=torch.device("cpu"), blend_mask_fn=_ones_blend_mask)
    tracker = ClipTracker(max_clip_size=60, temporal_overlap=0, iou_threshold=0.0)

    bbox = np.array([2.0, 2.0, 6.0, 6.0], dtype=np.float32)
    no_det = Detections(
        boxes_xyxy=[np.zeros((0, 4), dtype=np.float32)],
        masks=[torch.zeros((0, 8, 8), dtype=torch.bool)],
    )

    mosaic_det = Detections(
        boxes_xyxy=[np.array([bbox], dtype=np.float32)],
        masks=[torch.ones((1, 8, 8), dtype=torch.bool)],
    )

    phase = {"current": "mosaic1"}

    def detections_fn(frames_in: torch.Tensor, *, target_hw: tuple[int, int]) -> Detections:
        return mosaic_det if phase["current"].startswith("mosaic") else no_det

    raw_frame_context: dict[int, dict[int, torch.Tensor]] = {}
    ready_all: list[tuple[int, torch.Tensor, int]] = []
    frame_idx = 0

    # Phase 1: mosaic present for 20 frames
    phase["current"] = "mosaic1"
    for pts in range(20):
        frame_batch = torch.full((1, 3, 8, 8), original_value, dtype=torch.uint8)
        res = process_frame_batch(
            frames=frame_batch, pts_list=[pts], start_frame_idx=frame_idx,
            batch_size=1, target_hw=(8, 8), detections_fn=detections_fn,
            tracker=tracker, frame_buffer=fb, restoration_pipeline=pipeline,
            discard_margin=0, raw_frame_context=raw_frame_context,
        )
        ready_all.extend(res.ready_frames)
        frame_idx = res.next_frame_idx

    # Phase 2: no mosaic for 10 frames (gap triggers flush_track)
    phase["current"] = "gap"
    for pts in range(20, 30):
        frame_batch = torch.full((1, 3, 8, 8), original_value, dtype=torch.uint8)
        res = process_frame_batch(
            frames=frame_batch, pts_list=[pts], start_frame_idx=frame_idx,
            batch_size=1, target_hw=(8, 8), detections_fn=detections_fn,
            tracker=tracker, frame_buffer=fb, restoration_pipeline=pipeline,
            discard_margin=0, raw_frame_context=raw_frame_context,
        )
        ready_all.extend(res.ready_frames)
        frame_idx = res.next_frame_idx

    # Phase 3: mosaic reappears for 20 frames (new track, same worker)
    phase["current"] = "mosaic2"
    for pts in range(30, 50):
        frame_batch = torch.full((1, 3, 8, 8), original_value, dtype=torch.uint8)
        res = process_frame_batch(
            frames=frame_batch, pts_list=[pts], start_frame_idx=frame_idx,
            batch_size=1, target_hw=(8, 8), detections_fn=detections_fn,
            tracker=tracker, frame_buffer=fb, restoration_pipeline=pipeline,
            discard_margin=0, raw_frame_context=raw_frame_context,
        )
        ready_all.extend(res.ready_frames)
        frame_idx = res.next_frame_idx

    remaining = finalize_processing(
        tracker=tracker, frame_buffer=fb, restoration_pipeline=pipeline,
        discard_margin=0, raw_frame_context=raw_frame_context,
    )
    all_output = ready_all + remaining
    assert len(all_output) == 50

    for idx, blended, _ in all_output:
        region = blended[:, 2:6, 2:6]
        has_mosaic = phase_for_frame(idx) in ("mosaic1", "mosaic2")
        if has_mosaic:
            assert region.min().item() > 0, (
                f"frame {idx}: black pixel in mosaic region (min={region.min().item()}). "
                f"TVAI garbage frame after flush_track was blended into output."
            )


def phase_for_frame(idx: int) -> str:
    if idx < 20:
        return "mosaic1"
    elif idx < 30:
        return "gap"
    else:
        return "mosaic2"


def test_tvai_first_frame_garbage_two_workers(monkeypatch) -> None:
    """With 2 TVAI workers, both workers produce a garbage first frame. Verify
    that neither garbage frame is blended into the output.
    """
    import jasna.restorer.restoration_pipeline as rp
    monkeypatch.setattr(rp, "BORDER_RATIO", 0.0)
    monkeypatch.setattr(rp, "MIN_BORDER", 0)
    monkeypatch.setattr(rp, "MAX_EXPANSION_FACTOR", 0.0)

    restored_float = 0.5
    restored_u8 = int(round(restored_float * 255))
    original_value = 200

    class _ConstRestorer:
        dtype = torch.float32
        def raw_process(self, crops: list[torch.Tensor]) -> torch.Tensor:
            stacked = []
            for f in crops:
                stacked.append(torch.full(f.permute(2, 0, 1).shape, restored_float, dtype=torch.float32))
            return torch.stack(stacked, dim=0)

    tvai_fake = _TvaiLikeStreamingSecondary(latency=3, num_workers=2)
    pipeline = RestorationPipeline(
        restorer=_ConstRestorer(),  # type: ignore[arg-type]
        secondary_restorer=tvai_fake,
    )
    fb = FrameBuffer(device=torch.device("cpu"), blend_mask_fn=_ones_blend_mask)
    tracker = ClipTracker(max_clip_size=60, temporal_overlap=0, iou_threshold=0.0)

    bbox = np.array([2.0, 2.0, 6.0, 6.0], dtype=np.float32)

    def detections_fn(frames_in: torch.Tensor, *, target_hw: tuple[int, int]) -> Detections:
        bs = frames_in.shape[0]
        return Detections(
            boxes_xyxy=[np.array([bbox], dtype=np.float32) for _ in range(bs)],
            masks=[torch.ones((1, 8, 8), dtype=torch.bool) for _ in range(bs)],
        )

    num_frames = 60
    raw_frame_context: dict[int, dict[int, torch.Tensor]] = {}
    ready_all: list[tuple[int, torch.Tensor, int]] = []
    frame_idx = 0

    for pts in range(num_frames):
        frame_batch = torch.full((1, 3, 8, 8), original_value, dtype=torch.uint8)
        res = process_frame_batch(
            frames=frame_batch, pts_list=[pts], start_frame_idx=frame_idx,
            batch_size=1, target_hw=(8, 8), detections_fn=detections_fn,
            tracker=tracker, frame_buffer=fb, restoration_pipeline=pipeline,
            discard_margin=0, raw_frame_context=raw_frame_context,
        )
        ready_all.extend(res.ready_frames)
        frame_idx = res.next_frame_idx

    remaining = finalize_processing(
        tracker=tracker, frame_buffer=fb, restoration_pipeline=pipeline,
        discard_margin=0, raw_frame_context=raw_frame_context,
    )
    all_output = ready_all + remaining
    assert len(all_output) == num_frames

    for idx, blended, _ in all_output:
        region = blended[:, 2:6, 2:6]
        assert region.min().item() > 0, (
            f"frame {idx}: black pixel in mosaic region (min={region.min().item()}). "
            f"TVAI first-frame garbage from one of 2 workers was blended into output."
        )
        assert (region.float() - restored_u8).abs().max().item() <= 2, (
            f"frame {idx}: mosaic pixel deviates from expected {restored_u8} "
            f"(actual range [{region.min().item()}, {region.max().item()}]). "
            f"TVAI first-frame garbage from one of 2 workers was blended into output."
        )
