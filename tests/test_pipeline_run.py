import threading
from contextlib import ExitStack
from fractions import Fraction
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace

from factories import make_pipeline, pass_context, primary_result, video_metadata
from jasna import pipeline_threads
from jasna.accelerator import AcceleratorVendor
from jasna.crop_buffer import RawCrop
from jasna.media.encoder_settings import AMF_MIN_SIZE
from jasna.media.probe import UnsupportedVideoError
from jasna.pipeline import Pipeline
from jasna.pipeline_items import ClipRestoreItem, FrameMeta, _SENTINEL
from jasna.pipeline_processing import BatchProcessResult
from jasna.pipeline_threads import earliest_blocking_seqs, run_async_secondary, starvation_action
from jasna.restorer import RestorationPipeline
from jasna.segments import SegmentRange
from jasna.tracking.clip_tracker import TrackedClip


@pytest.fixture(autouse=True)
def _mock_gpu_monitoring():
    """Exercise the real thread flow on CPU without GPU memory monitoring."""
    with (
        patch("jasna.pipeline_threads.VramOffloader", autospec=True),
        patch.object(Pipeline, "_log_memory_at_end"),
    ):
        yield


def _fake_metadata():
    return video_metadata(is_10bit=True)


class _WhitePrimary:
    """Primary restorer that turns every crop white."""

    device = torch.device("cpu")
    input_dtype = torch.float32

    def __init__(self, error: Exception | None = None):
        self.error = error

    def raw_process(self, crops):
        if self.error:
            raise self.error
        return torch.ones((len(crops), 3, 256, 256))


class _FailingSecondary:
    name = "failing"
    prefers_cpu_input = False
    num_workers = 1

    def restore(self, frames_256, *, keep_start, keep_end):
        raise RuntimeError("secondary boom")


class _BufferingAsyncRestorer:
    """Async secondary restorer that, like TVAI, returns a clip only once it is flushed.

    It answers a flush only for the clips it names. ``on_pop`` runs after each forwarded batch;
    ``pop_error`` is raised once a flushed clip is ready.
    """

    name = "buffering"
    prefers_cpu_input = True
    num_workers = 2

    def __init__(
        self, *, push_error: Exception | None = None, pop_error: Exception | None = None, on_pop=lambda restorer: None
    ):
        self.push_error = push_error
        self.pop_error = pop_error
        self.on_pop = on_pop
        self.frames: dict[int, int] = {}
        self.ready: list[int] = []
        self.returned: list[int] = []
        self.flushes: list[set[int]] = []

    def push_clip(self, frames_256, *, keep_start, keep_end):
        if self.push_error:
            raise self.push_error
        seq = len(self.frames)
        self.frames[seq] = keep_end - keep_start
        return seq

    def pop_completed(self):
        if self.pop_error and self.ready:
            raise self.pop_error
        completed = [(seq, [np.full((256, 256, 3), 255, np.uint8)] * self.frames[seq]) for seq in self.ready]
        self.returned += self.ready
        self.ready = []
        if completed:
            self.on_pop(self)
        return completed

    @property
    def has_pending(self) -> bool:
        return len(self.returned) + len(self.ready) < len(self.frames)

    def flush_pending(self, target_seqs=None) -> bool:
        if target_seqs is None:
            return False
        self.flushes.append(target_seqs)
        self.ready += sorted(target_seqs)
        return True

    def flush_all(self) -> None:
        self.ready += [seq for seq in self.frames if seq not in self.returned and seq not in self.ready]

    def close(self) -> None:
        pass

    @staticmethod
    def _to_tensors(frames_np):
        return torch.from_numpy(np.stack(frames_np)).permute(0, 3, 1, 2) if frames_np else torch.empty(0)


def _pipeline(*, primary=None, secondary=None, **overrides) -> Pipeline:
    return make_pipeline(
        restoration_pipeline=RestorationPipeline(primary or _WhitePrimary(), secondary_restorer=secondary),
        batch_size=2,
        disable_progress=True,
        **overrides,
    )


def _reader(frames=None, error: Exception | None = None) -> MagicMock:
    reader = MagicMock()
    reader.__enter__.return_value = reader
    if error:
        reader.frames.side_effect = error
    else:
        reader.frames.return_value = iter([] if frames is None else [(frames, list(range(len(frames))))])
    return reader


def _detect_one_clip(**kwargs):
    """Detection that finds one mosaic covering the box (1, 1, 5, 5) on both frames."""
    clip = TrackedClip(
        track_id=42,
        start_frame=0,
        mask_resolution=(8, 8),
        bboxes=[np.array([1, 1, 5, 5], dtype=np.float32)] * 2,
        masks=[torch.ones((8, 8), dtype=torch.bool)] * 2,
    )
    for frame_idx in range(2):
        kwargs["blend_buffer"].register_frame(frame_idx, {42})
        kwargs["metadata_queue"].put(FrameMeta(frame_idx=frame_idx, pts=frame_idx))
    raw_crops = [
        RawCrop(crop=frame[:, 1:5, 1:5].clone(), enlarged_bbox=(1, 1, 5, 5), crop_shape=(4, 4))
        for frame in kwargs["frames"]
    ]
    kwargs["clip_queue"].put(
        ClipRestoreItem(clip=clip, raw_crops=raw_crops, frame_shape=(8, 8), keep_start=0, keep_end=2, crossfade_weights=None)
    )
    return BatchProcessResult(next_frame_idx=2, clips_emitted=1)


def _run(pipeline, *, metadata=None, frames=None, decode_error=None):
    """Run ``pipeline`` over ``frames`` with detection faked; returns the readers' and encoder's mocks."""
    source = _reader(frames, decode_error)
    blend = _reader(frames)
    reader_cls = MagicMock(side_effect=[source, blend])
    encoder = MagicMock()
    with ExitStack() as stack:
        stack.enter_context(patch("jasna.pipeline.get_video_meta_data", return_value=metadata or _fake_metadata()))
        stack.enter_context(patch("jasna.pipeline_threads.VideoReader", reader_cls))
        encoder_cls = stack.enter_context(patch("jasna.pipeline.VideoEncoder", return_value=encoder))
        stack.enter_context(patch("jasna.pipeline_threads.process_frame_batch", side_effect=_detect_one_clip))
        stack.enter_context(patch("jasna.pipeline_threads.finalize_processing"))
        stack.enter_context(patch("jasna.pipeline_threads.torch.cuda.set_device"))
        pipeline.run()
    return reader_cls, encoder_cls, encoder


@pytest.mark.parametrize("secondary", [None, _BufferingAsyncRestorer()], ids=["sync", "async"])
def test_run_blends_restored_mosaic_into_the_encoded_frames(secondary):
    progress = MagicMock()
    pipeline = _pipeline(secondary=secondary, progress_callback=progress)

    _, _, encoder = _run(pipeline, frames=torch.zeros((2, 3, 8, 8), dtype=torch.uint8))

    encoded = [call.args for call in encoder.encode.call_args_list]
    assert [pts for _, pts in encoded] == [0, 1]
    for frame, _ in encoded:
        assert frame[:, 1:5, 1:5].min() > 0
        assert frame[:, 0].max() == 0 and frame[:, :, 0].max() == 0
    assert pipeline.completed
    assert progress.call_args.args[0] == 100.0


def test_run_encodes_nothing_for_an_empty_video():
    _, _, encoder = _run(_pipeline())
    encoder.encode.assert_not_called()


@pytest.mark.parametrize(
    ("pipeline_kwargs", "decode_error", "message"),
    [
        ({}, RuntimeError("decode boom"), "decode boom"),
        ({"primary": _WhitePrimary(RuntimeError("primary boom"))}, None, "primary boom"),
        ({"secondary": _FailingSecondary()}, None, "secondary boom"),
        ({"secondary": _BufferingAsyncRestorer(push_error=RuntimeError("push boom"))}, None, "push boom"),
        ({"secondary": _BufferingAsyncRestorer(pop_error=RuntimeError("pop boom"))}, None, "pop boom"),
    ],
    ids=["decode", "primary", "secondary", "async-secondary-push", "async-secondary-pop"],
)
def test_run_raises_a_worker_thread_error(pipeline_kwargs, decode_error, message):
    with pytest.raises(RuntimeError, match=message):
        _run(_pipeline(**pipeline_kwargs), frames=torch.zeros((2, 3, 8, 8), dtype=torch.uint8), decode_error=decode_error)


def test_run_fails_promptly_when_blend_encode_dies_while_secondary_is_blocked():
    p = make_pipeline(batch_size=2, disable_progress=True)

    frames_t = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)
    r1 = _reader(frames_t)
    r2 = MagicMock()
    r2.__enter__.return_value = r2
    blend_reader_failed = threading.Event()

    def _failing_frames(seek_ts):
        blend_reader_failed.set()
        raise RuntimeError("AMF decode failed")

    r2.frames.side_effect = _failing_frames
    reader_cls = MagicMock(side_effect=[r1, r2])

    clip_frames = 40
    clip = TrackedClip(
        track_id=1, start_frame=0, mask_resolution=(2, 2),
        bboxes=[np.array([1, 1, 5, 5], dtype=np.float32)] * 2,
        masks=[torch.zeros((2, 2), dtype=torch.bool)] * 2,
    )

    def fake_process_batch(**kwargs):
        kwargs["metadata_queue"].put(FrameMeta(frame_idx=0, pts=0))
        for _ in range(2):
            kwargs["clip_queue"].put(
                ClipRestoreItem(clip=clip, raw_crops=[], frame_shape=(8, 8), keep_start=0, keep_end=clip_frames, crossfade_weights=None),
                frame_count=clip_frames,
            )
        return BatchProcessResult(next_frame_idx=2, clips_emitted=2)

    def _primary_after_blend_failed(*args):
        assert blend_reader_failed.wait(timeout=5)
        return MagicMock(keep_start=0, keep_end=clip_frames)

    p.restoration_pipeline.prepare_and_run_primary.side_effect = _primary_after_blend_failed
    p.restoration_pipeline.build_secondary_result.return_value = MagicMock(keep_end=clip_frames, frame_count=clip_frames)

    raised: list[BaseException] = []

    def _run_pipeline():
        try:
            p.run()
        except BaseException as exc:
            raised.append(exc)

    with (
        patch("jasna.pipeline.get_video_meta_data", return_value=_fake_metadata()),
        patch("jasna.pipeline_threads.VideoReader", reader_cls),
        patch("jasna.pipeline.VideoEncoder"),
        patch("jasna.pipeline_threads.process_frame_batch", side_effect=fake_process_batch),
        patch("jasna.pipeline_threads.finalize_processing"),
        patch("jasna.pipeline_threads.torch.cuda.set_device"),
    ):
        runner = threading.Thread(target=_run_pipeline, daemon=True)
        runner.start()
        runner.join(timeout=10)

    assert not runner.is_alive(), "restoration pass hung after a thread error"
    assert len(raised) == 1 and "AMF decode failed" in str(raised[0])


def test_retarget_high_fps_halves_the_rate_for_readers_encoder_and_progress():
    metadata = video_metadata(fps=60, num_frames=5)

    with patch("jasna.pipeline.Progressbar") as progress_cls:
        reader_cls, encoder_cls, _ = _run(_pipeline(retarget_high_fps=True), metadata=metadata)

    assert [call.kwargs["frame_stride"] for call in reader_cls.call_args_list] == [2, 2]
    assert encoder_cls.call_args.kwargs["output_fps"] == Fraction(30, 1)
    assert progress_cls.call_args.kwargs["total_frames"] == 3
    assert progress_cls.call_args.kwargs["video_fps"] == 30.0


def test_fmp4_is_dropped_for_segment_processing(tmp_path):
    p = _pipeline(
        fmp4=True, segments=(SegmentRange(0.0, 1.0),),
        input_video=tmp_path / "in.mp4", output_video=tmp_path / "out.mp4",
    )
    p.input_video.touch()

    with (
        patch("jasna.pipeline.get_video_meta_data", return_value=_fake_metadata()),
        patch.object(Pipeline, "_run_smart") as run_smart,
    ):
        p.run()

    run_smart.assert_called_once()
    assert p.fmp4 is False


@pytest.mark.parametrize(
    ("color_space", "supported"),
    [
        (AvColorspace.ITU709, True),
        (AvColorspace.ITU601, True),
        (AvColorspace.BT2020, True),
        (AvColorspace.SMPTE240M, False),
    ],
)
def test_validate_metadata_accepts_only_supported_color_spaces(color_space, supported):
    metadata = _fake_metadata()
    metadata.color_space = color_space
    if supported:
        make_pipeline().validate_metadata(metadata)
    else:
        with pytest.raises(UnsupportedVideoError, match="Only BT.709, BT.601, and BT.2020"):
            make_pipeline().validate_metadata(metadata)


def test_run_rejects_a_video_too_small_for_the_amd_encoder_before_processing():
    meta = _fake_metadata()
    meta.video_width, meta.video_height = AMF_MIN_SIZE["hevc"][0] - 2, 240

    with (
        patch("jasna.pipeline.get_video_meta_data", return_value=meta),
        patch("jasna.pipeline.vendor_for_device", return_value=AcceleratorVendor.AMD),
        pytest.raises(UnsupportedVideoError, match="AMD video encoder"),
    ):
        _pipeline().run()


def _run_async_secondary(restorer, results, *, primary_idle: bool):
    ctx = pass_context(restoration_pipeline=RestorationPipeline(_WhitePrimary(), secondary_restorer=restorer))
    for result in results:
        ctx.secondary_queue.put(result)
    if not primary_idle:
        ctx.secondary_queue.put(_SENTINEL)
    if primary_idle:
        ctx.primary_idle_event.set()
        restorer.on_pop = lambda r: ctx.secondary_queue.put(_SENTINEL) if len(r.returned) == len(results) else None
    stats = run_async_secondary(ctx)
    forwarded = []
    while not ctx.encode_queue.empty():
        forwarded.append(ctx.encode_queue.get())
    return stats, forwarded


def test_async_secondary_forwards_every_clip_in_order_once_flushed_at_the_end(monkeypatch):
    monkeypatch.setattr(pipeline_threads, "_ASYNC_POLL_TIMEOUT", 0.001)
    restorer = _BufferingAsyncRestorer()
    results = [primary_result(track_id=1, frames=1), primary_result(track_id=2, frames=170), primary_result(track_id=3, frames=80)]

    stats, forwarded = _run_async_secondary(restorer, results, primary_idle=False)

    assert [(sr.track_id, len(sr.restored_frames)) for sr in forwarded] == [(1, 1), (2, 170), (3, 80)]
    assert restorer.flushes == []
    assert (stats.clips_pushed, stats.clips_popped, stats.starvation_flushes) == (3, 3, 0)


def test_async_secondary_flushes_the_earliest_blocking_clip_when_starved(monkeypatch):
    monkeypatch.setattr(pipeline_threads, "_ASYNC_POLL_TIMEOUT", 0.001)
    monkeypatch.setattr(pipeline_threads, "_FLUSH_DELAY", 0.0)
    restorer = _BufferingAsyncRestorer()
    results = [primary_result(track_id=0, start_frame=0, frames=180), primary_result(track_id=1, start_frame=180, frames=180)]

    stats, forwarded = _run_async_secondary(restorer, results, primary_idle=True)

    assert restorer.flushes == [{0}, {1}]
    assert [sr.track_id for sr in forwarded] == [0, 1]


def test_async_secondary_stops_after_a_cancel_mid_pass_without_flushing(monkeypatch):
    monkeypatch.setattr(pipeline_threads, "_ASYNC_POLL_TIMEOUT", 0.001)
    stop = threading.Event()

    class _CancelOnFirstPush(_BufferingAsyncRestorer):
        def push_clip(self, frames_256, *, keep_start, keep_end):
            seq = super().push_clip(frames_256, keep_start=keep_start, keep_end=keep_end)
            stop.set()
            return seq

    restorer = _CancelOnFirstPush()
    ctx = pass_context(
        restoration_pipeline=RestorationPipeline(_WhitePrimary(), secondary_restorer=restorer), stop_event=stop
    )
    ctx.secondary_queue.put(primary_result(track_id=1))
    ctx.secondary_queue.put(primary_result(track_id=2))

    with patch("jasna.pipeline_threads.torch.cuda.set_device"):
        pipeline_threads.async_secondary_restore_loop(ctx)

    assert ctx.error_holder == []
    assert list(restorer.frames) == [0]
    assert restorer.ready == [] and restorer.returned == []
    assert ctx.secondary_queue.qsize() == 1
    assert ctx.encode_queue.get_nowait() is _SENTINEL
    assert ctx.encode_queue.empty()


@pytest.mark.parametrize(
    ("starved", "flushed", "since_push", "since_flush", "action"),
    [
        (False, False, 99.0, 99.0, None),
        (True, False, 1.0, 99.0, None),
        (True, False, 3.0, 99.0, "flush"),
        (True, True, 99.0, 1.0, None),
        (True, True, 99.0, 6.0, "retry"),
    ],
    ids=["clips-incoming", "short-gap", "starved", "already-flushed", "flush-stuck"],
)
def test_starvation_action(starved, flushed, since_push, since_flush, action):
    assert starvation_action(
        starved=starved, flushed_since_last_push=flushed, since_push=since_push, since_flush=since_flush
    ) == action


@pytest.mark.parametrize(
    ("clips", "expected"),
    [
        ({}, None),
        ({0: dict(start_frame=100, frames=180)}, {0}),
        ({0: dict(start_frame=100, frames=180), 1: dict(start_frame=100, frames=180)}, {0, 1}),
        ({0: dict(start_frame=0, frames=180), 1: dict(start_frame=200, frames=180)}, {0}),
        ({0: dict(start_frame=0, frames=180), 1: dict(start_frame=50, frames=180)}, {0}),
        ({0: dict(start_frame=0, frames=180, keep_start=20), 1: dict(start_frame=10, frames=180)}, {1}),
    ],
    ids=["empty", "single", "same-start", "disjoint", "overlapping", "keep-start-shifts"],
)
def test_earliest_blocking_seqs(clips, expected):
    assert earliest_blocking_seqs({seq: primary_result(**clip) for seq, clip in clips.items()}) == expected
