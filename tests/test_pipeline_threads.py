"""Tests for pipeline_threads.py shared thread functions and related frame writers."""
from __future__ import annotations

import logging
import threading
from queue import Queue
from unittest.mock import MagicMock, patch

import pytest
import torch

from factories import make_pipeline, video_metadata
from jasna.blend_buffer import BlendBuffer
from jasna.frame_queue import FrameQueue
from jasna.pipeline import _OfflineFrameWriter
from jasna.pipeline_items import FrameMeta, _SENTINEL
from jasna.pipeline_processing import BatchProcessResult
from jasna.pipeline_threads import (
    blend_encode_loop,
    decode_detect_loop,
    primary_restore_loop,
    secondary_restore_loop,
)
from jasna.streaming_pipeline import _StreamingFrameWriter, _run_streaming_pass, _streaming_loop, run_streaming


def _reader(batches) -> MagicMock:
    """A VideoReader whose ``frames`` yields ``batches`` (a list or a generator function)."""
    reader = MagicMock()
    reader.__enter__.return_value = reader
    reader.frames.side_effect = batches if callable(batches) else lambda seek_ts=None: iter(batches)
    return reader


class _RecordingWriter:
    def __init__(self):
        self.written: list[tuple[torch.Tensor, int]] = []

    def write(self, frame: torch.Tensor, pts: int) -> None:
        self.written.append((frame, pts))

    def after_write(self, frames_written: int, stop_event: threading.Event) -> None:
        pass


def _register_frames(**kwargs):
    """Detection that finds nothing: every frame goes straight to blend-encode."""
    start = kwargs["start_frame_idx"]
    for offset, pts in enumerate(kwargs["pts_list"]):
        kwargs["blend_buffer"].register_frame(start + offset, set())
        kwargs["metadata_queue"].put(FrameMeta(frame_idx=start + offset, pts=int(pts)))
    return BatchProcessResult(next_frame_idx=start + len(kwargs["pts_list"]), clips_emitted=0)


def _decode(reader, *, process=_register_frames, metadata=None, **overrides):
    """Run ``decode_detect_loop`` over ``reader``; returns the metadata queue, error holder and detection mock."""
    metadata_queue = Queue(maxsize=999)
    arguments = dict(
        input_video="fake.mkv",
        batch_size=2,
        device=torch.device("cpu"),
        metadata=metadata or video_metadata(num_frames=100),
        detection_model=MagicMock(),
        max_clip_size=60,
        temporal_overlap=8,
        max_detection_gap=0,
        min_detection_duration=0,
        enable_crossfade=True,
        scene_detection=False,
        blend_buffer=BlendBuffer(device=torch.device("cpu")),
        crop_buffers={},
        clip_queue=FrameQueue(max_frames=999),
        metadata_queue=metadata_queue,
        error_holder=[],
        cancel_event=threading.Event(),
        debug_memory=MagicMock(),
        vr_mode="off",
        vr_projector=None,
        seek_ts=None,
        end_pts=None,
        effect_ranges=None,
        frame_stride=1,
        output_frame_count=None,
        output_fps=None,
        progress=None,
    )
    arguments.update(overrides)
    with (
        patch("jasna.pipeline_threads.VideoReader", return_value=reader),
        patch("jasna.pipeline_threads.torch.cuda.set_device"),
        patch("jasna.pipeline_threads.process_frame_batch", side_effect=process) as detect,
        patch("jasna.pipeline_threads.finalize_processing"),
    ):
        decode_detect_loop(**arguments)
    metas = []
    while (item := metadata_queue.get_nowait()) is not _SENTINEL:
        metas.append(item)
    return metas, arguments["error_holder"], detect


def test_segment_pass_trims_at_end_and_detects_only_exact_effect_frames():
    frames = torch.zeros((6, 3, 8, 8), dtype=torch.uint8)

    metas, _, detect = _decode(
        _reader([(frames, [60, 61, 62, 63, 64, 65])]),
        batch_size=6,
        seek_ts=2.5,
        end_pts=64,
        effect_ranges=((61, 63),),
    )

    assert [call.kwargs["pts_list"] for call in detect.call_args_list] == [[61, 62]]
    assert [meta.pts for meta in metas] == [60, 61, 62, 63]
    assert [meta.apply_effect for meta in metas] == [False, True, True, False]


def test_decode_starts_frame_numbering_at_the_seek_position():
    _, _, detect = _decode(_reader([(torch.zeros((2, 3, 8, 8), dtype=torch.uint8), [0, 1])]), seek_ts=2.0)

    assert detect.call_args.kwargs["start_frame_idx"] == 48


def test_decode_stops_reading_once_cancelled():
    cancel = threading.Event()
    batches_read = []

    def batches(seek_ts=None):
        for index in range(10):
            batches_read.append(index)
            if index == 1:
                cancel.set()
            yield torch.zeros((2, 3, 8, 8), dtype=torch.uint8), [2 * index, 2 * index + 1]

    _, errors, _ = _decode(_reader(batches), cancel_event=cancel)

    assert not errors
    assert batches_read == [0, 1]


def test_decode_stops_on_an_error_from_another_thread():
    batches_read = []

    def batches(seek_ts=None):
        for index in range(5):
            batches_read.append(index)
            yield torch.zeros((2, 3, 8, 8), dtype=torch.uint8), [2 * index, 2 * index + 1]

    failure = RuntimeError("boom")
    _, errors, _ = _decode(_reader(batches), error_holder=[failure])

    assert errors == [failure, failure]
    assert batches_read == [0]


def test_primary_loop_exits_on_cancel_and_releases_the_secondary_thread():
    cancel = threading.Event()
    cancel.set()
    secondary_queue = FrameQueue(max_frames=999)
    errors = []

    with patch("jasna.pipeline_threads.torch.cuda.set_device"):
        primary_restore_loop(
            device=torch.device("cpu"),
            restoration_pipeline=MagicMock(),
            clip_queue=FrameQueue(max_frames=999),
            secondary_queue=secondary_queue,
            error_holder=errors,
            primary_idle_event=threading.Event(),
            cancel_event=cancel,
            debug_memory=MagicMock(),
        )

    assert not errors
    assert secondary_queue.get() is _SENTINEL


def test_primary_loop_logs_its_timing_summary(caplog):
    clip_queue = FrameQueue(max_frames=999)
    clip_queue.put(_SENTINEL)

    with (
        patch("jasna.pipeline_threads.torch.cuda.set_device"),
        caplog.at_level(logging.INFO, logger="jasna.pipeline_threads"),
    ):
        primary_restore_loop(
            device=torch.device("cpu"),
            restoration_pipeline=MagicMock(),
            clip_queue=clip_queue,
            secondary_queue=FrameQueue(max_frames=999),
            error_holder=[],
            primary_idle_event=threading.Event(),
            cancel_event=threading.Event(),
            debug_memory=MagicMock(),
        )

    assert "[timing] primary" in caplog.text


def test_secondary_loop_exits_on_cancel_and_releases_the_encode_thread():
    cancel = threading.Event()
    cancel.set()
    encode_queue = FrameQueue(max_frames=999)
    errors = []

    with patch("jasna.pipeline_threads.torch.cuda.set_device"):
        secondary_restore_loop(
            device=torch.device("cpu"),
            restoration_pipeline=MagicMock(),
            secondary_queue=FrameQueue(max_frames=999),
            encode_queue=encode_queue,
            error_holder=errors,
            cancel_event=cancel,
            debug_memory=MagicMock(),
        )

    assert not errors
    assert encode_queue.get() is _SENTINEL


def _blend_encode(*, pending_tracks=None, encode_items=(), error_holder=None, cancel_event=None):
    """Run ``blend_encode_loop`` over one frame that waits on ``pending_tracks``; returns what it wrote."""
    blend_buffer = BlendBuffer(device=torch.device("cpu"))
    if pending_tracks is not None:
        blend_buffer.register_frame(0, pending_tracks)
    encode_queue = FrameQueue(max_frames=999)
    for item in encode_items:
        encode_queue.put(item)
    metadata_queue = Queue(maxsize=999)
    metadata_queue.put(FrameMeta(frame_idx=0, pts=0))
    metadata_queue.put(_SENTINEL)
    writer = _RecordingWriter()
    error_holder = [] if error_holder is None else error_holder

    with (
        patch("jasna.pipeline_threads.VideoReader", return_value=_reader([(torch.zeros((1, 3, 8, 8), dtype=torch.uint8), [0])])),
        patch("jasna.pipeline_threads.torch.cuda.set_device"),
    ):
        blend_encode_loop(
            input_video="fake.mkv",
            batch_size=1,
            device=torch.device("cpu"),
            metadata=video_metadata(),
            blend_buffer=blend_buffer,
            encode_queue=encode_queue,
            metadata_queue=metadata_queue,
            error_holder=error_holder,
            frame_writer=writer,
            cancel_event=cancel_event or threading.Event(),
            vram_offloader=MagicMock(),
            seek_ts=None,
            frame_stride=1,
        )
    return writer.written, error_holder


def test_blend_encode_writes_nothing_once_cancelled():
    cancel = threading.Event()
    cancel.set()

    written, errors = _blend_encode(cancel_event=cancel)

    assert written == [] and errors == []


def test_blend_encode_stops_waiting_for_a_restored_frame_when_another_thread_failed():
    failure = RuntimeError("downstream boom")

    written, errors = _blend_encode(pending_tracks={99}, error_holder=[failure])

    assert written == []
    assert errors == [failure, failure]


def test_blend_encode_writes_the_source_frame_when_secondary_ends_without_its_restoration():
    written, errors = _blend_encode(pending_tracks={99}, encode_items=[_SENTINEL])

    assert [pts for _, pts in written] == [0]
    assert errors == []


def test_offline_frame_writer_opens_the_encoder_on_first_write_and_closes_it_once():
    encoder = MagicMock()
    heartbeat = [0.0]
    writer = _OfflineFrameWriter(encoder, heartbeat)
    writer.close()
    assert encoder.__enter__.call_count == 0 and encoder.__exit__.call_count == 0

    writer.write(torch.zeros(3, 8, 8), pts=10)
    writer.write(torch.zeros(3, 8, 8), pts=20)
    writer.close()
    writer.close()

    assert [call.args[1] for call in encoder.encode.call_args_list] == [10, 20]
    assert (encoder.__enter__.call_count, encoder.__exit__.call_count) == (1, 1)
    assert heartbeat[0] > 0


@pytest.mark.parametrize(("frames_written", "segment"), [(1, 3), (9, 3), (10, 4), (25, 5), (100, 13)])
def test_streaming_frame_writer_reports_the_segment_being_produced(monkeypatch, frames_written, segment):
    monkeypatch.setattr("jasna.streaming_pipeline.time.monotonic", lambda: 50.0)
    encoder = MagicMock(frames_per_segment=10)
    server = MagicMock()
    stop = threading.Event()

    _StreamingFrameWriter(encoder, server, start_segment=3).after_write(frames_written, stop)

    server.update_production.assert_called_once_with(segment)
    server.wait_for_demand.assert_called_once_with(segment, 3, stop)


def test_streaming_frame_writer_raises_an_encoder_failure_before_reporting_progress():
    encoder = MagicMock(frames_per_segment=120)
    encoder.raise_if_failed.side_effect = RuntimeError("writer failed")
    server = MagicMock()

    with pytest.raises(RuntimeError, match="writer failed"):
        _StreamingFrameWriter(encoder, server, start_segment=5).after_write(1, threading.Event())

    server.update_production.assert_not_called()


def _streaming_pipeline(metadata):
    pipeline = make_pipeline(vr_mode="off", batch_size=2, max_detection_gap=0, min_detection_duration=0)
    pipeline.configure_vr(metadata)
    return pipeline


def _streaming_server() -> MagicMock:
    server = MagicMock()
    server.video_change = threading.Event()
    server.consume_seek_for_pass.return_value = None
    return server


def test_streaming_pass_worker_failure_stops_a_writer_waiting_for_player_demand():
    demand_entered = threading.Event()
    demand_stopped = []
    failure = RuntimeError("decode failed")
    cancel = threading.Event()
    server = _streaming_server()

    def wait_for_demand(segment, max_ahead, stop_event):
        demand_entered.set()
        demand_stopped.append(stop_event.wait(timeout=2))

    def fail_decode(**kwargs):
        kwargs["metadata_queue"].put(FrameMeta(0, 0, apply_effect=False))
        assert demand_entered.wait(timeout=2)
        kwargs["error_holder"].append(failure)

    def wait_for_stop(**kwargs):
        assert kwargs["cancel_event"].wait(timeout=3)

    server.wait_for_demand.side_effect = wait_for_demand
    metadata = video_metadata(num_frames=1)
    with (
        patch("jasna.pipeline_threads.VideoReader", return_value=_reader([(torch.zeros(1, 3, 8, 8), [0])])),
        patch("jasna.pipeline_threads.torch.cuda.set_device"),
        patch("jasna.pipeline_threads.decode_detect_loop", side_effect=fail_decode),
        patch("jasna.pipeline_threads.primary_restore_loop", side_effect=wait_for_stop),
        patch("jasna.pipeline_threads.secondary_restore_loop", side_effect=wait_for_stop),
        patch("jasna.pipeline_threads.VramOffloader"),
        pytest.raises(RuntimeError, match="decode failed") as caught,
    ):
        _run_streaming_pass(
            pipeline=_streaming_pipeline(metadata),
            metadata=metadata,
            hls_server=server,
            streaming_encoder=MagicMock(frames_per_segment=1),
            start_segment=0,
            start_time=0.0,
            cancel_event=cancel,
        )

    assert demand_stopped == [True]
    assert caught.value is failure
    assert not cancel.is_set()


@pytest.mark.parametrize("seek", [None, 10], ids=["completes", "seeks"])
def test_streaming_pass_returns_the_segment_the_player_seeked_to(seek):
    stall = threading.Event()

    def batches(seek_ts=None):
        yield torch.zeros((2, 3, 8, 8), dtype=torch.uint8), [0, 1]
        if seek is not None:
            stall.wait(timeout=5.0)

    metadata = video_metadata(num_frames=2 if seek is None else 200)
    encoder = MagicMock(frames_per_segment=120)
    server = _streaming_server()
    server.consume_seek_for_pass.side_effect = lambda start_segment: (stall.set(), seek)[1]
    with (
        patch("jasna.pipeline_threads.VideoReader", return_value=_reader(batches)),
        patch("jasna.pipeline_threads.torch.cuda.set_device"),
        patch("jasna.pipeline_threads.process_frame_batch", side_effect=_register_frames),
        patch("jasna.pipeline_threads.finalize_processing"),
        patch("jasna.pipeline_threads.VramOffloader"),
    ):
        result = _run_streaming_pass(
            pipeline=_streaming_pipeline(metadata),
            metadata=metadata,
            hls_server=server,
            streaming_encoder=encoder,
            start_segment=0,
            start_time=0.0,
            cancel_event=threading.Event(),
        )

    assert result == seek
    if seek is None:
        assert [call.args[1] for call in encoder.write_frame.call_args_list] == [0, 1]


def test_run_streaming_uses_the_given_server_without_starting_or_stopping_it():
    pipeline = make_pipeline(vr_mode="off")
    server = MagicMock(segments_dir="segments", segment_duration=2.0)

    with (
        patch("jasna.streaming_pipeline.get_video_meta_data", return_value=video_metadata()),
        patch("jasna.streaming_pipeline.StreamingEncoder") as encoder_cls,
        patch("jasna.streaming_pipeline._streaming_loop") as loop,
    ):
        run_streaming(pipeline, server)

    server.load_video.assert_called_once()
    loop.assert_called_once()
    encoder_cls.return_value.stop.assert_called_once()
    server.start.assert_not_called()
    server.stop.assert_not_called()


def _streaming_loop_with(passes, *, initial_start_segment=0, consume_seek=lambda: None):
    """Run ``_streaming_loop`` where each pass returns the next value of ``passes`` and the
    video changes after the last; returns the server and encoder mocks."""
    server = MagicMock(initial_start_segment=initial_start_segment)
    server.video_change = threading.Event()
    server.segment_start_time.return_value = 0.0
    server.consume_seek.side_effect = consume_seek
    encoder = MagicMock()
    results = iter(passes)
    start_segments = []

    def run_pass(**kwargs):
        start_segments.append(kwargs["start_segment"])
        result = next(results, None)
        if len(start_segments) == len(passes):
            server.video_change.set()
        return result

    with patch("jasna.streaming_pipeline._run_streaming_pass", side_effect=run_pass):
        _streaming_loop(pipeline=MagicMock(), metadata=MagicMock(), hls_server=server, streaming_encoder=encoder)
    return server, encoder, start_segments


def test_streaming_loop_starts_at_the_requested_segment():
    _, encoder, start_segments = _streaming_loop_with([None], initial_start_segment=10)

    assert start_segments == [10]
    encoder.start.assert_called_once_with(start_number=10)


@pytest.mark.parametrize("target", [5, 0])
def test_streaming_loop_restarts_the_encoder_at_a_seek_target(target):
    _, encoder, start_segments = _streaming_loop_with([target, None])

    assert start_segments == [0, target]
    encoder.start.assert_called_once_with(start_number=0)
    encoder.flush_and_restart.assert_called_once_with(start_number=target)


def test_streaming_loop_finishes_the_stream_and_restarts_on_a_later_seek():
    seeks = iter([None, 3])
    server, encoder, start_segments = _streaming_loop_with([None, None], consume_seek=lambda: next(seeks, None))

    assert start_segments == [0, 3]
    server.mark_finished.assert_called_once()
    encoder.stop.assert_called_once()


def test_streaming_loop_finishes_the_stream_when_the_video_changes_after_completion():
    server = MagicMock(initial_start_segment=0)
    server.video_change = threading.Event()
    server.consume_seek.side_effect = lambda: server.video_change.set()

    with patch("jasna.streaming_pipeline._run_streaming_pass", return_value=None):
        _streaming_loop(pipeline=MagicMock(), metadata=MagicMock(), hls_server=server, streaming_encoder=MagicMock())

    server.mark_finished.assert_called_once()
