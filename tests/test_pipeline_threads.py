"""Tests for pipeline_threads.py shared thread functions and related frame writers."""
from __future__ import annotations

import threading
import time
from fractions import Fraction
from queue import Queue, Empty
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.blend_buffer import BlendBuffer
from jasna.crop_buffer import CropBuffer, RawCrop
from jasna.frame_queue import FrameQueue
from jasna.media import VideoMetadata
from jasna.pipeline_items import ClipRestoreItem, FrameMeta, PrimaryRestoreResult, SecondaryRestoreResult, _SENTINEL
from jasna.pipeline_threads import (
    FrameWriter,
    decode_detect_loop,
    primary_restore_loop,
    secondary_restore_loop,
    blend_encode_loop,
    _estimate_start_frame,
)
from jasna.tracking.clip_tracker import TrackedClip


def _fake_metadata(num_frames=4, fps=24.0) -> VideoMetadata:
    return VideoMetadata(
        video_file="fake.mkv",
        num_frames=num_frames,
        video_fps=fps,
        average_fps=fps,
        video_fps_exact=Fraction(24, 1),
        codec_name="hevc",
        duration=num_frames / fps,
        video_width=8,
        video_height=8,
        time_base=Fraction(1, 24),
        start_pts=0,
        color_space=AvColorspace.ITU709,
        color_range=AvColorRange.MPEG,
        is_10bit=True,
    )


def _mock_reader(batches, seek_ts_check=None):
    r = MagicMock()
    r.__enter__ = MagicMock(return_value=r)
    r.__exit__ = MagicMock(return_value=False)
    def _frames(seek_ts=None):
        if seek_ts_check is not None:
            seek_ts_check(seek_ts)
        return iter(batches)
    r.frames = _frames
    return r


class _RecordingWriter:
    def __init__(self):
        self.written: list[tuple[torch.Tensor, int]] = []
        self.after_write_calls: list[int] = []

    def write(self, frame: torch.Tensor, pts: int) -> None:
        self.written.append((frame, pts))

    def after_write(self, frames_written: int) -> None:
        self.after_write_calls.append(frames_written)


# ---------------------------------------------------------------------------
# _estimate_start_frame
# ---------------------------------------------------------------------------

class TestEstimateStartFrame:
    def test_basic(self):
        meta = _fake_metadata(fps=30.0)
        assert _estimate_start_frame(meta, 2.0) == 60

    def test_zero(self):
        meta = _fake_metadata(fps=24.0)
        assert _estimate_start_frame(meta, 0.0) == 0


# ---------------------------------------------------------------------------
# decode_detect_loop — cancel_event & seek_ts paths
# ---------------------------------------------------------------------------

class TestDecodeDetectLoop:
    def test_cancel_event_breaks_loop(self):
        cancel = threading.Event()
        frames_t = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)

        call_count = 0
        def _batches(seek_ts=None):
            nonlocal call_count
            for _ in range(10):
                call_count += 1
                if call_count >= 2:
                    cancel.set()
                yield frames_t, [call_count * 2 - 2, call_count * 2 - 1]

        reader = MagicMock()
        reader.__enter__ = MagicMock(return_value=reader)
        reader.__exit__ = MagicMock(return_value=False)
        reader.frames = _batches

        clip_queue = FrameQueue(max_frames=999)
        metadata_queue = Queue(maxsize=999)
        error_holder = []
        frame_shape = []

        from jasna.pipeline_processing import BatchProcessResult

        with (
            patch("jasna.pipeline_threads.NvidiaVideoReader", return_value=reader),
            patch("jasna.pipeline_threads.torch.cuda.set_device"),
            patch("jasna.pipeline_threads.torch.inference_mode", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))),
            patch("jasna.pipeline_threads.process_frame_batch", return_value=BatchProcessResult(next_frame_idx=2, clips_emitted=0)),
            patch("jasna.pipeline_threads.finalize_processing"),
        ):
            decode_detect_loop(
                input_video="fake.mkv",
                batch_size=2,
                device=torch.device("cpu"),
                metadata=_fake_metadata(num_frames=20),
                detection_model=MagicMock(),
                max_clip_size=60,
                temporal_overlap=8,
                enable_crossfade=True,
                blend_buffer=BlendBuffer(device=torch.device("cpu")),
                crop_buffers={},
                clip_queue=clip_queue,
                metadata_queue=metadata_queue,
                error_holder=error_holder,
                frame_shape=frame_shape,
                cancel_event=cancel,
            )

        assert not error_holder
        assert call_count < 10

    def test_seek_ts_sets_start_frame(self):
        frames_t = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)
        received_seek = []
        reader = _mock_reader([(frames_t, [0, 1])], seek_ts_check=lambda st: received_seek.append(st))

        clip_queue = FrameQueue(max_frames=999)
        metadata_queue = Queue(maxsize=999)
        frame_shape = []

        from jasna.pipeline_processing import BatchProcessResult

        with (
            patch("jasna.pipeline_threads.NvidiaVideoReader", return_value=reader),
            patch("jasna.pipeline_threads.torch.cuda.set_device"),
            patch("jasna.pipeline_threads.torch.inference_mode", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))),
            patch("jasna.pipeline_threads.process_frame_batch", return_value=BatchProcessResult(next_frame_idx=50, clips_emitted=0)) as mock_pfb,
            patch("jasna.pipeline_threads.finalize_processing"),
        ):
            decode_detect_loop(
                input_video="fake.mkv",
                batch_size=2,
                device=torch.device("cpu"),
                metadata=_fake_metadata(num_frames=100, fps=24.0),
                detection_model=MagicMock(),
                max_clip_size=60,
                temporal_overlap=8,
                enable_crossfade=True,
                blend_buffer=BlendBuffer(device=torch.device("cpu")),
                crop_buffers={},
                clip_queue=clip_queue,
                metadata_queue=metadata_queue,
                error_holder=[],
                frame_shape=frame_shape,
                seek_ts=2.0,
            )

        assert received_seek == [2.0]
        call_kwargs = mock_pfb.call_args.kwargs
        assert call_kwargs["start_frame_idx"] == 48

    def test_error_holder_raises_in_loop(self):
        frames_t = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)

        call_count = 0
        def _batches(seek_ts=None):
            nonlocal call_count
            for _ in range(5):
                call_count += 1
                yield frames_t, [call_count * 2 - 2, call_count * 2 - 1]

        reader = MagicMock()
        reader.__enter__ = MagicMock(return_value=reader)
        reader.__exit__ = MagicMock(return_value=False)
        reader.frames = _batches

        clip_queue = FrameQueue(max_frames=999)
        metadata_queue = Queue(maxsize=999)
        error_holder = [RuntimeError("boom")]

        from jasna.pipeline_processing import BatchProcessResult

        with (
            patch("jasna.pipeline_threads.NvidiaVideoReader", return_value=reader),
            patch("jasna.pipeline_threads.torch.cuda.set_device"),
            patch("jasna.pipeline_threads.torch.inference_mode", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))),
            patch("jasna.pipeline_threads.process_frame_batch", return_value=BatchProcessResult(next_frame_idx=2, clips_emitted=0)),
        ):
            decode_detect_loop(
                input_video="fake.mkv",
                batch_size=2,
                device=torch.device("cpu"),
                metadata=_fake_metadata(num_frames=20),
                detection_model=MagicMock(),
                max_clip_size=60,
                temporal_overlap=8,
                enable_crossfade=True,
                blend_buffer=BlendBuffer(device=torch.device("cpu")),
                crop_buffers={},
                clip_queue=clip_queue,
                metadata_queue=metadata_queue,
                error_holder=error_holder,
                frame_shape=[],
            )

        assert len(error_holder) == 2
        assert call_count == 1


# ---------------------------------------------------------------------------
# primary_restore_loop — cancel_event path & secondary_prefers_cpu_input
# ---------------------------------------------------------------------------

class TestPrimaryRestoreLoop:
    def test_cancel_event_stops_loop(self):
        cancel = threading.Event()
        clip_queue = FrameQueue(max_frames=999)
        secondary_queue = FrameQueue(max_frames=999)
        error_holder = []
        primary_idle = threading.Event()

        def _set_cancel_later():
            time.sleep(0.15)
            cancel.set()

        t = threading.Thread(target=_set_cancel_later, daemon=True)
        t.start()

        with patch("jasna.pipeline_threads.torch.cuda.set_device"):
            primary_restore_loop(
                device=torch.device("cpu"),
                restoration_pipeline=MagicMock(),
                clip_queue=clip_queue,
                secondary_queue=secondary_queue,
                error_holder=error_holder,
                primary_idle_event=primary_idle,
                cancel_event=cancel,
            )

        t.join(timeout=3)
        assert not error_holder
        assert secondary_queue.get() is _SENTINEL

    def test_secondary_prefers_cpu_input(self):
        clip_queue = FrameQueue(max_frames=999)
        secondary_queue = FrameQueue(max_frames=999)
        error_holder = []
        primary_idle = threading.Event()

        clip = TrackedClip(
            track_id=1, start_frame=0, mask_resolution=(2, 2),
            bboxes=[np.array([1, 1, 5, 5], dtype=np.float32)] * 2,
            masks=[torch.zeros((2, 2), dtype=torch.bool)] * 2,
        )
        raw_crops = [
            RawCrop(crop=torch.zeros(3, 4, 4, dtype=torch.uint8), enlarged_bbox=(1, 1, 5, 5), crop_shape=(4, 4))
            for _ in range(2)
        ]
        clip_queue.put(ClipRestoreItem(
            clip=clip, raw_crops=raw_crops, frame_shape=(8, 8),
            keep_start=0, keep_end=2, crossfade_weights=None,
        ))
        clip_queue.put(_SENTINEL)

        mock_pipeline = MagicMock()
        mock_pipeline.secondary_prefers_cpu_input = True
        pr_result = MagicMock()
        pr_result.keep_end = 2
        pr_result.keep_start = 0
        pr_result.primary_raw = torch.zeros(2, 3, 8, 8)
        mock_pipeline.prepare_and_run_primary.return_value = pr_result

        with patch("jasna.pipeline_threads.torch.cuda.set_device"):
            primary_restore_loop(
                device=torch.device("cpu"),
                restoration_pipeline=mock_pipeline,
                clip_queue=clip_queue,
                secondary_queue=secondary_queue,
                error_holder=error_holder,
                primary_idle_event=primary_idle,
            )

        assert not error_holder
        item = secondary_queue.get()
        assert item is not _SENTINEL
        assert item.primary_raw.device.type == "cpu"


# ---------------------------------------------------------------------------
# secondary_restore_loop — cancel_event path
# ---------------------------------------------------------------------------

class TestSecondaryRestoreLoop:
    def test_cancel_event_stops_loop(self):
        cancel = threading.Event()
        secondary_queue = FrameQueue(max_frames=999)
        encode_queue = FrameQueue(max_frames=999)
        error_holder = []

        def _set_cancel_later():
            time.sleep(0.15)
            cancel.set()

        t = threading.Thread(target=_set_cancel_later, daemon=True)
        t.start()

        with patch("jasna.pipeline_threads.torch.cuda.set_device"):
            secondary_restore_loop(
                device=torch.device("cpu"),
                restoration_pipeline=MagicMock(),
                secondary_queue=secondary_queue,
                encode_queue=encode_queue,
                error_holder=error_holder,
                cancel_event=cancel,
            )

        t.join(timeout=3)
        assert not error_holder
        assert encode_queue.get() is _SENTINEL


# ---------------------------------------------------------------------------
# blend_encode_loop — cancel_event, error_holder, vram_offloader
# ---------------------------------------------------------------------------

class TestBlendEncodeLoop:
    def _run_blend_encode(self, *, cancel_event=None, error_holder=None,
                          frame_writer=None, vram_offloader=None,
                          encode_items=None, metadata_items=None,
                          seek_ts=None):
        frames_t = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)
        reader = _mock_reader([(frames_t, [0, 1])])

        blend_buffer = BlendBuffer(device=torch.device("cpu"))
        encode_queue = FrameQueue(max_frames=999)
        metadata_queue = Queue(maxsize=999)

        if metadata_items:
            for item in metadata_items:
                metadata_queue.put(item)
        if encode_items:
            for item in encode_items:
                encode_queue.put(item)

        if error_holder is None:
            error_holder = []
        if frame_writer is None:
            frame_writer = _RecordingWriter()

        with (
            patch("jasna.pipeline_threads.NvidiaVideoReader", return_value=reader),
            patch("jasna.pipeline_threads.torch.cuda.set_device"),
        ):
            blend_encode_loop(
                input_video="fake.mkv",
                batch_size=2,
                device=torch.device("cpu"),
                metadata=_fake_metadata(),
                blend_buffer=blend_buffer,
                encode_queue=encode_queue,
                metadata_queue=metadata_queue,
                error_holder=error_holder,
                frame_writer=frame_writer,
                cancel_event=cancel_event,
                seek_ts=seek_ts,
                vram_offloader=vram_offloader,
            )

        return frame_writer, error_holder

    def test_cancel_event_breaks_loop(self):
        cancel = threading.Event()
        cancel.set()
        writer, errors = self._run_blend_encode(cancel_event=cancel, metadata_items=[_SENTINEL])
        assert not errors
        assert len(writer.written) == 0

    def test_vram_offloader_pause_called(self):
        vram_offloader = MagicMock()
        self._run_blend_encode(
            metadata_items=[_SENTINEL],
            vram_offloader=vram_offloader,
        )
        vram_offloader.pause_stall_check.assert_called_once()

    def test_seek_ts_passed_to_reader(self):
        received_seek = []
        frames_t = torch.randint(0, 256, (1, 3, 8, 8), dtype=torch.uint8)
        reader = MagicMock()
        reader.__enter__ = MagicMock(return_value=reader)
        reader.__exit__ = MagicMock(return_value=False)
        def _frames(seek_ts=None):
            received_seek.append(seek_ts)
            return iter([(frames_t, [0])])
        reader.frames = _frames

        blend_buffer = BlendBuffer(device=torch.device("cpu"))
        blend_buffer.register_frame(0, set())
        encode_queue = FrameQueue(max_frames=999)
        metadata_queue = Queue(maxsize=999)
        metadata_queue.put(FrameMeta(frame_idx=0, pts=0))
        metadata_queue.put(_SENTINEL)

        with (
            patch("jasna.pipeline_threads.NvidiaVideoReader", return_value=reader),
            patch("jasna.pipeline_threads.torch.cuda.set_device"),
        ):
            blend_encode_loop(
                input_video="fake.mkv",
                batch_size=2,
                device=torch.device("cpu"),
                metadata=_fake_metadata(),
                blend_buffer=blend_buffer,
                encode_queue=encode_queue,
                metadata_queue=metadata_queue,
                error_holder=[],
                frame_writer=_RecordingWriter(),
                seek_ts=5.0,
            )

        assert received_seek == [5.0]

    def test_error_holder_propagates_in_wait_loop(self):
        blend_buffer = BlendBuffer(device=torch.device("cpu"))
        blend_buffer.register_frame(0, {99})

        frames_t = torch.randint(0, 256, (1, 3, 8, 8), dtype=torch.uint8)
        reader = _mock_reader([(frames_t, [0])])

        encode_queue = FrameQueue(max_frames=999)
        metadata_queue = Queue(maxsize=999)
        metadata_queue.put(FrameMeta(frame_idx=0, pts=0))
        metadata_queue.put(_SENTINEL)

        error_holder = []

        def _inject_error():
            time.sleep(0.15)
            error_holder.append(RuntimeError("downstream boom"))

        t = threading.Thread(target=_inject_error, daemon=True)
        t.start()

        with (
            patch("jasna.pipeline_threads.NvidiaVideoReader", return_value=reader),
            patch("jasna.pipeline_threads.torch.cuda.set_device"),
        ):
            blend_encode_loop(
                input_video="fake.mkv",
                batch_size=1,
                device=torch.device("cpu"),
                metadata=_fake_metadata(),
                blend_buffer=blend_buffer,
                encode_queue=encode_queue,
                metadata_queue=metadata_queue,
                error_holder=error_holder,
                frame_writer=_RecordingWriter(),
            )

        t.join(timeout=3)
        assert len(error_holder) >= 1

    def test_secondary_done_before_frame_ready_logs_error(self):
        blend_buffer = BlendBuffer(device=torch.device("cpu"))
        blend_buffer.register_frame(0, {99})

        frames_t = torch.randint(0, 256, (1, 3, 8, 8), dtype=torch.uint8)
        reader = _mock_reader([(frames_t, [0])])

        encode_queue = FrameQueue(max_frames=999)
        encode_queue.put(_SENTINEL)
        metadata_queue = Queue(maxsize=999)
        metadata_queue.put(FrameMeta(frame_idx=0, pts=0))
        metadata_queue.put(_SENTINEL)

        writer = _RecordingWriter()

        with (
            patch("jasna.pipeline_threads.NvidiaVideoReader", return_value=reader),
            patch("jasna.pipeline_threads.torch.cuda.set_device"),
        ):
            blend_encode_loop(
                input_video="fake.mkv",
                batch_size=1,
                device=torch.device("cpu"),
                metadata=_fake_metadata(),
                blend_buffer=blend_buffer,
                encode_queue=encode_queue,
                metadata_queue=metadata_queue,
                error_holder=[],
                frame_writer=writer,
            )

        assert len(writer.written) == 1


# ---------------------------------------------------------------------------
# _OfflineFrameWriter
# ---------------------------------------------------------------------------

class TestOfflineFrameWriter:
    def test_write_enters_ctx_once_and_encodes(self):
        from jasna.pipeline import _OfflineFrameWriter
        mock_enc = MagicMock()
        mock_enc.__enter__ = MagicMock(return_value=mock_enc)
        mock_enc.__exit__ = MagicMock(return_value=False)

        heartbeat = [0.0]
        writer = _OfflineFrameWriter(mock_enc, heartbeat)

        frame = torch.zeros(3, 8, 8)
        writer.write(frame, pts=10)
        writer.write(frame, pts=20)

        mock_enc.__enter__.assert_called_once()
        assert mock_enc.encode.call_count == 2
        assert heartbeat[0] > 0

    def test_after_write_is_noop(self):
        from jasna.pipeline import _OfflineFrameWriter
        writer = _OfflineFrameWriter(MagicMock(), [0.0])
        writer.after_write(1)

    def test_close_exits_ctx(self):
        from jasna.pipeline import _OfflineFrameWriter
        mock_enc = MagicMock()
        mock_enc.__enter__ = MagicMock(return_value=mock_enc)
        mock_enc.__exit__ = MagicMock(return_value=False)

        writer = _OfflineFrameWriter(mock_enc, [0.0])
        writer.write(torch.zeros(3, 8, 8), pts=0)
        writer.close()

        mock_enc.__exit__.assert_called_once_with(None, None, None)

    def test_close_without_write_is_noop(self):
        from jasna.pipeline import _OfflineFrameWriter
        mock_enc = MagicMock()
        writer = _OfflineFrameWriter(mock_enc, [0.0])
        writer.close()
        mock_enc.__exit__.assert_not_called()


# ---------------------------------------------------------------------------
# _StreamingFrameWriter
# ---------------------------------------------------------------------------

class TestStreamingFrameWriter:
    def test_write_delegates_to_encoder(self):
        from jasna.streaming_pipeline import _StreamingFrameWriter
        mock_enc = MagicMock()
        mock_server = MagicMock()
        mock_server.frames_per_segment.return_value = 120

        writer = _StreamingFrameWriter(mock_enc, mock_server, start_segment=0)
        frame = torch.zeros(3, 8, 8)
        writer.write(frame, pts=42)

        mock_enc.write_frame.assert_called_once_with(frame, 42)

    def test_after_write_updates_production_and_waits(self):
        from jasna.streaming_pipeline import _StreamingFrameWriter
        mock_enc = MagicMock()
        mock_server = MagicMock()
        mock_server.frames_per_segment.return_value = 120

        cancel = threading.Event()
        writer = _StreamingFrameWriter(mock_enc, mock_server, start_segment=5)
        writer.set_cancel_event(cancel)

        writer.after_write(1)
        mock_server.update_production.assert_called_once_with(5)
        mock_server.wait_for_demand.assert_called_once()

    def test_after_write_segment_calculation(self):
        from jasna.streaming_pipeline import _StreamingFrameWriter
        mock_enc = MagicMock()
        mock_server = MagicMock()
        mock_server.frames_per_segment.return_value = 10

        writer = _StreamingFrameWriter(mock_enc, mock_server, start_segment=3)

        writer.after_write(25)
        mock_server.update_production.assert_called_with(5)

    def test_after_write_first_frame_logging(self):
        from jasna.streaming_pipeline import _StreamingFrameWriter
        mock_enc = MagicMock()
        mock_server = MagicMock()
        mock_server.frames_per_segment.return_value = 120

        writer = _StreamingFrameWriter(mock_enc, mock_server, start_segment=0)
        writer.after_write(1)

        mock_server.update_production.assert_called_once()

    def test_after_write_100th_frame_logging(self):
        from jasna.streaming_pipeline import _StreamingFrameWriter
        mock_enc = MagicMock()
        mock_server = MagicMock()
        mock_server.frames_per_segment.return_value = 120

        writer = _StreamingFrameWriter(mock_enc, mock_server, start_segment=0)
        writer.after_write(100)

        mock_server.update_production.assert_called_once()


# ---------------------------------------------------------------------------
# streaming_pipeline._run_streaming_pass
# ---------------------------------------------------------------------------

class TestRunStreamingPass:
    def test_pass_completes_normally(self):
        from jasna.streaming_pipeline import _run_streaming_pass
        from jasna.pipeline_processing import BatchProcessResult

        frames_t = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)
        reader = _mock_reader([(frames_t, [0, 1])])
        reader_cls = MagicMock(return_value=reader)

        mock_pipeline = MagicMock()
        mock_pipeline.max_clip_size = 60
        mock_pipeline.temporal_overlap = 8
        mock_pipeline.enable_crossfade = True
        mock_pipeline.batch_size = 2
        mock_pipeline.input_video = "fake.mkv"
        mock_pipeline.restoration_pipeline = MagicMock()
        mock_pipeline.restoration_pipeline.secondary_num_workers = 1
        mock_pipeline.restoration_pipeline.secondary_prefers_cpu_input = False
        mock_pipeline.detection_model = MagicMock()

        def fake_pfb(**kwargs):
            bb = kwargs["blend_buffer"]
            mq = kwargs["metadata_queue"]
            pts_list = kwargs["pts_list"]
            start_idx = kwargs["start_frame_idx"]
            for i, pts in enumerate(pts_list):
                bb.register_frame(start_idx + i, set())
                mq.put(FrameMeta(frame_idx=start_idx + i, pts=int(pts)))
            return BatchProcessResult(next_frame_idx=start_idx + len(pts_list), clips_emitted=0)

        mock_server = MagicMock()
        mock_server.video_change = threading.Event()
        mock_server.consume_seek.return_value = None
        mock_server.frames_per_segment.return_value = 120
        mock_enc = MagicMock()
        cancel = threading.Event()

        with (
            patch("jasna.pipeline_threads.NvidiaVideoReader", reader_cls),
            patch("jasna.pipeline_threads.torch.cuda.set_device"),
            patch("jasna.pipeline_threads.torch.inference_mode", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))),
            patch("jasna.pipeline_threads.process_frame_batch", side_effect=fake_pfb),
            patch("jasna.pipeline_threads.finalize_processing"),
            patch("jasna.streaming_pipeline.VramOffloader"),
        ):
            result = _run_streaming_pass(
                pipeline=mock_pipeline,
                device=torch.device("cpu"),
                metadata=_fake_metadata(),
                hls_server=mock_server,
                streaming_encoder=mock_enc,
                start_segment=0,
                start_frame=0,
                start_time=0.0,
                cancel_event=cancel,
            )

        assert result is None

    def test_seek_during_pass(self):
        from jasna.streaming_pipeline import _run_streaming_pass
        from jasna.pipeline_processing import BatchProcessResult

        frames_t = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)
        stall = threading.Event()

        def _batches(seek_ts=None):
            yield frames_t, [0, 1]
            stall.wait(timeout=5.0)

        reader = MagicMock()
        reader.__enter__ = MagicMock(return_value=reader)
        reader.__exit__ = MagicMock(return_value=False)
        reader.frames = _batches
        reader_cls = MagicMock(return_value=reader)

        mock_pipeline = MagicMock()
        mock_pipeline.max_clip_size = 60
        mock_pipeline.temporal_overlap = 8
        mock_pipeline.enable_crossfade = True
        mock_pipeline.batch_size = 2
        mock_pipeline.input_video = "fake.mkv"
        mock_pipeline.restoration_pipeline = MagicMock()
        mock_pipeline.restoration_pipeline.secondary_num_workers = 1
        mock_pipeline.restoration_pipeline.secondary_prefers_cpu_input = False
        mock_pipeline.detection_model = MagicMock()

        def fake_pfb(**kwargs):
            bb = kwargs["blend_buffer"]
            mq = kwargs["metadata_queue"]
            pts_list = kwargs["pts_list"]
            start_idx = kwargs["start_frame_idx"]
            for i, pts in enumerate(pts_list):
                bb.register_frame(start_idx + i, set())
                mq.put(FrameMeta(frame_idx=start_idx + i, pts=int(pts)))
            return BatchProcessResult(next_frame_idx=start_idx + len(pts_list), clips_emitted=0)

        mock_server = MagicMock()
        mock_server.video_change = threading.Event()
        mock_server.consume_seek.return_value = 10
        mock_server.frames_per_segment.return_value = 120
        mock_enc = MagicMock()
        cancel = threading.Event()

        with (
            patch("jasna.pipeline_threads.NvidiaVideoReader", reader_cls),
            patch("jasna.pipeline_threads.torch.cuda.set_device"),
            patch("jasna.pipeline_threads.torch.inference_mode", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))),
            patch("jasna.pipeline_threads.process_frame_batch", side_effect=fake_pfb),
            patch("jasna.pipeline_threads.finalize_processing"),
            patch("jasna.streaming_pipeline.VramOffloader"),
        ):
            result = _run_streaming_pass(
                pipeline=mock_pipeline,
                device=torch.device("cpu"),
                metadata=_fake_metadata(num_frames=200),
                hls_server=mock_server,
                streaming_encoder=mock_enc,
                start_segment=0,
                start_frame=0,
                start_time=0.0,
                cancel_event=cancel,
            )

        assert result == 10


# ---------------------------------------------------------------------------
# streaming_pipeline.run_streaming
# ---------------------------------------------------------------------------

class TestRunStreaming:
    def test_run_streaming_creates_server_when_none(self):
        from jasna.streaming_pipeline import run_streaming

        mock_pipeline = MagicMock()
        mock_pipeline.device = torch.device("cpu")
        mock_pipeline.input_video = MagicMock()
        mock_pipeline.input_video.name = "fake.mkv"

        meta = _fake_metadata()
        meta.color_space = AvColorspace.ITU709

        with (
            patch("jasna.streaming_pipeline.get_video_meta_data", return_value=meta),
            patch("jasna.streaming_pipeline.HlsStreamingServer") as mock_server_cls,
            patch("jasna.streaming_pipeline.StreamingEncoder") as mock_enc_cls,
            patch("jasna.streaming_pipeline._streaming_loop") as mock_loop,
            patch("jasna.streaming_pipeline.torch.cuda.empty_cache"),
            patch("jasna.streaming_pipeline.torch.cuda.ipc_collect"),
            patch("jasna.streaming_pipeline.torch.cuda.reset_peak_memory_stats"),
        ):
            server_inst = mock_server_cls.return_value
            server_inst.segments_dir = "/tmp/segs"
            server_inst.start.return_value = "http://localhost:8765/stream.m3u8"

            run_streaming(mock_pipeline, port=8765, segment_duration=4.0)

            mock_server_cls.assert_called_once_with(segment_duration=4.0, port=8765, max_segments_ahead=3)
            server_inst.load_video.assert_called_once_with(meta)
            server_inst.start.assert_called_once()
            mock_loop.assert_called_once()
            mock_enc_cls.return_value.stop.assert_called_once()
            server_inst.stop.assert_called_once()

    def test_run_streaming_uses_provided_server(self):
        from jasna.streaming_pipeline import run_streaming

        mock_pipeline = MagicMock()
        mock_pipeline.device = torch.device("cpu")
        mock_pipeline.input_video = MagicMock()
        mock_pipeline.input_video.name = "fake.mkv"

        meta = _fake_metadata()
        meta.color_space = AvColorspace.ITU709

        mock_server = MagicMock()
        mock_server.segments_dir = "/tmp/segs"

        with (
            patch("jasna.streaming_pipeline.get_video_meta_data", return_value=meta),
            patch("jasna.streaming_pipeline.StreamingEncoder") as mock_enc_cls,
            patch("jasna.streaming_pipeline._streaming_loop"),
            patch("jasna.streaming_pipeline.torch.cuda.empty_cache"),
            patch("jasna.streaming_pipeline.torch.cuda.ipc_collect"),
            patch("jasna.streaming_pipeline.torch.cuda.reset_peak_memory_stats"),
        ):
            run_streaming(mock_pipeline, hls_server=mock_server)

            mock_server.load_video.assert_called_once_with(meta)
            mock_server.start.assert_not_called()
            mock_server.stop.assert_not_called()


# ---------------------------------------------------------------------------
# _streaming_loop
# ---------------------------------------------------------------------------

class TestStreamingLoop:
    def _make_mocks(self):
        mock_server = MagicMock()
        mock_server.video_change = threading.Event()
        mock_server.segment_start_time.return_value = 0.0
        mock_server.segment_start_frame.return_value = 0
        mock_server.consume_seek.return_value = None
        mock_enc = MagicMock()
        mock_pipeline = MagicMock()
        return mock_pipeline, mock_server, mock_enc

    def test_video_change_during_pass(self):
        from jasna.streaming_pipeline import _streaming_loop
        pipeline, server, enc = self._make_mocks()

        def _fake_pass(**kwargs):
            server.video_change.set()
            return None

        with patch("jasna.streaming_pipeline._run_streaming_pass", side_effect=_fake_pass):
            _streaming_loop(
                pipeline=pipeline,
                device=torch.device("cpu"),
                metadata=MagicMock(),
                hls_server=server,
                streaming_encoder=enc,
            )

        enc.start.assert_called_once_with(start_number=0)

    def test_seek_during_pass_loops(self):
        from jasna.streaming_pipeline import _streaming_loop
        pipeline, server, enc = self._make_mocks()

        call_count = [0]
        def _fake_pass(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return 5
            server.video_change.set()
            return None

        server.segment_start_time.return_value = 20.0
        server.segment_start_frame.return_value = 600

        with patch("jasna.streaming_pipeline._run_streaming_pass", side_effect=_fake_pass):
            _streaming_loop(
                pipeline=pipeline,
                device=torch.device("cpu"),
                metadata=MagicMock(),
                hls_server=server,
                streaming_encoder=enc,
            )

        assert call_count[0] == 2
        enc.flush_and_restart.assert_called_once_with(start_number=5)

    def test_completion_then_seek(self):
        from jasna.streaming_pipeline import _streaming_loop
        pipeline, server, enc = self._make_mocks()

        call_count = [0]
        def _fake_pass(**kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                server.video_change.set()
            return None

        seek_calls = [0]
        def _consume_seek():
            seek_calls[0] += 1
            if seek_calls[0] >= 2:
                return 3
            return None
        server.consume_seek.side_effect = _consume_seek

        with patch("jasna.streaming_pipeline._run_streaming_pass", side_effect=_fake_pass):
            _streaming_loop(
                pipeline=pipeline,
                device=torch.device("cpu"),
                metadata=MagicMock(),
                hls_server=server,
                streaming_encoder=enc,
            )

        assert call_count[0] == 2
        server.mark_finished.assert_called_once()
        enc.stop.assert_called_once()

    def test_completion_then_video_change(self):
        from jasna.streaming_pipeline import _streaming_loop
        pipeline, server, enc = self._make_mocks()

        def _fake_pass(**kwargs):
            return None

        def _consume_seek():
            server.video_change.set()
            return None
        server.consume_seek.side_effect = _consume_seek

        with patch("jasna.streaming_pipeline._run_streaming_pass", side_effect=_fake_pass):
            _streaming_loop(
                pipeline=pipeline,
                device=torch.device("cpu"),
                metadata=MagicMock(),
                hls_server=server,
                streaming_encoder=enc,
            )

        server.mark_finished.assert_called_once()


# ---------------------------------------------------------------------------
# Pipeline.run_streaming thin wrapper
# ---------------------------------------------------------------------------

class TestPipelineRunStreamingWrapper:
    def test_delegates_to_streaming_pipeline(self):
        from jasna.pipeline import Pipeline

        with (
            patch("jasna.pipeline.RfDetrMosaicDetectionModel"),
            patch("jasna.pipeline.YoloMosaicDetectionModel"),
        ):
            p = Pipeline(
                input_video=MagicMock(),
                output_video=MagicMock(),
                detection_model_name="rfdetr-v5",
                detection_model_path=MagicMock(),
                detection_score_threshold=0.25,
                restoration_pipeline=MagicMock(secondary_restorer=None, secondary_num_workers=1),
                codec="hevc",
                encoder_settings={},
                batch_size=2,
                device=torch.device("cpu"),
                max_clip_size=60,
                temporal_overlap=8,
                fp16=True,
            )

        with patch("jasna.streaming_pipeline.run_streaming") as mock_rs:
            p.run_streaming(port=9999, segment_duration=2.0, hls_server="fake_server")
            mock_rs.assert_called_once_with(p, port=9999, segment_duration=2.0, hls_server="fake_server")
