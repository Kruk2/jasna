from fractions import Fraction
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, patch, call

import numpy as np
import torch
import pytest
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.media import VideoMetadata
from jasna.pipeline import Pipeline
from jasna.pipeline_items import ClipRestoreItem, PrimaryRestoreResult, SecondaryRestoreResult, _SENTINEL
from jasna.restorer.secondary_restorer import AsyncSecondaryRestorer
from jasna.tracking.clip_tracker import TrackedClip


def _fake_metadata() -> VideoMetadata:
    return VideoMetadata(
        video_file="fake_input.mkv",
        num_frames=4,
        video_fps=24.0,
        average_fps=24.0,
        video_fps_exact=Fraction(24, 1),
        codec_name="hevc",
        duration=4.0 / 24.0,
        video_width=8,
        video_height=8,
        time_base=Fraction(1, 24),
        start_pts=0,
        color_space=AvColorspace.ITU709,
        color_range=AvColorRange.MPEG,
        is_10bit=True,
    )


def _make_pipeline():
    with (
        patch("jasna.pipeline.RfDetrMosaicDetectionModel"),
        patch("jasna.pipeline.YoloMosaicDetectionModel"),
    ):
        rest_pipeline = MagicMock()
        rest_pipeline.secondary_restorer = None
        rest_pipeline.secondary_num_workers = 1
        p = Pipeline(
            input_video=Path("in.mp4"),
            output_video=Path("out.mkv"),
            detection_model_name="rfdetr-v5",
            detection_model_path=Path("model.onnx"),
            detection_score_threshold=0.25,
            restoration_pipeline=rest_pipeline,
            codec="hevc",
            encoder_settings={},
            batch_size=2,
            device=torch.device("cuda:0"),
            max_clip_size=60,
            temporal_overlap=8,
            fp16=True,
            disable_progress=True,
        )
    return p


class TestPipelineRun:
    def test_run_no_frames(self):
        p = _make_pipeline()

        mock_reader = MagicMock()
        mock_reader.__enter__ = MagicMock(return_value=mock_reader)
        mock_reader.__exit__ = MagicMock(return_value=False)
        mock_reader.frames.return_value = iter([])

        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)

        with (
            patch("jasna.pipeline.get_video_meta_data", return_value=_fake_metadata()),
            patch("jasna.pipeline.NvidiaVideoReader", return_value=mock_reader),
            patch("jasna.pipeline.NvidiaVideoEncoder", return_value=mock_encoder),
            patch("jasna.pipeline.torch.cuda.set_device"),
            patch("jasna.pipeline.torch.inference_mode", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))),
        ):
            p.run()

        mock_encoder.encode.assert_not_called()

    def test_run_full_thread_flow(self):
        """Exercise all four thread bodies: decode->primary->secondary->encode."""
        p = _make_pipeline()

        frames_t = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)
        mock_reader = MagicMock()
        mock_reader.__enter__ = MagicMock(return_value=mock_reader)
        mock_reader.__exit__ = MagicMock(return_value=False)
        mock_reader.frames.return_value = iter([(frames_t, [0, 1])])

        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)

        clip = TrackedClip(
            track_id=42,
            start_frame=0,
            mask_resolution=(2, 2),
            bboxes=[np.array([1, 1, 5, 5], dtype=np.float32)] * 2,
            masks=[torch.zeros((2, 2), dtype=torch.bool)] * 2,
        )

        from jasna.pipeline_processing import BatchProcessResult

        def fake_process_batch(**kwargs):
            fb = kwargs["frame_buffer"]
            cq = kwargs["clip_queue"]
            fb.add_frame(0, pts=0, frame=frames_t[0], clip_track_ids={42})
            fb.add_frame(1, pts=1, frame=frames_t[1], clip_track_ids={42})
            cq.put(ClipRestoreItem(
                clip=clip,
                frames=[frames_t[0], frames_t[1]],
                keep_start=0,
                keep_end=2,
                crossfade_weights=None,
            ))
            return BatchProcessResult(next_frame_idx=2)

        pr_result = PrimaryRestoreResult(
            clip=clip,
            frame_count=2,
            frame_shape=(8, 8),
            frame_device=frames_t[0].device,
            primary_raw=torch.zeros((2, 3, 256, 256)),
            keep_start=0,
            keep_end=2,
            crossfade_weights=None,
            enlarged_bboxes=[(1, 1, 5, 5)] * 2,
            crop_shapes=[(4, 4)] * 2,
            pad_offsets=[(126, 126)] * 2,
            resize_shapes=[(4, 4)] * 2,
        )
        p.restoration_pipeline.prepare_and_run_primary.return_value = pr_result

        sr_result = SecondaryRestoreResult(
            clip=clip,
            frame_count=2,
            frame_shape=(8, 8),
            frame_device=frames_t[0].device,
            restored_frames=[torch.randint(0, 255, (3, 256, 256), dtype=torch.uint8)] * 2,
            keep_start=0,
            keep_end=2,
            crossfade_weights=None,
            enlarged_bboxes=[(1, 1, 5, 5)] * 2,
            crop_shapes=[(4, 4)] * 2,
            pad_offsets=[(126, 126)] * 2,
            resize_shapes=[(4, 4)] * 2,
        )
        restored = [torch.randint(0, 255, (3, 256, 256), dtype=torch.uint8)] * 2
        restorer = MagicMock(spec=AsyncSecondaryRestorer)
        restorer.push_clip.return_value = 0
        restorer.pop_completed.side_effect = [[], [(0, restored)], []]
        restorer.flush_all.return_value = None
        p.restoration_pipeline.secondary_restorer = restorer
        p.restoration_pipeline.build_secondary_result.return_value = sr_result

        def fake_blend(sr, fb):
            for i in range(2):
                pending = fb.frames.get(i)
                if pending:
                    pending.pending_clips.discard(42)
                yield i

        p.restoration_pipeline.blend_secondary_result.side_effect = fake_blend

        with (
            patch("jasna.pipeline.get_video_meta_data", return_value=_fake_metadata()),
            patch("jasna.pipeline.NvidiaVideoReader", return_value=mock_reader),
            patch("jasna.pipeline.NvidiaVideoEncoder", return_value=mock_encoder),
            patch("jasna.pipeline.process_frame_batch", side_effect=fake_process_batch),
            patch("jasna.pipeline.finalize_processing"),
            patch("jasna.pipeline.torch.cuda.set_device"),
            patch("jasna.pipeline.torch.inference_mode", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))),
        ):
            p.run()

        p.restoration_pipeline.prepare_and_run_primary.assert_called_once()
        p.restoration_pipeline.build_secondary_result.assert_called_once()
        assert mock_encoder.encode.call_count == 2

    def test_run_processes_frames(self):
        p = _make_pipeline()

        frames = torch.zeros((2, 3, 8, 8), dtype=torch.uint8)
        mock_reader = MagicMock()
        mock_reader.__enter__ = MagicMock(return_value=mock_reader)
        mock_reader.__exit__ = MagicMock(return_value=False)
        mock_reader.frames.return_value = iter([(frames, [0, 1])])

        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)

        from jasna.pipeline_processing import BatchProcessResult
        batch_result = BatchProcessResult(next_frame_idx=2)

        restorer = MagicMock(spec=AsyncSecondaryRestorer)
        restorer.push_clip.return_value = 0
        restorer.pop_completed.return_value = []
        restorer.flush_all.return_value = None
        p.restoration_pipeline.secondary_restorer = restorer
        p.restoration_pipeline.build_secondary_result.return_value = MagicMock(spec=SecondaryRestoreResult)

        with (
            patch("jasna.pipeline.get_video_meta_data", return_value=_fake_metadata()),
            patch("jasna.pipeline.NvidiaVideoReader", return_value=mock_reader),
            patch("jasna.pipeline.NvidiaVideoEncoder", return_value=mock_encoder),
            patch("jasna.pipeline.process_frame_batch", return_value=batch_result),
            patch("jasna.pipeline.finalize_processing"),
            patch("jasna.pipeline.torch.cuda.set_device"),
            patch("jasna.pipeline.torch.inference_mode", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))),
        ):
            p.run()

    def test_run_propagates_decode_error(self):
        p = _make_pipeline()

        mock_reader = MagicMock()
        mock_reader.__enter__ = MagicMock(return_value=mock_reader)
        mock_reader.__exit__ = MagicMock(return_value=False)
        mock_reader.frames.side_effect = RuntimeError("decode boom")

        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)

        with (
            patch("jasna.pipeline.get_video_meta_data", return_value=_fake_metadata()),
            patch("jasna.pipeline.NvidiaVideoReader", return_value=mock_reader),
            patch("jasna.pipeline.NvidiaVideoEncoder", return_value=mock_encoder),
            patch("jasna.pipeline.torch.cuda.set_device"),
            patch("jasna.pipeline.torch.inference_mode", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))),
        ):
            with pytest.raises(RuntimeError, match="decode boom"):
                p.run()

    def test_run_primary_error_propagates(self):
        """Cover lines 175-176: error in primary_restore_thread."""
        p = _make_pipeline()

        frames_t = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)
        mock_reader = MagicMock()
        mock_reader.__enter__ = MagicMock(return_value=mock_reader)
        mock_reader.__exit__ = MagicMock(return_value=False)
        mock_reader.frames.return_value = iter([(frames_t, [0, 1])])

        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)

        clip = TrackedClip(
            track_id=1, start_frame=0, mask_resolution=(2, 2),
            bboxes=[np.array([1, 1, 5, 5], dtype=np.float32)] * 2,
            masks=[torch.zeros((2, 2), dtype=torch.bool)] * 2,
        )

        from jasna.pipeline_processing import BatchProcessResult

        def fake_process_batch(**kwargs):
            fb = kwargs["frame_buffer"]
            cq = kwargs["clip_queue"]
            fb.add_frame(0, pts=0, frame=frames_t[0], clip_track_ids={1})
            fb.add_frame(1, pts=1, frame=frames_t[1], clip_track_ids={1})
            cq.put(ClipRestoreItem(clip=clip, frames=[frames_t[0], frames_t[1]], keep_start=0, keep_end=2, crossfade_weights=None))
            return BatchProcessResult(next_frame_idx=2)

        p.restoration_pipeline.prepare_and_run_primary.side_effect = RuntimeError("primary boom")

        with (
            patch("jasna.pipeline.get_video_meta_data", return_value=_fake_metadata()),
            patch("jasna.pipeline.NvidiaVideoReader", return_value=mock_reader),
            patch("jasna.pipeline.NvidiaVideoEncoder", return_value=mock_encoder),
            patch("jasna.pipeline.process_frame_batch", side_effect=fake_process_batch),
            patch("jasna.pipeline.finalize_processing"),
            patch("jasna.pipeline.torch.cuda.set_device"),
            patch("jasna.pipeline.torch.inference_mode", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))),
        ):
            with pytest.raises(RuntimeError, match="primary boom"):
                p.run()

    def test_run_secondary_error_propagates(self):
        """Cover lines 195-196: error in secondary_restore_thread."""
        p = _make_pipeline()

        frames_t = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)
        mock_reader = MagicMock()
        mock_reader.__enter__ = MagicMock(return_value=mock_reader)
        mock_reader.__exit__ = MagicMock(return_value=False)
        mock_reader.frames.return_value = iter([(frames_t, [0, 1])])

        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)

        clip = TrackedClip(
            track_id=1, start_frame=0, mask_resolution=(2, 2),
            bboxes=[np.array([1, 1, 5, 5], dtype=np.float32)] * 2,
            masks=[torch.zeros((2, 2), dtype=torch.bool)] * 2,
        )

        from jasna.pipeline_processing import BatchProcessResult

        def fake_process_batch(**kwargs):
            fb = kwargs["frame_buffer"]
            cq = kwargs["clip_queue"]
            fb.add_frame(0, pts=0, frame=frames_t[0], clip_track_ids={1})
            fb.add_frame(1, pts=1, frame=frames_t[1], clip_track_ids={1})
            cq.put(ClipRestoreItem(clip=clip, frames=[frames_t[0], frames_t[1]], keep_start=0, keep_end=2, crossfade_weights=None))
            return BatchProcessResult(next_frame_idx=2)

        pr_result = PrimaryRestoreResult(
            clip=clip, frame_count=2, frame_shape=(8, 8), frame_device=frames_t[0].device,
            primary_raw=torch.zeros((2, 3, 256, 256)),
            keep_start=0, keep_end=2, crossfade_weights=None,
            enlarged_bboxes=[(1, 1, 5, 5)] * 2, crop_shapes=[(4, 4)] * 2,
            pad_offsets=[(126, 126)] * 2, resize_shapes=[(4, 4)] * 2,
        )
        p.restoration_pipeline.prepare_and_run_primary.return_value = pr_result
        restorer = MagicMock(spec=AsyncSecondaryRestorer)
        restorer.push_clip.return_value = 0
        restorer.pop_completed.side_effect = [[], [(0, [])], []]
        restorer.flush_all.return_value = None
        p.restoration_pipeline.secondary_restorer = restorer
        p.restoration_pipeline.build_secondary_result.side_effect = RuntimeError("secondary boom")

        with (
            patch("jasna.pipeline.get_video_meta_data", return_value=_fake_metadata()),
            patch("jasna.pipeline.NvidiaVideoReader", return_value=mock_reader),
            patch("jasna.pipeline.NvidiaVideoEncoder", return_value=mock_encoder),
            patch("jasna.pipeline.process_frame_batch", side_effect=fake_process_batch),
            patch("jasna.pipeline.finalize_processing"),
            patch("jasna.pipeline.torch.cuda.set_device"),
            patch("jasna.pipeline.torch.inference_mode", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))),
        ):
            with pytest.raises(RuntimeError, match="secondary boom"):
                p.run()

    def test_run_secondary_loop(self):
        """Cover _run_secondary_loop: push_clip → flush → pop_completed → build_secondary_result."""
        p = _make_pipeline()

        clip = TrackedClip(
            track_id=1, start_frame=0, mask_resolution=(2, 2),
            bboxes=[np.array([1, 1, 5, 5], dtype=np.float32)] * 2,
            masks=[torch.zeros((2, 2), dtype=torch.bool)] * 2,
        )
        pr = PrimaryRestoreResult(
            clip=clip, frame_count=2, frame_shape=(8, 8), frame_device=torch.device("cpu"),
            primary_raw=torch.zeros((2, 3, 256, 256)),
            keep_start=0, keep_end=2, crossfade_weights=None,
            enlarged_bboxes=[(1, 1, 5, 5)] * 2, crop_shapes=[(4, 4)] * 2,
            pad_offsets=[(126, 126)] * 2, resize_shapes=[(4, 4)] * 2,
        )

        restored = [torch.randint(0, 255, (3, 256, 256), dtype=torch.uint8)] * 2
        sr_result = SecondaryRestoreResult(
            clip=clip, frame_count=pr.frame_count, frame_shape=pr.frame_shape, frame_device=pr.frame_device,
            restored_frames=restored,
            keep_start=0, keep_end=2, crossfade_weights=None,
            enlarged_bboxes=pr.enlarged_bboxes, crop_shapes=pr.crop_shapes,
            pad_offsets=pr.pad_offsets, resize_shapes=pr.resize_shapes,
        )

        restorer = MagicMock(spec=AsyncSecondaryRestorer)
        restorer.num_workers = 2
        restorer.push_clip.return_value = 0
        restorer.pop_completed.side_effect = [[], [(0, restored)], []]
        restorer.flush_all.return_value = None
        p.restoration_pipeline.secondary_restorer = restorer
        p.restoration_pipeline.build_secondary_result.return_value = sr_result

        secondary_queue: Queue = Queue()
        encode_queue: Queue = Queue()
        secondary_queue.put(pr)
        secondary_queue.put(_SENTINEL)

        p._run_secondary_loop(secondary_queue, encode_queue)

        restorer.push_clip.assert_called_once()
        assert not encode_queue.empty()
        result = encode_queue.get()
        assert result is sr_result

    def test_run_secondary_loop_idle_gap_triggers_flush_pending(self):
        """Idle timeout triggers flush_pending when has_pending is True."""
        p = _make_pipeline()
        p._ASYNC_POLL_TIMEOUT = 0.01
        p._ASYNC_FLUSH_TIMEOUT = 0.03

        clip = TrackedClip(
            track_id=1, start_frame=0, mask_resolution=(2, 2),
            bboxes=[np.array([1, 1, 5, 5], dtype=np.float32)] * 2,
            masks=[torch.zeros((2, 2), dtype=torch.bool)] * 2,
        )
        pr = PrimaryRestoreResult(
            clip=clip, frame_count=2, frame_shape=(8, 8), frame_device=torch.device("cpu"),
            primary_raw=torch.zeros((2, 3, 256, 256)),
            keep_start=0, keep_end=2, crossfade_weights=None,
            enlarged_bboxes=[(1, 1, 5, 5)] * 2, crop_shapes=[(4, 4)] * 2,
            pad_offsets=[(126, 126)] * 2, resize_shapes=[(4, 4)] * 2,
        )

        restored = [torch.randint(0, 255, (3, 256, 256), dtype=torch.uint8)] * 2
        sr_result = SecondaryRestoreResult(
            clip=clip, frame_count=pr.frame_count, frame_shape=pr.frame_shape, frame_device=pr.frame_device,
            restored_frames=restored,
            keep_start=0, keep_end=2, crossfade_weights=None,
            enlarged_bboxes=pr.enlarged_bboxes, crop_shapes=pr.crop_shapes,
            pad_offsets=pr.pad_offsets, resize_shapes=pr.resize_shapes,
        )

        flush_pending_called = False
        result_returned = False

        def mock_pop():
            nonlocal flush_pending_called, result_returned
            if flush_pending_called and not result_returned:
                result_returned = True
                return [(0, restored)]
            return []

        restorer = MagicMock(spec=AsyncSecondaryRestorer)
        restorer.num_workers = 2
        restorer.push_clip.return_value = 0
        restorer.pop_completed.side_effect = mock_pop
        restorer.has_pending = True
        restorer.flush_all.return_value = None

        def on_flush_pending():
            nonlocal flush_pending_called
            flush_pending_called = True
            restorer.has_pending = False

        restorer.flush_pending.side_effect = on_flush_pending
        p.restoration_pipeline.secondary_restorer = restorer
        p.restoration_pipeline.build_secondary_result.return_value = sr_result

        secondary_queue: Queue = Queue()
        encode_queue: Queue = Queue()
        secondary_queue.put(pr)

        import threading
        def put_sentinel_later():
            import time
            time.sleep(0.3)
            secondary_queue.put(_SENTINEL)

        t = threading.Thread(target=put_sentinel_later, daemon=True)
        t.start()

        p._run_secondary_loop(secondary_queue, encode_queue)
        t.join(timeout=3)

        restorer.flush_pending.assert_called()
        restorer.flush_all.assert_called_once()
        assert not encode_queue.empty()

    def test_run_secondary_loop_no_gap_flush_when_items_arrive(self):
        """No flush_pending when clips arrive without gaps."""
        p = _make_pipeline()

        clip = TrackedClip(
            track_id=1, start_frame=0, mask_resolution=(2, 2),
            bboxes=[np.array([1, 1, 5, 5], dtype=np.float32)] * 2,
            masks=[torch.zeros((2, 2), dtype=torch.bool)] * 2,
        )
        pr = PrimaryRestoreResult(
            clip=clip, frame_count=2, frame_shape=(8, 8), frame_device=torch.device("cpu"),
            primary_raw=torch.zeros((2, 3, 256, 256)),
            keep_start=0, keep_end=2, crossfade_weights=None,
            enlarged_bboxes=[(1, 1, 5, 5)] * 2, crop_shapes=[(4, 4)] * 2,
            pad_offsets=[(126, 126)] * 2, resize_shapes=[(4, 4)] * 2,
        )

        restored = [torch.randint(0, 255, (3, 256, 256), dtype=torch.uint8)] * 2
        sr_result = SecondaryRestoreResult(
            clip=clip, frame_count=pr.frame_count, frame_shape=pr.frame_shape, frame_device=pr.frame_device,
            restored_frames=restored,
            keep_start=0, keep_end=2, crossfade_weights=None,
            enlarged_bboxes=pr.enlarged_bboxes, crop_shapes=pr.crop_shapes,
            pad_offsets=pr.pad_offsets, resize_shapes=pr.resize_shapes,
        )

        restorer = MagicMock(spec=AsyncSecondaryRestorer)
        restorer.num_workers = 2
        restorer.push_clip.return_value = 0
        restorer.pop_completed.side_effect = [[], [(0, restored)], []]
        restorer.flush_all.return_value = None
        restorer.has_pending = True
        p.restoration_pipeline.secondary_restorer = restorer
        p.restoration_pipeline.build_secondary_result.return_value = sr_result

        secondary_queue: Queue = Queue()
        encode_queue: Queue = Queue()
        secondary_queue.put(pr)
        secondary_queue.put(_SENTINEL)

        p._run_secondary_loop(secondary_queue, encode_queue)

        restorer.flush_pending.assert_not_called()
        restorer.flush_all.assert_called_once()
        assert not encode_queue.empty()
        result = encode_queue.get()
        assert result is sr_result

    def test_run_secondary_loop_self_priming_prevents_deadlock(self):
        """3 clips on 2 workers: clip 2 primes clip 0's buffered tail, preventing deadlock."""
        p = _make_pipeline()

        def _make_pr(track_id, n_frames):
            clip = TrackedClip(
                track_id=track_id, start_frame=0, mask_resolution=(2, 2),
                bboxes=[np.array([1, 1, 5, 5], dtype=np.float32)] * n_frames,
                masks=[torch.zeros((2, 2), dtype=torch.bool)] * n_frames,
            )
            return PrimaryRestoreResult(
                clip=clip, frame_count=n_frames, frame_shape=(8, 8), frame_device=torch.device("cpu"),
                primary_raw=torch.zeros((n_frames, 3, 256, 256)),
                keep_start=0, keep_end=n_frames, crossfade_weights=None,
                enlarged_bboxes=[(1, 1, 5, 5)] * n_frames, crop_shapes=[(4, 4)] * n_frames,
                pad_offsets=[(126, 126)] * n_frames, resize_shapes=[(4, 4)] * n_frames,
            )

        pr0 = _make_pr(1, 50)
        pr1 = _make_pr(2, 60)
        pr2 = _make_pr(3, 40)

        push_count = 0
        completed_seqs: set[int] = set()

        def mock_push_clip(frames, keep_start, keep_end):
            nonlocal push_count
            seq = push_count
            push_count += 1
            return seq

        def mock_pop_completed():
            if push_count >= 3 and 0 not in completed_seqs:
                completed_seqs.add(0)
                return [(0, [torch.zeros((3, 8, 8), dtype=torch.uint8)] * 50)]
            return []

        restorer = MagicMock(spec=AsyncSecondaryRestorer)
        restorer.num_workers = 2
        restorer.push_clip.side_effect = mock_push_clip
        restorer.pop_completed.side_effect = mock_pop_completed
        restorer.flush_all.return_value = None
        p.restoration_pipeline.secondary_restorer = restorer
        p.restoration_pipeline.build_secondary_result.side_effect = lambda pr, frames: SecondaryRestoreResult(
            clip=pr.clip, frame_count=pr.frame_count, frame_shape=pr.frame_shape, frame_device=pr.frame_device,
            restored_frames=frames, keep_start=pr.keep_start, keep_end=pr.keep_end,
            crossfade_weights=None, enlarged_bboxes=pr.enlarged_bboxes,
            crop_shapes=pr.crop_shapes, pad_offsets=pr.pad_offsets, resize_shapes=pr.resize_shapes,
        )

        secondary_queue: Queue = Queue()
        encode_queue: Queue = Queue()
        secondary_queue.put(pr0)
        secondary_queue.put(pr1)
        secondary_queue.put(pr2)
        secondary_queue.put(_SENTINEL)

        p._run_secondary_loop(secondary_queue, encode_queue)

        assert restorer.push_clip.call_count == 3
        assert not encode_queue.empty()
        restorer.flush_all.assert_called_once()

    def test_run_secondary_loop_tiny_and_large_clip_no_deadlock(self):
        """Reproduces original deadlock: 1-frame + 170-frame clips on 2 workers.

        With max_pending=num_workers+1=3, the 3rd clip primes worker 0,
        releasing the tiny clip's buffered tail.
        """
        p = _make_pipeline()

        def _make_pr(track_id, n_frames):
            clip = TrackedClip(
                track_id=track_id, start_frame=0, mask_resolution=(2, 2),
                bboxes=[np.array([1, 1, 5, 5], dtype=np.float32)] * n_frames,
                masks=[torch.zeros((2, 2), dtype=torch.bool)] * n_frames,
            )
            return PrimaryRestoreResult(
                clip=clip, frame_count=n_frames, frame_shape=(8, 8), frame_device=torch.device("cpu"),
                primary_raw=torch.zeros((n_frames, 3, 256, 256)),
                keep_start=0, keep_end=n_frames, crossfade_weights=None,
                enlarged_bboxes=[(1, 1, 5, 5)] * n_frames, crop_shapes=[(4, 4)] * n_frames,
                pad_offsets=[(126, 126)] * n_frames, resize_shapes=[(4, 4)] * n_frames,
            )

        pr_tiny = _make_pr(1, 1)
        pr_large = _make_pr(2, 170)
        pr_next = _make_pr(3, 80)

        push_count = 0
        completed_seqs: set[int] = set()

        def mock_push_clip(frames, keep_start, keep_end):
            nonlocal push_count
            seq = push_count
            push_count += 1
            return seq

        def mock_pop_completed():
            if push_count >= 3 and 0 not in completed_seqs:
                completed_seqs.add(0)
                return [(0, [torch.zeros((3, 8, 8), dtype=torch.uint8)] * 1)]
            return []

        restorer = MagicMock(spec=AsyncSecondaryRestorer)
        restorer.num_workers = 2
        restorer.push_clip.side_effect = mock_push_clip
        restorer.pop_completed.side_effect = mock_pop_completed
        restorer.flush_all.return_value = None
        p.restoration_pipeline.secondary_restorer = restorer
        p.restoration_pipeline.build_secondary_result.side_effect = lambda pr, frames: SecondaryRestoreResult(
            clip=pr.clip, frame_count=pr.frame_count, frame_shape=pr.frame_shape, frame_device=pr.frame_device,
            restored_frames=frames, keep_start=pr.keep_start, keep_end=pr.keep_end,
            crossfade_weights=None, enlarged_bboxes=pr.enlarged_bboxes,
            crop_shapes=pr.crop_shapes, pad_offsets=pr.pad_offsets, resize_shapes=pr.resize_shapes,
        )

        secondary_queue: Queue = Queue()
        encode_queue: Queue = Queue()
        secondary_queue.put(pr_tiny)
        secondary_queue.put(pr_large)
        secondary_queue.put(pr_next)
        secondary_queue.put(_SENTINEL)

        p._run_secondary_loop(secondary_queue, encode_queue)

        assert restorer.push_clip.call_count == 3
        assert not encode_queue.empty()
        first_result = encode_queue.get()
        assert first_result.frame_count == 1
        restorer.flush_all.assert_called_once()

    def test_run_with_progress_callback(self):
        cb = MagicMock()
        with (
            patch("jasna.pipeline.RfDetrMosaicDetectionModel"),
            patch("jasna.pipeline.YoloMosaicDetectionModel"),
        ):
            rest_pipeline = MagicMock()
            rest_pipeline.secondary_num_workers = 1
            rest_pipeline.secondary_restorer.num_workers = 2
            p = Pipeline(
                input_video=Path("in.mp4"),
                output_video=Path("out.mkv"),
                detection_model_name="rfdetr-v5",
                detection_model_path=Path("model.onnx"),
                detection_score_threshold=0.25,
                restoration_pipeline=rest_pipeline,
                codec="hevc",
                encoder_settings={},
                batch_size=2,
                device=torch.device("cuda:0"),
                max_clip_size=60,
                temporal_overlap=8,
                fp16=True,
                disable_progress=True,
                progress_callback=cb,
            )

        mock_reader = MagicMock()
        mock_reader.__enter__ = MagicMock(return_value=mock_reader)
        mock_reader.__exit__ = MagicMock(return_value=False)
        mock_reader.frames.return_value = iter([])

        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)

        with (
            patch("jasna.pipeline.get_video_meta_data", return_value=_fake_metadata()),
            patch("jasna.pipeline.NvidiaVideoReader", return_value=mock_reader),
            patch("jasna.pipeline.NvidiaVideoEncoder", return_value=mock_encoder),
            patch("jasna.pipeline.torch.cuda.set_device"),
            patch("jasna.pipeline.torch.inference_mode", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))),
        ):
            p.run()
