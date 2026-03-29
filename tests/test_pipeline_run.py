from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from jasna.crop_buffer import RawCrop
from jasna.frame_queue import FrameQueue

import numpy as np
import torch
import pytest
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.media import VideoMetadata
from jasna.pipeline import Pipeline
from jasna.pipeline_items import ClipRestoreItem, FrameMeta, PrimaryRestoreResult, SecondaryRestoreResult, _SENTINEL
from jasna.restorer.secondary_restorer import AsyncSecondaryRestorer
from jasna.tracking.clip_tracker import TrackedClip


def _mock_async_restorer(**kwargs) -> MagicMock:
    m = MagicMock(spec=AsyncSecondaryRestorer, **kwargs)
    def _real_to_tensors(frames_np):
        if not frames_np:
            return torch.empty(0)
        if isinstance(frames_np[0], np.ndarray):
            batch = np.stack(frames_np)
            batch = np.ascontiguousarray(batch.transpose(0, 3, 1, 2))
            return torch.from_numpy(batch)
        return torch.stack(frames_np)
    m._to_tensors.side_effect = _real_to_tensors
    return m


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
        rest_pipeline.secondary_prefers_cpu_input = False
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


def _make_two_readers(frames_batches: list[tuple[torch.Tensor, list[int]]]):
    def _make_reader(batches):
        r = MagicMock()
        r.__enter__ = MagicMock(return_value=r)
        r.__exit__ = MagicMock(return_value=False)
        r.frames.return_value = iter(batches)
        return r

    flat_frames = []
    for batch, pts in frames_batches:
        for i in range(len(pts)):
            flat_frames.append(batch[i])

    reader1 = _make_reader(list(frames_batches))
    reader2 = _make_reader([(torch.stack(flat_frames), list(range(len(flat_frames))))] if flat_frames else [])
    readers = iter([reader1, reader2])
    return MagicMock(side_effect=lambda *a, **kw: next(readers)), reader1, reader2


class TestPipelineColorspaceCheck:
    @patch("jasna.pipeline.get_video_meta_data")
    def test_run_raises_on_bt601_colorspace(self, mock_meta):
        meta = _fake_metadata()
        meta.color_space = AvColorspace.ITU601
        mock_meta.return_value = meta
        p = _make_pipeline()

        from jasna.media import UnsupportedColorspaceError
        with pytest.raises(UnsupportedColorspaceError, match="Only BT.709 is supported"):
            p.run()

    @patch("jasna.pipeline.get_video_meta_data")
    def test_run_raises_on_smpte240m_colorspace(self, mock_meta):
        meta = _fake_metadata()
        meta.color_space = AvColorspace.SMPTE240M
        mock_meta.return_value = meta
        p = _make_pipeline()

        from jasna.media import UnsupportedColorspaceError
        with pytest.raises(UnsupportedColorspaceError, match="Only BT.709 is supported"):
            p.run()


class TestPipelineRun:
    def test_run_no_frames(self):
        p = _make_pipeline()

        reader_cls, _, _ = _make_two_readers([])

        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)

        with (
            patch("jasna.pipeline.get_video_meta_data", return_value=_fake_metadata()),
            patch("jasna.pipeline_threads.NvidiaVideoReader", reader_cls),
            patch("jasna.pipeline.NvidiaVideoEncoder", return_value=mock_encoder),
            patch("jasna.pipeline_threads.torch.cuda.set_device"),
            patch("jasna.pipeline_threads.torch.inference_mode", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))),
        ):
            p.run()

        mock_encoder.encode.assert_not_called()

    def test_run_full_thread_flow(self):
        """Exercise all four thread bodies: decode->primary->secondary->encode."""
        p = _make_pipeline()

        frames_t = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)
        batches = [(frames_t, [0, 1])]
        reader_cls, _, _ = _make_two_readers(batches)

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
            bb = kwargs["blend_buffer"]
            mq = kwargs["metadata_queue"]
            cq = kwargs["clip_queue"]
            frames = kwargs["frames"]
            bb.register_frame(0, {42})
            bb.register_frame(1, {42})
            mq.put(FrameMeta(frame_idx=0, pts=0))
            mq.put(FrameMeta(frame_idx=1, pts=1))
            raw_crops = [
                RawCrop(crop=frames[i][:, 1:5, 1:5].clone(), enlarged_bbox=(1, 1, 5, 5), crop_shape=(4, 4))
                for i in range(2)
            ]
            cq.put(ClipRestoreItem(
                clip=clip,
                raw_crops=raw_crops,
                frame_shape=(8, 8),
                keep_start=0,
                keep_end=2,
                crossfade_weights=None,
            ))
            return BatchProcessResult(next_frame_idx=2, clips_emitted=1)

        pr_result = PrimaryRestoreResult(
            track_id=clip.track_id,
            start_frame=clip.start_frame,
            frame_count=2,
            frame_shape=(8, 8),
            frame_device=frames_t[0].device,
            masks=clip.masks,
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
            track_id=clip.track_id,
            start_frame=clip.start_frame,
            frame_count=2,
            frame_shape=(8, 8),
            frame_device=frames_t[0].device,
            masks=clip.masks,
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
        restorer = _mock_async_restorer()
        restorer.push_clip.return_value = 0
        _pop_done = set()
        def _pop_full():
            if restorer.push_clip.called and 0 not in _pop_done:
                _pop_done.add(0)
                return [(0, restored)]
            return []
        restorer.pop_completed.side_effect = _pop_full
        restorer.flush_all.return_value = None
        p.restoration_pipeline.secondary_restorer = restorer
        p.restoration_pipeline.build_secondary_result.return_value = sr_result

        with (
            patch("jasna.pipeline.get_video_meta_data", return_value=_fake_metadata()),
            patch("jasna.pipeline_threads.NvidiaVideoReader", reader_cls),
            patch("jasna.pipeline.NvidiaVideoEncoder", return_value=mock_encoder),
            patch("jasna.pipeline_threads.process_frame_batch", side_effect=fake_process_batch),
            patch("jasna.pipeline_threads.finalize_processing"),
            patch("jasna.pipeline_threads.torch.cuda.set_device"),
            patch("jasna.pipeline_threads.torch.inference_mode", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))),
        ):
            p.run()

        p.restoration_pipeline.prepare_and_run_primary.assert_called_once()
        p.restoration_pipeline.build_secondary_result.assert_called_once()
        assert mock_encoder.encode.call_count == 2

    def test_run_processes_frames(self):
        p = _make_pipeline()

        frames = torch.zeros((2, 3, 8, 8), dtype=torch.uint8)
        batches = [(frames, [0, 1])]
        reader_cls, _, _ = _make_two_readers(batches)

        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)

        from jasna.pipeline_processing import BatchProcessResult

        def fake_process_batch(**kwargs):
            bb = kwargs["blend_buffer"]
            mq = kwargs["metadata_queue"]
            pts_list = kwargs["pts_list"]
            start_idx = kwargs["start_frame_idx"]
            for i, pts in enumerate(pts_list):
                bb.register_frame(start_idx + i, set())
                mq.put(FrameMeta(frame_idx=start_idx + i, pts=int(pts)))
            return BatchProcessResult(next_frame_idx=start_idx + len(pts_list), clips_emitted=0)

        restorer = _mock_async_restorer()
        restorer.push_clip.return_value = 0
        restorer.pop_completed.return_value = []
        restorer.flush_all.return_value = None
        p.restoration_pipeline.secondary_restorer = restorer
        p.restoration_pipeline.build_secondary_result.return_value = MagicMock(spec=SecondaryRestoreResult)

        with (
            patch("jasna.pipeline.get_video_meta_data", return_value=_fake_metadata()),
            patch("jasna.pipeline_threads.NvidiaVideoReader", reader_cls),
            patch("jasna.pipeline.NvidiaVideoEncoder", return_value=mock_encoder),
            patch("jasna.pipeline_threads.process_frame_batch", side_effect=fake_process_batch),
            patch("jasna.pipeline_threads.finalize_processing"),
            patch("jasna.pipeline_threads.torch.cuda.set_device"),
            patch("jasna.pipeline_threads.torch.inference_mode", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))),
        ):
            p.run()

    def test_run_propagates_decode_error(self):
        p = _make_pipeline()

        def _make_reader_with_error():
            r1 = MagicMock()
            r1.__enter__ = MagicMock(return_value=r1)
            r1.__exit__ = MagicMock(return_value=False)
            r1.frames.side_effect = RuntimeError("decode boom")
            r2 = MagicMock()
            r2.__enter__ = MagicMock(return_value=r2)
            r2.__exit__ = MagicMock(return_value=False)
            r2.frames.return_value = iter([])
            readers = iter([r1, r2])
            return MagicMock(side_effect=lambda *a, **kw: next(readers))

        reader_cls = _make_reader_with_error()

        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)

        with (
            patch("jasna.pipeline.get_video_meta_data", return_value=_fake_metadata()),
            patch("jasna.pipeline_threads.NvidiaVideoReader", reader_cls),
            patch("jasna.pipeline.NvidiaVideoEncoder", return_value=mock_encoder),
            patch("jasna.pipeline_threads.torch.cuda.set_device"),
            patch("jasna.pipeline_threads.torch.inference_mode", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))),
        ):
            with pytest.raises(RuntimeError, match="decode boom"):
                p.run()

    def test_run_primary_error_propagates(self):
        """Cover lines 175-176: error in primary_restore_thread."""
        p = _make_pipeline()

        frames_t = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)
        batches = [(frames_t, [0, 1])]
        reader_cls, _, _ = _make_two_readers(batches)

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
            bb = kwargs["blend_buffer"]
            mq = kwargs["metadata_queue"]
            cq = kwargs["clip_queue"]
            frames = kwargs["frames"]
            bb.register_frame(0, {1})
            bb.register_frame(1, {1})
            mq.put(FrameMeta(frame_idx=0, pts=0))
            mq.put(FrameMeta(frame_idx=1, pts=1))
            raw_crops = [
                RawCrop(crop=frames[i][:, 1:5, 1:5].clone(), enlarged_bbox=(1, 1, 5, 5), crop_shape=(4, 4))
                for i in range(2)
            ]
            cq.put(ClipRestoreItem(clip=clip, raw_crops=raw_crops, frame_shape=(8, 8), keep_start=0, keep_end=2, crossfade_weights=None))
            return BatchProcessResult(next_frame_idx=2, clips_emitted=1)

        p.restoration_pipeline.prepare_and_run_primary.side_effect = RuntimeError("primary boom")

        with (
            patch("jasna.pipeline.get_video_meta_data", return_value=_fake_metadata()),
            patch("jasna.pipeline_threads.NvidiaVideoReader", reader_cls),
            patch("jasna.pipeline.NvidiaVideoEncoder", return_value=mock_encoder),
            patch("jasna.pipeline_threads.process_frame_batch", side_effect=fake_process_batch),
            patch("jasna.pipeline_threads.finalize_processing"),
            patch("jasna.pipeline_threads.torch.cuda.set_device"),
            patch("jasna.pipeline_threads.torch.inference_mode", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))),
        ):
            with pytest.raises(RuntimeError, match="primary boom"):
                p.run()

    def test_run_secondary_error_propagates(self):
        """Cover lines 195-196: error in secondary_restore_thread."""
        p = _make_pipeline()

        frames_t = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)
        batches = [(frames_t, [0, 1])]
        reader_cls, _, _ = _make_two_readers(batches)

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
            bb = kwargs["blend_buffer"]
            mq = kwargs["metadata_queue"]
            cq = kwargs["clip_queue"]
            frames = kwargs["frames"]
            bb.register_frame(0, {1})
            bb.register_frame(1, {1})
            mq.put(FrameMeta(frame_idx=0, pts=0))
            mq.put(FrameMeta(frame_idx=1, pts=1))
            raw_crops = [
                RawCrop(crop=frames[i][:, 1:5, 1:5].clone(), enlarged_bbox=(1, 1, 5, 5), crop_shape=(4, 4))
                for i in range(2)
            ]
            cq.put(ClipRestoreItem(clip=clip, raw_crops=raw_crops, frame_shape=(8, 8), keep_start=0, keep_end=2, crossfade_weights=None))
            return BatchProcessResult(next_frame_idx=2, clips_emitted=1)

        pr_result = PrimaryRestoreResult(
            track_id=clip.track_id, start_frame=clip.start_frame, frame_count=2, frame_shape=(8, 8), frame_device=frames_t[0].device,
            masks=clip.masks, primary_raw=torch.zeros((2, 3, 256, 256)),
            keep_start=0, keep_end=2, crossfade_weights=None,
            enlarged_bboxes=[(1, 1, 5, 5)] * 2, crop_shapes=[(4, 4)] * 2,
            pad_offsets=[(126, 126)] * 2, resize_shapes=[(4, 4)] * 2,
        )
        p.restoration_pipeline.prepare_and_run_primary.return_value = pr_result
        restorer = _mock_async_restorer()
        restorer.push_clip.return_value = 0
        _pop_done_err = set()
        def _pop_err():
            if restorer.push_clip.called and 0 not in _pop_done_err:
                _pop_done_err.add(0)
                return [(0, [])]
            return []
        restorer.pop_completed.side_effect = _pop_err
        restorer.flush_all.return_value = None
        p.restoration_pipeline.secondary_restorer = restorer
        p.restoration_pipeline.build_secondary_result.side_effect = RuntimeError("secondary boom")

        with (
            patch("jasna.pipeline.get_video_meta_data", return_value=_fake_metadata()),
            patch("jasna.pipeline_threads.NvidiaVideoReader", reader_cls),
            patch("jasna.pipeline.NvidiaVideoEncoder", return_value=mock_encoder),
            patch("jasna.pipeline_threads.process_frame_batch", side_effect=fake_process_batch),
            patch("jasna.pipeline_threads.finalize_processing"),
            patch("jasna.pipeline_threads.torch.cuda.set_device"),
            patch("jasna.pipeline_threads.torch.inference_mode", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))),
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
            track_id=clip.track_id, start_frame=clip.start_frame,
            frame_count=2, frame_shape=(8, 8), frame_device=torch.device("cpu"),
            masks=clip.masks, primary_raw=torch.zeros((2, 3, 256, 256)),
            keep_start=0, keep_end=2, crossfade_weights=None,
            enlarged_bboxes=[(1, 1, 5, 5)] * 2, crop_shapes=[(4, 4)] * 2,
            pad_offsets=[(126, 126)] * 2, resize_shapes=[(4, 4)] * 2,
        )

        restored = [torch.randint(0, 255, (3, 256, 256), dtype=torch.uint8)] * 2
        sr_result = SecondaryRestoreResult(
            track_id=pr.track_id, start_frame=pr.start_frame,
            frame_count=pr.frame_count, frame_shape=pr.frame_shape, frame_device=pr.frame_device,
            masks=pr.masks, restored_frames=restored,
            keep_start=0, keep_end=2, crossfade_weights=None,
            enlarged_bboxes=pr.enlarged_bboxes, crop_shapes=pr.crop_shapes,
            pad_offsets=pr.pad_offsets, resize_shapes=pr.resize_shapes,
        )

        restorer = _mock_async_restorer()
        restorer.num_workers = 2
        restorer.push_clip.return_value = 0
        _pop_done = set()
        def _pop():
            if restorer.push_clip.called and 0 not in _pop_done:
                _pop_done.add(0)
                return [(0, restored)]
            return []
        restorer.pop_completed.side_effect = _pop
        restorer.flush_all.return_value = None
        p.restoration_pipeline.secondary_restorer = restorer
        p.restoration_pipeline.build_secondary_result.return_value = sr_result

        secondary_queue= FrameQueue(max_frames=9999)
        encode_queue= FrameQueue(max_frames=9999)
        secondary_queue.put(pr)
        secondary_queue.put(_SENTINEL)

        p._ASYNC_POLL_TIMEOUT = 0.001
        p._run_secondary_loop(secondary_queue, encode_queue)

        restorer.push_clip.assert_called_once()
        assert not encode_queue.empty()
        result = encode_queue.get()
        assert result is sr_result

    def test_run_secondary_loop_no_flush_when_primary_busy(self):
        """No flush_pending when secondary_queue is empty but primary is busy (not idle)."""
        import threading
        p = _make_pipeline()
        p._ASYNC_POLL_TIMEOUT = 0.01
        p._FLUSH_DELAY = 0.05

        clip = TrackedClip(
            track_id=1, start_frame=0, mask_resolution=(2, 2),
            bboxes=[np.array([1, 1, 5, 5], dtype=np.float32)] * 2,
            masks=[torch.zeros((2, 2), dtype=torch.bool)] * 2,
        )
        pr = PrimaryRestoreResult(
            track_id=clip.track_id, start_frame=clip.start_frame,
            frame_count=2, frame_shape=(8, 8), frame_device=torch.device("cpu"),
            masks=clip.masks, primary_raw=torch.zeros((2, 3, 256, 256)),
            keep_start=0, keep_end=2, crossfade_weights=None,
            enlarged_bboxes=[(1, 1, 5, 5)] * 2, crop_shapes=[(4, 4)] * 2,
            pad_offsets=[(126, 126)] * 2, resize_shapes=[(4, 4)] * 2,
        )

        restorer = _mock_async_restorer()
        restorer.num_workers = 2
        restorer.push_clip.return_value = 0
        restorer.pop_completed.return_value = []
        restorer.has_pending = True
        restorer.flush_all.return_value = None
        p.restoration_pipeline.secondary_restorer = restorer

        secondary_queue= FrameQueue(max_frames=9999)
        encode_queue= FrameQueue(max_frames=9999)
        cq= FrameQueue(max_frames=9999)
        primary_idle = threading.Event()
        secondary_queue.put(pr)

        def put_sentinel_later():
            import time
            time.sleep(0.15)
            secondary_queue.put(_SENTINEL)

        t = threading.Thread(target=put_sentinel_later, daemon=True)
        t.start()

        p._run_secondary_loop(secondary_queue, encode_queue, clip_queue=cq, primary_idle_event=primary_idle)
        t.join(timeout=3)

        restorer.flush_pending.assert_not_called()
        restorer.flush_all.assert_called_once()

    def test_run_secondary_loop_no_flush_short_gap(self):
        """No flush_pending when gap is shorter than FLUSH_DELAY."""
        import threading
        p = _make_pipeline()
        p._ASYNC_POLL_TIMEOUT = 0.01

        clip = TrackedClip(
            track_id=1, start_frame=0, mask_resolution=(2, 2),
            bboxes=[np.array([1, 1, 5, 5], dtype=np.float32)] * 2,
            masks=[torch.zeros((2, 2), dtype=torch.bool)] * 2,
        )
        pr = PrimaryRestoreResult(
            track_id=clip.track_id, start_frame=clip.start_frame,
            frame_count=2, frame_shape=(8, 8), frame_device=torch.device("cpu"),
            masks=clip.masks, primary_raw=torch.zeros((2, 3, 256, 256)),
            keep_start=0, keep_end=2, crossfade_weights=None,
            enlarged_bboxes=[(1, 1, 5, 5)] * 2, crop_shapes=[(4, 4)] * 2,
            pad_offsets=[(126, 126)] * 2, resize_shapes=[(4, 4)] * 2,
        )

        restorer = _mock_async_restorer()
        restorer.num_workers = 2
        restorer.push_clip.return_value = 0
        restorer.pop_completed.return_value = []
        restorer.has_pending = True
        restorer.flush_all.return_value = None
        p.restoration_pipeline.secondary_restorer = restorer

        secondary_queue= FrameQueue(max_frames=9999)
        encode_queue= FrameQueue(max_frames=9999)
        cq= FrameQueue(max_frames=9999)
        primary_idle = threading.Event()
        primary_idle.set()
        secondary_queue.put(pr)

        def put_sentinel_later():
            import time
            time.sleep(0.15)
            secondary_queue.put(_SENTINEL)

        t = threading.Thread(target=put_sentinel_later, daemon=True)
        t.start()

        p._run_secondary_loop(secondary_queue, encode_queue, clip_queue=cq, primary_idle_event=primary_idle)
        t.join(timeout=3)

        restorer.flush_pending.assert_not_called()
        restorer.flush_all.assert_called_once()

    def test_run_secondary_loop_no_gap_flush_when_items_arrive(self):
        """No flush_pending when clips arrive without gaps."""
        p = _make_pipeline()

        clip = TrackedClip(
            track_id=1, start_frame=0, mask_resolution=(2, 2),
            bboxes=[np.array([1, 1, 5, 5], dtype=np.float32)] * 2,
            masks=[torch.zeros((2, 2), dtype=torch.bool)] * 2,
        )
        pr = PrimaryRestoreResult(
            track_id=clip.track_id, start_frame=clip.start_frame,
            frame_count=2, frame_shape=(8, 8), frame_device=torch.device("cpu"),
            masks=clip.masks, primary_raw=torch.zeros((2, 3, 256, 256)),
            keep_start=0, keep_end=2, crossfade_weights=None,
            enlarged_bboxes=[(1, 1, 5, 5)] * 2, crop_shapes=[(4, 4)] * 2,
            pad_offsets=[(126, 126)] * 2, resize_shapes=[(4, 4)] * 2,
        )

        restored = [torch.randint(0, 255, (3, 256, 256), dtype=torch.uint8)] * 2
        sr_result = SecondaryRestoreResult(
            track_id=pr.track_id, start_frame=pr.start_frame,
            frame_count=pr.frame_count, frame_shape=pr.frame_shape, frame_device=pr.frame_device,
            masks=pr.masks, restored_frames=restored,
            keep_start=0, keep_end=2, crossfade_weights=None,
            enlarged_bboxes=pr.enlarged_bboxes, crop_shapes=pr.crop_shapes,
            pad_offsets=pr.pad_offsets, resize_shapes=pr.resize_shapes,
        )

        restorer = _mock_async_restorer()
        restorer.num_workers = 2
        restorer.push_clip.return_value = 0
        _pop_done = set()
        def _pop():
            if restorer.push_clip.called and 0 not in _pop_done:
                _pop_done.add(0)
                return [(0, restored)]
            return []
        restorer.pop_completed.side_effect = _pop
        restorer.flush_all.return_value = None
        restorer.has_pending = True
        p.restoration_pipeline.secondary_restorer = restorer
        p.restoration_pipeline.build_secondary_result.return_value = sr_result

        secondary_queue= FrameQueue(max_frames=9999)
        encode_queue= FrameQueue(max_frames=9999)
        secondary_queue.put(pr)
        secondary_queue.put(_SENTINEL)

        p._ASYNC_POLL_TIMEOUT = 0.001
        p._run_secondary_loop(secondary_queue, encode_queue)

        restorer.flush_pending.assert_not_called()
        restorer.flush_all.assert_called_once()
        assert not encode_queue.empty()
        result = encode_queue.get()
        assert result is sr_result

    def test_run_secondary_loop_flush_called_once_per_starvation(self):
        """flush_pending called only once while starved, even if has_pending stays true."""
        import threading
        p = _make_pipeline()
        p._ASYNC_POLL_TIMEOUT = 0.01
        p._FLUSH_DELAY = 0.05
        p._FLUSH_RETRY_TIMEOUT = 999

        clip = TrackedClip(
            track_id=1, start_frame=0, mask_resolution=(2, 2),
            bboxes=[np.array([1, 1, 5, 5], dtype=np.float32)] * 2,
            masks=[torch.zeros((2, 2), dtype=torch.bool)] * 2,
        )
        pr = PrimaryRestoreResult(
            track_id=clip.track_id, start_frame=clip.start_frame,
            frame_count=2, frame_shape=(8, 8), frame_device=torch.device("cpu"),
            masks=clip.masks, primary_raw=torch.zeros((2, 3, 256, 256)),
            keep_start=0, keep_end=2, crossfade_weights=None,
            enlarged_bboxes=[(1, 1, 5, 5)] * 2, crop_shapes=[(4, 4)] * 2,
            pad_offsets=[(126, 126)] * 2, resize_shapes=[(4, 4)] * 2,
        )

        restorer = _mock_async_restorer()
        restorer.num_workers = 2
        restorer.push_clip.return_value = 0
        restorer.pop_completed.return_value = []
        restorer.has_pending = True
        restorer.flush_all.return_value = None
        restorer.flush_pending.return_value = True
        p.restoration_pipeline.secondary_restorer = restorer

        secondary_queue= FrameQueue(max_frames=9999)
        encode_queue= FrameQueue(max_frames=9999)
        cq= FrameQueue(max_frames=9999)
        primary_idle = threading.Event()
        primary_idle.set()
        secondary_queue.put(pr)

        def put_sentinel_later():
            import time
            time.sleep(0.2)
            secondary_queue.put(_SENTINEL)

        t = threading.Thread(target=put_sentinel_later, daemon=True)
        t.start()

        p._run_secondary_loop(secondary_queue, encode_queue, clip_queue=cq, primary_idle_event=primary_idle)
        t.join(timeout=3)

        restorer.flush_pending.assert_called_once_with(target_seqs={0})

    def test_run_secondary_loop_flush_retry_after_timeout(self):
        """flush_pending retried after _FLUSH_RETRY_TIMEOUT if first flush didn't unstick."""
        import threading
        p = _make_pipeline()
        p._ASYNC_POLL_TIMEOUT = 0.01
        p._FLUSH_DELAY = 0.02
        p._FLUSH_RETRY_TIMEOUT = 0.08

        clip = TrackedClip(
            track_id=1, start_frame=0, mask_resolution=(2, 2),
            bboxes=[np.array([1, 1, 5, 5], dtype=np.float32)] * 2,
            masks=[torch.zeros((2, 2), dtype=torch.bool)] * 2,
        )
        pr = PrimaryRestoreResult(
            track_id=clip.track_id, start_frame=clip.start_frame,
            frame_count=2, frame_shape=(8, 8), frame_device=torch.device("cpu"),
            masks=clip.masks, primary_raw=torch.zeros((2, 3, 256, 256)),
            keep_start=0, keep_end=2, crossfade_weights=None,
            enlarged_bboxes=[(1, 1, 5, 5)] * 2, crop_shapes=[(4, 4)] * 2,
            pad_offsets=[(126, 126)] * 2, resize_shapes=[(4, 4)] * 2,
        )

        restorer = _mock_async_restorer()
        restorer.num_workers = 2
        restorer.push_clip.return_value = 0
        restorer.pop_completed.return_value = []
        restorer.has_pending = True
        restorer.flush_all.return_value = None
        restorer.flush_pending.return_value = True
        p.restoration_pipeline.secondary_restorer = restorer

        secondary_queue = FrameQueue(max_frames=9999)
        encode_queue = FrameQueue(max_frames=9999)
        cq = FrameQueue(max_frames=9999)
        primary_idle = threading.Event()
        primary_idle.set()
        secondary_queue.put(pr)

        def put_sentinel_later():
            import time
            time.sleep(0.4)
            secondary_queue.put(_SENTINEL)

        t = threading.Thread(target=put_sentinel_later, daemon=True)
        t.start()

        p._run_secondary_loop(secondary_queue, encode_queue, clip_queue=cq, primary_idle_event=primary_idle)
        t.join(timeout=3)

        assert restorer.flush_pending.call_count >= 2, (
            f"Expected flush retry but flush_pending called {restorer.flush_pending.call_count} time(s)"
        )

    def test_run_secondary_loop_pipeline_starved_triggers_flush(self):
        """flush_pending when primary idle, clip_queue empty, and FLUSH_DELAY elapsed."""
        import threading
        p = _make_pipeline()
        p._ASYNC_POLL_TIMEOUT = 0.01
        p._FLUSH_DELAY = 0.05

        clip = TrackedClip(
            track_id=1, start_frame=0, mask_resolution=(2, 2),
            bboxes=[np.array([1, 1, 5, 5], dtype=np.float32)] * 2,
            masks=[torch.zeros((2, 2), dtype=torch.bool)] * 2,
        )
        pr = PrimaryRestoreResult(
            track_id=clip.track_id, start_frame=clip.start_frame,
            frame_count=2, frame_shape=(8, 8), frame_device=torch.device("cpu"),
            masks=clip.masks, primary_raw=torch.zeros((2, 3, 256, 256)),
            keep_start=0, keep_end=2, crossfade_weights=None,
            enlarged_bboxes=[(1, 1, 5, 5)] * 2, crop_shapes=[(4, 4)] * 2,
            pad_offsets=[(126, 126)] * 2, resize_shapes=[(4, 4)] * 2,
        )

        restored = [torch.randint(0, 255, (3, 256, 256), dtype=torch.uint8)] * 2
        sr_result = SecondaryRestoreResult(
            track_id=pr.track_id, start_frame=pr.start_frame,
            frame_count=pr.frame_count, frame_shape=pr.frame_shape, frame_device=pr.frame_device,
            masks=pr.masks, restored_frames=restored,
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

        restorer = _mock_async_restorer()
        restorer.num_workers = 2
        restorer.push_clip.return_value = 0
        restorer.pop_completed.side_effect = mock_pop
        restorer.has_pending = True
        restorer.flush_all.return_value = None

        def on_flush_pending(target_seqs=None):
            nonlocal flush_pending_called
            flush_pending_called = True
            restorer.has_pending = False
            return True

        restorer.flush_pending.side_effect = on_flush_pending
        p.restoration_pipeline.secondary_restorer = restorer
        p.restoration_pipeline.build_secondary_result.return_value = sr_result

        secondary_queue= FrameQueue(max_frames=9999)
        encode_queue= FrameQueue(max_frames=9999)
        cq= FrameQueue(max_frames=9999)
        primary_idle = threading.Event()
        primary_idle.set()
        secondary_queue.put(pr)

        def put_sentinel_later():
            import time
            time.sleep(0.3)
            secondary_queue.put(_SENTINEL)

        t = threading.Thread(target=put_sentinel_later, daemon=True)
        t.start()

        p._run_secondary_loop(secondary_queue, encode_queue, clip_queue=cq, primary_idle_event=primary_idle)
        t.join(timeout=3)

        restorer.flush_pending.assert_called_with(target_seqs={0})
        restorer.flush_all.assert_called_once()
        assert not encode_queue.empty()

    def test_run_secondary_loop_self_priming_prevents_deadlock(self):
        """3 clips on 2 workers: clip 2 primes clip 0's buffered tail, preventing deadlock."""
        p = _make_pipeline()
        p._ASYNC_POLL_TIMEOUT = 0.001

        def _make_pr(track_id, n_frames):
            masks = [torch.zeros((2, 2), dtype=torch.bool)] * n_frames
            return PrimaryRestoreResult(
                track_id=track_id, start_frame=0,
                frame_count=n_frames, frame_shape=(8, 8), frame_device=torch.device("cpu"),
                masks=masks, primary_raw=torch.zeros((n_frames, 3, 256, 256)),
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

        restorer = _mock_async_restorer()
        restorer.num_workers = 2
        restorer.push_clip.side_effect = mock_push_clip
        restorer.pop_completed.side_effect = mock_pop_completed
        restorer.flush_all.return_value = None
        p.restoration_pipeline.secondary_restorer = restorer
        p.restoration_pipeline.build_secondary_result.side_effect = lambda pr, frames: SecondaryRestoreResult(
            track_id=pr.track_id, start_frame=pr.start_frame,
            frame_count=pr.frame_count, frame_shape=pr.frame_shape, frame_device=pr.frame_device,
            masks=pr.masks, restored_frames=frames, keep_start=pr.keep_start, keep_end=pr.keep_end,
            crossfade_weights=None, enlarged_bboxes=pr.enlarged_bboxes,
            crop_shapes=pr.crop_shapes, pad_offsets=pr.pad_offsets, resize_shapes=pr.resize_shapes,
        )

        secondary_queue= FrameQueue(max_frames=9999)
        encode_queue= FrameQueue(max_frames=9999)
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

        With pusher thread, all 3 clips are pushed without blocking.
        The 3rd clip primes worker 0, releasing the tiny clip's buffered tail.
        """
        p = _make_pipeline()
        p._ASYNC_POLL_TIMEOUT = 0.001

        def _make_pr(track_id, n_frames):
            masks = [torch.zeros((2, 2), dtype=torch.bool)] * n_frames
            return PrimaryRestoreResult(
                track_id=track_id, start_frame=0,
                frame_count=n_frames, frame_shape=(8, 8), frame_device=torch.device("cpu"),
                masks=masks, primary_raw=torch.zeros((n_frames, 3, 256, 256)),
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

        restorer = _mock_async_restorer()
        restorer.num_workers = 2
        restorer.push_clip.side_effect = mock_push_clip
        restorer.pop_completed.side_effect = mock_pop_completed
        restorer.flush_all.return_value = None
        p.restoration_pipeline.secondary_restorer = restorer
        p.restoration_pipeline.build_secondary_result.side_effect = lambda pr, frames: SecondaryRestoreResult(
            track_id=pr.track_id, start_frame=pr.start_frame,
            frame_count=pr.frame_count, frame_shape=pr.frame_shape, frame_device=pr.frame_device,
            masks=pr.masks, restored_frames=frames, keep_start=pr.keep_start, keep_end=pr.keep_end,
            crossfade_weights=None, enlarged_bboxes=pr.enlarged_bboxes,
            crop_shapes=pr.crop_shapes, pad_offsets=pr.pad_offsets, resize_shapes=pr.resize_shapes,
        )

        secondary_queue= FrameQueue(max_frames=9999)
        encode_queue= FrameQueue(max_frames=9999)
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
            rest_pipeline.secondary_prefers_cpu_input = False
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

        reader_cls, _, _ = _make_two_readers([])

        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)

        with (
            patch("jasna.pipeline.get_video_meta_data", return_value=_fake_metadata()),
            patch("jasna.pipeline_threads.NvidiaVideoReader", reader_cls),
            patch("jasna.pipeline.NvidiaVideoEncoder", return_value=mock_encoder),
            patch("jasna.pipeline_threads.torch.cuda.set_device"),
            patch("jasna.pipeline_threads.torch.inference_mode", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))),
        ):
            p.run()


    def test_run_secondary_loop_reflush_after_forwarding(self):
        """After flushing worker 0 and forwarding its results, flush fires again for worker 1."""
        import threading
        p = _make_pipeline()
        p._ASYNC_POLL_TIMEOUT = 0.01
        p._FLUSH_DELAY = 0.05

        def _make_pr(track_id, start_frame, n_frames):
            masks = [torch.zeros((2, 2), dtype=torch.bool)] * n_frames
            return PrimaryRestoreResult(
                track_id=track_id, start_frame=start_frame,
                frame_count=n_frames, frame_shape=(8, 8), frame_device=torch.device("cpu"),
                masks=masks, primary_raw=torch.zeros((n_frames, 3, 256, 256)),
                keep_start=0, keep_end=n_frames, crossfade_weights=None,
                enlarged_bboxes=[(1, 1, 5, 5)] * n_frames, crop_shapes=[(4, 4)] * n_frames,
                pad_offsets=[(126, 126)] * n_frames, resize_shapes=[(4, 4)] * n_frames,
            )

        pr0 = _make_pr(track_id=0, start_frame=0, n_frames=180)
        pr1 = _make_pr(track_id=1, start_frame=180, n_frames=180)

        push_count = 0
        flush_calls: list[set[int] | None] = []
        returned_seqs: set[int] = set()

        def mock_push(frames, keep_start, keep_end):
            nonlocal push_count
            seq = push_count
            push_count += 1
            return seq

        def mock_flush(target_seqs=None):
            flush_calls.append(target_seqs)
            return True

        def mock_pop():
            if len(flush_calls) >= 1 and 0 not in returned_seqs:
                returned_seqs.add(0)
                return [(0, [torch.zeros((3, 8, 8), dtype=torch.uint8)] * 180)]
            if len(flush_calls) >= 2 and 1 not in returned_seqs:
                returned_seqs.add(1)
                return [(1, [torch.zeros((3, 8, 8), dtype=torch.uint8)] * 180)]
            return []

        restorer = _mock_async_restorer()
        restorer.num_workers = 2
        restorer.push_clip.side_effect = mock_push
        restorer.pop_completed.side_effect = mock_pop
        restorer.flush_pending.side_effect = mock_flush
        restorer.flush_all.return_value = None

        @property
        def _has_pending(self):
            return len(returned_seqs) < push_count

        type(restorer).has_pending = property(lambda self: len(returned_seqs) < push_count)

        p.restoration_pipeline.secondary_restorer = restorer
        p.restoration_pipeline.build_secondary_result.side_effect = lambda pr, frames: SecondaryRestoreResult(
            track_id=pr.track_id, start_frame=pr.start_frame,
            frame_count=pr.frame_count, frame_shape=pr.frame_shape, frame_device=pr.frame_device,
            masks=pr.masks, restored_frames=frames, keep_start=pr.keep_start, keep_end=pr.keep_end,
            crossfade_weights=None, enlarged_bboxes=pr.enlarged_bboxes,
            crop_shapes=pr.crop_shapes, pad_offsets=pr.pad_offsets, resize_shapes=pr.resize_shapes,
        )

        secondary_queue= FrameQueue(max_frames=9999)
        encode_queue= FrameQueue(max_frames=9999)
        cq= FrameQueue(max_frames=9999)
        primary_idle = threading.Event()
        primary_idle.set()

        secondary_queue.put(pr0)
        secondary_queue.put(pr1)

        def put_sentinel_later():
            import time
            time.sleep(0.5)
            secondary_queue.put(_SENTINEL)

        t = threading.Thread(target=put_sentinel_later, daemon=True)
        t.start()

        p._run_secondary_loop(secondary_queue, encode_queue, clip_queue=cq, primary_idle_event=primary_idle)
        t.join(timeout=3)

        assert len(flush_calls) == 2, f"Expected 2 flush calls, got {len(flush_calls)}: {flush_calls}"
        assert flush_calls[0] == {0}
        assert flush_calls[1] == {1}
        assert encode_queue.qsize() == 2


class TestEarliestBlockingSeqs:
    def _make_pr(self, start_frame, n_frames, keep_start=0, keep_end=None):
        if keep_end is None:
            keep_end = n_frames
        masks = [torch.zeros((2, 2), dtype=torch.bool)] * n_frames
        return PrimaryRestoreResult(
            track_id=start_frame, start_frame=start_frame,
            frame_count=n_frames, frame_shape=(8, 8), frame_device=torch.device("cpu"),
            masks=masks, primary_raw=torch.zeros((n_frames, 3, 256, 256)),
            keep_start=keep_start, keep_end=keep_end, crossfade_weights=None,
            enlarged_bboxes=[(1, 1, 5, 5)] * n_frames, crop_shapes=[(4, 4)] * n_frames,
            pad_offsets=[(126, 126)] * n_frames, resize_shapes=[(4, 4)] * n_frames,
        )

    def test_empty_returns_none(self):
        assert Pipeline._earliest_blocking_seqs({}) is None

    def test_single_clip(self):
        pr = self._make_pr(start_frame=100, n_frames=180)
        result = Pipeline._earliest_blocking_seqs({0: pr})
        assert result == {0}

    def test_two_clips_same_start_frame(self):
        pr0 = self._make_pr(start_frame=100, n_frames=180)
        pr1 = self._make_pr(start_frame=100, n_frames=180)
        result = Pipeline._earliest_blocking_seqs({0: pr0, 1: pr1})
        assert result == {0, 1}

    def test_non_overlapping_clips_returns_earliest_only(self):
        pr0 = self._make_pr(start_frame=0, n_frames=180)
        pr1 = self._make_pr(start_frame=200, n_frames=180)
        result = Pipeline._earliest_blocking_seqs({0: pr0, 1: pr1})
        assert result == {0}

    def test_overlapping_clips_different_starts(self):
        pr0 = self._make_pr(start_frame=0, n_frames=180)
        pr1 = self._make_pr(start_frame=50, n_frames=180)
        result = Pipeline._earliest_blocking_seqs({0: pr0, 1: pr1})
        assert result == {0}

    def test_keep_start_shifts_earliest_frame(self):
        pr0 = self._make_pr(start_frame=0, n_frames=180, keep_start=20, keep_end=180)
        pr1 = self._make_pr(start_frame=10, n_frames=180, keep_start=0, keep_end=180)
        result = Pipeline._earliest_blocking_seqs({0: pr0, 1: pr1})
        assert result == {1}
