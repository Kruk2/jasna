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
            frames=[frames_t[0], frames_t[1]],
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
            frames=[frames_t[0], frames_t[1]],
            restored_frames=[torch.randint(0, 255, (3, 256, 256), dtype=torch.uint8)] * 2,
            keep_start=0,
            keep_end=2,
            crossfade_weights=None,
            enlarged_bboxes=[(1, 1, 5, 5)] * 2,
            crop_shapes=[(4, 4)] * 2,
            pad_offsets=[(126, 126)] * 2,
            resize_shapes=[(4, 4)] * 2,
        )
        p.restoration_pipeline.run_secondary_from_primary.return_value = sr_result

        def fake_blend(sr, fb):
            for i in range(2):
                pending = fb.frames.get(i)
                if pending:
                    pending.pending_clips.discard(42)

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
        p.restoration_pipeline.run_secondary_from_primary.assert_called_once()
        p.restoration_pipeline.blend_secondary_result.assert_called_once()
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

        def fake_secondary(pr):
            return MagicMock(spec=SecondaryRestoreResult)

        p.restoration_pipeline.run_secondary_from_primary = fake_secondary

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
            clip=clip, frames=[frames_t[0], frames_t[1]],
            primary_raw=torch.zeros((2, 3, 256, 256)),
            keep_start=0, keep_end=2, crossfade_weights=None,
            enlarged_bboxes=[(1, 1, 5, 5)] * 2, crop_shapes=[(4, 4)] * 2,
            pad_offsets=[(126, 126)] * 2, resize_shapes=[(4, 4)] * 2,
        )
        p.restoration_pipeline.prepare_and_run_primary.return_value = pr_result
        p.restoration_pipeline.run_secondary_from_primary.side_effect = RuntimeError("secondary boom")

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

    def test_run_pooled_secondary(self):
        """Cover lines 186, 283: pooled secondary path with num_workers > 1."""
        with (
            patch("jasna.pipeline.RfDetrMosaicDetectionModel"),
            patch("jasna.pipeline.YoloMosaicDetectionModel"),
        ):
            rest_pipeline = MagicMock()
            rest_pipeline.secondary_restorer = None
            rest_pipeline.secondary_num_workers = 2
            p = Pipeline(
                input_video=Path("in.mp4"), output_video=Path("out.mkv"),
                detection_model_name="rfdetr-v5", detection_model_path=Path("model.onnx"),
                detection_score_threshold=0.25, restoration_pipeline=rest_pipeline,
                codec="hevc", encoder_settings={}, batch_size=2,
                device=torch.device("cuda:0"), max_clip_size=60, temporal_overlap=8,
                fp16=True, disable_progress=True,
            )

        frames_t = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)
        mock_reader = MagicMock()
        mock_reader.__enter__ = MagicMock(return_value=mock_reader)
        mock_reader.__exit__ = MagicMock(return_value=False)
        mock_reader.frames.return_value = iter([(frames_t, [0, 1])])

        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)

        clip = TrackedClip(
            track_id=42, start_frame=0, mask_resolution=(2, 2),
            bboxes=[np.array([1, 1, 5, 5], dtype=np.float32)] * 2,
            masks=[torch.zeros((2, 2), dtype=torch.bool)] * 2,
        )

        from jasna.pipeline_processing import BatchProcessResult

        def fake_process_batch(**kwargs):
            fb = kwargs["frame_buffer"]
            cq = kwargs["clip_queue"]
            fb.add_frame(0, pts=0, frame=frames_t[0], clip_track_ids={42})
            fb.add_frame(1, pts=1, frame=frames_t[1], clip_track_ids={42})
            cq.put(ClipRestoreItem(clip=clip, frames=[frames_t[0], frames_t[1]], keep_start=0, keep_end=2, crossfade_weights=None))
            return BatchProcessResult(next_frame_idx=2)

        pr_result = PrimaryRestoreResult(
            clip=clip, frames=[frames_t[0], frames_t[1]],
            primary_raw=torch.zeros((2, 3, 256, 256)),
            keep_start=0, keep_end=2, crossfade_weights=None,
            enlarged_bboxes=[(1, 1, 5, 5)] * 2, crop_shapes=[(4, 4)] * 2,
            pad_offsets=[(126, 126)] * 2, resize_shapes=[(4, 4)] * 2,
        )
        rest_pipeline.prepare_and_run_primary.return_value = pr_result

        sr_result = SecondaryRestoreResult(
            clip=clip, frames=[frames_t[0], frames_t[1]],
            restored_frames=[torch.randint(0, 255, (3, 256, 256), dtype=torch.uint8)] * 2,
            keep_start=0, keep_end=2, crossfade_weights=None,
            enlarged_bboxes=[(1, 1, 5, 5)] * 2, crop_shapes=[(4, 4)] * 2,
            pad_offsets=[(126, 126)] * 2, resize_shapes=[(4, 4)] * 2,
        )
        rest_pipeline.run_secondary_from_primary.return_value = sr_result

        def fake_blend(sr, fb):
            for i in range(2):
                pending = fb.frames.get(i)
                if pending:
                    pending.pending_clips.discard(42)

        rest_pipeline.blend_secondary_result.side_effect = fake_blend

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

        rest_pipeline.run_secondary_from_primary.assert_called_once()
        assert mock_encoder.encode.call_count == 2

    def test_run_pooled_secondary_error_drains_queue(self):
        with (
            patch("jasna.pipeline.RfDetrMosaicDetectionModel"),
            patch("jasna.pipeline.YoloMosaicDetectionModel"),
        ):
            rest_pipeline = MagicMock()
            rest_pipeline.secondary_num_workers = 2
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

        clip = TrackedClip(
            track_id=7,
            start_frame=0,
            mask_resolution=(2, 2),
            bboxes=[np.array([1, 1, 5, 5], dtype=np.float32)] * 2,
            masks=[torch.zeros((2, 2), dtype=torch.bool)] * 2,
        )
        pr = PrimaryRestoreResult(
            clip=clip,
            frames=[torch.zeros((3, 8, 8), dtype=torch.uint8)] * 2,
            primary_raw=torch.zeros((2, 3, 256, 256)),
            keep_start=0,
            keep_end=2,
            crossfade_weights=None,
            enlarged_bboxes=[(1, 1, 5, 5)] * 2,
            crop_shapes=[(4, 4)] * 2,
            pad_offsets=[(126, 126)] * 2,
            resize_shapes=[(4, 4)] * 2,
        )

        secondary_queue: Queue[PrimaryRestoreResult | object] = Queue()
        encode_queue: Queue[SecondaryRestoreResult | object] = Queue()
        secondary_queue.put(pr)
        secondary_queue.put(pr)
        secondary_queue.put(_SENTINEL)

        calls = {"n": 0}

        def fail_first(_):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("secondary boom")
            return MagicMock(spec=SecondaryRestoreResult)

        rest_pipeline.run_secondary_from_primary.side_effect = fail_first

        with pytest.raises(RuntimeError, match="secondary boom"):
            p._run_pooled_secondary(2, secondary_queue, encode_queue)

        assert secondary_queue.empty()

    def test_run_async_secondary(self):
        """Cover _run_async_secondary: push_clip → flush → pop_completed → build_secondary_result."""
        p = _make_pipeline()

        clip = TrackedClip(
            track_id=1, start_frame=0, mask_resolution=(2, 2),
            bboxes=[np.array([1, 1, 5, 5], dtype=np.float32)] * 2,
            masks=[torch.zeros((2, 2), dtype=torch.bool)] * 2,
        )
        pr = PrimaryRestoreResult(
            clip=clip, frames=[torch.zeros((3, 8, 8), dtype=torch.uint8)] * 2,
            primary_raw=torch.zeros((2, 3, 256, 256)),
            keep_start=0, keep_end=2, crossfade_weights=None,
            enlarged_bboxes=[(1, 1, 5, 5)] * 2, crop_shapes=[(4, 4)] * 2,
            pad_offsets=[(126, 126)] * 2, resize_shapes=[(4, 4)] * 2,
        )

        restored = [torch.randint(0, 255, (3, 256, 256), dtype=torch.uint8)] * 2
        sr_result = SecondaryRestoreResult(
            clip=clip, frames=pr.frames,
            restored_frames=restored,
            keep_start=0, keep_end=2, crossfade_weights=None,
            enlarged_bboxes=pr.enlarged_bboxes, crop_shapes=pr.crop_shapes,
            pad_offsets=pr.pad_offsets, resize_shapes=pr.resize_shapes,
        )

        restorer = MagicMock()
        restorer.push_clip.return_value = 0
        restorer.pop_completed.side_effect = [[], [(0, restored)], []]
        restorer.flush_all.return_value = None
        p.restoration_pipeline.secondary_restorer = restorer
        p.restoration_pipeline.build_secondary_result.return_value = sr_result

        secondary_queue: Queue = Queue()
        encode_queue: Queue = Queue()
        secondary_queue.put(pr)
        secondary_queue.put(_SENTINEL)

        p._run_async_secondary(secondary_queue, encode_queue)

        restorer.push_clip.assert_called_once()
        assert not encode_queue.empty()
        result = encode_queue.get()
        assert result is sr_result

    def test_run_async_secondary_gap_flush(self):
        """Cover gap-based flush in _run_async_secondary when no new clips arrive."""
        p = _make_pipeline()
        p._FLUSH_GAP_SECONDS = 0.0

        clip = TrackedClip(
            track_id=1, start_frame=0, mask_resolution=(2, 2),
            bboxes=[np.array([1, 1, 5, 5], dtype=np.float32)] * 2,
            masks=[torch.zeros((2, 2), dtype=torch.bool)] * 2,
        )
        pr = PrimaryRestoreResult(
            clip=clip, frames=[torch.zeros((3, 8, 8), dtype=torch.uint8)] * 2,
            primary_raw=torch.zeros((2, 3, 256, 256)),
            keep_start=0, keep_end=2, crossfade_weights=None,
            enlarged_bboxes=[(1, 1, 5, 5)] * 2, crop_shapes=[(4, 4)] * 2,
            pad_offsets=[(126, 126)] * 2, resize_shapes=[(4, 4)] * 2,
        )

        restored = [torch.randint(0, 255, (3, 256, 256), dtype=torch.uint8)] * 2
        sr_result = SecondaryRestoreResult(
            clip=clip, frames=pr.frames,
            restored_frames=restored,
            keep_start=0, keep_end=2, crossfade_weights=None,
            enlarged_bboxes=pr.enlarged_bboxes, crop_shapes=pr.crop_shapes,
            pad_offsets=pr.pad_offsets, resize_shapes=pr.resize_shapes,
        )

        pop_results = iter([[], [], [(0, restored)]])

        restorer = MagicMock()
        restorer.push_clip.return_value = 0
        restorer.pop_completed.side_effect = lambda: next(pop_results, [])
        restorer.flush_all.return_value = None
        p.restoration_pipeline.secondary_restorer = restorer
        p.restoration_pipeline.build_secondary_result.return_value = sr_result

        secondary_queue: Queue = Queue()
        encode_queue: Queue = Queue()
        secondary_queue.put(pr)

        import threading
        def put_sentinel_later():
            import time
            time.sleep(1.5)
            secondary_queue.put(_SENTINEL)

        t = threading.Thread(target=put_sentinel_later, daemon=True)
        t.start()

        p._run_async_secondary(secondary_queue, encode_queue)
        t.join(timeout=3)

        restorer.flush_all.assert_called()
        assert not encode_queue.empty()

    def test_run_with_progress_callback(self):
        cb = MagicMock()
        with (
            patch("jasna.pipeline.RfDetrMosaicDetectionModel"),
            patch("jasna.pipeline.YoloMosaicDetectionModel"),
        ):
            rest_pipeline = MagicMock()
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
