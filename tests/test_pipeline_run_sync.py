from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.crop_buffer import RawCrop
from jasna.media import VideoMetadata
from jasna.pipeline import Pipeline
from jasna.pipeline_items import ClipRestoreItem, FrameMeta, PrimaryRestoreResult, SecondaryRestoreResult
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


def _make_pipeline() -> Pipeline:
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


def _mock_inference_mode() -> MagicMock:
    return MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))


def _make_two_readers(frames_batches: list[tuple[torch.Tensor, list[int]]]):
    """Create a side_effect for NvidiaVideoReader that returns two independent readers."""
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


class TestPipelineRunSync:
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
            patch("jasna.pipeline_threads.torch.inference_mode", return_value=_mock_inference_mode()),
            patch("jasna.pipeline.torch.cuda.mem_get_info", return_value=(8 * 1024**3, 24 * 1024**3)),
        ):
            p.run()

        mock_encoder.encode.assert_not_called()

    def test_run_full_thread_flow(self):
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

        restored_frames = [torch.randint(0, 255, (3, 256, 256), dtype=torch.uint8)] * 2
        sr_result = SecondaryRestoreResult(
            track_id=clip.track_id,
            start_frame=clip.start_frame,
            frame_count=2,
            frame_shape=(8, 8),
            frame_device=frames_t[0].device,
            masks=clip.masks,
            restored_frames=restored_frames,
            keep_start=0,
            keep_end=2,
            crossfade_weights=None,
            enlarged_bboxes=[(1, 1, 5, 5)] * 2,
            crop_shapes=[(4, 4)] * 2,
            pad_offsets=[(126, 126)] * 2,
            resize_shapes=[(4, 4)] * 2,
        )
        p.restoration_pipeline._run_secondary.return_value = restored_frames
        p.restoration_pipeline.build_secondary_result.return_value = sr_result

        with (
            patch("jasna.pipeline.get_video_meta_data", return_value=_fake_metadata()),
            patch("jasna.pipeline_threads.NvidiaVideoReader", reader_cls),
            patch("jasna.pipeline.NvidiaVideoEncoder", return_value=mock_encoder),
            patch("jasna.pipeline_threads.process_frame_batch", side_effect=fake_process_batch),
            patch("jasna.pipeline_threads.finalize_processing"),
            patch("jasna.pipeline_threads.torch.cuda.set_device"),
            patch("jasna.pipeline_threads.torch.inference_mode", return_value=_mock_inference_mode()),
            patch("jasna.pipeline.torch.cuda.mem_get_info", return_value=(8 * 1024**3, 24 * 1024**3)),
        ):
            p.run()

        p.restoration_pipeline.prepare_and_run_primary.assert_called_once()
        p.restoration_pipeline._run_secondary.assert_called_once()
        p.restoration_pipeline.build_secondary_result.assert_called_once()
        assert mock_encoder.encode.call_count == 2

    def test_run_primary_error_propagates(self):
        p = _make_pipeline()

        frames_t = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)
        batches = [(frames_t, [0, 1])]
        reader_cls, _, _ = _make_two_readers(batches)

        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)

        clip = TrackedClip(
            track_id=1,
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
            patch("jasna.pipeline_threads.torch.inference_mode", return_value=_mock_inference_mode()),
            patch("jasna.pipeline.torch.cuda.mem_get_info", return_value=(8 * 1024**3, 24 * 1024**3)),
        ):
            with pytest.raises(RuntimeError, match="primary boom"):
                p.run()

    def test_run_secondary_error_propagates(self):
        p = _make_pipeline()

        frames_t = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)
        batches = [(frames_t, [0, 1])]
        reader_cls, _, _ = _make_two_readers(batches)

        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)

        clip = TrackedClip(
            track_id=1,
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
        p.restoration_pipeline._run_secondary.side_effect = RuntimeError("secondary boom")

        with (
            patch("jasna.pipeline.get_video_meta_data", return_value=_fake_metadata()),
            patch("jasna.pipeline_threads.NvidiaVideoReader", reader_cls),
            patch("jasna.pipeline.NvidiaVideoEncoder", return_value=mock_encoder),
            patch("jasna.pipeline_threads.process_frame_batch", side_effect=fake_process_batch),
            patch("jasna.pipeline_threads.finalize_processing"),
            patch("jasna.pipeline_threads.torch.cuda.set_device"),
            patch("jasna.pipeline_threads.torch.inference_mode", return_value=_mock_inference_mode()),
            patch("jasna.pipeline.torch.cuda.mem_get_info", return_value=(8 * 1024**3, 24 * 1024**3)),
        ):
            with pytest.raises(RuntimeError, match="secondary boom"):
                p.run()

