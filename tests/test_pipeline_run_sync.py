import threading
from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.frame_queue import FrameQueue
from jasna.media import VideoMetadata
from jasna.pipeline import Pipeline
from jasna.pipeline_items import ClipRestoreItem, PrimaryRestoreResult, SecondaryRestoreResult
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
        rest_pipeline.secondary_preferred_queue_size = 2
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


class TestPipelineRunSync:
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
            patch("jasna.pipeline.torch.inference_mode", return_value=_mock_inference_mode()),
            patch("jasna.pipeline.torch.cuda.mem_get_info", return_value=(8 * 1024**3, 24 * 1024**3)),
        ):
            p.run()

        mock_encoder.encode.assert_not_called()

    def test_run_decode_gap_backpressure_stalls_when_no_clips_emitted(self):
        p = _make_pipeline()
        p.max_clip_size = 2
        p.temporal_overlap = 0

        frames_t = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)
        mock_reader = MagicMock()
        mock_reader.__enter__ = MagicMock(return_value=mock_reader)
        mock_reader.__exit__ = MagicMock(return_value=False)
        mock_reader.frames.return_value = iter([
            (frames_t, [0, 1]),
            (frames_t, [2, 3]),
            (frames_t, [4, 5]),
            (frames_t, [6, 7]),
            (frames_t, [8, 9]),
        ])

        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)

        from jasna.pipeline_processing import BatchProcessResult

        call_count = 0

        def fake_process_batch(**kwargs):
            nonlocal call_count
            call_count += 1
            fb = kwargs["frame_buffer"]
            frames = kwargs["frames"]
            pts_list = kwargs["pts_list"]
            start_idx = kwargs["start_frame_idx"]
            for i, pts in enumerate(pts_list):
                fb.add_frame(start_idx + i, pts=int(pts), frame=frames[i], clip_track_ids={999})
            return BatchProcessResult(next_frame_idx=start_idx + len(pts_list), clips_emitted=0)

        with (
            patch("jasna.pipeline.get_video_meta_data", return_value=_fake_metadata()),
            patch("jasna.pipeline.NvidiaVideoReader", return_value=mock_reader),
            patch("jasna.pipeline.NvidiaVideoEncoder", return_value=mock_encoder),
            patch("jasna.pipeline.process_frame_batch", side_effect=fake_process_batch),
            patch("jasna.pipeline.finalize_processing"),
            patch("jasna.pipeline.torch.cuda.set_device"),
            patch("jasna.pipeline.torch.inference_mode", return_value=_mock_inference_mode()),
            patch("jasna.pipeline.torch.cuda.mem_get_info", return_value=(8 * 1024**3, 24 * 1024**3)),
            patch.object(p, "_wait_for_decode_fb_drain", side_effect=RuntimeError("decode stalled")),
        ):
            with pytest.raises(RuntimeError, match="decode stalled"):
                p.run()

        assert call_count == 2

    def test_run_decode_gap_backpressure_resets_on_clip_emit(self):
        p = _make_pipeline()
        p.max_clip_size = 4
        p.temporal_overlap = 0

        frames_t = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)
        mock_reader = MagicMock()
        mock_reader.__enter__ = MagicMock(return_value=mock_reader)
        mock_reader.__exit__ = MagicMock(return_value=False)
        mock_reader.frames.return_value = iter([
            (frames_t, [0, 1]),
            (frames_t, [2, 3]),
            (frames_t, [4, 5]),
            (frames_t, [6, 7]),
            (frames_t, [8, 9]),
        ])

        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)

        from jasna.pipeline_processing import BatchProcessResult

        call_count = 0

        def fake_process_batch(**kwargs):
            nonlocal call_count
            call_count += 1
            fb = kwargs["frame_buffer"]
            frames = kwargs["frames"]
            pts_list = kwargs["pts_list"]
            start_idx = kwargs["start_frame_idx"]
            for i, pts in enumerate(pts_list):
                fb.add_frame(start_idx + i, pts=int(pts), frame=frames[i], clip_track_ids={999})
            return BatchProcessResult(next_frame_idx=start_idx + len(pts_list), clips_emitted=1)

        with (
            patch("jasna.pipeline.get_video_meta_data", return_value=_fake_metadata()),
            patch("jasna.pipeline.NvidiaVideoReader", return_value=mock_reader),
            patch("jasna.pipeline.NvidiaVideoEncoder", return_value=mock_encoder),
            patch("jasna.pipeline.process_frame_batch", side_effect=fake_process_batch),
            patch("jasna.pipeline.finalize_processing"),
            patch("jasna.pipeline.torch.cuda.set_device"),
            patch("jasna.pipeline.torch.inference_mode", return_value=_mock_inference_mode()),
            patch("jasna.pipeline.torch.cuda.mem_get_info", return_value=(8 * 1024**3, 24 * 1024**3)),
            patch.object(p, "_wait_for_decode_fb_drain", side_effect=RuntimeError("decode stalled")),
        ):
            p.run()

        assert call_count == 5

    def test_should_offload_frames_true_when_free_below_headroom(self):
        p = _make_pipeline()
        p._VRAM_FREE_HEADROOM_BYTES = 1024 ** 3
        p._VRAM_LIMIT_OVERRIDE_GB = None
        free = 512 * 1024**2
        with patch("jasna.pipeline.torch.cuda.mem_get_info", return_value=(free, 24 * 1024**3)):
            should, used, threshold = p._should_offload_frames()
            assert should is True

    def test_should_offload_frames_false_when_free_above_headroom(self):
        p = _make_pipeline()
        p._VRAM_FREE_HEADROOM_BYTES = 1024 ** 3
        p._VRAM_LIMIT_OVERRIDE_GB = None
        free = 2 * 1024**3
        with patch("jasna.pipeline.torch.cuda.mem_get_info", return_value=(free, 24 * 1024**3)):
            should, used, threshold = p._should_offload_frames()
            assert should is False

    def test_should_offload_frames_with_vram_limit_override(self):
        p = _make_pipeline()
        p._VRAM_FREE_HEADROOM_BYTES = 1024 ** 3
        p._VRAM_LIMIT_OVERRIDE_GB = 10.0
        total = 24 * 1024**3
        used_over = int(9.5 * 1024**3)
        with patch("jasna.pipeline.torch.cuda.mem_get_info", return_value=(total - used_over, total)):
            should, used, threshold = p._should_offload_frames()
            assert should is True
        used_under = int(8 * 1024**3)
        with patch("jasna.pipeline.torch.cuda.mem_get_info", return_value=(total - used_under, total)):
            should, used, threshold = p._should_offload_frames()
            assert should is False

    def test_run_full_thread_flow(self):
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

        def fake_blend(sr, fb):
            for i in range(2):
                pending = fb.frames.get(i)
                if pending:
                    pending.pending_clips.discard(42)
                    yield

        p.restoration_pipeline.blend_secondary_result.side_effect = fake_blend

        with (
            patch("jasna.pipeline.get_video_meta_data", return_value=_fake_metadata()),
            patch("jasna.pipeline.NvidiaVideoReader", return_value=mock_reader),
            patch("jasna.pipeline.NvidiaVideoEncoder", return_value=mock_encoder),
            patch("jasna.pipeline.process_frame_batch", side_effect=fake_process_batch),
            patch("jasna.pipeline.finalize_processing"),
            patch("jasna.pipeline.torch.cuda.set_device"),
            patch("jasna.pipeline.torch.inference_mode", return_value=_mock_inference_mode()),
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
        mock_reader = MagicMock()
        mock_reader.__enter__ = MagicMock(return_value=mock_reader)
        mock_reader.__exit__ = MagicMock(return_value=False)
        mock_reader.frames.return_value = iter([(frames_t, [0, 1])])

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
            fb = kwargs["frame_buffer"]
            cq = kwargs["clip_queue"]
            fb.add_frame(0, pts=0, frame=frames_t[0], clip_track_ids={1})
            fb.add_frame(1, pts=1, frame=frames_t[1], clip_track_ids={1})
            cq.put(ClipRestoreItem(clip=clip, frames=[frames_t[0], frames_t[1]], keep_start=0, keep_end=2, crossfade_weights=None))
            return BatchProcessResult(next_frame_idx=2, clips_emitted=1)

        p.restoration_pipeline.prepare_and_run_primary.side_effect = RuntimeError("primary boom")

        with (
            patch("jasna.pipeline.get_video_meta_data", return_value=_fake_metadata()),
            patch("jasna.pipeline.NvidiaVideoReader", return_value=mock_reader),
            patch("jasna.pipeline.NvidiaVideoEncoder", return_value=mock_encoder),
            patch("jasna.pipeline.process_frame_batch", side_effect=fake_process_batch),
            patch("jasna.pipeline.finalize_processing"),
            patch("jasna.pipeline.torch.cuda.set_device"),
            patch("jasna.pipeline.torch.inference_mode", return_value=_mock_inference_mode()),
            patch("jasna.pipeline.torch.cuda.mem_get_info", return_value=(8 * 1024**3, 24 * 1024**3)),
        ):
            with pytest.raises(RuntimeError, match="primary boom"):
                p.run()

    def test_run_secondary_error_propagates(self):
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
            track_id=1,
            start_frame=0,
            mask_resolution=(2, 2),
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
            patch("jasna.pipeline.NvidiaVideoReader", return_value=mock_reader),
            patch("jasna.pipeline.NvidiaVideoEncoder", return_value=mock_encoder),
            patch("jasna.pipeline.process_frame_batch", side_effect=fake_process_batch),
            patch("jasna.pipeline.finalize_processing"),
            patch("jasna.pipeline.torch.cuda.set_device"),
            patch("jasna.pipeline.torch.inference_mode", return_value=_mock_inference_mode()),
            patch("jasna.pipeline.torch.cuda.mem_get_info", return_value=(8 * 1024**3, 24 * 1024**3)),
        ):
            with pytest.raises(RuntimeError, match="secondary boom"):
                p.run()


class TestVramBackpressure:
    def test_primary_starving_predicate(self):
        primary_idle = threading.Event()
        q = FrameQueue(max_frames=100)

        assert not (primary_idle.is_set() and q.empty())

        primary_idle.set()
        assert primary_idle.is_set() and q.empty()

        q.put("item", frame_count=1)
        assert not (primary_idle.is_set() and q.empty())

        q.get()
        primary_idle.clear()
        assert not (primary_idle.is_set() and q.empty())

    def test_offload_bytes_to_free_uses_minimal_excess(self):
        used = 10 * 1024 ** 3
        threshold = 9 * 1024 ** 3
        excess = used - threshold

        bytes_to_free = int(excess * 1.2)
        assert bytes_to_free == int((used - threshold) * 1.2)
        assert bytes_to_free < used - threshold + 512 * 1024 ** 2

    def test_hysteresis_band_no_clear_between_soft_and_hard(self):
        p = _make_pipeline()
        p._VRAM_FREE_HEADROOM_BYTES = 1024 ** 3
        p._VRAM_LIMIT_OVERRIDE_GB = 10.0
        p._VRAM_PRESSURE_HYSTERESIS_BYTES = 512 * 1024 ** 2
        total = 24 * 1024 ** 3

        with patch("jasna.pipeline.torch.cuda.mem_get_info", return_value=(total - int(10.5 * 1024 ** 3), total)):
            over, used, threshold = p._should_offload_frames()
            assert over is True
            soft = threshold - p._VRAM_PRESSURE_HYSTERESIS_BYTES

        used_in_band = threshold - 256 * 1024 ** 2
        assert used_in_band > soft
        assert used_in_band < threshold

        used_below_soft = soft - 100 * 1024 ** 2
        assert used_below_soft < soft

    def test_vram_backpressure_no_deadlock_with_tight_vram(self):
        p = _make_pipeline()
        p._VRAM_LIMIT_OVERRIDE_GB = 2.0
        p._VRAM_FREE_HEADROOM_BYTES = 512 * 1024 ** 2
        p._VRAM_PRESSURE_HYSTERESIS_BYTES = 256 * 1024 ** 2
        p._VRAM_BP_MAX_WAIT_SECONDS = 0.1

        frames_t = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)
        mock_reader = MagicMock()
        mock_reader.__enter__ = MagicMock(return_value=mock_reader)
        mock_reader.__exit__ = MagicMock(return_value=False)
        mock_reader.frames.return_value = iter([
            (frames_t, [0, 1]),
            (frames_t, [2, 3]),
        ])

        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)

        from jasna.pipeline_processing import BatchProcessResult

        call_count = 0

        def fake_process_batch(**kwargs):
            nonlocal call_count
            call_count += 1
            fb = kwargs["frame_buffer"]
            frames = kwargs["frames"]
            pts_list = kwargs["pts_list"]
            start_idx = kwargs["start_frame_idx"]
            for i, pts in enumerate(pts_list):
                fb.add_frame(start_idx + i, pts=int(pts), frame=frames[i], clip_track_ids=set())
            return BatchProcessResult(next_frame_idx=start_idx + len(pts_list), clips_emitted=0)

        total = 24 * 1024 ** 3
        tight_free = 200 * 1024 ** 2

        with (
            patch("jasna.pipeline.get_video_meta_data", return_value=_fake_metadata()),
            patch("jasna.pipeline.NvidiaVideoReader", return_value=mock_reader),
            patch("jasna.pipeline.NvidiaVideoEncoder", return_value=mock_encoder),
            patch("jasna.pipeline.process_frame_batch", side_effect=fake_process_batch),
            patch("jasna.pipeline.finalize_processing"),
            patch("jasna.pipeline.torch.cuda.set_device"),
            patch("jasna.pipeline.torch.inference_mode", return_value=_mock_inference_mode()),
            patch("jasna.pipeline.torch.cuda.mem_get_info", return_value=(tight_free, total)),
        ):
            p.run()

        assert call_count == 2

    def test_vram_starvation_escape_when_primary_idle(self):
        p = _make_pipeline()
        p._VRAM_LIMIT_OVERRIDE_GB = 2.0
        p._VRAM_FREE_HEADROOM_BYTES = 512 * 1024 ** 2
        p._VRAM_PRESSURE_HYSTERESIS_BYTES = 256 * 1024 ** 2
        p._VRAM_BP_MAX_WAIT_SECONDS = 30.0

        frames_t = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)
        mock_reader = MagicMock()
        mock_reader.__enter__ = MagicMock(return_value=mock_reader)
        mock_reader.__exit__ = MagicMock(return_value=False)
        mock_reader.frames.return_value = iter([
            (frames_t, [0, 1]),
            (frames_t, [2, 3]),
        ])

        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)

        from jasna.pipeline_processing import BatchProcessResult

        call_count = 0

        def fake_process_batch(**kwargs):
            nonlocal call_count
            call_count += 1
            fb = kwargs["frame_buffer"]
            frames = kwargs["frames"]
            pts_list = kwargs["pts_list"]
            start_idx = kwargs["start_frame_idx"]
            for i, pts in enumerate(pts_list):
                fb.add_frame(start_idx + i, pts=int(pts), frame=frames[i], clip_track_ids=set())
            return BatchProcessResult(next_frame_idx=start_idx + len(pts_list), clips_emitted=0)

        total = 24 * 1024 ** 3
        tight_free = 200 * 1024 ** 2

        with (
            patch("jasna.pipeline.get_video_meta_data", return_value=_fake_metadata()),
            patch("jasna.pipeline.NvidiaVideoReader", return_value=mock_reader),
            patch("jasna.pipeline.NvidiaVideoEncoder", return_value=mock_encoder),
            patch("jasna.pipeline.process_frame_batch", side_effect=fake_process_batch),
            patch("jasna.pipeline.finalize_processing"),
            patch("jasna.pipeline.torch.cuda.set_device"),
            patch("jasna.pipeline.torch.inference_mode", return_value=_mock_inference_mode()),
            patch("jasna.pipeline.torch.cuda.mem_get_info", return_value=(tight_free, total)),
        ):
            p.run()

        assert call_count == 2

