from fractions import Fraction
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, patch, call

import torch
import pytest
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.media import VideoMetadata
from jasna.pipeline import Pipeline
from jasna.pipeline_items import ClipRestoreItem, PrimaryRestoreResult, SecondaryRestoreResult, _SENTINEL


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
