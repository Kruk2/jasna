from pathlib import Path
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from jasna.pipeline import Pipeline
from jasna.vr180 import SbsDetectionAdapter


def _make_pipeline(**overrides):
    defaults = dict(
        input_video=Path("in.mp4"),
        output_video=Path("out.mkv"),
        detection_model_name="rfdetr-v5",
        detection_model_path=Path("model.onnx"),
        detection_score_threshold=0.25,
        restoration_pipeline=MagicMock(),
        codec="hevc",
        encoder_settings={},
        batch_size=4,
        device=torch.device("cpu"),
        max_clip_size=60,
        temporal_overlap=8,
        enable_crossfade=True,
        fp16=True,
    )
    defaults.update(overrides)

    with (
        patch("jasna.mosaic.rfdetr.RfDetrMosaicDetectionModel"),
        patch("jasna.mosaic.yolo.YoloMosaicDetectionModel"),
    ):
        return Pipeline(**defaults)


class TestPipelineInit:
    def test_stores_basic_attributes(self):
        p = _make_pipeline(batch_size=2, max_clip_size=30, temporal_overlap=4)
        assert p.batch_size == 2
        assert p.max_clip_size == 30
        assert p.temporal_overlap == 4
        assert p.codec == "hevc"
        assert p.enable_crossfade is True

    def test_rfdetr_model_created(self):
        with (
            patch("jasna.mosaic.rfdetr.RfDetrMosaicDetectionModel") as mock_rf,
            patch("jasna.mosaic.yolo.YoloMosaicDetectionModel") as mock_yolo,
        ):
            Pipeline(
                input_video=Path("in.mp4"),
                output_video=Path("out.mkv"),
                detection_model_name="rfdetr-v5",
                detection_model_path=Path("model.onnx"),
                detection_score_threshold=0.25,
                restoration_pipeline=MagicMock(),
                codec="hevc",
                encoder_settings={},
                batch_size=4,
                device=torch.device("cpu"),
                max_clip_size=60,
                temporal_overlap=8,
                fp16=True,
            )
            mock_rf.assert_called_once()
            mock_yolo.assert_not_called()

    def test_yolo_model_created(self):
        with (
            patch("jasna.mosaic.rfdetr.RfDetrMosaicDetectionModel") as mock_rf,
            patch("jasna.mosaic.yolo.YoloMosaicDetectionModel") as mock_yolo,
        ):
            Pipeline(
                input_video=Path("in.mp4"),
                output_video=Path("out.mkv"),
                detection_model_name="lada-yolo-v4",
                detection_model_path=Path("model.pt"),
                detection_score_threshold=0.25,
                restoration_pipeline=MagicMock(),
                codec="hevc",
                encoder_settings={},
                batch_size=4,
                device=torch.device("cpu"),
                max_clip_size=60,
                temporal_overlap=8,
                fp16=True,
            )
            mock_yolo.assert_called_once()
            mock_rf.assert_not_called()

    def test_crossfade_disabled(self):
        p = _make_pipeline(enable_crossfade=False)
        assert p.enable_crossfade is False

    def test_codec_forwarded_unchanged(self):
        for codec in ("hevc", "h264", "av1"):
            assert _make_pipeline(codec=codec).codec == codec

    def test_progress_callback(self):
        cb = MagicMock()
        p = _make_pipeline(progress_callback=cb)
        assert p.progress_callback is cb

    def test_retarget_high_fps_defaults_off_and_can_be_enabled(self):
        assert _make_pipeline().retarget_high_fps is False
        assert _make_pipeline(retarget_high_fps=True).retarget_high_fps is True

    def test_configure_vr_wraps_detector_for_direct_sbs(self):
        pipeline = _make_pipeline(
            input_video=Path("VRKM-0001.mp4"),
            vr_mode="auto",
        )
        metadata = SimpleNamespace(
            video_width=200,
            video_height=100,
            sample_aspect_ratio=Fraction(1, 1),
            stereo_layout="",
            spherical_projection="",
        )

        pipeline.configure_vr(metadata)

        assert pipeline._vr_resolution.resolved == "sbs"
        assert isinstance(pipeline._job_detection_model, SbsDetectionAdapter)
        assert pipeline._vr_projector is None

    def test_configure_vr_builds_fisheye_projector(self):
        pipeline = _make_pipeline(
            input_video=Path("FSVSS-0001.mp4"),
            vr_mode="auto",
        )
        metadata = SimpleNamespace(
            video_width=200,
            video_height=100,
            sample_aspect_ratio=Fraction(1, 1),
            stereo_layout="",
            spherical_projection="",
        )

        pipeline.configure_vr(metadata)

        assert pipeline._vr_resolution.resolved == "sbs-fisheye"
        assert isinstance(pipeline._job_detection_model, SbsDetectionAdapter)
        assert pipeline._vr_projector.eye_width == 100
