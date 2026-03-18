from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from jasna.pipeline import Pipeline


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
        patch("jasna.pipeline.RfDetrMosaicDetectionModel"),
        patch("jasna.pipeline.YoloMosaicDetectionModel"),
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
            patch("jasna.pipeline.RfDetrMosaicDetectionModel") as mock_rf,
            patch("jasna.pipeline.YoloMosaicDetectionModel") as mock_yolo,
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
            patch("jasna.pipeline.RfDetrMosaicDetectionModel") as mock_rf,
            patch("jasna.pipeline.YoloMosaicDetectionModel") as mock_yolo,
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

    def test_working_directory(self):
        p = _make_pipeline(working_directory=Path("/tmp/work"))
        assert p.working_directory == Path("/tmp/work")

    def test_progress_callback(self):
        cb = MagicMock()
        p = _make_pipeline(progress_callback=cb)
        assert p.progress_callback is cb


