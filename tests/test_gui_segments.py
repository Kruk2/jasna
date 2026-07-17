from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from jasna.gui.models import AppSettings, JobItem, JobStatus
from jasna.gui.processor import Processor
from jasna.gui.video_session import VideoSession
from jasna.segments import SegmentRange


def test_pending_job_segments_can_be_replaced() -> None:
    job = JobItem(Path("video.mp4"))
    segments = (SegmentRange(1, 2),)
    assert job.try_set_segments(segments)
    assert job.snapshot_segments() == segments


def test_begin_processing_atomically_freezes_segments() -> None:
    job = JobItem(
        Path("video.mp4"),
        segments=(SegmentRange(1, 2),),
        detection_model="lada-yolo-v4",
        detection_score_threshold=0.4,
    )
    snapshot = job.begin_processing()
    assert snapshot.segments == (SegmentRange(1, 2),)
    assert snapshot.detection_model == "lada-yolo-v4"
    assert snapshot.detection_score_threshold == 0.4
    assert job.status is JobStatus.PROCESSING
    assert not job.try_set_segments((SegmentRange(3, 4),))
    assert job.begin_processing() is None


def test_processor_passes_frozen_segments_to_video_job(tmp_path) -> None:
    source = tmp_path / "video.mp4"
    source.touch()
    segments = (SegmentRange(1, 2),)
    job = JobItem(source, segments=segments)
    processor = Processor()
    processor._settings = AppSettings()
    processor._output_pattern = "{original}_restored.mp4"

    with (
        patch.object(processor, "_run_pipeline") as run_pipeline,
        patch("jasna.gui.processor._cleanup_torch"),
    ):
        processor._process_job(job)

    assert run_pipeline.call_args.kwargs["segments"] == segments
    assert job.status is JobStatus.COMPLETED


def test_processor_uses_each_videos_detection_overrides(tmp_path) -> None:
    source = tmp_path / "video.mp4"
    source.touch()
    job = JobItem(
        source,
        detection_model="lada-yolo-v4",
        detection_score_threshold=0.55,
    )
    processor = Processor()
    processor._settings = AppSettings(
        detection_model="rfdetr-v5",
        detection_score_threshold=0.25,
    )
    processor._output_pattern = "{original}_restored.mp4"

    with (
        patch.object(processor, "_run_pipeline") as run_pipeline,
        patch("jasna.gui.processor._cleanup_torch"),
    ):
        processor._process_job(job)

    settings = run_pipeline.call_args.kwargs["settings"]
    assert settings.detection_model == "lada-yolo-v4"
    assert settings.detection_score_threshold == 0.55
    assert processor._settings.detection_model == "rfdetr-v5"
    assert processor._settings.detection_score_threshold == 0.25


def test_video_job_passes_precomputed_splice_plan_to_pipeline(tmp_path) -> None:
    input_path = tmp_path / "video.mp4"
    output_path = tmp_path / "output.mp4"
    segments = (SegmentRange(1, 2),)
    metadata = MagicMock(codec_name="h264", duration=10.0)
    splice_plan = MagicMock()
    pipeline = MagicMock()
    processor = Processor()
    processor._settings = AppSettings()
    processor._video_session = VideoSession(
        device=MagicMock(),
        det_name="detector",
        detection_model_path=tmp_path / "detector.engine",
        restoration_pipeline=MagicMock(),
        secondary_restorer=None,
        lut_path=None,
    )
    processor._ensure_video_session = MagicMock()
    processor._prepare_job_detector = MagicMock(
        return_value=("detector", tmp_path / "detector.engine")
    )
    processor._build_encoder_settings = MagicMock(return_value={})

    with (
        patch("jasna.media.get_video_meta_data", return_value=metadata),
        patch("jasna.media.splice.validate_smart_render"),
        patch("jasna.media.splice.probe_keyframes", return_value=MagicMock()),
        patch("jasna.media.splice.build_splice_plan", return_value=splice_plan),
        patch("jasna.pipeline.Pipeline", return_value=pipeline) as pipeline_cls,
    ):
        processor._run_video_job(1, input_path, output_path, segments=segments)

    assert pipeline_cls.call_args.kwargs["splice_plan"] is splice_plan


def test_ensure_video_session_delegates_to_factory_and_close_unloads() -> None:
    processor = Processor()
    processor._settings = AppSettings()
    session = MagicMock()

    with patch("jasna.gui.processor.build_video_session", return_value=session) as build:
        processor._ensure_video_session()
        processor._ensure_video_session()

    build.assert_called_once()
    assert build.call_args.kwargs["disable_basicvsrpp_tensorrt"] is False
    assert processor._video_session is session

    processor._close_video_session()
    session.close.assert_called_once_with()
    assert processor._video_session is None
