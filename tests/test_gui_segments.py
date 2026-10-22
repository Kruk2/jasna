from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jasna.gui.models import AppSettings, JobItem, JobStatus
from jasna.gui.processor import Processor
from jasna.segments import SegmentRange
from jasna.session_factory import RestorationSession

from factories import session_config, no_resume_conflict

pytestmark = pytest.mark.usefixtures("nvidia_vendor")


def test_pending_job_segments_can_be_replaced() -> None:
    job = JobItem(Path("video.mp4"))
    segments = (SegmentRange(1, 2),)
    assert job.try_set_video_options(
        segments, detection_model="rfdetr-v6", detection_score_threshold=0.35, vr_projection="auto"
    )
    assert job.snapshot_segments() == segments


def test_begin_processing_atomically_freezes_segments() -> None:
    job = JobItem(
        Path("video.mp4"),
        segments=(SegmentRange(1, 2),),
        detection_model="lada-yolo-v4",
        detection_score_threshold=0.4,
        vr_projection="gnomonic",
    )
    snapshot = job.begin_processing()
    assert snapshot.segments == (SegmentRange(1, 2),)
    assert snapshot.detection_model == "lada-yolo-v4"
    assert snapshot.detection_score_threshold == 0.4
    assert snapshot.vr_projection == "gnomonic"
    assert job.status is JobStatus.PROCESSING
    assert not job.try_set_video_options(
        (SegmentRange(3, 4),), detection_model="rfdetr-v6", detection_score_threshold=0.35, vr_projection="auto"
    )
    assert job.begin_processing() is None


def test_processor_passes_frozen_segments_to_video_job(tmp_path) -> None:
    source = tmp_path / "video.mp4"
    source.touch()
    segments = (SegmentRange(1, 2),)
    job = JobItem(source, segments=segments)
    processor = Processor(on_resume_conflict=no_resume_conflict)
    processor._settings = AppSettings()
    processor._output_pattern = "{original}_restored.mp4"

    with (
        patch.object(processor, "_run_pipeline") as run_pipeline,
        patch("jasna.gui.processor._cleanup_torch"),
    ):
        processor._process_job(job)

    assert run_pipeline.call_args.kwargs["segments"] == segments
    assert job.status is JobStatus.COMPLETED


def test_processor_uses_each_videos_detection_and_projection_overrides(tmp_path) -> None:
    source = tmp_path / "video.mp4"
    source.touch()
    job = JobItem(
        source,
        detection_model="lada-yolo-v4",
        detection_score_threshold=0.55,
        vr_projection="fisheye",
    )
    processor = Processor(on_resume_conflict=no_resume_conflict)
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
    assert settings.vr_projection == "fisheye"
    assert processor._settings.detection_model == "rfdetr-v5"
    assert processor._settings.detection_score_threshold == 0.25
    assert processor._settings.vr_projection == "auto"


def test_video_job_passes_precomputed_splice_plan_to_pipeline(tmp_path) -> None:
    input_path = tmp_path / "video.mp4"
    output_path = tmp_path / "output.mp4"
    segments = (SegmentRange(1, 2),)
    metadata = MagicMock(codec_name="h264", duration=10.0)
    splice_plan = MagicMock()
    pipeline = MagicMock()
    processor = Processor(on_resume_conflict=no_resume_conflict)
    processor._settings = AppSettings()
    processor._video_session = RestorationSession(device=MagicMock(), restoration_pipeline=MagicMock())
    processor._ensure_video_session = MagicMock()
    processor._prepare_job_detector = MagicMock()
    
    with (
        patch("jasna.media.probe.get_video_meta_data", return_value=metadata),
        patch("jasna.media.splice.validate_smart_render"),
        patch("jasna.media.splice.probe_keyframes", return_value=MagicMock()),
        patch("jasna.media.splice.build_splice_plan", return_value=splice_plan),
        patch(
            "jasna.mosaic.detection_registry.coerce_detection_model_name",
            side_effect=lambda name: name,
        ),
        patch(
            "jasna.mosaic.detection_registry.require_detection_model_weights",
            return_value=tmp_path / "detector.engine",
        ),
        patch("jasna.pipeline.Pipeline", return_value=pipeline) as pipeline_cls,
    ):
        processor._run_video_job(1, input_path, output_path, segments=segments, settings=AppSettings())

    assert pipeline_cls.call_args.kwargs["splice_plan"] is splice_plan


def test_ensure_video_session_delegates_to_factory_and_close_unloads() -> None:
    processor = Processor(on_resume_conflict=no_resume_conflict)
    processor._settings = AppSettings()
    session = MagicMock()

    with patch("jasna.gui.processor.build_video_session", return_value=session) as build:
        processor._ensure_video_session(processor._settings)
        processor._ensure_video_session(processor._settings)

    build.assert_called_once()
    assert processor._video_session is session

    with patch("jasna.gui.processor.release_session_memory") as release:
        processor._close_video_session()
    session.close.assert_called_once_with()
    release.assert_called_once_with(session.device)
    assert processor._video_session is None


def _run_ltx_job(tmp_path, segments, split):
    pipeline = MagicMock()
    processor = Processor(on_resume_conflict=no_resume_conflict)
    settings = AppSettings(
        restoration_model="ltx",
        ltx_auto_split=True,
        ltx_auto_split_percent=20,
        codec="av1",
        encoder_cq=38,
        encoder_custom_args="tune=uhq",
    )
    processor._settings = settings
    processor._video_session = RestorationSession(device=MagicMock(), restoration_pipeline=MagicMock())
    processor._ensure_video_session = MagicMock()
    processor._prepare_job_detector = MagicMock()
    with (
        patch("jasna.gui.processor.ltx_auto_split", return_value=split) as auto_split,
        patch("jasna.media.probe.get_video_meta_data", return_value=MagicMock(codec_name="hevc", duration=10.0)),
        patch("jasna.media.splice.validate_smart_render"),
        patch("jasna.media.splice.probe_keyframes", return_value=MagicMock()),
        patch("jasna.media.splice.build_splice_plan", return_value=MagicMock()),
        patch("jasna.mosaic.detection_registry.coerce_detection_model_name", side_effect=lambda name: name),
        patch("jasna.mosaic.detection_registry.require_detection_model_weights", return_value=tmp_path / "d.engine"),
        patch("jasna.session_factory.provide_restoration_models"),
        patch("jasna.pipeline.Pipeline", return_value=pipeline) as pipeline_cls,
    ):
        processor._run_video_job(1, tmp_path / "in.mp4", tmp_path / "out.mp4", segments=segments, settings=settings)
    return auto_split, pipeline_cls.call_args.kwargs


def test_ltx_auto_split_job_runs_the_split_ranges_in_the_source_codec(tmp_path) -> None:
    from jasna.segments import SegmentRestoration
    from jasna.session_factory import AutoSplit

    split = AutoSplit(session_config(codec="h264"), (SegmentRange(1, 2, SegmentRestoration("ltx", 1)),), MagicMock())
    auto_split, kwargs = _run_ltx_job(tmp_path, (), split)

    assert auto_split.call_args.args[0].ltx_auto_split_percent == 20.0
    assert kwargs["config"] is split.config
    assert kwargs["segments"] == split.segments
    assert kwargs["splice_plan"] is split.splice_plan


def test_ranges_set_in_the_editor_override_the_ltx_auto_split(tmp_path) -> None:
    auto_split, kwargs = _run_ltx_job(tmp_path, (SegmentRange(1, 2),), None)

    auto_split.assert_not_called()
    assert kwargs["segments"] == (SegmentRange(1, 2),)
    assert kwargs["config"].codec == "hevc"


def test_editor_ranges_give_the_source_codec_its_default_cq_not_settings_meant_for_another(tmp_path) -> None:
    from jasna.accelerator import vendor_for_device
    from jasna.media.encoder_settings import encoder_cq_spec

    _, kwargs = _run_ltx_job(tmp_path, (SegmentRange(1, 2),), None)

    assert kwargs["config"].encoder_settings == {"cq": encoder_cq_spec("hevc", vendor_for_device()).default}
