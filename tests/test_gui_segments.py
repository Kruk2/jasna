from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from jasna.gui.models import AppSettings, JobItem, JobStatus
from jasna.gui.processor import Processor
from jasna.segments import SegmentRange


def test_pending_job_segments_can_be_replaced() -> None:
    job = JobItem(Path("video.mp4"))
    segments = (SegmentRange(1, 2),)
    assert job.try_set_segments(segments)
    assert job.snapshot_segments() == segments


def test_begin_processing_atomically_freezes_segments() -> None:
    job = JobItem(Path("video.mp4"), segments=(SegmentRange(1, 2),))
    assert job.begin_processing() == (SegmentRange(1, 2),)
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


def test_video_job_passes_precomputed_splice_plan_to_pipeline(tmp_path) -> None:
    input_path = tmp_path / "video.mp4"
    output_path = tmp_path / "output.mp4"
    segments = (SegmentRange(1, 2),)
    metadata = MagicMock(codec_name="h264", duration=10.0)
    splice_plan = MagicMock()
    pipeline = MagicMock()
    processor = Processor()
    processor._settings = AppSettings()
    processor._video_session = {
        "det_name": "detector",
        "detection_model_path": tmp_path / "detector.engine",
        "restoration_pipeline": MagicMock(),
        "device": MagicMock(),
        "lut_path": None,
    }
    processor._ensure_video_session = MagicMock()
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
