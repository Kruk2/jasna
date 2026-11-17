from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from jasna.gui.models import AppSettings, JobItem, JobStatus
from jasna.gui.processor import Processor
from jasna.segments import SegmentRange

from factories import no_resume_conflict

pytestmark = pytest.mark.usefixtures("nvidia_vendor", "no_gpu_cleanup", "tk_garbage_collected")


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


def _run_one_job(job: JobItem, settings: AppSettings):
    """Run ``job`` through a real processor queue and return the pipeline call it made."""
    processor = Processor(on_resume_conflict=no_resume_conflict)
    with patch.object(processor, "_run_pipeline") as run_pipeline:
        processor.start([job], settings, output_folder="", output_pattern="{original}_restored.mp4")
        processor.join(timeout=5.0)
    return run_pipeline.call_args


def test_processor_passes_frozen_segments_to_video_job(tmp_path) -> None:
    source = tmp_path / "video.mp4"
    source.touch()
    segments = (SegmentRange(1, 2),)
    job = JobItem(source, segments=segments)

    call = _run_one_job(job, AppSettings())

    assert call.kwargs["segments"] == segments
    assert job.status is JobStatus.COMPLETED


def test_processor_uses_each_videos_detection_and_projection_overrides(tmp_path) -> None:
    source = tmp_path / "video.mp4"
    source.touch()
    job = JobItem(source, detection_model="lada-yolo-v4", detection_score_threshold=0.55, vr_projection="fisheye")
    queue_settings = AppSettings(detection_model="rfdetr-v5", detection_score_threshold=0.25)

    settings = _run_one_job(job, queue_settings).kwargs["settings"]

    assert (settings.detection_model, settings.detection_score_threshold, settings.vr_projection) == (
        "lada-yolo-v4", 0.55, "fisheye"
    )
    assert (queue_settings.detection_model, queue_settings.vr_projection) == ("rfdetr-v5", "auto")
