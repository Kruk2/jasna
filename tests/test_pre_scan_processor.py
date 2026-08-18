from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jasna.gui.models import (
    AppSettings,
    JobItem,
    JobStatus,
    SegmentSelectionMode,
)
from jasna.gui.pre_scan_routing import PreScanFailed, PreScanOutcome
from jasna.gui.processor import Processor
from jasna.segments import SegmentRange


def _processor(tmp_path, settings: AppSettings):
    processor = Processor()
    processor._settings = settings
    processor._output_folder = str(tmp_path)
    processor._output_pattern = "{original}_restored.mp4"
    processor._validate_completed_video_output = MagicMock()
    return processor


def test_auto_scan_supplies_dynamic_segments_to_existing_smart_render(tmp_path):
    source = tmp_path / "video.mp4"
    source.touch()
    job = JobItem(source)
    processor = _processor(tmp_path, AppSettings())
    processor._run_pipeline = MagicMock(return_value="smart")
    coordinator = MagicMock()
    coordinator.run.return_value = PreScanOutcome(
        "smart",
        segments=(SegmentRange(10, 40),),
        coverage=0.25,
    )

    with (
        patch(
            "jasna.gui.pre_scan_routing.PreScanCoordinator",
            return_value=coordinator,
        ),
        patch("jasna.media.get_video_meta_data", return_value=MagicMock()),
        patch("jasna.gui.processor._cleanup_torch"),
    ):
        processor._process_job(job)

    assert job.status is JobStatus.COMPLETED
    assert processor._run_pipeline.call_args.kwargs["segments"] == (
        SegmentRange(10, 40),
    )
    assert processor._run_pipeline.call_args.kwargs["automatic_segments"] is True
    assert processor.completed_processing_path(job.id) == "smart"
    coordinator.close.assert_called_once()


def test_auto_scan_failure_falls_back_to_full_processing(tmp_path):
    source = tmp_path / "video.mp4"
    source.touch()
    job = JobItem(source)
    processor = _processor(tmp_path, AppSettings())
    processor._run_pipeline = MagicMock(return_value="full")
    coordinator = MagicMock()
    coordinator.run.side_effect = PreScanFailed("detector unavailable")

    with (
        patch(
            "jasna.gui.pre_scan_routing.PreScanCoordinator",
            return_value=coordinator,
        ),
        patch("jasna.media.get_video_meta_data", return_value=MagicMock()),
        patch("jasna.gui.processor._cleanup_torch"),
    ):
        processor._process_job(job)

    assert job.status is JobStatus.COMPLETED
    assert "segments" not in processor._run_pipeline.call_args.kwargs
    assert processor.completed_processing_path(job.id) == "full"


def test_forced_scan_failure_is_not_silently_changed_to_full(tmp_path):
    source = tmp_path / "video.mp4"
    source.touch()
    job = JobItem(source)
    processor = _processor(tmp_path, AppSettings(pre_scan_policy="scan"))
    processor._run_pipeline = MagicMock(return_value="full")
    coordinator = MagicMock()
    coordinator.run.side_effect = PreScanFailed("detector unavailable")

    with (
        patch(
            "jasna.gui.pre_scan_routing.PreScanCoordinator",
            return_value=coordinator,
        ),
        patch("jasna.media.get_video_meta_data", return_value=MagicMock()),
        patch("jasna.gui.processor._cleanup_torch"),
    ):
        processor._process_job(job)

    assert job.status is JobStatus.ERROR
    processor._run_pipeline.assert_not_called()


def test_manual_segments_take_priority_over_global_auto_scan(tmp_path):
    source = tmp_path / "video.mp4"
    source.touch()
    ranges = (SegmentRange(1, 3),)
    job = JobItem(
        source,
        segments=ranges,
        segment_selection_mode=SegmentSelectionMode.MANUAL,
    )
    processor = _processor(tmp_path, AppSettings())
    processor._run_pipeline = MagicMock(return_value="smart")

    with (
        patch("jasna.gui.pre_scan_routing.PreScanCoordinator") as coordinator,
        patch("jasna.gui.processor._cleanup_torch"),
    ):
        processor._process_job(job)

    coordinator.assert_not_called()
    assert processor._run_pipeline.call_args.kwargs["segments"] == ranges
    assert "automatic_segments" not in processor._run_pipeline.call_args.kwargs


def test_manual_empty_selection_forces_full_processing(tmp_path):
    source = tmp_path / "video.mp4"
    source.touch()
    job = JobItem(
        source,
        segment_selection_mode=SegmentSelectionMode.FULL,
    )
    processor = _processor(tmp_path, AppSettings())
    processor._run_pipeline = MagicMock(return_value="full")

    with (
        patch("jasna.gui.pre_scan_routing.PreScanCoordinator") as coordinator,
        patch("jasna.gui.processor._cleanup_torch"),
    ):
        processor._process_job(job)

    coordinator.assert_not_called()
    assert "segments" not in processor._run_pipeline.call_args.kwargs


def test_off_policy_bypasses_pre_scan_and_processes_full_video(tmp_path):
    source = tmp_path / "video.mp4"
    source.touch()
    job = JobItem(source)
    processor = _processor(tmp_path, AppSettings(pre_scan_policy="off"))
    processor._run_pipeline = MagicMock(return_value="full")

    with (
        patch("jasna.gui.pre_scan_routing.PreScanCoordinator") as coordinator,
        patch("jasna.media.get_video_meta_data") as metadata,
        patch("jasna.gui.processor._cleanup_torch"),
    ):
        processor._process_job(job)

    coordinator.assert_not_called()
    metadata.assert_not_called()
    assert "segments" not in processor._run_pipeline.call_args.kwargs
    assert processor.completed_processing_path(job.id) == "full"


def test_zero_hit_copy_does_not_load_restoration_pipeline(tmp_path):
    source = tmp_path / "video.mp4"
    source.touch()
    job = JobItem(source)
    processor = _processor(tmp_path, AppSettings())
    processor._run_pipeline = MagicMock()
    processor._copy_source_video = MagicMock()
    coordinator = MagicMock()
    coordinator.run.return_value = PreScanOutcome("copy", reason="no mosaic")

    with (
        patch(
            "jasna.gui.pre_scan_routing.PreScanCoordinator",
            return_value=coordinator,
        ),
        patch("jasna.media.get_video_meta_data", return_value=MagicMock()),
        patch("jasna.gui.processor._cleanup_torch"),
    ):
        processor._process_job(job)

    processor._copy_source_video.assert_called_once()
    processor._run_pipeline.assert_not_called()
    assert processor.completed_processing_path(job.id) == "copy"


def test_automatic_ranges_fall_back_when_smart_render_is_incompatible(tmp_path):
    from jasna.media.splice import SmartRenderCompatibilityError

    source = tmp_path / "video.mp4"
    output = tmp_path / "output.mp4"
    processor = Processor()
    processor._settings = AppSettings()
    processor._video_session = MagicMock()
    processor._ensure_video_session = MagicMock()
    processor._prepare_job_detector = MagicMock()
    processor._build_encoder_settings = MagicMock(return_value={})
    pipeline = MagicMock(cancel_requested=False, completed=True)

    with (
        patch(
            "jasna.media.get_video_meta_data",
            return_value=MagicMock(codec_name="h264", duration=10.0),
        ),
        patch(
            "jasna.media.splice.validate_smart_render",
            side_effect=SmartRenderCompatibilityError("unsupported"),
        ),
        patch("jasna.media.splice.commit_video_output", create=True),
        patch("jasna.gui.processor.video_session_config", return_value=MagicMock()),
        patch("jasna.gui.processor.build_pipeline", return_value=pipeline) as build,
    ):
        path = processor._run_video_job(
            1,
            source,
            output,
            segments=(SegmentRange(1, 2),),
            automatic_segments=True,
        )

    assert path == "full"
    assert build.call_args.kwargs["segments"] is None
    assert build.call_args.kwargs["splice_plan"] is None


def test_manual_ranges_keep_smart_render_incompatibility_strict(tmp_path):
    from jasna.media.splice import SmartRenderCompatibilityError

    source = tmp_path / "video.mp4"
    output = tmp_path / "output.mp4"
    processor = Processor()
    processor._settings = AppSettings()

    with (
        patch(
            "jasna.media.get_video_meta_data",
            return_value=MagicMock(codec_name="h264", duration=10.0),
        ),
        patch(
            "jasna.media.splice.validate_smart_render",
            side_effect=SmartRenderCompatibilityError("unsupported"),
        ),
        patch("jasna.gui.processor.build_pipeline") as build,
        pytest.raises(SmartRenderCompatibilityError, match="unsupported"),
    ):
        processor._run_video_job(
            1,
            source,
            output,
            segments=(SegmentRange(1, 2),),
            automatic_segments=False,
        )

    build.assert_not_called()
