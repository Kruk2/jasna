from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jasna.accelerator import AcceleratorVendor
from jasna.gui.models import (
    ENCODER_RATE_MODE_AUTO_SOURCE,
    AppSettings,
    JobItem,
    JobStatus,
    SegmentSelectionMode,
)
from jasna.gui.pre_scan_routing import PreScanFailed, PreScanOutcome
from jasna.gui.processor import Processor
from jasna.media.splice import KeyframeIndex, SplicePlan, SpliceSpan
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
        patch("jasna.media.probe.get_video_meta_data", return_value=MagicMock()),
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
        patch("jasna.media.probe.get_video_meta_data", return_value=MagicMock()),
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
        patch("jasna.media.probe.get_video_meta_data", return_value=MagicMock()),
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
        patch("jasna.media.probe.get_video_meta_data") as metadata,
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
        patch("jasna.media.probe.get_video_meta_data", return_value=MagicMock()),
        patch("jasna.gui.processor._cleanup_torch"),
    ):
        processor._process_job(job)

    processor._copy_source_video.assert_called_once()
    processor._run_pipeline.assert_not_called()
    assert processor.completed_processing_path(job.id) == "copy"


@pytest.mark.parametrize("processing_path", ("copy", "full", "smart"))
def test_preserved_subfolder_path_reaches_every_auto_scan_route(
    tmp_path,
    processing_path,
):
    root = tmp_path / "input"
    source = root / "season" / "video.mp4"
    source.parent.mkdir(parents=True)
    source.touch()
    job = JobItem(source, input_root=root)
    output_root = tmp_path / "output"
    expected = output_root / "season" / "video_restored.mp4"
    processor = _processor(output_root, AppSettings())
    processor._preserve_input_structure = True
    processor._copy_source_video = MagicMock()
    processor._run_pipeline = MagicMock(return_value=processing_path)
    coordinator = MagicMock()
    coordinator.run.return_value = PreScanOutcome(
        processing_path,
        segments=(SegmentRange(1, 2),) if processing_path == "smart" else (),
        reason="route test",
    )

    with (
        patch(
            "jasna.gui.pre_scan_routing.PreScanCoordinator",
            return_value=coordinator,
        ),
        patch("jasna.media.probe.get_video_meta_data", return_value=MagicMock()),
        patch("jasna.gui.processor._cleanup_torch"),
    ):
        processor._process_job(job)

    assert job.status is JobStatus.COMPLETED
    assert job.output_path == expected
    if processing_path == "copy":
        processor._copy_source_video.assert_called_once_with(source, expected)
        processor._run_pipeline.assert_not_called()
    else:
        assert processor._run_pipeline.call_args.args[2] == expected


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
            "jasna.media.probe.get_video_meta_data",
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


def test_automatic_ranges_fall_back_before_gpu_when_h264_gop_is_incompatible(
    tmp_path,
):
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
    plan = _mixed_splice_plan()

    with (
        patch(
            "jasna.media.probe.get_video_meta_data",
            return_value=MagicMock(codec_name="h264", duration=3.0),
        ),
        patch("jasna.media.splice.validate_smart_render"),
        patch("jasna.media.splice.probe_keyframes"),
        patch("jasna.media.splice.build_splice_plan", return_value=plan),
        patch(
            "jasna.media.splice.resolve_smart_encoder_settings",
            side_effect=SmartRenderCompatibilityError(
                "AMF H.264 smart rendering supports at most 3 consecutive "
                "B-frames; source uses 4"
            ),
        ) as resolve_settings,
        patch(
            "jasna.media.video_decoder.auto_amf_interop_eligible",
            return_value=True,
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
    resolve_settings.assert_called_once()
    assert build.call_args.kwargs["segments"] is None
    assert build.call_args.kwargs["splice_plan"] is None
    assert build.call_args.kwargs["effect_ranges"] == ((30, 60),)
    pipeline.run.assert_called_once()


def test_manual_ranges_keep_smart_render_incompatibility_strict(tmp_path):
    from jasna.media.splice import SmartRenderCompatibilityError

    source = tmp_path / "video.mp4"
    output = tmp_path / "output.mp4"
    processor = Processor()
    processor._settings = AppSettings()

    with (
        patch(
            "jasna.media.probe.get_video_meta_data",
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


def _mixed_splice_plan() -> SplicePlan:
    segments = (SegmentRange(1, 2),)
    return SplicePlan(
        index=KeyframeIndex((0, 30, 60), Fraction(1, 30), 0, 90),
        spans=(
            SpliceSpan("copy", 0, 30),
            SpliceSpan("render", 30, 60, ((30, 60),)),
            SpliceSpan("copy", 60, 90),
        ),
        segments=segments,
    )


def test_automatic_linux_amd_hevc_mixed_plan_remains_smart(
    tmp_path,
):
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
            "jasna.media.probe.get_video_meta_data",
            return_value=MagicMock(codec_name="hevc", duration=3.0),
        ),
        patch("jasna.media.splice.validate_smart_render"),
        patch("jasna.media.splice.probe_keyframes"),
        patch(
            "jasna.media.splice.build_splice_plan",
            return_value=_mixed_splice_plan(),
        ),
        patch(
            "jasna.media.video_decoder.auto_amf_interop_eligible",
            return_value=True,
        ),
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

    assert path == "smart"
    assert build.call_args.kwargs["segments"] == (SegmentRange(1, 2),)


def test_automatic_linux_amd_hevc_dual_gop_mixed_plan_remains_smart(
    tmp_path,
):
    source = tmp_path / "video.mp4"
    output = tmp_path / "output.mp4"
    processor = Processor()
    processor._settings = AppSettings(
        amd_dual_gop_encode=True,
        encoder_rate_mode=ENCODER_RATE_MODE_AUTO_SOURCE,
    )
    processor._video_session = MagicMock()
    processor._ensure_video_session = MagicMock()
    processor._prepare_job_detector = MagicMock()
    processor._build_encoder_settings = MagicMock(return_value={})
    pipeline = MagicMock(cancel_requested=False, completed=True)
    metadata = MagicMock(
        codec_name="hevc",
        duration=3.0,
        video_width=8192,
        video_height=4096,
        is_10bit=True,
        pixel_format="p010le",
        profile="Main 10",
        video_bitrate=12_000_000,
    )

    with (
        patch("jasna.gui.processor.sys.platform", "linux"),
        patch("jasna.accelerator.vendor_for_device", return_value=AcceleratorVendor.AMD),
        patch("jasna.media.probe.get_video_meta_data", return_value=metadata),
        patch("jasna.media.splice.validate_smart_render"),
        patch("jasna.media.splice.probe_keyframes"),
        patch(
            "jasna.media.splice.build_splice_plan",
            return_value=_mixed_splice_plan(),
        ),
        patch(
            "jasna.media.video_decoder.auto_amf_interop_eligible",
            return_value=True,
        ),
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

    assert path == "smart"
    assert build.call_args.kwargs["segments"] == (SegmentRange(1, 2),)
    assert build.call_args.kwargs["splice_plan"] == _mixed_splice_plan()
    assert build.call_args.kwargs["effect_ranges"] is None
    processor._ensure_video_session.assert_called_once()


def test_manual_linux_amd_hevc_mixed_plan_remains_smart(
    tmp_path,
):
    source = tmp_path / "video.mp4"
    output = tmp_path / "output.mp4"
    processor = Processor()
    processor._settings = AppSettings()
    processor._video_session = MagicMock()
    processor._ensure_video_session = MagicMock()
    processor._prepare_job_detector = MagicMock()
    processor._build_encoder_settings = MagicMock(return_value={})
    pipeline = MagicMock(cancel_requested=False, completed=True)
    plan = _mixed_splice_plan()

    with (
        patch(
            "jasna.media.probe.get_video_meta_data",
            return_value=MagicMock(codec_name="hevc", duration=3.0),
        ),
        patch("jasna.media.splice.validate_smart_render"),
        patch("jasna.media.splice.probe_keyframes"),
        patch(
            "jasna.media.splice.build_splice_plan",
            return_value=plan,
        ),
        patch(
            "jasna.media.video_decoder.auto_amf_interop_eligible",
            return_value=True,
        ),
        patch("jasna.gui.processor.video_session_config", return_value=MagicMock()),
        patch("jasna.gui.processor.build_pipeline", return_value=pipeline) as build,
    ):
        path = processor._run_video_job(
            1,
            source,
            output,
            segments=(SegmentRange(1, 2),),
            automatic_segments=False,
        )

    assert path == "smart"
    assert build.call_args.kwargs["segments"] == (SegmentRange(1, 2),)
    assert build.call_args.kwargs["splice_plan"] is plan
    processor._ensure_video_session.assert_called_once()


@pytest.mark.parametrize(
    ("codec", "native_linux_amd"),
    [("h264", True), ("av1", True), ("hevc", False)],
)
def test_other_smart_render_backends_and_codecs_remain_enabled(
    tmp_path,
    codec,
    native_linux_amd,
):
    source = tmp_path / "video.mp4"
    output = tmp_path / "output.mp4"
    processor = Processor()
    processor._settings = AppSettings()
    processor._video_session = MagicMock()
    processor._ensure_video_session = MagicMock()
    processor._prepare_job_detector = MagicMock()
    processor._build_encoder_settings = MagicMock(return_value={})
    pipeline = MagicMock(cancel_requested=False, completed=True)
    plan = _mixed_splice_plan()

    with (
        patch(
            "jasna.media.probe.get_video_meta_data",
            return_value=MagicMock(codec_name=codec, duration=3.0),
        ),
        patch("jasna.media.splice.validate_smart_render"),
        patch("jasna.media.splice.probe_keyframes"),
        patch("jasna.media.splice.build_splice_plan", return_value=plan),
        patch(
            "jasna.media.video_decoder.auto_amf_interop_eligible",
            return_value=native_linux_amd,
        ),
        patch("jasna.gui.processor.video_session_config", return_value=MagicMock()),
        patch("jasna.gui.processor.build_pipeline", return_value=pipeline) as build,
    ):
        path = processor._run_video_job(
            1,
            source,
            output,
            segments=(SegmentRange(1, 2),),
            automatic_segments=False,
        )

    assert path == "smart"
    assert build.call_args.kwargs["segments"] == (SegmentRange(1, 2),)
    assert build.call_args.kwargs["splice_plan"] is plan


def test_automatic_ranges_do_not_retry_full_after_runtime_smart_incompatibility(
    tmp_path,
):
    from jasna.media.splice import SmartRenderCompatibilityError

    source = tmp_path / "video.mp4"
    output = tmp_path / "output.mp4"
    processor = Processor()
    processor._settings = AppSettings()
    processor._video_session = MagicMock()
    processor._ensure_video_session = MagicMock()
    processor._prepare_job_detector = MagicMock()
    processor._build_encoder_settings = MagicMock(return_value={})
    pipeline = MagicMock(cancel_requested=False, completed=False)
    pipeline.run.side_effect = SmartRenderCompatibilityError("late seam conflict")

    with (
        patch(
            "jasna.media.probe.get_video_meta_data",
            return_value=MagicMock(codec_name="hevc", duration=10.0),
        ),
        patch("jasna.media.splice.validate_smart_render"),
        patch("jasna.media.splice.probe_keyframes", return_value=(0.0, 10.0)),
        patch("jasna.media.splice.build_splice_plan", return_value=MagicMock()),
        patch(
            "jasna.media.video_decoder.auto_amf_interop_eligible",
            return_value=True,
        ),
        patch("jasna.gui.processor.video_session_config", return_value=MagicMock()),
        patch("jasna.gui.processor.build_pipeline", return_value=pipeline) as build,
        pytest.raises(SmartRenderCompatibilityError, match="late seam conflict"),
    ):
        processor._run_video_job(
            1,
            source,
            output,
            segments=(SegmentRange(1, 2),),
            automatic_segments=True,
        )

    build.assert_called_once()
    assert build.call_args.kwargs["segments"] == (SegmentRange(1, 2),)
    pipeline.close.assert_called_once()
    assert processor._current_pipeline is None
    assert "Close and restart Jasna" in processor.restart_required_reason()


def test_native_runtime_smart_failure_blocks_reusing_the_processor(tmp_path):
    processor = Processor()
    processor._restart_required_reason = "restart required"

    with pytest.raises(RuntimeError, match="restart required"):
        processor.start(
            [],
            AppSettings(),
            str(tmp_path),
            "{original}_restored.mp4",
            disable_basicvsrpp_tensorrt=False,
        )
