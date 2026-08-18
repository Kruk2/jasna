import io
import json
import os
from pathlib import Path
import stat
from unittest.mock import patch

import pytest

from jasna.gui.models import (
    AppSettings,
    JobItem,
    JobProcessingSnapshot,
    JobStatus,
)
from jasna.gui.processor import Processor
from jasna.gui.video_job_process import (
    EVENT_PREFIX,
    RESULT_SCHEMA_VERSION,
    VideoJobResult,
    _emit_event,
    _load_request,
    build_video_job_request,
    parse_event_line,
    parse_video_job_result,
    run_video_job_file,
    write_video_job_request,
)
from jasna.segments import SegmentRange


def _request(tmp_path: Path):
    job = JobItem(
        id=41,
        path=tmp_path / "input video.mp4",
        duration_seconds=12.5,
    )
    snapshot = JobProcessingSnapshot(
        segments=(SegmentRange(1.0, 2.5),),
        detection_model="rfdetr-v6",
        detection_score_threshold=0.45,
        vr_projection="fisheye180",
    )
    settings = AppSettings(post_export_action="command", post_export_command="echo ok")
    return build_video_job_request(
        job,
        snapshot,
        settings,
        output_folder=str(tmp_path / "output"),
        output_pattern="{original}_restored.mp4",
        disable_basicvsrpp_tensorrt=True,
    )


def test_request_round_trip_preserves_job_snapshot_and_settings(tmp_path: Path) -> None:
    path = tmp_path / "request.json"
    request = _request(tmp_path)

    write_video_job_request(path, request)
    job, settings, payload = _load_request(path)

    assert job.id == 41
    assert job.path == tmp_path / "input video.mp4"
    assert job.segments == (SegmentRange(1.0, 2.5),)
    assert job.detection_score_threshold == 0.45
    assert settings.post_export_action == "command"
    assert payload["disable_basicvsrpp_tensorrt"] is True
    if os.name != "nt" and stat.S_IMODE(path.stat().st_mode):
        assert stat.S_IMODE(path.stat().st_mode) & 0o077 == 0


def test_event_protocol_keeps_non_protocol_output_separate() -> None:
    stream = io.StringIO()
    event = {"type": "progress", "update": {"status": "processing"}}

    _emit_event(stream, event)

    line = stream.getvalue().strip()
    assert line.startswith(EVENT_PREFIX)
    assert parse_event_line(line) == event
    assert parse_event_line("ordinary native log") is None


def test_event_parser_rejects_non_object_payload() -> None:
    line = EVENT_PREFIX + json.dumps(["not", "an", "object"])
    try:
        parse_event_line(line)
    except ValueError as error:
        assert "JSON object" in str(error)
    else:
        raise AssertionError("non-object protocol payload was accepted")


def test_parent_does_not_complete_from_unvalidated_child_result_event(
    tmp_path: Path,
) -> None:
    processor = Processor()
    job = JobItem(path=tmp_path / "input.mp4", status=JobStatus.PROCESSING)
    output = tmp_path / "output.mp4"

    complete = processor._apply_isolated_event(
        job,
        {
            "type": "result",
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": JobStatus.COMPLETED.value,
            "output_path": str(output),
        },
    )

    assert complete is True
    assert job.status is JobStatus.PROCESSING
    assert job.output_path is None


def test_child_result_requires_current_schema_and_expected_output_path(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "output.mp4"
    result = parse_video_job_result(
        {
            "type": "result",
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": JobStatus.COMPLETED.value,
            "output_path": str(expected),
            "processing_path": "smart",
        },
        expected_output_path=expected,
    )

    assert result == VideoJobResult(JobStatus.COMPLETED, expected, "smart")


def test_isolated_child_defers_per_video_post_export_to_parent(
    monkeypatch,
    tmp_path: Path,
) -> None:
    request_path = tmp_path / "request.json"
    request = _request(tmp_path)
    request["settings"]["post_export_video_command"] = "after {output}"
    write_video_job_request(request_path, request)
    seen_commands: list[str] = []

    class FakeProcessor:
        def __init__(self, *, on_complete=None, **_kwargs):
            self._on_complete = on_complete

        def stop(self):
            pass

        def is_paused(self):
            return False

        def pause(self):
            pass

        def completed_processing_path(self, _job_id):
            return "full"

        def _run(self):
            seen_commands.append(self._settings.post_export_video_command)
            self._jobs[0].status = JobStatus.COMPLETED
            self._jobs[0].output_path = tmp_path / "output.mp4"

    monkeypatch.setattr("jasna.gui.processor.Processor", FakeProcessor)
    output = io.StringIO()

    assert run_video_job_file(request_path, input_stream=io.StringIO(), output_stream=output) == 0

    result_events = [
        parse_event_line(line)
        for line in output.getvalue().splitlines()
        if line.startswith(EVENT_PREFIX)
    ]
    assert seen_commands == [""]
    assert result_events[-1]["schema_version"] == RESULT_SCHEMA_VERSION


@pytest.mark.parametrize(
    "event, message",
    [
        (
            {
                "type": "result",
                "status": JobStatus.COMPLETED.value,
                "output_path": "output.mp4",
            },
            "schema",
        ),
        (
            {
                "type": "result",
                "schema_version": RESULT_SCHEMA_VERSION,
                "status": JobStatus.COMPLETED.value,
                "output_path": "wrong-output.mp4",
                "processing_path": "full",
            },
            "unexpected output path",
        ),
        (
            {
                "type": "result",
                "schema_version": RESULT_SCHEMA_VERSION,
                "status": JobStatus.PROCESSING.value,
                "output_path": None,
            },
            "non-terminal status",
        ),
        (
            {
                "type": "result",
                "schema_version": RESULT_SCHEMA_VERSION,
                "status": JobStatus.COMPLETED.value,
                "output_path": "output.mp4",
            },
            "processing path",
        ),
    ],
)
def test_child_result_rejects_wrong_protocol_or_path(
    tmp_path: Path,
    event: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_video_job_result(
            event,
            expected_output_path=tmp_path / "output.mp4",
        )


def test_progress_completion_event_cannot_complete_parent_job(tmp_path: Path) -> None:
    updates = []
    processor = Processor(on_progress=updates.append)
    job = JobItem(path=tmp_path / "input.mp4", status=JobStatus.PROCESSING)

    processor._apply_isolated_event(
        job,
        {
            "type": "progress",
            "update": {"status": JobStatus.COMPLETED.value, "progress": 100.0},
        },
    )

    assert job.status is JobStatus.PROCESSING
    assert updates[-1].status is JobStatus.PROCESSING


def _isolated_completion_context(tmp_path: Path, *, post_export: bool = False):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source")
    output = tmp_path / "output.mp4"
    job = JobItem(path=source)
    snapshot = job.begin_processing()
    assert snapshot is not None
    processor = Processor()
    processor._settings = AppSettings(
        codec="hevc",
        post_export_video_command="after {output}" if post_export else "",
    )
    return processor, job, snapshot, source, output


def test_parent_completes_only_after_verified_output_then_post_export(
    tmp_path: Path,
) -> None:
    processor, job, snapshot, source, output = _isolated_completion_context(
        tmp_path,
        post_export=True,
    )
    output.write_bytes(b"old")
    previous = processor._output_fingerprint(output)
    output.write_bytes(b"new completed output")
    events: list[str] = []

    def validate(*_args, **_kwargs):
        events.append("validate")
        assert job.status is JobStatus.PROCESSING
        assert job.output_path is None

    with (
        patch.object(
            processor,
            "_validate_completed_video_output",
            side_effect=validate,
        ) as verify,
        patch(
            "jasna.post_export_action.run_post_export_video_command",
            side_effect=lambda *_args: events.append("post-export"),
        ),
    ):
        processor._complete_isolated_video_job(
            job,
            snapshot,
            VideoJobResult(JobStatus.COMPLETED, output, "full"),
            expected_output_path=output,
            previous_output_fingerprint=previous,
            settings=processor._settings,
        )

    assert events == ["validate", "post-export"]
    verify.assert_called_once_with(
        source,
        output,
        codec="hevc",
        smart_render=False,
        previous_fingerprint=previous,
    )
    assert job.status is JobStatus.COMPLETED
    assert job.output_path == output
    assert processor.completed_processing_path(job.id) == "full"


@pytest.mark.parametrize(
    ("processing_path", "expected_codec", "expected_smart_render"),
    [
        ("copy", None, False),
        ("smart", "hevc", True),
        ("full", "hevc", False),
    ],
)
def test_parent_validates_isolated_output_for_actual_processing_path(
    tmp_path: Path,
    processing_path: str,
    expected_codec: str | None,
    expected_smart_render: bool,
) -> None:
    processor, job, snapshot, source, output = _isolated_completion_context(tmp_path)
    output.write_bytes(b"completed output")

    with patch.object(processor, "_validate_completed_video_output") as verify:
        processor._complete_isolated_video_job(
            job,
            snapshot,
            VideoJobResult(JobStatus.COMPLETED, output, processing_path),
            expected_output_path=output,
            previous_output_fingerprint=None,
            settings=processor._settings,
        )

    verify.assert_called_once_with(
        source,
        output,
        codec=expected_codec,
        smart_render=expected_smart_render,
        previous_fingerprint=None,
    )
    assert job.status is JobStatus.COMPLETED
    assert processor.completed_processing_path(job.id) == processing_path


def test_isolated_output_prediction_keeps_flat_auto_rename(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    expected = tmp_path / "clip_restored.mp4"
    expected.write_bytes(b"existing")
    processor = Processor()
    processor._output_pattern = "{original}_restored.mp4"

    output = processor._expected_isolated_video_output_path(
        JobItem(source),
        AppSettings(file_conflict="auto_rename"),
    )

    assert output == tmp_path / "clip_restored (1).mp4"


def test_parent_rejects_stale_isolated_output(tmp_path: Path) -> None:
    processor, job, snapshot, _source, output = _isolated_completion_context(tmp_path)
    output.write_bytes(b"stale output")
    previous = processor._output_fingerprint(output)

    processor._complete_isolated_video_job(
        job,
        snapshot,
        VideoJobResult(JobStatus.COMPLETED, output, "full"),
        expected_output_path=output,
        previous_output_fingerprint=previous,
        settings=processor._settings,
    )

    assert job.status is JobStatus.ERROR
    assert job.output_path is None


def test_parent_rejects_invalid_isolated_media_before_post_export(tmp_path: Path) -> None:
    processor, job, snapshot, _source, output = _isolated_completion_context(
        tmp_path,
        post_export=True,
    )
    output.write_bytes(b"old")
    previous = processor._output_fingerprint(output)
    output.write_bytes(b"invalid media")

    with (
        patch.object(
            processor,
            "_validate_completed_video_output",
            side_effect=RuntimeError("invalid final media"),
        ),
        patch("jasna.post_export_action.run_post_export_video_command") as command,
    ):
        processor._complete_isolated_video_job(
            job,
            snapshot,
            VideoJobResult(JobStatus.COMPLETED, output, "full"),
            expected_output_path=output,
            previous_output_fingerprint=previous,
            settings=processor._settings,
        )

    assert job.status is JobStatus.ERROR
    assert job.output_path is None
    command.assert_not_called()


def test_stop_during_isolated_post_export_prevents_completion_markers(
    tmp_path: Path,
) -> None:
    processor, job, snapshot, _source, output = _isolated_completion_context(
        tmp_path,
        post_export=True,
    )
    output.write_bytes(b"completed output")

    with (
        patch.object(processor, "_validate_completed_video_output"),
        patch(
            "jasna.post_export_action.run_post_export_video_command",
            side_effect=lambda *_args: processor.stop(),
        ),
    ):
        processor._complete_isolated_video_job(
            job,
            snapshot,
            VideoJobResult(JobStatus.COMPLETED, output, "full"),
            expected_output_path=output,
            previous_output_fingerprint=None,
            settings=processor._settings,
        )

    assert job.status is JobStatus.PENDING
    assert job.output_path is None
    assert processor.completed_processing_path(job.id) is None


def test_parent_does_not_complete_isolated_result_after_cancellation(
    tmp_path: Path,
) -> None:
    processor, job, snapshot, _source, output = _isolated_completion_context(tmp_path)
    processor._stop_event.set()

    processor._complete_isolated_video_job(
        job,
        snapshot,
        VideoJobResult(JobStatus.COMPLETED, output, "full"),
        expected_output_path=output,
        previous_output_fingerprint=None,
        settings=processor._settings,
    )

    assert job.status is JobStatus.PENDING
    assert job.output_path is None


def test_linux_amd_isolation_never_routes_images(monkeypatch, tmp_path: Path) -> None:
    processor = Processor(video_job_isolation="linux-amd")
    monkeypatch.setattr(
        "jasna.gui.processor._is_linux_amd_runtime",
        lambda: True,
    )

    assert processor._should_isolate_video_job(JobItem(path=tmp_path / "clip.mp4"))
    assert not processor._should_isolate_video_job(JobItem(path=tmp_path / "still.png"))


def test_isolation_policy_is_inert_when_not_requested(monkeypatch, tmp_path: Path) -> None:
    processor = Processor()
    called = False

    def runtime_probe():
        nonlocal called
        called = True
        return True

    monkeypatch.setattr("jasna.gui.processor._is_linux_amd_runtime", runtime_probe)

    assert not processor._should_isolate_video_job(JobItem(path=tmp_path / "clip.mp4"))
    assert called is False
