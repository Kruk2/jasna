from __future__ import annotations

import io
from pathlib import Path
import threading
from unittest.mock import MagicMock

from jasna.gui.models import (
    AppSettings,
    JobItem,
    JobStatus,
    SegmentSelectionMode,
)
from jasna.gui.processor import ProgressUpdate
from jasna.gui.video_job_process import (
    EVENT_PREFIX,
    _load_request,
    build_video_job_request,
    parse_event_line,
    write_video_job_request,
)
from jasna.native_worker import (
    HostMemoryPressureError,
    ISOLATED_VIDEO_JOB_ENV,
    NATIVE_PRESSURE_RECYCLE_EXIT_CODE,
    NativeWorkerRecycleRequested,
)
from jasna.segments import SegmentRange


class _BlockingInput:
    def __iter__(self):
        return self

    def __next__(self):
        threading.Event().wait(60)
        raise StopIteration


def _request(tmp_path: Path):
    job = JobItem(
        id=42,
        path=tmp_path / "clip.mp4",
        duration_seconds=12.5,
        segments=(SegmentRange(1.0, 2.5),),
        segment_selection_mode=SegmentSelectionMode.MANUAL,
        detection_model="detector-a",
        detection_score_threshold=0.45,
        vr_projection="fisheye",
    )
    snapshot = job.begin_processing()
    assert snapshot is not None
    return build_video_job_request(
        job,
        snapshot,
        AppSettings(batch_size=8, post_export_action="shutdown"),
        output_folder=str(tmp_path / "out"),
        output_pattern="{original}.mkv",
        disable_basicvsrpp_tensorrt=True,
    )


def test_video_job_request_round_trip_preserves_current_job_snapshot(tmp_path) -> None:
    path = tmp_path / "request.json"
    write_video_job_request(path, _request(tmp_path))

    job, settings, payload = _load_request(path)

    assert job.id == 42
    assert job.status is JobStatus.PENDING
    assert job.path == tmp_path / "clip.mp4"
    assert job.segments == (SegmentRange(1.0, 2.5),)
    assert job.segment_selection_mode is SegmentSelectionMode.MANUAL
    assert job.detection_model == "detector-a"
    assert job.detection_score_threshold == 0.45
    assert job.vr_projection == "fisheye"
    assert settings.batch_size == 8
    assert settings.post_export_action == "shutdown"
    assert payload["disable_basicvsrpp_tensorrt"] is True


def test_event_protocol_ignores_native_output_and_parses_json() -> None:
    assert parse_event_line("native diagnostic") is None
    assert parse_event_line(EVENT_PREFIX + '{"type":"result","status":"completed"}') == {
        "type": "result",
        "status": "completed",
    }


def test_video_job_command_supports_source_and_frozen(monkeypatch, tmp_path) -> None:
    import jasna.gui.video_job_process as module

    request_path = tmp_path / "request.json"
    monkeypatch.setattr(module.sys, "executable", "/opt/jasna/python")
    monkeypatch.setattr(module, "is_frozen", lambda: False)
    assert module.video_job_command(request_path) == [
        "/opt/jasna/python",
        "-m",
        "jasna.gui.video_job_process",
        str(request_path),
    ]

    monkeypatch.setattr(module, "is_frozen", lambda: True)
    assert module.video_job_command(request_path) == [
        "/opt/jasna/python",
        "--isolated-video-job",
        str(request_path),
    ]


def test_child_emits_current_progress_and_completed_result_without_queue_action(
    monkeypatch,
    tmp_path,
) -> None:
    import jasna.gui.processor as processor_module
    import jasna.gui.video_job_process as video_job_module

    captured_settings = []
    frozen_patch = MagicMock()

    class FakeProcessor:
        def __init__(self, on_progress, on_log, on_complete):
            self._on_progress = on_progress
            self._on_log = on_log
            self._on_complete = on_complete
            self._paused = False

        def is_paused(self):
            return self._paused

        def pause(self):
            self._paused = not self._paused

        def stop(self):
            pass

        def _run(self):
            captured_settings.append(self._settings)
            job = self._jobs[0]
            job.status = video_job_module.JobStatus.PROCESSING
            self._on_progress(
                processor_module.ProgressUpdate(
                    job.id,
                    video_job_module.JobStatus.PROCESSING,
                    progress=25.0,
                    phase="fine_scan",
                )
            )
            self._on_log("INFO", "child log")
            job.output_path = tmp_path / "out" / "clip.mkv"
            job.status = video_job_module.JobStatus.COMPLETED
            self._on_progress(
                processor_module.ProgressUpdate(
                    job.id,
                    video_job_module.JobStatus.COMPLETED,
                    progress=100.0,
                )
            )
            self._on_complete()

        def completed_processing_path(self, _job_id):
            return "smart"

    monkeypatch.setattr(processor_module, "Processor", FakeProcessor)
    monkeypatch.setattr(video_job_module, "is_frozen", lambda: True)
    monkeypatch.setattr("jasna._frozen.patch_frozen_torch", frozen_patch)
    request_path = tmp_path / "request.json"
    write_video_job_request(request_path, _request(tmp_path))
    output = io.StringIO()

    assert video_job_module.run_video_job_file(
        request_path,
        input_stream=_BlockingInput(),
        output_stream=output,
    ) == 0

    events = [
        parse_event_line(line)
        for line in output.getvalue().splitlines()
        if line.startswith(EVENT_PREFIX)
    ]
    assert [event["type"] for event in events] == [
        "progress",
        "log",
        "progress",
        "result",
    ]
    assert events[0]["update"]["phase"] == "fine_scan"
    assert events[-1] == {
        "type": "result",
        "status": "completed",
        "output_path": str(tmp_path / "out" / "clip.mkv"),
        "processing_path": "smart",
    }
    frozen_patch.assert_called_once_with()
    assert captured_settings[0].post_export_action == "none"
    assert captured_settings[0].post_export_command == ""


def test_child_reports_retryable_native_pressure_without_marking_job_failed(
    monkeypatch,
    tmp_path,
) -> None:
    import jasna.gui.processor as processor_module
    import jasna.gui.video_job_process as video_job_module

    class RecyclingProcessor:
        def __init__(self, on_progress, on_log, on_complete):
            pass

        def _run(self):
            raise NativeWorkerRecycleRequested("completed one pressured span")

    monkeypatch.setattr(processor_module, "Processor", RecyclingProcessor)
    monkeypatch.delenv(ISOLATED_VIDEO_JOB_ENV, raising=False)
    request_path = tmp_path / "request.json"
    write_video_job_request(request_path, _request(tmp_path))
    output = io.StringIO()

    assert video_job_module.run_video_job_file(
        request_path,
        input_stream=_BlockingInput(),
        output_stream=output,
    ) == NATIVE_PRESSURE_RECYCLE_EXIT_CODE

    events = [
        parse_event_line(line)
        for line in output.getvalue().splitlines()
        if line.startswith(EVENT_PREFIX)
    ]
    assert events == [
        {
            "type": "retry",
            "reason": "native_pressure",
            "message": "completed one pressured span",
        }
    ]
    assert ISOLATED_VIDEO_JOB_ENV not in video_job_module.os.environ


def test_child_reports_structured_host_memory_pressure(
    monkeypatch,
    tmp_path,
) -> None:
    import jasna.gui.processor as processor_module
    import jasna.gui.video_job_process as video_job_module

    class PressuredProcessor:
        def __init__(self, on_progress, on_log, on_complete):
            pass

        def _run(self):
            raise HostMemoryPressureError("host RAM reserve exhausted")

    monkeypatch.setattr(processor_module, "Processor", PressuredProcessor)
    request_path = tmp_path / "request.json"
    write_video_job_request(request_path, _request(tmp_path))
    output = io.StringIO()

    assert video_job_module.run_video_job_file(
        request_path,
        input_stream=_BlockingInput(),
        output_stream=output,
    ) == 1
    events = [
        parse_event_line(line)
        for line in output.getvalue().splitlines()
        if line.startswith(EVENT_PREFIX)
    ]
    assert events[0]["type"] == "fatal"
    assert events[0]["reason"] == "host_memory_pressure"
    assert events[0]["message"] == "host RAM reserve exhausted"


def test_child_preserves_non_pressure_native_recycle_reason(
    monkeypatch,
    tmp_path,
) -> None:
    import jasna.gui.processor as processor_module
    import jasna.gui.video_job_process as video_job_module

    class RecyclingProcessor:
        def __init__(self, on_progress, on_log, on_complete):
            pass

        def _run(self):
            raise NativeWorkerRecycleRequested(
                "bounded AMF session completed",
                reason="amf_session_limit",
            )

    monkeypatch.setattr(processor_module, "Processor", RecyclingProcessor)
    request_path = tmp_path / "request.json"
    write_video_job_request(request_path, _request(tmp_path))
    output = io.StringIO()

    assert video_job_module.run_video_job_file(
        request_path,
        input_stream=_BlockingInput(),
        output_stream=output,
    ) == NATIVE_PRESSURE_RECYCLE_EXIT_CODE
    events = [
        parse_event_line(line)
        for line in output.getvalue().splitlines()
        if line.startswith(EVENT_PREFIX)
    ]
    assert events == [
        {
            "type": "retry",
            "reason": "amf_session_limit",
            "message": "bounded AMF session completed",
        }
    ]
