from __future__ import annotations

import io
import json
from pathlib import Path
import signal
import subprocess
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jasna.gui.models import AppSettings, JobItem, JobStatus
from jasna.gui.processor import Processor
from jasna.gui.video_job_process import EVENT_PREFIX
from jasna.native_worker import (
    NATIVE_ENCODE_STALL_EXIT_CODE,
    NATIVE_OPEN_STALL_EXIT_CODE,
    NATIVE_PRESSURE_RECYCLE_EXIT_CODE,
)


def _event(payload: dict) -> str:
    return EVENT_PREFIX + json.dumps(payload) + "\n"


def _result(
    job_id: int,
    *,
    status: str = "completed",
    output_path: Path | None = None,
    processing_path: str = "full",
) -> str:
    lines = [
        _event(
            {
                "type": "progress",
                "update": {
                    "job_id": job_id,
                    "status": "processing",
                    "progress": 10.0,
                    "phase": "coarse_scan",
                },
            }
        )
    ]
    if status == "completed":
        lines.append(
            _event(
                {
                    "type": "progress",
                    "update": {
                        "job_id": job_id,
                        "status": "completed",
                        "progress": 100.0,
                    },
                }
            )
        )
    event = {"type": "result", "status": status}
    if output_path is not None:
        event["output_path"] = str(output_path)
        event["processing_path"] = processing_path
    lines.append(_event(event))
    return "".join(lines)


def _retry(message: str = "native pressure") -> str:
    return _event(
        {
            "type": "retry",
            "reason": "native_pressure",
            "message": message,
        }
    )


def _session_retry(message: str = "bounded AMF session completed") -> str:
    return _event(
        {
            "type": "retry",
            "reason": "amf_session_limit",
            "message": message,
        }
    )


class _FakeProcess:
    _next_pid = 41000

    def __init__(self, output: str, returncode: int = 0):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(output)
        self._returncode = returncode
        self._finished = False
        self.terminated = False
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid

    def poll(self):
        return self._returncode if self._finished else None

    def wait(self, timeout=None):
        self._finished = True
        return self._returncode

    def terminate(self):
        self.terminated = True
        self._finished = True

    def kill(self):
        self.terminated = True
        self._finished = True


class _HungProcess(_FakeProcess):
    def __init__(self):
        super().__init__("")
        self.terminated_event = threading.Event()
        self.killed_event = threading.Event()

    def wait(self, timeout=None):
        if self.killed_event.is_set():
            self._finished = True
            return -9
        threading.Event().wait(timeout or 0)
        raise subprocess.TimeoutExpired("video-job", timeout)

    def terminate(self):
        self.terminated = True
        self.terminated_event.set()

    def kill(self):
        self.killed_event.set()


def _processor(tmp_path: Path, jobs: list[JobItem]) -> Processor:
    processor = Processor(video_job_isolation="linux-amd")
    processor._jobs = jobs
    processor._settings = AppSettings()
    processor._output_folder = str(tmp_path / "output")
    processor._output_pattern = "{original}_restored.mp4"
    processor._validate_isolated_completed_output = MagicMock()
    return processor


def _canonical_output(tmp_path: Path, job: JobItem) -> Path:
    return tmp_path / "output" / f"{job.path.stem}_restored.mp4"


def test_linux_amd_batch_uses_fresh_process_and_preserves_processing_path(
    monkeypatch,
    tmp_path,
) -> None:
    import jasna.gui.processor as module

    jobs = [
        JobItem(path=tmp_path / "a.mp4"),
        JobItem(path=tmp_path / "b.mp4"),
        JobItem(path=tmp_path / "c.mp4"),
    ]
    processing_paths = ("smart", "full", "copy")
    processes = [
        _FakeProcess(
            _result(
                job.id,
                output_path=_canonical_output(tmp_path, job),
                processing_path=processing_paths[index],
            )
        )
        for index, job in enumerate(jobs)
    ]
    popen = MagicMock(side_effect=processes)
    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    monkeypatch.setattr(module.subprocess, "Popen", popen)

    processor = _processor(tmp_path, jobs)
    processor._run()

    assert popen.call_count == 3
    assert len({process.pid for process in processes}) == 3
    assert all(process._finished for process in processes)
    assert [job.status for job in jobs] == [
        JobStatus.COMPLETED,
        JobStatus.COMPLETED,
        JobStatus.COMPLETED,
    ]
    assert [processor.completed_processing_path(job.id) for job in jobs] == list(
        processing_paths
    )
    assert [job.output_path for job in jobs] == [
        _canonical_output(tmp_path, job) for job in jobs
    ]
    assert all(call.kwargs["start_new_session"] for call in popen.call_args_list)


def test_child_job_error_does_not_poison_parent_or_block_next_video(
    monkeypatch,
    tmp_path,
) -> None:
    import jasna.gui.processor as module

    jobs = [JobItem(path=tmp_path / "oom.mp4"), JobItem(path=tmp_path / "next.mp4")]
    processes = [
        _FakeProcess(_result(jobs[0].id, status="error")),
        _FakeProcess(
            _result(jobs[1].id, output_path=_canonical_output(tmp_path, jobs[1]))
        ),
    ]
    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        MagicMock(side_effect=processes),
    )

    processor = _processor(tmp_path, jobs)
    processor._run()

    assert [job.status for job in jobs] == [JobStatus.ERROR, JobStatus.COMPLETED]
    assert processor.restart_required_reason() is None


def test_native_pressure_recycles_worker_and_resumes_same_job(
    monkeypatch,
    tmp_path,
) -> None:
    import jasna.gui.processor as module

    job = JobItem(path=tmp_path / "clip.mp4")
    output = _canonical_output(tmp_path, job)
    processes = [
        _FakeProcess(
            _retry("completed span under whole-card pressure"),
            returncode=NATIVE_PRESSURE_RECYCLE_EXIT_CODE,
        ),
        _FakeProcess(_result(job.id, output_path=output, processing_path="smart")),
    ]
    popen = MagicMock(side_effect=processes)
    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    monkeypatch.setattr(module.subprocess, "Popen", popen)

    processor = _processor(tmp_path, [job])
    processor._wait_for_isolated_gpu_recovery = MagicMock(return_value=True)
    processor._run()

    assert popen.call_count == 2
    processor._wait_for_isolated_gpu_recovery.assert_called_once_with()
    assert job.status is JobStatus.COMPLETED
    assert processor.completed_processing_path(job.id) == "smart"


def test_native_abort_recycles_worker_and_resumes_same_job(
    monkeypatch,
    tmp_path,
) -> None:
    import jasna.gui.processor as module

    job = JobItem(path=tmp_path / "clip.mp4")
    output = _canonical_output(tmp_path, job)
    processes = [
        _FakeProcess("", returncode=-signal.SIGABRT),
        _FakeProcess(_result(job.id, output_path=output, processing_path="full")),
    ]
    popen = MagicMock(side_effect=processes)
    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    monkeypatch.setattr(module.subprocess, "Popen", popen)

    processor = _processor(tmp_path, [job])
    processor._wait_for_isolated_gpu_recovery = MagicMock(return_value=True)
    processor._run()

    assert popen.call_count == 2
    processor._wait_for_isolated_gpu_recovery.assert_called_once_with()
    assert job.status is JobStatus.COMPLETED
    assert processor.completed_processing_path(job.id) == "full"


def test_repeated_native_abort_stops_after_bounded_retries(
    monkeypatch,
    tmp_path,
) -> None:
    import jasna.gui.processor as module

    job = JobItem(path=tmp_path / "clip.mp4")
    processes = [
        _FakeProcess("", returncode=-signal.SIGABRT)
        for _ in range(module._ISOLATED_NATIVE_ABORT_RETRY_LIMIT + 1)
    ]
    popen = MagicMock(side_effect=processes)
    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    monkeypatch.setattr(module.subprocess, "Popen", popen)

    processor = _processor(tmp_path, [job])
    processor._wait_for_isolated_gpu_recovery = MagicMock(return_value=True)
    processor._run()

    assert popen.call_count == module._ISOLATED_NATIVE_ABORT_RETRY_LIMIT + 1
    assert processor._wait_for_isolated_gpu_recovery.call_count == module._ISOLATED_NATIVE_ABORT_RETRY_LIMIT
    assert job.status is JobStatus.ERROR
    assert "Close and restart Jasna" in processor.restart_required_reason()


def test_amf_open_stall_restarts_worker_and_resumes_same_job(
    monkeypatch,
    tmp_path,
) -> None:
    import jasna.gui.processor as module

    job = JobItem(path=tmp_path / "clip.mp4")
    output = _canonical_output(tmp_path, job)
    processes = [
        _FakeProcess("", returncode=NATIVE_OPEN_STALL_EXIT_CODE),
        _FakeProcess(_result(job.id, output_path=output, processing_path="smart")),
    ]
    popen = MagicMock(side_effect=processes)
    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    monkeypatch.setattr(module.subprocess, "Popen", popen)

    processor = _processor(tmp_path, [job])
    processor._wait_for_isolated_gpu_recovery = MagicMock(return_value=True)
    processor._run()

    assert popen.call_count == 2
    processor._wait_for_isolated_gpu_recovery.assert_called_once_with()
    assert job.status is JobStatus.COMPLETED


def test_repeated_amf_open_stall_stops_after_bounded_retries(
    monkeypatch,
    tmp_path,
) -> None:
    import jasna.gui.processor as module

    job = JobItem(path=tmp_path / "clip.mp4")
    processes = [
        _FakeProcess("", returncode=NATIVE_OPEN_STALL_EXIT_CODE)
        for _ in range(module._ISOLATED_NATIVE_OPEN_STALL_RETRY_LIMIT + 1)
    ]
    popen = MagicMock(side_effect=processes)
    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    monkeypatch.setattr(module.subprocess, "Popen", popen)

    processor = _processor(tmp_path, [job])
    processor._wait_for_isolated_gpu_recovery = MagicMock(return_value=True)
    processor._run()

    assert popen.call_count == 3
    assert processor._wait_for_isolated_gpu_recovery.call_count == 2
    assert job.status is JobStatus.ERROR


def test_amf_encode_stall_restarts_worker_and_resumes_same_job(
    monkeypatch,
    tmp_path,
) -> None:
    import jasna.gui.processor as module

    job = JobItem(path=tmp_path / "clip.mp4")
    output = _canonical_output(tmp_path, job)
    processes = [
        _FakeProcess("", returncode=NATIVE_ENCODE_STALL_EXIT_CODE),
        _FakeProcess(_result(job.id, output_path=output, processing_path="smart")),
    ]
    popen = MagicMock(side_effect=processes)
    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    monkeypatch.setattr(module.subprocess, "Popen", popen)

    processor = _processor(tmp_path, [job])
    processor._wait_for_isolated_gpu_recovery = MagicMock(return_value=True)
    processor._run()

    assert popen.call_count == 2
    processor._wait_for_isolated_gpu_recovery.assert_called_once_with()
    assert job.status is JobStatus.COMPLETED


def test_repeated_amf_encode_stall_stops_after_bounded_retries(
    monkeypatch,
    tmp_path,
) -> None:
    import jasna.gui.processor as module

    job = JobItem(path=tmp_path / "clip.mp4")
    processes = [
        _FakeProcess("", returncode=NATIVE_ENCODE_STALL_EXIT_CODE)
        for _ in range(module._ISOLATED_NATIVE_ENCODE_STALL_RETRY_LIMIT + 1)
    ]
    popen = MagicMock(side_effect=processes)
    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    monkeypatch.setattr(module.subprocess, "Popen", popen)

    processor = _processor(tmp_path, [job])
    processor._wait_for_isolated_gpu_recovery = MagicMock(return_value=True)
    processor._run()

    assert popen.call_count == module._ISOLATED_NATIVE_ENCODE_STALL_RETRY_LIMIT + 1
    assert (
        processor._wait_for_isolated_gpu_recovery.call_count
        == module._ISOLATED_NATIVE_ENCODE_STALL_RETRY_LIMIT
    )
    assert job.status is JobStatus.ERROR


def test_completed_session_fragment_resets_encode_stall_retry_budget(
    monkeypatch,
    tmp_path,
) -> None:
    import jasna.gui.processor as module

    job = JobItem(path=tmp_path / "clip.mp4")
    output = _canonical_output(tmp_path, job)
    processes = [
        _FakeProcess("", returncode=NATIVE_ENCODE_STALL_EXIT_CODE),
        _FakeProcess(
            _session_retry(),
            returncode=NATIVE_PRESSURE_RECYCLE_EXIT_CODE,
        ),
        _FakeProcess("", returncode=NATIVE_ENCODE_STALL_EXIT_CODE),
        _FakeProcess(
            _session_retry(),
            returncode=NATIVE_PRESSURE_RECYCLE_EXIT_CODE,
        ),
        _FakeProcess("", returncode=NATIVE_ENCODE_STALL_EXIT_CODE),
        _FakeProcess(_result(job.id, output_path=output, processing_path="smart")),
    ]
    popen = MagicMock(side_effect=processes)
    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    monkeypatch.setattr(module.subprocess, "Popen", popen)

    processor = _processor(tmp_path, [job])
    processor._wait_for_isolated_gpu_recovery = MagicMock(return_value=True)
    processor._run()

    assert popen.call_count == len(processes)
    assert job.status is JobStatus.COMPLETED


def test_expected_session_recycle_keeps_progress_monotonic_and_setup_quiet(
    monkeypatch,
    tmp_path,
) -> None:
    import jasna.gui.processor as module

    job = JobItem(path=tmp_path / "clip.mp4")
    output = _canonical_output(tmp_path, job)
    first_attempt = "".join(
        [
            _event(
                {
                    "type": "progress",
                    "update": {
                        "job_id": job.id,
                        "status": "processing",
                        "progress": 55.0,
                        "fps": 7.0,
                        "eta_seconds": 100.0,
                        "phase": "restoring",
                    },
                }
            ),
            _session_retry(),
        ]
    )
    second_attempt = "".join(
        [
            _event(
                {
                    "type": "log",
                    "level": "INFO",
                    "message": "loading_models",
                }
            ),
            _event(
                {
                    "type": "progress",
                    "update": {
                        "job_id": job.id,
                        "status": "processing",
                        "progress": 5.0,
                        "fps": 40.0,
                        "eta_seconds": 10.0,
                        "phase": "coarse_scan",
                    },
                }
            ),
            _event(
                {
                    "type": "progress",
                    "update": {
                        "job_id": job.id,
                        "status": "processing",
                        "progress": 55.0,
                        "fps": 7.5,
                        "eta_seconds": 90.0,
                        "phase": "restoring",
                    },
                }
            ),
            _event(
                {
                    "type": "result",
                    "status": "completed",
                    "output_path": str(output),
                    "processing_path": "smart",
                }
            ),
        ]
    )
    processes = [
        _FakeProcess(first_attempt, returncode=NATIVE_PRESSURE_RECYCLE_EXIT_CODE),
        _FakeProcess(second_attempt),
    ]
    popen = MagicMock(side_effect=processes)
    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    updates = []
    logs = []
    processor = Processor(
        video_job_isolation="linux-amd",
        on_progress=updates.append,
        on_log=lambda level, message: logs.append((level, message)),
    )
    processor._jobs = [job]
    processor._settings = AppSettings()
    processor._output_folder = str(tmp_path / "output")
    processor._output_pattern = "{original}_restored.mp4"
    processor._validate_isolated_completed_output = MagicMock()
    processor._wait_for_isolated_gpu_recovery = MagicMock(return_value=True)

    processor._run()

    processing_progress = [
        update.progress
        for update in updates
        if update.status is JobStatus.PROCESSING
    ]
    assert processing_progress == sorted(processing_progress)
    assert 5.0 not in processing_progress
    assert not any("loading_models" in message for _level, message in logs)
    assert not any("bounded AMF session completed" in message for _level, message in logs)
    assert job.status is JobStatus.COMPLETED


def test_quiet_resume_does_not_hide_terminal_error_progress(tmp_path) -> None:
    job = JobItem(path=tmp_path / "clip.mp4", status=JobStatus.PROCESSING)
    updates = []
    processor = Processor(on_progress=updates.append)
    resume_state = {
        "progress_high_water": 55.0,
        "quiet_resume": True,
    }

    processor._apply_isolated_event(
        job,
        {
            "type": "progress",
            "update": {
                "job_id": job.id,
                "status": "error",
                "progress": 0.0,
                "message": "worker failed",
            },
        },
        resume_state=resume_state,
    )

    assert len(updates) == 1
    assert updates[0].status is JobStatus.ERROR
    assert job.status is JobStatus.ERROR


def test_initial_restoration_progress_is_not_blocked_by_scan_percentage(
    tmp_path,
) -> None:
    job = JobItem(path=tmp_path / "clip.mp4", status=JobStatus.PROCESSING)
    updates = []
    processor = Processor(on_progress=updates.append)
    resume_state = {
        "progress_high_water": 0.0,
        "quiet_resume": False,
    }

    for progress, phase in ((15.0, "coarse_scan"), (1.0, "restoring")):
        processor._apply_isolated_event(
            job,
            {
                "type": "progress",
                "update": {
                    "job_id": job.id,
                    "status": "processing",
                    "progress": progress,
                    "phase": phase,
                },
            },
            resume_state=resume_state,
        )

    assert [(update.progress, update.phase) for update in updates] == [
        (15.0, "coarse_scan"),
        (1.0, "restoring"),
    ]
    assert resume_state["progress_high_water"] == 1.0


def test_resumed_frame_work_keeps_speed_until_new_timing_samples(tmp_path):
    job = JobItem(path=tmp_path / "clip.mp4", status=JobStatus.PROCESSING)
    updates = []
    processor = Processor(on_progress=updates.append)
    state = {"progress_high_water": 0.0, "quiet_resume": False}

    def report(percent, frames, fps, eta=0.0):
        processor._apply_isolated_event(job, {"type": "progress", "update": {
            "status": "processing", "phase": "restoring", "progress": percent,
            "frames_processed": frames, "total_frames": 100, "fps": fps,
            "eta_seconds": eta,
        }}, resume_state=state)

    report(30.0, 30, 20.0, 3.5)
    state["quiet_resume"] = True
    report(30.0, 30, 0.0)  # Replayed workspace prefix is not new work.
    assert len(updates) == 1
    assert state["quiet_resume"] is True
    report(31.0, 31, 0.0)
    assert updates[-1].fps == 20.0
    assert updates[-1].eta_seconds == pytest.approx(69 / 20)
    report(32.0, 32, 12.5, 5.44)
    assert updates[-1].fps == 12.5
    assert updates[-1].eta_seconds == 5.44
    assert state["frame_speed_warmup"] is False


def test_resume_speed_estimate_does_not_enter_ltx_stages(tmp_path):
    job = JobItem(path=tmp_path / "clip.mp4", status=JobStatus.PROCESSING)
    updates = []
    processor = Processor(on_progress=updates.append)
    state = {"progress_high_water": 30.0, "quiet_resume": False,
             "frame_speed_warmup": True, "last_frame_fps": 20.0}
    processor._apply_isolated_event(job, {"type": "progress", "update": {
        "status": "processing", "phase": "restoring", "progress": 40.0,
        "frames_processed": 40, "total_frames": 100, "fps": 0.0,
        "eta_seconds": 12.0, "stage": "denoise",
    }}, resume_state=state)
    assert updates[-1].stage == "denoise"
    assert updates[-1].fps == 0.0
    assert updates[-1].eta_seconds == 12.0


def test_gpu_recovery_wait_requires_stable_startup_headroom(monkeypatch) -> None:
    import jasna.gui.processor as module
    import jasna.vram_offloader as offloader_module

    gib = 1024 ** 3
    samples = iter(
        [
            (21 * gib, 24 * gib),
            (19 * gib, 24 * gib),
            (19 * gib, 24 * gib),
        ]
    )
    monkeypatch.setattr(
        offloader_module,
        "read_linux_amd_system_vram",
        lambda: next(samples),
    )
    monkeypatch.setattr(module, "_ISOLATED_GPU_RECOVERY_POLL_SECONDS", 0.001)

    assert Processor()._wait_for_isolated_gpu_recovery()


def test_gpu_recovery_wait_fails_when_startup_budget_never_returns(
    monkeypatch,
) -> None:
    import jasna.gui.processor as module
    import jasna.vram_offloader as offloader_module

    gib = 1024 ** 3
    logs = []
    monkeypatch.setattr(
        offloader_module,
        "read_linux_amd_system_vram",
        lambda: (23 * gib, 24 * gib),
    )
    monkeypatch.setattr(module, "_ISOLATED_GPU_RECOVERY_TIMEOUT_SECONDS", 0.001)
    processor = Processor(on_log=lambda level, message: logs.append((level, message)))

    assert not processor._wait_for_isolated_gpu_recovery()
    assert logs and logs[-1][0] == "ERROR"


def test_completed_child_waits_for_parent_validation_and_commit(
    monkeypatch,
    tmp_path,
) -> None:
    import jasna.gui.processor as module

    job = JobItem(path=tmp_path / "clip.mp4")
    output = _canonical_output(tmp_path, job)
    updates = []
    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        MagicMock(return_value=_FakeProcess(_result(job.id, output_path=output))),
    )

    processor = _processor(tmp_path, [job])
    processor._on_progress = updates.append
    processor._run()

    assert [update.status for update in updates[-2:]] == [
        JobStatus.PROCESSING,
        JobStatus.COMPLETED,
    ]
    assert [update.progress for update in updates[-2:]] == [99.9, 100.0]
    assert job.output_path == output


def test_preserved_folder_resume_skips_valid_output_before_starting_child(
    monkeypatch,
    tmp_path,
) -> None:
    import jasna.gui.processor as module
    import jasna.gui.resume_validation as resume_module

    root = tmp_path / "input"
    source = root / "season" / "clip.mp4"
    source.parent.mkdir(parents=True)
    source.touch()
    job = JobItem(path=source, input_root=root)
    output = tmp_path / "output" / "season" / "clip_restored.mp4"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"validated by test double")
    popen = MagicMock()
    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(
        resume_module,
        "validate_resume_video_output",
        MagicMock(),
    )

    processor = _processor(tmp_path, [job])
    processor._preserve_input_structure = True
    processor._run()

    assert job.status is JobStatus.SKIPPED
    assert job.output_path is None
    popen.assert_not_called()


def test_preserved_folder_invalid_output_is_replaced_at_exact_child_path(
    monkeypatch,
    tmp_path,
) -> None:
    import jasna.gui.processor as module
    import jasna.gui.video_job_process as video_job_module

    root = tmp_path / "input"
    source = root / "season" / "clip.mp4"
    source.parent.mkdir(parents=True)
    source.touch()
    job = JobItem(path=source, input_root=root)
    output = tmp_path / "output" / "season" / "clip_restored.mp4"
    output.parent.mkdir(parents=True)
    output.touch()
    captured = {}
    original_builder = video_job_module.build_video_job_request

    def capture_request(job, snapshot, settings, **kwargs):
        captured["settings"] = settings
        captured.update(kwargs)
        return original_builder(job, snapshot, settings, **kwargs)

    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        MagicMock(return_value=_FakeProcess(_result(job.id, output_path=output))),
    )
    monkeypatch.setattr(video_job_module, "build_video_job_request", capture_request)

    processor = _processor(tmp_path, [job])
    processor._preserve_input_structure = True
    processor._handle_existing_final_output = MagicMock(return_value="replace")
    processor._run()

    assert job.status is JobStatus.COMPLETED
    assert job.output_path == output
    assert captured["output_folder"] == str(output.parent)
    assert captured["output_pattern"] == output.name
    assert captured["settings"].file_conflict == "overwrite"


def test_stop_during_parent_completion_gate_keeps_job_pending(
    monkeypatch,
    tmp_path,
) -> None:
    import jasna.gui.processor as module

    job = JobItem(path=tmp_path / "clip.mp4")
    output = _canonical_output(tmp_path, job)
    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        MagicMock(return_value=_FakeProcess(_result(job.id, output_path=output))),
    )
    processor = _processor(tmp_path, [job])
    processor._validate_isolated_completed_output = lambda *_args, **_kwargs: (
        processor.stop()
    )

    processor._run()

    assert job.status is JobStatus.PENDING
    assert job.output_path is None
    assert processor.completed_processing_path(job.id) is None


def test_auto_renamed_result_requires_preexisting_canonical_output(
    monkeypatch,
    tmp_path,
) -> None:
    import jasna.gui.processor as module

    job = JobItem(path=tmp_path / "clip.mp4")
    reported = tmp_path / "output" / "clip_restored (1).mp4"
    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        MagicMock(return_value=_FakeProcess(_result(job.id, output_path=reported))),
    )

    processor = _processor(tmp_path, [job])
    processor._run()

    assert job.status is JobStatus.ERROR


def test_auto_renamed_result_accepts_preexisting_canonical_output(
    monkeypatch,
    tmp_path,
) -> None:
    import jasna.gui.processor as module

    job = JobItem(path=tmp_path / "clip.mp4")
    canonical = _canonical_output(tmp_path, job)
    canonical.parent.mkdir(parents=True)
    canonical.touch()
    reported = canonical.with_name("clip_restored (1).mp4")
    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        MagicMock(return_value=_FakeProcess(_result(job.id, output_path=reported))),
    )

    processor = _processor(tmp_path, [job])
    processor._run()

    assert job.status is JobStatus.COMPLETED
    assert job.output_path == reported


def test_completed_child_outside_expected_folder_is_rejected(
    monkeypatch,
    tmp_path,
) -> None:
    import jasna.gui.processor as module

    job = JobItem(path=tmp_path / "clip.mp4")
    outside = tmp_path / "other" / "clip_restored.mp4"
    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        MagicMock(return_value=_FakeProcess(_result(job.id, output_path=outside))),
    )

    processor = _processor(tmp_path, [job])
    processor._run()

    assert job.status is JobStatus.ERROR


def test_pause_and_stop_commands_are_forwarded_to_child(monkeypatch) -> None:
    import jasna.gui.processor as module

    process = _FakeProcess("")
    processor = Processor(video_job_isolation="linux-amd")
    processor._isolated_process = process
    monkeypatch.setattr(processor, "_start_isolated_stop_reaper", MagicMock())

    processor.pause()
    processor.stop()

    commands = [json.loads(line) for line in process.stdin.getvalue().splitlines()]
    assert commands == [
        {"command": "set_paused", "paused": True},
        {"command": "stop"},
    ]


def test_stop_reaper_terminates_hung_process_group(monkeypatch) -> None:
    import jasna.gui.processor as module

    if module.os.name != "posix":
        pytest.skip("tests POSIX process-group termination")
    monkeypatch.setattr(module, "_ISOLATED_STOP_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(module, "_ISOLATED_TERMINATE_GRACE_SECONDS", 0.01)
    signals = []
    monkeypatch.setattr(
        module.os,
        "killpg",
        lambda process_group, signal_number: signals.append(
            (process_group, signal_number)
        ),
    )
    process = _HungProcess()
    processor = Processor(video_job_isolation="linux-amd")
    processor._isolated_process = process

    processor.stop()

    reaper = processor._isolated_stop_reaper
    assert reaper is not None
    reaper.join(timeout=1.0)
    assert not reaper.is_alive()
    assert signals == [
        (process.pid, module.signal.SIGTERM),
        (process.pid, module.signal.SIGKILL),
    ]


def test_linux_amd_isolation_is_gui_opt_in_and_video_only(monkeypatch, tmp_path) -> None:
    import jasna.gui.processor as module

    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    isolated = Processor(video_job_isolation="linux-amd")

    assert isolated._should_isolate_video_job(JobItem(path=tmp_path / "clip.mp4"))
    assert not isolated._should_isolate_video_job(JobItem(path=tmp_path / "still.png"))
    assert not Processor()._should_isolate_video_job(
        JobItem(path=tmp_path / "clip.mp4")
    )
