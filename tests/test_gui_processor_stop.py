import threading

import pytest

from jasna.gui.models import AppSettings, JobItem, JobStatus
from jasna.gui.processor import Processor, ProgressUpdate
from factories import no_resume_conflict

pytestmark = pytest.mark.usefixtures("no_gpu_cleanup", "tk_garbage_collected")


class _FakePipeline:
    def __init__(self):
        self.cancel_requested = False
        self.completed = False

    def cancel(self):
        self.cancel_requested = True


def _processor_with_job(tmp_path, run_pipeline):
    job = JobItem(path=tmp_path / "clip.mp4")
    updates: list[ProgressUpdate] = []
    processor = Processor(on_resume_conflict=no_resume_conflict, on_progress=updates.append)
    processor._run_pipeline = run_pipeline
    processor._jobs = [job]
    processor._settings = AppSettings()
    processor._output_folder = str(tmp_path)
    processor._output_pattern = "{original}_restored.mp4"
    return processor, job, updates


def test_stopped_job_leaves_processing_state(tmp_path):
    from jasna.gui.processor import ProcessingStopped

    started = threading.Event()
    release = threading.Event()

    def fake_run_pipeline(job_id, input_path, output_path, **kwargs):
        started.set()
        release.wait(5)
        if processor._stop_event.is_set():
            raise ProcessingStopped("Processing stopped")

    processor, job, updates = _processor_with_job(tmp_path, fake_run_pipeline)

    worker = threading.Thread(target=processor._run, daemon=True)
    worker.start()
    assert started.wait(5)
    assert job.status is JobStatus.PROCESSING

    processor.stop()
    release.set()
    worker.join(5)

    assert not worker.is_alive()
    assert job.status is JobStatus.PENDING
    assert updates[-1].status is JobStatus.PENDING


def test_completed_job_stays_completed(tmp_path):
    processor, job, updates = _processor_with_job(
        tmp_path,
        lambda job_id, input_path, output_path, **kwargs: None,
    )

    processor._run()

    assert job.status is JobStatus.COMPLETED
    assert job.output_path == tmp_path / "clip_restored.mp4"
    assert updates[-1].status is JobStatus.COMPLETED


def test_stop_cancels_the_running_pipeline(tmp_path):
    processor, _job, _updates = _processor_with_job(
        tmp_path,
        lambda *args, **kwargs: None,
    )
    pipeline = _FakePipeline()
    processor._current_pipeline = pipeline

    processor.stop()

    assert pipeline.cancel_requested


def test_interrupted_pipeline_marks_job_pending(tmp_path):
    from jasna.gui.processor import ProcessingStopped

    runs = []

    def fake_run_pipeline(job_id, input_path, output_path, **kwargs):
        runs.append(job_id)
        raise ProcessingStopped("Processing stopped")

    processor, job, updates = _processor_with_job(tmp_path, fake_run_pipeline)

    processor._run()

    assert runs == [job.id]  # a job left pending must not be picked up again

    assert job.status is JobStatus.PENDING
    assert updates[-1].status is JobStatus.PENDING


def test_failing_job_still_marked_error(tmp_path):
    def fake_run_pipeline(job_id, input_path, output_path, **kwargs):
        raise RuntimeError("boom")

    processor, job, updates = _processor_with_job(tmp_path, fake_run_pipeline)

    processor._run()

    assert job.status is JobStatus.ERROR
    assert updates[-1].status is JobStatus.ERROR


@pytest.mark.parametrize("accepted", [False, True])
def test_changed_progress_requires_confirmation_and_declining_skips_only_that_job(tmp_path, accepted):
    from concurrent.futures import Future
    from unittest.mock import MagicMock

    from jasna.job_resume import ResumeMismatchError

    directory = tmp_path / ".saved.jasna-resume"
    directory.mkdir()
    marker = directory / "saved"
    marker.touch()
    processor, job, updates = _processor_with_job(tmp_path, lambda *_a, **_k: None)
    next_job = JobItem(path=tmp_path / "next.mp4")
    processor._jobs.append(next_job)
    error = ResumeMismatchError(directory)
    processor._run_video_pipeline = MagicMock(side_effect=[error, None, None])
    processor._run_pipeline = processor._run_video_job
    answer = Future()
    answer.set_result(accepted)
    processor._on_resume_conflict = MagicMock(return_value=answer)
    completions = []
    processor._on_complete = completions.append
    processor._run()
    processor._on_resume_conflict.assert_called_once_with(directory)
    assert marker.exists() is not accepted
    assert processor._run_video_pipeline.call_count == (3 if accepted else 2)
    assert job.status is (JobStatus.COMPLETED if accepted else JobStatus.SKIPPED)
    assert next_job.status is JobStatus.COMPLETED
    assert completions == [True]


@pytest.mark.parametrize("file_conflict", ["auto_rename", "skip"])
def test_saved_progress_wins_over_file_conflict(tmp_path, monkeypatch, file_conflict):
    from fractions import Fraction
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from factories import session_config
    from jasna.job_resume import ResumeJob

    run = MagicMock()
    processor, job, _updates = _processor_with_job(tmp_path, run)
    job.path.touch()
    output = tmp_path / "clip_restored.mp4"
    output.touch()
    processor._settings.restoration_model = "ltx"
    processor._settings.file_conflict = file_conflict
    config = session_config(restoration_model_name="ltx")
    metadata = SimpleNamespace(num_frames=120, duration=4.0, time_base=Fraction(1,30))
    ResumeJob(config, job.path, output, metadata, ()).prepare()
    monkeypatch.setattr("jasna.gui.processor.video_session_config", lambda *_a, **_k: config)
    monkeypatch.setattr("jasna.gui.processor.build_job_encoder_settings", lambda *_a: {})
    monkeypatch.setattr("jasna.media.probe.get_video_meta_data", lambda *_a: metadata)
    processor._run()
    assert run.call_args.args[2] == output
    assert job.status is JobStatus.COMPLETED


def test_control_bar_shows_pause_for_ltx_and_uses_pause_confirmation():
    from tkinter import TclError
    from unittest.mock import MagicMock, patch

    import customtkinter as ctk

    from jasna.gui.control_bar import ControlBar
    from jasna.gui.locales import t

    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    try:
        bar = ControlBar(root)
        stop = MagicMock()
        bar.set_callbacks(on_stop=stop)
        bar.set_resumable(True)
        assert bar._stop_btn.cget("text") == "Ⅱ"
        with patch("jasna.gui.control_bar.messagebox.askyesno", return_value=True) as ask:
            bar._handle_stop()
        assert ask.call_args.args == (t("pause_confirm_title"), t("pause_confirm"))
        stop.assert_called_once()
        bar.set_resumable(False)
        assert bar._stop_btn.cget("text") == "■"
    finally:
        root.destroy()


@pytest.mark.parametrize("cancel_before_show", [False, True])
def test_resume_confirmation_is_posted_to_main_thread_and_can_be_cancelled(cancel_before_show):
    from types import SimpleNamespace
    from unittest.mock import patch

    from jasna.gui.app import JasnaApp

    posted = []
    app = SimpleNamespace(_main_thread=SimpleNamespace(post=posted.append))
    answer = JasnaApp._confirm_resume_restart(app, None)
    assert not answer.done()
    if cancel_before_show:
        assert answer.cancel()
    with patch("tkinter.messagebox.askyesno", return_value=True) as ask:
        posted.pop()()
    assert ask.called is not cancel_before_show
    if not cancel_before_show:
        assert answer.result() is True


@pytest.mark.parametrize("cancel_requested,completed,expected", [
    (False, True, False),
    (True, False, True),
    (True, True, False),
])
def test_pipeline_stop_detection(cancel_requested, completed, expected):
    from jasna.gui.processor import _pipeline_was_stopped

    pipeline = _FakePipeline()
    pipeline.cancel_requested = cancel_requested
    pipeline.completed = completed

    assert _pipeline_was_stopped(pipeline) is expected


@pytest.mark.parametrize("queue_finished", [False, True])
def test_only_a_finished_queue_is_reported_as_completed(queue_finished):
    from tkinter import TclError
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import customtkinter as ctk

    from jasna.gui.app import JasnaApp
    from jasna.gui.control_bar import ControlBar

    try:
        root = ctk.CTk()
    except TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    try:
        bar = ControlBar(root)
        bar.set_running(True)
        bar.update_progress("clip.mp4", 40.0)
        logs: list[str] = []
        post_export = MagicMock()
        app = SimpleNamespace(
            _status_pill=MagicMock(),
            _processing_start_time=0.0,
            _control_bar=bar,
            _update_start_button_state=lambda: None,
            _log_panel=SimpleNamespace(info=logs.append),
            _settings_panel=MagicMock(),
            _queue_panel=MagicMock(),
            _run_post_export_action=post_export,
        )

        JasnaApp._handle_complete(app, queue_finished)

        assert (bar._percent_label.cget("text") == "100%") is queue_finished
        assert (logs == ["All jobs completed"]) is queue_finished
        assert post_export.called is queue_finished
    finally:
        root.destroy()
