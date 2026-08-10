import threading

import pytest

from jasna.gui.models import AppSettings, JobItem, JobStatus
from jasna.gui.processor import Processor, ProgressUpdate


class _FakePipeline:
    def __init__(self):
        self.cancel_requested = False
        self.completed = False

    def cancel(self):
        self.cancel_requested = True


def _processor_with_job(tmp_path, run_pipeline):
    job = JobItem(path=tmp_path / "clip.mp4")
    updates: list[ProgressUpdate] = []
    processor = Processor(on_progress=updates.append)
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
