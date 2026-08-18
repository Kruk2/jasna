import threading
from unittest.mock import MagicMock, patch

import pytest

from jasna.gui.models import AppSettings, JobItem, JobStatus
from jasna.gui.processor import ProcessingStopped, Processor, ProgressUpdate


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
    processor._validate_completed_video_output = MagicMock()

    processor._run()

    assert job.status is JobStatus.COMPLETED
    assert job.output_path == tmp_path / "clip_restored.mp4"
    assert updates[-1].status is JobStatus.COMPLETED


def test_pipeline_return_without_output_is_not_completed(tmp_path):
    processor, job, updates = _processor_with_job(
        tmp_path,
        lambda job_id, input_path, output_path, **kwargs: None,
    )

    processor._run()

    assert job.status is JobStatus.ERROR
    assert updates[-1].status is JobStatus.ERROR
    assert "completed output is missing" in updates[-1].message


def test_overwrite_pipeline_cannot_claim_unchanged_preexisting_output(tmp_path):
    existing = tmp_path / "clip_restored.mp4"
    existing.write_bytes(b"old completed output")
    processor, job, updates = _processor_with_job(
        tmp_path,
        lambda job_id, input_path, output_path, **kwargs: None,
    )
    processor._settings = AppSettings(file_conflict="overwrite")

    processor._run()

    assert job.status is JobStatus.ERROR
    assert updates[-1].status is JobStatus.ERROR
    assert "not created or changed" in updates[-1].message


def test_changed_video_output_is_synced_before_completion(tmp_path):
    existing = tmp_path / "clip_restored.mp4"
    existing.write_bytes(b"old")

    def write_new_output(job_id, input_path, output_path, **kwargs):
        output_path.write_bytes(b"new completed output")

    processor, job, updates = _processor_with_job(tmp_path, write_new_output)
    processor._settings = AppSettings(file_conflict="overwrite", codec="hevc")

    with patch("jasna.media.splice.sync_and_validate_final_output") as validate:
        processor._run()

    assert job.status is JobStatus.COMPLETED
    assert updates[-1].status is JobStatus.COMPLETED
    validate.assert_called_once_with(
        existing,
        source=job.path,
        expected_codec="hevc",
    )


def test_video_post_export_runs_after_final_output_validation(tmp_path):
    events: list[str] = []
    output_paths_during_validation = []

    def write_new_output(job_id, input_path, output_path, **kwargs):
        output_path.write_bytes(b"new completed output")

    processor, job, _updates = _processor_with_job(tmp_path, write_new_output)
    processor._settings = AppSettings(post_export_video_command="remux {output}")

    def validate(*_args, **_kwargs):
        events.append("validate")
        output_paths_during_validation.append(job.output_path)

    with (
        patch(
            "jasna.media.splice.sync_and_validate_final_output",
            side_effect=validate,
        ),
        patch(
            "jasna.post_export_action.run_post_export_video_command",
            side_effect=lambda *_args, **_kwargs: events.append("post-export"),
        ),
    ):
        processor._run()

    assert job.status is JobStatus.COMPLETED
    assert events == ["validate", "post-export"]
    assert output_paths_during_validation == [None]


def test_stop_during_final_validation_prevents_completion_markers(tmp_path):
    def write_new_output(job_id, input_path, output_path, **kwargs):
        output_path.write_bytes(b"new completed output")

    processor, job, updates = _processor_with_job(tmp_path, write_new_output)
    processor._settings.pre_scan_policy = "off"

    def validate(*_args, **_kwargs):
        processor.stop()

    with patch(
        "jasna.media.splice.sync_and_validate_final_output",
        side_effect=validate,
    ):
        processor._run()

    assert processor._stop_event.is_set()
    assert job.status is JobStatus.PENDING
    assert job.output_path is None
    assert getattr(processor, "completed_processing_path", lambda _job_id: None)(job.id) is None
    assert updates[-1].status is JobStatus.PENDING


def test_stop_during_post_export_prevents_completion_markers(tmp_path):
    def write_new_output(job_id, input_path, output_path, **kwargs):
        output_path.write_bytes(b"new completed output")

    processor, job, updates = _processor_with_job(tmp_path, write_new_output)
    processor._settings = AppSettings(post_export_video_command="remux {output}")
    processor._settings.pre_scan_policy = "off"

    with (
        patch("jasna.media.splice.sync_and_validate_final_output"),
        patch(
            "jasna.post_export_action.run_post_export_video_command",
            side_effect=lambda *_args, **_kwargs: processor.stop(),
        ),
    ):
        processor._run()

    assert processor._stop_event.is_set()
    assert job.status is JobStatus.PENDING
    assert job.output_path is None
    assert getattr(processor, "completed_processing_path", lambda _job_id: None)(job.id) is None
    assert updates[-1].status is JobStatus.PENDING


def test_validation_failure_skips_video_post_export_command(tmp_path):
    def write_new_output(job_id, input_path, output_path, **kwargs):
        output_path.write_bytes(b"new completed output")

    processor, job, updates = _processor_with_job(tmp_path, write_new_output)
    processor._settings = AppSettings(post_export_video_command="remux {output}")

    with (
        patch(
            "jasna.media.splice.sync_and_validate_final_output",
            side_effect=RuntimeError("final output invalid"),
        ),
        patch("jasna.post_export_action.run_post_export_video_command") as command,
    ):
        processor._run()

    assert job.status is JobStatus.ERROR
    assert updates[-1].status is JobStatus.ERROR
    command.assert_not_called()


def test_image_jobs_do_not_require_video_output_validation(tmp_path):
    job = JobItem(path=tmp_path / "image.png")
    processor = Processor()
    processor._jobs = [job]
    processor._settings = AppSettings()
    processor._output_folder = str(tmp_path)
    processor._output_pattern = "{original}_restored.mp4"
    processor._run_pipeline = MagicMock()
    processor._validate_completed_video_output = MagicMock()

    processor._run()

    assert job.status is JobStatus.COMPLETED
    assert job.output_path == tmp_path / "image_restored.png"
    processor._validate_completed_video_output.assert_not_called()


def test_stop_after_job_selection_does_not_claim_job_or_create_output_directory(
    tmp_path,
):
    nested_output = tmp_path / "not-created" / "nested"
    processor, job, _updates = _processor_with_job(
        tmp_path,
        MagicMock(),
    )
    processor._output_folder = str(nested_output)
    original_next_pending_job = processor._next_pending_job

    def select_then_stop():
        selected = original_next_pending_job()
        processor.stop()
        return selected

    processor._next_pending_job = select_then_stop

    processor._run()

    assert job.status is JobStatus.PENDING
    assert not nested_output.exists()
    processor._run_pipeline.assert_not_called()


def test_stop_after_job_claim_does_not_create_output_directory(tmp_path):
    nested_output = tmp_path / "not-created" / "nested"
    processor, job, updates = _processor_with_job(
        tmp_path,
        MagicMock(),
    )
    processor._output_folder = str(nested_output)
    original_create_parent = processor._create_output_parent_unless_stopped

    def stop_then_create_parent(output_path):
        processor.stop()
        original_create_parent(output_path)

    processor._create_output_parent_unless_stopped = stop_then_create_parent

    processor._run()

    assert job.status is JobStatus.PENDING
    assert updates[-1].status is JobStatus.PENDING
    assert not nested_output.exists()
    processor._run_pipeline.assert_not_called()


def test_full_render_is_published_from_same_directory_staging(tmp_path):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    staging_path = tmp_path / ".output.jasna-full-test.mp4"
    pipeline = MagicMock(cancel_requested=False, completed=True)
    pipeline.run.side_effect = lambda: staging_path.write_bytes(b"complete video")
    processor = Processor()
    processor._settings = AppSettings(codec="hevc")
    processor._video_session = MagicMock()
    processor._ensure_video_session = MagicMock()
    processor._prepare_job_detector = MagicMock()
    processor._build_encoder_settings = MagicMock(return_value={})
    processor._full_render_staging_path = MagicMock(return_value=staging_path)

    def publish(staging, destination, **_kwargs):
        staging.replace(destination)

    with (
        patch("jasna.gui.processor.video_session_config", return_value=MagicMock()),
        patch("jasna.gui.processor.build_pipeline", return_value=pipeline) as build,
        patch("jasna.media.splice.commit_video_output", side_effect=publish) as commit,
    ):
        processor._run_video_job(1, input_path, output_path)

    assert build.call_args.args[3] == staging_path
    commit.assert_called_once_with(
        staging_path,
        output_path,
        source=input_path,
        codec="hevc",
    )
    assert output_path.read_bytes() == b"complete video"
    assert not staging_path.exists()


def test_stop_before_full_render_publish_keeps_existing_output(tmp_path):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    output_path.write_bytes(b"previous complete video")
    staging_path = tmp_path / ".output.jasna-full-test.mp4"
    pipeline = MagicMock(cancel_requested=False, completed=True)
    processor = Processor()
    processor._settings = AppSettings(codec="hevc")
    processor._video_session = MagicMock()
    processor._ensure_video_session = MagicMock()
    processor._prepare_job_detector = MagicMock()
    processor._build_encoder_settings = MagicMock(return_value={})
    processor._full_render_staging_path = MagicMock(return_value=staging_path)

    def finish_after_stop():
        staging_path.write_bytes(b"partial replacement")
        processor.stop()

    pipeline.run.side_effect = finish_after_stop

    with (
        patch("jasna.gui.processor.video_session_config", return_value=MagicMock()),
        patch("jasna.gui.processor.build_pipeline", return_value=pipeline),
        patch("jasna.media.splice.commit_video_output") as commit,
        pytest.raises(ProcessingStopped, match="Processing stopped"),
    ):
        processor._run_video_job(1, input_path, output_path)

    commit.assert_not_called()
    assert output_path.read_bytes() == b"previous complete video"
    assert not staging_path.exists()


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
