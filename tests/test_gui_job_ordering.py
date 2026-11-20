"""Tests for GUI job ordering — stable job_id tracking during reorder/remove/add."""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from jasna.gui.job_queue import JobQueue
from jasna.gui.processor import Processor, ProgressUpdate
from jasna.gui.models import JobItem, JobStatus, AppSettings
from jasna.post_export_action import PostExportVideoCommandError
from factories import gui_processor, job_queue


def _make_jobs(*names: str) -> list[JobItem]:
    return [JobItem(path=Path(n)) for n in names]


def _run(processor: Processor, jobs: list[JobItem] | JobQueue, settings: AppSettings = AppSettings(), output_folder: str = "") -> None:
    processor.start(jobs if isinstance(jobs, JobQueue) else job_queue(jobs), settings, output_folder=output_folder, output_pattern="{original}_restored.mp4")
    processor.join(timeout=5.0)
    assert not processor.is_running()


def _restored(fake_job_pipelines) -> list[str]:
    return [pipeline.input_path.name for pipeline in fake_job_pipelines.built]


class TestProcessorPullLoop:
    def test_processes_jobs_in_order_by_status(self, fake_job_pipelines):
        jobs = _make_jobs("a.mp4", "b.mp4", "c.mp4")

        _run(gui_processor(), jobs)

        assert _restored(fake_job_pipelines) == ["a.mp4", "b.mp4", "c.mp4"]
        assert all(j.status == JobStatus.COMPLETED for j in jobs)

    def test_unsupported_video_is_skipped_with_its_reason_and_the_queue_continues(self, fake_job_pipelines):
        from jasna.media.probe import UnsupportedVideoError

        updates: list[ProgressUpdate] = []
        jobs = _make_jobs("a_vr.mp4", "b_clip.mp4")
        reason = "LTX does not support VR180 side-by-side video; restore it with BasicVSR++"

        def restore(pipeline):
            if pipeline.input_path.name == "a_vr.mp4":
                raise UnsupportedVideoError(reason)

        fake_job_pipelines.restore = restore
        _run(gui_processor(on_progress=updates.append), jobs)

        assert [j.status for j in jobs] == [JobStatus.SKIPPED, JobStatus.COMPLETED]
        assert any(u.job_id == jobs[0].id and u.status is JobStatus.SKIPPED and u.message == reason for u in updates)

    def test_skips_removed_pending_job(self, fake_job_pipelines):
        jobs = _make_jobs("a.mp4", "b.mp4", "c.mp4")
        queue = job_queue(jobs)

        def restore(pipeline):
            if pipeline.input_path.name == "a.mp4":
                queue.remove(jobs[1])  # the GUI removes b.mp4 while a.mp4 is processing

        fake_job_pipelines.restore = restore
        _run(gui_processor(), queue)

        assert _restored(fake_job_pipelines) == ["a.mp4", "c.mp4"]

    def test_progress_update_carries_job_id(self, fake_job_pipelines):
        updates: list[ProgressUpdate] = []
        jobs = _make_jobs("a.mp4")

        _run(gui_processor(on_progress=updates.append), jobs)

        assert all(u.job_id == jobs[0].id for u in updates)
        assert any(u.status == JobStatus.PROCESSING for u in updates)
        assert any(u.status == JobStatus.COMPLETED for u in updates)

    def test_reorder_during_processing_respects_new_order(self, fake_job_pipelines):
        jobs = _make_jobs("a.mp4", "b.mp4", "c.mp4")
        queue = job_queue(jobs)

        def restore(pipeline):
            if pipeline.input_path.name == "a.mp4":
                queue.move(2, 1)

        fake_job_pipelines.restore = restore
        _run(gui_processor(), queue)

        assert _restored(fake_job_pipelines) == ["a.mp4", "c.mp4", "b.mp4"]

    @pytest.mark.parametrize("stop_during_first_job", [False, True])
    def test_reports_whether_the_queue_finished(self, fake_job_pipelines, stop_during_first_job):
        finished: list[bool] = []
        processor = gui_processor(on_complete=finished.append)
        jobs = _make_jobs("a.mp4", "b.mp4")
        if stop_during_first_job:
            fake_job_pipelines.restore = lambda _pipeline: processor.stop()

        _run(processor, jobs, AppSettings(post_export_action="shutdown"))

        assert finished == [not stop_during_first_job]
        expected = [JobStatus.PENDING] * 2 if stop_during_first_job else [JobStatus.COMPLETED] * 2
        assert [job.status for job in jobs] == expected

    def test_skips_jobs_that_already_finished(self, fake_job_pipelines):
        jobs = _make_jobs("a.mp4", "b.mp4")
        jobs[0].status = JobStatus.COMPLETED

        _run(gui_processor(), jobs)

        assert _restored(fake_job_pipelines) == ["b.mp4"]

    def test_runs_post_export_video_command_after_each_video(self, fake_job_pipelines, tmp_path):
        calls: list[tuple[str, Path, Path]] = []
        jobs = _make_jobs(str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4"))

        with patch(
            "jasna.post_export_action.run_post_export_video_command",
            lambda command, input_path, output_path, _cancel, **_kwargs: calls.append((command, input_path, output_path)),
        ):
            _run(
                gui_processor(), jobs,
                AppSettings(post_export_video_command="remux {output}"), output_folder=str(tmp_path),
            )

        assert calls == [
            ("remux {output}", tmp_path / "a.mp4", tmp_path / "a_restored.mp4"),
            ("remux {output}", tmp_path / "b.mp4", tmp_path / "b_restored.mp4"),
        ]
        assert all(job.status is JobStatus.COMPLETED for job in jobs)

    def test_post_export_video_command_failure_marks_job_error_and_continues(self, fake_job_pipelines, tmp_path):
        jobs = _make_jobs(str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4"))

        def run_command(_command, _input_path, output_path, _cancel, **_kwargs):
            if output_path.name == "a_restored.mp4":
                raise PostExportVideoCommandError("exit code 7")

        with patch("jasna.post_export_action.run_post_export_video_command", side_effect=run_command):
            _run(
                gui_processor(), jobs,
                AppSettings(post_export_video_command="remux {output}"), output_folder=str(tmp_path),
            )

        assert jobs[0].status is JobStatus.ERROR
        assert jobs[1].status is JobStatus.COMPLETED

    def test_post_export_video_command_skips_images(self, fake_job_pipelines, tmp_path):
        from PIL import Image

        commands: list[Path] = []
        jobs = _make_jobs(str(tmp_path / "image.png"))
        Image.new("RGB", (8, 8)).save(jobs[0].path)

        with patch(
            "jasna.post_export_action.run_post_export_video_command",
            lambda _command, input_path, *_args, **_kwargs: commands.append(input_path),
        ):
            _run(
                gui_processor(), jobs,
                AppSettings(post_export_video_command="remux {output}"), output_folder=str(tmp_path),
            )

        assert jobs[0].status is JobStatus.COMPLETED
        assert commands == []

    def test_post_export_video_command_receives_auto_renamed_output(self, fake_job_pipelines, tmp_path):
        (tmp_path / "clip_restored.mp4").touch()
        outputs: list[Path] = []
        jobs = _make_jobs(str(tmp_path / "clip.mp4"))

        with patch(
            "jasna.post_export_action.run_post_export_video_command",
            lambda _command, _input_path, output_path, *_args, **_kwargs: outputs.append(output_path),
        ):
            _run(
                gui_processor(), jobs,
                AppSettings(post_export_video_command="remux {output}", file_conflict="auto_rename"),
                output_folder=str(tmp_path),
            )

        assert outputs == [tmp_path / "clip_restored (1).mp4"]

    def test_log_placeholder_gets_a_file_with_only_that_videos_log(self, fake_job_pipelines, tmp_path):
        logs_seen_by_command: dict[str, tuple[Path, str]] = {}
        jobs = _make_jobs(str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4"))
        fake_job_pipelines.restore = lambda pipeline: logging.getLogger("jasna.restorer").warning(
            "restoring %s", pipeline.input_path.name
        )

        def run_command(_command, input_path, _output_path, _cancel, *, log_path):
            logs_seen_by_command[input_path.name] = (log_path, log_path.read_text(encoding="utf-8"))

        with patch("jasna.post_export_action.run_post_export_video_command", side_effect=run_command):
            _run(
                gui_processor(), jobs,
                AppSettings(post_export_video_command="copy {log} logs"), output_folder=str(tmp_path),
            )

        a_log_path, a_log = logs_seen_by_command["a.mp4"]
        b_log_path, b_log = logs_seen_by_command["b.mp4"]
        assert (a_log_path, b_log_path) == (tmp_path / "a_restored.log", tmp_path / "b_restored.log")
        assert "Started processing a.mp4" in a_log and "restoring a.mp4" in a_log
        assert "Started processing b.mp4" in b_log and "restoring b.mp4" in b_log
        assert "a.mp4" not in b_log
        assert "b.mp4" not in a_log
        assert all(job.status is JobStatus.COMPLETED for job in jobs)

    def test_command_without_log_placeholder_writes_no_log_file(self, fake_job_pipelines, tmp_path):
        log_paths: list[Path | None] = []
        jobs = _make_jobs(str(tmp_path / "a.mp4"))

        with patch(
            "jasna.post_export_action.run_post_export_video_command",
            lambda *_args, log_path: log_paths.append(log_path),
        ):
            _run(
                gui_processor(), jobs,
                AppSettings(post_export_video_command="remux {output}"), output_folder=str(tmp_path),
            )

        assert log_paths == [None]
        assert list(tmp_path.glob("*.log")) == []


class TestJobItemId:
    def test_unique_ids(self):
        jobs = _make_jobs("a.mp4", "b.mp4", "c.mp4")
        ids = [j.id for j in jobs]
        assert len(set(ids)) == 3
