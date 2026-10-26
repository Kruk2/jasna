"""Tests for GUI job ordering — stable job_id tracking during reorder/remove/add."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from jasna.gui.processor import Processor, ProgressUpdate
from jasna.gui.models import JobItem, JobStatus, AppSettings
from jasna.post_export_action import PostExportVideoCommandError
from factories import no_resume_conflict

pytestmark = pytest.mark.usefixtures("no_gpu_cleanup", "tk_garbage_collected")


def _make_jobs(*names: str) -> list[JobItem]:
    return [JobItem(path=Path(n)) for n in names]


class TestProcessorPullLoop:
    def test_processes_jobs_in_order_by_status(self):
        processed_ids: list[int] = []
        p = Processor(on_resume_conflict=no_resume_conflict)
        jobs = _make_jobs("a.mp4", "b.mp4", "c.mp4")

        def fake_pipeline(job_id, inp, out, **_kwargs):
            processed_ids.append(job_id)

        with patch.object(p, "_run_pipeline", side_effect=fake_pipeline):
            p.start(
                jobs,
                AppSettings(),
                output_folder="",
                output_pattern="{original}_restored.mp4",
            )
            p.join(timeout=5.0)

        assert processed_ids == [jobs[0].id, jobs[1].id, jobs[2].id]
        assert all(j.status == JobStatus.COMPLETED for j in jobs)

    def test_unsupported_video_is_skipped_with_its_reason_and_the_queue_continues(self):
        from jasna.media.probe import UnsupportedVideoError

        updates: list[ProgressUpdate] = []
        p = Processor(on_resume_conflict=no_resume_conflict, on_progress=updates.append)
        jobs = _make_jobs("a_vr.mp4", "b_clip.mp4")
        reason = "LTX does not support VR180 side-by-side video; restore it with BasicVSR++"

        with patch.object(p, "_run_pipeline", side_effect=[UnsupportedVideoError(reason), None]):
            p.start(jobs, AppSettings(), output_folder="", output_pattern="{original}_restored.mp4")
            p.join(timeout=5.0)

        assert [j.status for j in jobs] == [JobStatus.SKIPPED, JobStatus.COMPLETED]
        assert any(u.job_id == jobs[0].id and u.status is JobStatus.SKIPPED and u.message == reason for u in updates)

    def test_skips_removed_pending_job(self):
        processed_ids: list[int] = []
        p = Processor(on_resume_conflict=no_resume_conflict)
        jobs = _make_jobs("a.mp4", "b.mp4", "c.mp4")

        call_count = [0]

        def fake_pipeline(job_id, inp, out, **_kwargs):
            processed_ids.append(job_id)
            call_count[0] += 1
            if call_count[0] == 1:
                # Simulate GUI removing b.mp4 while a is processing
                del jobs[1]

        with patch.object(p, "_run_pipeline", side_effect=fake_pipeline):
            p.start(
                jobs,
                AppSettings(),
                output_folder="",
                output_pattern="{original}_restored.mp4",
            )
            p.join(timeout=5.0)

        assert len(processed_ids) == 2
        filenames = [j.filename for j in jobs if j.status == JobStatus.COMPLETED]
        assert "b.mp4" not in filenames

    def test_progress_update_carries_job_id(self):
        updates: list[ProgressUpdate] = []
        p = Processor(on_resume_conflict=no_resume_conflict, on_progress=lambda u: updates.append(u))
        jobs = _make_jobs("a.mp4")

        with patch.object(p, "_run_pipeline"):
            p.start(
                jobs,
                AppSettings(),
                output_folder="",
                output_pattern="{original}_restored.mp4",
            )
            p.join(timeout=5.0)

        assert all(u.job_id == jobs[0].id for u in updates)
        assert any(u.status == JobStatus.PROCESSING for u in updates)
        assert any(u.status == JobStatus.COMPLETED for u in updates)

    def test_reorder_during_processing_respects_new_order(self):
        processed_filenames: list[str] = []
        p = Processor(on_resume_conflict=no_resume_conflict)
        jobs = _make_jobs("a.mp4", "b.mp4", "c.mp4")

        call_count = [0]

        def fake_pipeline(job_id, inp, out, **_kwargs):
            processed_filenames.append(inp.name)
            call_count[0] += 1
            if call_count[0] == 1:
                # After processing a.mp4, reorder: move c before b
                pending = [j for j in jobs if j.status == JobStatus.PENDING]
                if len(pending) >= 2:
                    idx_b = jobs.index(pending[0])
                    idx_c = jobs.index(pending[1])
                    jobs[idx_b], jobs[idx_c] = jobs[idx_c], jobs[idx_b]

        with patch.object(p, "_run_pipeline", side_effect=fake_pipeline):
            p.start(
                jobs,
                AppSettings(),
                output_folder="",
                output_pattern="{original}_restored.mp4",
            )
            p.join(timeout=5.0)

        assert processed_filenames == ["a.mp4", "c.mp4", "b.mp4"]

    @pytest.mark.parametrize("stop_during_first_job", [False, True])
    def test_reports_whether_the_queue_finished(self, stop_during_first_job):
        finished: list[bool] = []
        p = Processor(on_resume_conflict=no_resume_conflict, on_complete=finished.append)
        jobs = _make_jobs("a.mp4", "b.mp4")

        def run_pipeline(*_args, **_kwargs):
            if stop_during_first_job:
                p.stop()

        with patch.object(p, "_run_pipeline", side_effect=run_pipeline):
            p.start(jobs, AppSettings(post_export_action="shutdown"), output_folder="", output_pattern="{original}_restored.mp4")
            p.join(timeout=5.0)

        assert finished == [not stop_during_first_job]
        assert (jobs[1].status is JobStatus.PENDING) is stop_during_first_job

    def test_skips_jobs_that_already_finished(self):
        processed: list[str] = []
        p = Processor(on_resume_conflict=no_resume_conflict)
        jobs = _make_jobs("a.mp4", "b.mp4")
        jobs[0].status = JobStatus.COMPLETED

        with patch.object(p, "_run_pipeline", side_effect=lambda job_id, inp, out, **_kwargs: processed.append(inp.name)):
            p.start(jobs, AppSettings(), output_folder="", output_pattern="{original}_restored.mp4")
            p.join(timeout=5.0)

        assert processed == ["b.mp4"]

    def test_runs_post_export_video_command_after_each_video(self, tmp_path):
        calls: list[tuple[str, Path, Path]] = []
        p = Processor(on_resume_conflict=no_resume_conflict)
        jobs = _make_jobs(str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4"))

        with (
            patch.object(p, "_run_pipeline"),
            patch(
                "jasna.post_export_action.run_post_export_video_command",
                lambda command, input_path, output_path, _cancel: calls.append(
                    (command, input_path, output_path)
                ),
            ),
        ):
            p.start(
                jobs,
                AppSettings(post_export_video_command="remux {output}"),
                output_folder=str(tmp_path),
                output_pattern="{original}_restored.mp4",
            )
            p.join(timeout=5.0)

        assert calls == [
            (
                "remux {output}",
                tmp_path / "a.mp4",
                tmp_path / "a_restored.mp4",
            ),
            (
                "remux {output}",
                tmp_path / "b.mp4",
                tmp_path / "b_restored.mp4",
            ),
        ]
        assert all(job.status is JobStatus.COMPLETED for job in jobs)

    def test_post_export_video_command_failure_marks_job_error_and_continues(self, tmp_path):
        p = Processor(on_resume_conflict=no_resume_conflict)
        jobs = _make_jobs(str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4"))

        def run_command(_command, _input_path, output_path, _cancel):
            if output_path.name == "a_restored.mp4":
                raise PostExportVideoCommandError("exit code 7")

        with (
            patch.object(p, "_run_pipeline"),
            patch(
                "jasna.post_export_action.run_post_export_video_command",
                side_effect=run_command,
            ),
        ):
            p.start(
                jobs,
                AppSettings(post_export_video_command="remux {output}"),
                output_folder=str(tmp_path),
                output_pattern="{original}_restored.mp4",
            )
            p.join(timeout=5.0)

        assert jobs[0].status is JobStatus.ERROR
        assert jobs[1].status is JobStatus.COMPLETED

    def test_post_export_video_command_skips_images(self, tmp_path):
        command = MagicMock()
        p = Processor(on_resume_conflict=no_resume_conflict)
        jobs = _make_jobs(str(tmp_path / "image.png"))

        with (
            patch.object(p, "_run_pipeline"),
            patch("jasna.post_export_action.run_post_export_video_command", command),
        ):
            p.start(
                jobs,
                AppSettings(post_export_video_command="remux {output}"),
                output_folder=str(tmp_path),
                output_pattern="{original}_restored.mp4",
            )
            p.join(timeout=5.0)

        command.assert_not_called()

    def test_post_export_video_command_receives_auto_renamed_output(self, tmp_path):
        (tmp_path / "clip_restored.mp4").touch()
        command = MagicMock()
        p = Processor(on_resume_conflict=no_resume_conflict)
        jobs = _make_jobs(str(tmp_path / "clip.mp4"))

        with (
            patch.object(p, "_run_pipeline"),
            patch("jasna.post_export_action.run_post_export_video_command", command),
        ):
            p.start(
                jobs,
                AppSettings(
                    post_export_video_command="remux {output}",
                    file_conflict="auto_rename",
                ),
                output_folder=str(tmp_path),
                output_pattern="{original}_restored.mp4",
            )
            p.join(timeout=5.0)

        assert command.call_args.args[2] == tmp_path / "clip_restored (1).mp4"


class TestJobItemId:
    def test_unique_ids(self):
        jobs = _make_jobs("a.mp4", "b.mp4", "c.mp4")
        ids = [j.id for j in jobs]
        assert len(set(ids)) == 3
