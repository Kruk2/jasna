"""Tests for GUI job ordering — stable job_id tracking during reorder/remove/add."""

import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

from jasna.gui.processor import Processor, ProgressUpdate
from jasna.gui.models import JobItem, JobStatus, AppSettings


def _make_jobs(*names: str) -> list[JobItem]:
    return [JobItem(path=Path(n)) for n in names]


class TestNextPendingJob:
    def test_returns_first_pending(self):
        p = Processor()
        jobs = _make_jobs("a.mp4", "b.mp4", "c.mp4")
        jobs[0].status = JobStatus.COMPLETED
        p._jobs = jobs
        assert p._next_pending_job() is jobs[1]

    def test_returns_none_when_all_done(self):
        p = Processor()
        jobs = _make_jobs("a.mp4", "b.mp4")
        jobs[0].status = JobStatus.COMPLETED
        jobs[1].status = JobStatus.ERROR
        p._jobs = jobs
        assert p._next_pending_job() is None

    def test_respects_reordered_list(self):
        p = Processor()
        jobs = _make_jobs("a.mp4", "b.mp4", "c.mp4")
        jobs[0].status = JobStatus.COMPLETED
        # Simulate reorder: move c before b
        jobs[1], jobs[2] = jobs[2], jobs[1]
        p._jobs = jobs
        result = p._next_pending_job()
        assert result.filename == "c.mp4"

    def test_picks_up_newly_added_job(self):
        p = Processor()
        jobs = _make_jobs("a.mp4")
        jobs[0].status = JobStatus.COMPLETED
        p._jobs = jobs
        assert p._next_pending_job() is None
        new_job = JobItem(path=Path("b.mp4"))
        jobs.append(new_job)
        assert p._next_pending_job() is new_job


class TestProcessorPullLoop:
    def test_processes_jobs_in_order_by_status(self):
        processed_ids: list[int] = []
        p = Processor()
        jobs = _make_jobs("a.mp4", "b.mp4", "c.mp4")

        def fake_pipeline(job_id, inp, out):
            processed_ids.append(job_id)

        with patch.object(p, "_run_pipeline", side_effect=fake_pipeline):
            p.start(
                jobs,
                AppSettings(),
                output_folder="",
                output_pattern="{original}_restored.mp4",
                disable_basicvsrpp_tensorrt=False,
            )
            p.join(timeout=5.0)

        assert processed_ids == [jobs[0].id, jobs[1].id, jobs[2].id]
        assert all(j.status == JobStatus.COMPLETED for j in jobs)

    def test_skips_removed_pending_job(self):
        processed_ids: list[int] = []
        p = Processor()
        jobs = _make_jobs("a.mp4", "b.mp4", "c.mp4")

        call_count = [0]

        def fake_pipeline(job_id, inp, out):
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
                disable_basicvsrpp_tensorrt=False,
            )
            p.join(timeout=5.0)

        assert len(processed_ids) == 2
        filenames = [j.filename for j in jobs if j.status == JobStatus.COMPLETED]
        assert "b.mp4" not in filenames

    def test_progress_update_carries_job_id(self):
        updates: list[ProgressUpdate] = []
        p = Processor(on_progress=lambda u: updates.append(u))
        jobs = _make_jobs("a.mp4")

        with patch.object(p, "_run_pipeline"):
            p.start(
                jobs,
                AppSettings(),
                output_folder="",
                output_pattern="{original}_restored.mp4",
                disable_basicvsrpp_tensorrt=False,
            )
            p.join(timeout=5.0)

        assert all(u.job_id == jobs[0].id for u in updates)
        assert any(u.status == JobStatus.PROCESSING for u in updates)
        assert any(u.status == JobStatus.COMPLETED for u in updates)

    def test_reorder_during_processing_respects_new_order(self):
        processed_filenames: list[str] = []
        p = Processor()
        jobs = _make_jobs("a.mp4", "b.mp4", "c.mp4")

        call_count = [0]

        def fake_pipeline(job_id, inp, out):
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
                disable_basicvsrpp_tensorrt=False,
            )
            p.join(timeout=5.0)

        assert processed_filenames == ["a.mp4", "c.mp4", "b.mp4"]


class TestJobItemId:
    def test_unique_ids(self):
        jobs = _make_jobs("a.mp4", "b.mp4", "c.mp4")
        ids = [j.id for j in jobs]
        assert len(set(ids)) == 3

    def test_id_survives_list_reorder(self):
        jobs = _make_jobs("a.mp4", "b.mp4")
        id_a, id_b = jobs[0].id, jobs[1].id
        jobs.reverse()
        assert jobs[0].id == id_b
        assert jobs[1].id == id_a
