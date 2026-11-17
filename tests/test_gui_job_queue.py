from pathlib import Path

from jasna.gui.models import JobItem, JobStatus
from factories import job_queue


def test_removing_an_earlier_job_during_a_scan_keeps_the_pending_job_in_the_scan():
    finished = JobItem(path=Path("a.mp4"), status=JobStatus.COMPLETED)
    pending = JobItem(path=Path("b.mp4"))
    queue = job_queue([finished, pending])

    scanned = []
    for job in queue:
        if job is finished:
            queue.remove(finished)
        scanned.append(job)

    assert scanned == [finished, pending]
    assert list(queue) == [pending]
    assert queue.next_pending() is pending


def test_add_unique_skips_inputs_that_are_already_queued():
    first = JobItem(path=Path("a.mp4"))
    queue = job_queue([first])

    added = queue.add_unique([JobItem(path=Path("a.mp4")), JobItem(path=Path("b.mp4")), JobItem(path=Path("b.mp4"))])

    assert [job.path for job in added] == [Path("b.mp4")]
    assert list(queue) == [first, *added]


def test_next_pending_follows_the_order_after_a_move():
    a, b, c = (JobItem(path=Path(name)) for name in ("a.mp4", "b.mp4", "c.mp4"))
    a.status = JobStatus.COMPLETED
    queue = job_queue([a, b, c])

    queue.move(2, 1)

    assert list(queue) == [a, c, b]
    assert queue.next_pending() is c


def test_requeue_moves_only_finished_jobs_to_the_end_as_pending():
    finished = JobItem(path=Path("a.mp4"), status=JobStatus.ERROR, error_message="failed", progress=0.5)
    pending = JobItem(path=Path("b.mp4"))
    queue = job_queue([finished, pending])

    assert queue.requeue(pending) is None
    assert queue.requeue(finished) == 0

    assert list(queue) == [pending, finished]
    assert (finished.status, finished.error_message, finished.progress) == (JobStatus.PENDING, "", 0.0)


def test_clear_finished_keeps_pending_and_processing_jobs():
    jobs = [JobItem(path=Path(f"{status.value}.mp4"), status=status) for status in JobStatus]
    queue = job_queue(jobs)

    queue.clear_finished()

    assert [job.status for job in queue] == [JobStatus.PENDING, JobStatus.PROCESSING]
