"""The ordered job list the queue panel shows and the processor works through."""

import threading
from typing import Iterable, Iterator

from jasna.gui.models import JobItem, JobStatus

FINISHED_STATUSES = frozenset({JobStatus.COMPLETED, JobStatus.ERROR, JobStatus.SKIPPED})


class JobQueue:
    """Job order shared by the UI thread, which edits it, and the processor thread,
    which picks the next pending job while a run is going."""

    def __init__(self) -> None:
        self._jobs: list[JobItem] = []
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._jobs)

    def __iter__(self) -> Iterator[JobItem]:
        with self._lock:
            return iter(list(self._jobs))

    def __getitem__(self, index: int) -> JobItem:
        with self._lock:
            return self._jobs[index]

    def add_unique(self, jobs: Iterable[JobItem]) -> list[JobItem]:
        """Appends the jobs whose input is not queued yet and returns them."""
        jobs = list(jobs)
        added = []
        with self._lock:
            queued = {job.path for job in self._jobs}
            for job in jobs:
                if job.path in queued:
                    continue
                queued.add(job.path)
                self._jobs.append(job)
                added.append(job)
        return added

    def index_of(self, job_id: int) -> int | None:
        with self._lock:
            return self._index_of(job_id)

    def _index_of(self, job_id: int) -> int | None:
        for index, job in enumerate(self._jobs):
            if job.id == job_id:
                return index
        return None

    def remove(self, job: JobItem) -> int | None:
        """Removes the job; returns the index it had, or None when it was not queued."""
        with self._lock:
            index = self._index_of(job.id)
            if index is not None:
                del self._jobs[index]
            return index

    def move(self, from_index: int, to_index: int) -> None:
        with self._lock:
            self._jobs.insert(to_index, self._jobs.pop(from_index))

    def requeue(self, job: JobItem) -> int | None:
        """Moves a finished job to the end as pending again; returns the index it had,
        or None when the job is not finished or not queued."""
        with self._lock:
            if job.status not in FINISHED_STATUSES:
                return None
            index = self._index_of(job.id)
            if index is None:
                return None
            self._jobs.append(self._jobs.pop(index))
            job.error_message = ""
            job.output_path = None
            job.status = JobStatus.PENDING
            job.progress = 0.0
            job.elapsed_seconds = None
            return index

    def clear(self) -> None:
        with self._lock:
            self._jobs.clear()

    def clear_finished(self) -> None:
        with self._lock:
            self._jobs[:] = [job for job in self._jobs if job.status not in FINISHED_STATUSES]

    def next_pending(self) -> JobItem | None:
        with self._lock:
            return next((job for job in self._jobs if job.status is JobStatus.PENDING), None)
