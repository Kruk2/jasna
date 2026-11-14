"""Process-start stopwatch for diagnosing launch latency.

Imported as the first jasna module in ``__main__`` so ``PROCESS_START`` and ``ENTERED_AT`` are
captured close to process start. The GUI logs elapsed-since-start at a few milestones
(pre-window work, UI build, first paint), and the first job logs how long the app took to
reach it, so a slow frozen launch can be pinpointed on the target machine without attaching
a profiler.
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

PROCESS_START = time.perf_counter()
ENTERED_AT = time.time()
"""Wall clock when the app's own code began, comparable with other processes' clocks."""
_checked_at: float | None = None
_first_job_logged = False


def elapsed_ms() -> float:
    return (time.perf_counter() - PROCESS_START) * 1000.0


def mark_startup_checked() -> None:
    """Called once the release's startup check has passed."""
    global _checked_at
    _checked_at = time.time()


def app_startup_line(created_at: float, entered_at: float, checked_at: float | None, job_at: float) -> str:
    """Seconds from the process's creation to its first job, split at the app's own start and,
    in releases, at the end of the startup check (all wall-clock times)."""
    parts = [f"process start {entered_at - created_at:.1f}s"]
    if checked_at is not None:
        parts.append(f"startup check {checked_at - entered_at:.1f}s")
    parts.append(f"until the job {job_at - (entered_at if checked_at is None else checked_at):.1f}s")
    return f"[timing] app startup: {job_at - created_at:.1f}s to the first job ({', '.join(parts)})"


def log_first_job() -> None:
    """Log ``app_startup_line`` when the process starts its first job."""
    global _first_job_logged
    if _first_job_logged:
        return
    _first_job_logged = True
    import psutil

    logger.info(app_startup_line(psutil.Process().create_time(), ENTERED_AT, _checked_at, time.time()))
