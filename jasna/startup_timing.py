"""Process-start stopwatch for diagnosing GUI launch latency.

Imported as the first jasna module in ``__main__`` so ``PROCESS_START`` is captured close
to process start. The GUI logs elapsed-since-start at a few milestones (pre-window work,
UI build, first paint) so a slow frozen launch can be pinpointed on the target machine
without attaching a profiler.
"""
from __future__ import annotations

import time

PROCESS_START = time.perf_counter()


def elapsed_ms() -> float:
    return (time.perf_counter() - PROCESS_START) * 1000.0
