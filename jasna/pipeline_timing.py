from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager


class LoopTimer:
    """Accumulates wall time per category for one pipeline thread loop.

    Measures CPU-thread occupancy: queue waits show idle, work categories
    show where a thread spends its time. ~µs overhead per measure, safe to
    keep always on.
    """

    def __init__(self, name: str):
        self.name = name
        self.totals: dict[str, float] = {}

    @contextmanager
    def measure(self, category: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.totals[category] = self.totals.get(category, 0.0) + (time.perf_counter() - start)

    def timed_iter(self, iterable: Iterable, category: str) -> Iterator:
        iterator = iter(iterable)
        while True:
            with self.measure(category):
                try:
                    item = next(iterator)
                except StopIteration:
                    return
            yield item

    def summary(self) -> str:
        if not self.totals:
            return f"[timing] {self.name}: no samples"
        total = sum(self.totals.values())
        parts = ", ".join(f"{category}={seconds:.1f}s" for category, seconds in self.totals.items())
        return f"[timing] {self.name}: {parts} (tracked {total:.1f}s)"
