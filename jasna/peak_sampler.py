from __future__ import annotations

import threading
from collections.abc import Callable


class PeakSampler:
    """Calls ``read`` on a daemon thread as it starts and then every ``interval_seconds``
    until stopped, keeping the largest value it returned."""

    def __init__(self, read: Callable[[], int], interval_seconds: float) -> None:
        self._read = read
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._peak = 0
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="PeakSampler", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> int:
        """The peak read; raises what a read raised."""
        self._stop.set()
        self._thread.join()
        if self._error is not None:
            raise self._error
        return self._peak

    def _run(self) -> None:
        try:
            while True:
                self._peak = max(self._peak, self._read())
                if self._stop.wait(self._interval_seconds):
                    return
        except BaseException as error:
            self._error = error
