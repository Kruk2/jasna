"""Command-queue helpers shared by the GUI background workers."""
from __future__ import annotations

import queue
from collections.abc import Callable


def drain(pending: queue.Queue) -> None:
    """Discard everything currently queued."""
    while True:
        try:
            pending.get_nowait()
        except queue.Empty:
            return


def replace_pending(pending: queue.Queue, command: object) -> None:
    """Replace any queued command with ``command``; the GUI thread is the only producer."""
    drain(pending)
    pending.put_nowait(command)


class MainThreadCalls:
    """Runs callbacks posted by worker threads on the Tk main thread.

    Tk must only be touched from the main thread, so workers ``post`` and the
    main loop polls; call ``close`` before destroying ``widget``.
    """

    def __init__(self, widget, interval_ms: int) -> None:
        self._widget = widget
        self._interval_ms = interval_ms
        self._pending: queue.SimpleQueue = queue.SimpleQueue()
        self._after_id = widget.after(interval_ms, self._run_pending)

    def post(self, callback: Callable[[], object]) -> None:
        self._pending.put(callback)

    def _run_pending(self) -> None:
        self._after_id = self._widget.after(self._interval_ms, self._run_pending)
        while True:
            try:
                callback = self._pending.get_nowait()
            except queue.Empty:
                return
            callback()

    def close(self) -> None:
        self._widget.after_cancel(self._after_id)
