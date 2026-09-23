"""Command-queue helpers shared by the GUI background workers."""
from __future__ import annotations

import queue


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
