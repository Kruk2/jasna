"""Run blocking work off the Tk thread."""

import threading
from collections.abc import Callable
from typing import TypeVar

Result = TypeVar("Result")


def run_in_background(
    operation: Callable[[], Result],
    on_done: Callable[[Result | None, Exception | None], None],
    name: str | None = None,
) -> threading.Thread:
    """Run ``operation`` on a daemon thread, then ``on_done(result, None)`` or ``on_done(None, error)``
    on that same thread; callers post to the UI thread themselves."""

    def worker():
        try:
            result = operation()
        except Exception as exc:
            on_done(None, exc)
        else:
            on_done(result, None)

    thread = threading.Thread(target=worker, daemon=True, name=name)
    thread.start()
    return thread
