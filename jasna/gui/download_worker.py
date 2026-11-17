"""Background model downloads, kept off the widget layer."""

import threading
from typing import Callable

from jasna.gui.workers import run_in_background

DownloadProgress = Callable[[int, int | None], None]


def start_download(
    download: Callable[[DownloadProgress], None],
    on_percent: Callable[[int], None],
    on_done: Callable[[str | None], None],
) -> threading.Thread:
    """Run ``download(progress_callback)`` on a daemon thread.

    ``on_percent(percent)`` fires on every whole-percent change and
    ``on_done(error)`` fires once with ``None`` on success or the error text on
    failure. Both run on the worker thread; callers marshal to the UI thread.
    """
    progress_lock = threading.Lock()
    last_percent = -1

    def progress(downloaded: int, total: int | None):
        nonlocal last_percent
        if not total:
            return
        percent = max(0, min(100, int(downloaded * 100 / total)))
        with progress_lock:
            if percent == last_percent:
                return
            last_percent = percent
        on_percent(percent)

    return run_in_background(
        lambda: download(progress), lambda _result, error: on_done(None if error is None else str(error))
    )
