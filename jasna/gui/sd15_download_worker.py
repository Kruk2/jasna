"""Background SD15 bundle download, kept off the widget layer."""

import threading
from pathlib import Path

from jasna.restorer.sd15_download import download_sd15_bundle


def start_sd15_download(
    model_dir: Path,
    repo_id: str,
    on_percent,
    on_done,
) -> threading.Thread:
    """Download the SD15 bundle on a daemon thread.

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

    def worker():
        error = None
        try:
            download_sd15_bundle(model_dir, repo_id, progress_callback=progress)
        except Exception as exc:  # surface failure to the user
            error = str(exc)
        on_done(error)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread
