from __future__ import annotations

import queue
import threading
from pathlib import Path

from jasna.gui import video_session
from jasna.gui.image_restore_worker import ImageRestoreWorker
from jasna.gui.models import AppSettings


def test_worker_reports_a_failed_render_with_its_token_and_stops_on_close(monkeypatch) -> None:
    def missing_model(_settings, *, log):
        raise FileNotFoundError("model missing")

    monkeypatch.setattr(video_session, "build_image_session", missing_model)
    failures: queue.Queue = queue.Queue()
    stopped = threading.Event()
    worker = ImageRestoreWorker(
        [Path("image.png")],
        AppSettings(),
        on_rendered=lambda *_rendered: None,
        on_failed=lambda token, exc: failures.put((token, str(exc))),
        on_stopped=stopped.set,
    )
    worker.start()

    worker.request(7, 0, 42)

    assert failures.get(timeout=5) == (7, "model missing")
    worker.close()
    assert stopped.wait(5)
