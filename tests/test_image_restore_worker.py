from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from jasna.gui import video_session
from jasna.gui.image_restore_worker import ImageRestoreWorker, save_restored_image


def _start_worker(paths: list[Path], events: queue.Queue, stopped: threading.Event) -> ImageRestoreWorker:
    from jasna.gui.models import AppSettings

    worker = ImageRestoreWorker(
        paths,
        AppSettings(),
        on_first_image_size=lambda size: events.put(("size", size)),
        on_rendered=lambda *_rendered: None,
        on_failed=lambda token, exc: events.put((token, str(exc))),
        on_stopped=stopped.set,
    )
    worker.start()
    return worker


def test_worker_reports_the_first_image_size_then_a_failed_render_and_stops_on_close(
    monkeypatch, tmp_path: Path
) -> None:
    def missing_model(_settings, *, log):
        raise FileNotFoundError("model missing")

    monkeypatch.setattr(video_session, "build_image_session", missing_model)
    image_path = tmp_path / "image.png"
    Image.new("RGB", (64, 48)).save(image_path)
    events: queue.Queue = queue.Queue()
    stopped = threading.Event()
    worker = _start_worker([image_path], events, stopped)

    assert events.get(timeout=5) == ("size", (64, 48))
    worker.request(7, 0, 42)

    assert events.get(timeout=5) == (7, "model missing")
    worker.close()
    assert stopped.wait(5)


def test_worker_reports_an_unreadable_first_image_as_a_failure_before_any_request(tmp_path: Path) -> None:
    events: queue.Queue = queue.Queue()
    stopped = threading.Event()
    worker = _start_worker([tmp_path / "missing.png"], events, stopped)

    token, message = events.get(timeout=5)
    worker.close()

    assert token == 0
    assert "missing.png" in message
    assert stopped.wait(5)


def _save(source: Path, output_folder: str, image) -> tuple[Path | None, Exception | None]:
    done: queue.Queue = queue.Queue()
    save_restored_image(source, output_folder, "{original}_restored.mp4", image, lambda *result: done.put(result))
    return done.get(timeout=5)


def test_saving_creates_the_output_folder_and_never_overwrites(tmp_path: Path) -> None:
    source = tmp_path / "input.png"
    image = np.zeros((3, 8, 8), dtype=np.uint8)
    output_folder = tmp_path / "new" / "folder"

    first = _save(source, str(output_folder), image)
    second = _save(source, str(output_folder), image)

    assert first == (output_folder / "input_restored.png", None)
    assert second == (output_folder / "input_restored (1).png", None)
    assert Image.open(output_folder / "input_restored (1).png").size == (8, 8)


def test_saving_reports_a_write_failure(tmp_path: Path) -> None:
    path, error = _save(tmp_path / "input.png", "", np.zeros((8, 8), dtype=np.uint8))

    assert path is None
    assert isinstance(error, ValueError)
