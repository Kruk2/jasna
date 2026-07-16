from __future__ import annotations

import queue
import subprocess

import pytest

from jasna.gui.segment_preview import PreviewFrame, PreviewLoaded, SegmentPreviewWorker
from jasna.os_utils import resolve_executable, subprocess_no_window_kwargs


def _make_preview_source(path) -> None:
    subprocess.run(
        [
            resolve_executable("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=12:duration=1",
            "-c:v",
            "mpeg4",
            str(path),
        ],
        check=True,
        **subprocess_no_window_kwargs(),
    )


def _next_event(worker, event_type):
    while True:
        event = worker.events.get(timeout=5)
        if isinstance(event, event_type):
            return event


def test_preview_worker_loads_seeks_and_decodes_sequentially(tmp_path) -> None:
    source = tmp_path / "preview.mp4"
    _make_preview_source(source)
    worker = SegmentPreviewWorker(source, max_size=(80, 80))
    worker.start()

    try:
        loaded = _next_event(worker, PreviewLoaded)
        assert loaded.metadata.duration == 1

        worker.seek(0.5)
        sought = _next_event(worker, PreviewFrame)
        assert sought.seconds == 0.5
        assert sought.image.width <= 80
        assert sought.image.height <= 80

        worker.next_frame()
        following = _next_event(worker, PreviewFrame)
        assert following.seconds > sought.seconds
    finally:
        worker.close()


def test_preview_worker_coalesces_pending_seeks() -> None:
    worker = SegmentPreviewWorker("unused.mp4")

    first_generation = worker.seek(1)
    second_generation = worker.seek(2)

    command = worker._commands.get_nowait()
    assert command.seconds == 2
    assert command.generation == second_generation
    assert second_generation == first_generation + 1
    with pytest.raises(queue.Empty):
        worker._commands.get_nowait()
    worker.close()
