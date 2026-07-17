from __future__ import annotations

import queue
import subprocess

import pytest

from jasna.gui.segment_preview import (
    PreviewFrame,
    PreviewFullFrame,
    PreviewLoaded,
    SegmentPreviewWorker,
)
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


def _make_variable_rate_preview_source(path) -> None:
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
            "testsrc2=size=160x90:rate=10:duration=1",
            "-vf",
            r"setpts=if(lt(N\,5)\,N/(10*TB)\,(N+5)/(10*TB))",
            "-fps_mode",
            "vfr",
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


def test_preview_worker_finds_exact_previous_vfr_frame(tmp_path) -> None:
    source = tmp_path / "variable-rate-preview.mp4"
    _make_variable_rate_preview_source(source)
    worker = SegmentPreviewWorker(source, max_size=(80, 80))
    worker.start()

    try:
        _next_event(worker, PreviewLoaded)
        worker.seek(1.0)
        current = _next_event(worker, PreviewFrame)
        assert current.seconds == pytest.approx(1.0)

        worker.previous_frame(current.seconds)
        previous = _next_event(worker, PreviewFrame)

        assert previous.seconds == pytest.approx(0.4)
    finally:
        worker.close()


def test_grab_full_returns_native_resolution_of_shown_frame(tmp_path) -> None:
    source = tmp_path / "preview.mp4"
    _make_preview_source(source)
    worker = SegmentPreviewWorker(source, max_size=(80, 80))

    worker.start()

    try:
        _next_event(worker, PreviewLoaded)
        worker.seek(0.5)
        shown = _next_event(worker, PreviewFrame)
        assert shown.image.width <= 80

        worker.grab_full()
        full = _next_event(worker, PreviewFullFrame)
        assert (full.image.width, full.image.height) == (160, 90)
        assert full.seconds == shown.seconds
    finally:
        worker.close()


def test_preview_worker_uses_left_eye_for_explicit_sbs(tmp_path) -> None:
    source = tmp_path / "sbs-preview.mp4"
    _make_preview_source(source)
    worker = SegmentPreviewWorker(
        source,
        max_size=(80, 80),
        vr_mode="sbs",
    )
    worker.start()

    try:
        _next_event(worker, PreviewLoaded)
        worker.seek(0.0)
        frame = _next_event(worker, PreviewFrame)

        assert frame.image.size == (70, 80)
    finally:
        worker.close()
