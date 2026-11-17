from __future__ import annotations

import queue
import threading
from fractions import Fraction

import pytest
import torch

from factories import make_pipeline, video_metadata
from jasna.gui.engine_preflight import EnginePreflightResult
from jasna.gui.models import AppSettings, JobItem
from jasna.gui.queues import Cancel, Stop
from jasna.gui.restoration_preview import (
    RestorationPreviewWorker,
    _Request,
    _CenterFrameCollector,
    _PlaybackFrameCollector,
    playback_window,
    preview_window,
)


def _metadata(*, start_pts: int = 0):
    """A 10-second 30 fps 1080p video on a 90 kHz time base."""
    return video_metadata(
        fps=30,
        num_frames=300,
        video_file="video.mp4",
        video_width=1920,
        video_height=1080,
        codec_name="h264",
        time_base=Fraction(1, 90000),
        start_pts=start_pts,
    )


@pytest.mark.parametrize(
    ("seconds", "start_pts", "seek_ts", "center_pts", "end_pts"),
    [
        (5.0, 0, 3.5, round(5.0 * 90000), round(6.5 * 90000)),
        (0.2, 0, 0.0, round(0.2 * 90000), round(3.0 * 90000)),
        (9.8, 0, 7.0, round(9.8 * 90000), round(10.0 * 90000)),
        (5.0, 1000, 3.5, 1000 + round(5.0 * 90000), 1000 + round(6.5 * 90000)),
    ],
    ids=["centered", "clamped-at-start", "clamped-at-end", "start-pts-offset"],
)
def test_preview_window_centers_on_the_requested_time(seconds, start_pts, seek_ts, center_pts, end_pts) -> None:
    window = preview_window(_metadata(start_pts=start_pts), seconds, 90)

    assert (window.seek_ts, window.center_pts, window.end_pts) == (pytest.approx(seek_ts), center_pts, end_pts)


@pytest.mark.parametrize(
    ("seconds", "end_seconds"), [(5.0, 8.0), (9.0, 10.0)], ids=["three-seconds", "clamped-at-end"]
)
def test_playback_window_starts_at_the_requested_time(seconds, end_seconds) -> None:
    window = playback_window(_metadata(), seconds, 90)

    assert (window.seek_ts, window.center_pts, window.end_pts) == (
        pytest.approx(seconds), round(seconds * 90000), round(end_seconds * 90000)
    )


def _frame(value: int) -> torch.Tensor:
    return torch.full((3, 4, 4), value, dtype=torch.uint8)


def _shown_value(collector: _CenterFrameCollector) -> int:
    return collector.result_image((4, 4)).getpixel((0, 0))[0]


def test_collector_keeps_frame_closest_to_center_and_cancels() -> None:
    cancel = threading.Event()
    collector = _CenterFrameCollector(100, cancel)

    collector.write(_frame(1), 40)
    collector.write(_frame(2), 90)
    assert not collector.done
    collector.write(_frame(3), 130)

    assert collector.done
    assert cancel.is_set()
    assert _shown_value(collector) == 2
    collector.write(_frame(4), 160)
    assert _shown_value(collector) == 2


def test_collector_prefers_first_frame_past_center_when_closer() -> None:
    cancel = threading.Event()
    collector = _CenterFrameCollector(100, cancel)

    collector.write(_frame(1), 40)
    collector.write(_frame(2), 105)

    assert collector.done
    assert _shown_value(collector) == 2


def test_collector_exposes_nearest_frame_when_center_is_past_eof() -> None:
    collector = _CenterFrameCollector(100, threading.Event())

    collector.write(_frame(1), 40)
    collector.write(_frame(2), 90)

    assert not collector.done
    assert collector.has_result
    assert collector.result_image((10, 10)).getpixel((0, 0)) == (2, 2, 2)


def test_collector_result_image_caps_size() -> None:
    cancel = threading.Event()
    collector = _CenterFrameCollector(0, cancel)
    collector.write(torch.zeros((3, 1080, 1920), dtype=torch.uint8), 0)

    image = collector.result_image((960, 540))

    assert image.size == (960, 540)


def test_collector_applies_lut_before_copying_frame() -> None:
    class _Lut:
        def __init__(self) -> None:
            self.frames = []

        def apply(self, frame):
            self.frames.append(frame)
            return frame + 5

    lut = _Lut()
    collector = _CenterFrameCollector(0, threading.Event(), lut)
    frame = _frame(1)

    collector.write(frame, 0)
    image = collector.result_image((10, 10))

    assert lut.frames == [frame]
    assert image.getpixel((0, 0)) == (6, 6, 6)


def test_playback_collector_returns_timestamped_bounded_images() -> None:
    metadata = _metadata()
    collector = _PlaybackFrameCollector(metadata, (8, 8))

    collector.write(torch.zeros((3, 20, 40), dtype=torch.uint8), 90000)
    frames = collector.result_frames()

    assert len(frames) == 1
    assert frames[0].seconds == pytest.approx(1.0)
    assert frames[0].image.size == (8, 4)


def test_restoration_collectors_show_only_left_eye() -> None:
    frame = torch.zeros((3, 8, 16), dtype=torch.uint8)
    frame[:, :, :8] = 25
    frame[:, :, 8:] = 200
    still = _CenterFrameCollector(
        0,
        threading.Event(),
        stereo="sbs",
    )
    still.write(frame, 0)
    playback = _PlaybackFrameCollector(
        _metadata(),
        (16, 16),
        stereo="sbs",
    )
    playback.write(frame, 0)

    still_image = still.result_image((16, 16))
    playback_image = playback.result_frames()[0].image

    assert still_image.size == (8, 8)
    assert playback_image.size == (8, 8)
    assert still_image.getpixel((0, 0)) == (25, 25, 25)
    assert playback_image.getpixel((0, 0)) == (25, 25, 25)


def test_restoration_worker_coalesces_pending_requests() -> None:
    worker = RestorationPreviewWorker("unused.mp4", _metadata(), on_stopped=lambda: None)

    first_generation = worker.request(1.0, AppSettings(), projection="auto")
    second_generation = worker.request(2.0, AppSettings(), projection="fisheye")

    command = worker._commands.get_nowait()
    assert command.center_seconds == 2.0
    assert command.settings.vr_projection == "fisheye"
    assert command.generation == second_generation
    assert second_generation == first_generation + 1
    with pytest.raises(queue.Empty):
        worker._commands.get_nowait()
    worker.close()


def test_restoration_worker_marks_playback_requests() -> None:
    worker = RestorationPreviewWorker("unused.mp4", _metadata(), on_stopped=lambda: None)

    worker.request(2.0, AppSettings(), projection="raw", playback=True)

    command = worker._commands.get_nowait()
    assert command.playback is True
    assert command.settings.vr_projection == "raw"
    worker.close()


def _preview_pipeline():
    pipeline = make_pipeline()
    pipeline.configure_vr(_metadata())
    return pipeline


@pytest.mark.parametrize(
    ("action", "queued"),
    [
        (lambda worker: worker.request(2.0, AppSettings(), projection="auto"), _Request),
        (lambda worker: worker.cancel(), Cancel),
        (lambda worker: worker.close(), Stop),
    ],
    ids=["request", "cancel", "close"],
)
def test_restoration_worker_cancels_the_running_pass(monkeypatch, action, queued) -> None:
    from jasna import pipeline_threads

    entered = threading.Event()
    cancelled: list[bool] = []

    def run_pass(pipeline, metadata, collector, cancel_event, **_kwargs):
        entered.set()
        cancelled.append(cancel_event.wait(5))

    monkeypatch.setattr(pipeline_threads, "run_restoration_pass", run_pass)
    worker = RestorationPreviewWorker("unused.mp4", _metadata(), on_stopped=lambda: None)
    worker.request(1.0, AppSettings(), projection="auto")
    running = threading.Thread(target=worker._run_preview_pass, args=(worker._commands.get_nowait(), _preview_pipeline()))
    running.start()
    assert entered.wait(5)

    action(worker)
    running.join(5)

    assert cancelled == [True]
    assert isinstance(worker._commands.get_nowait(), queued)
    worker.close()


def test_restoration_worker_reports_completed_teardown() -> None:
    stopped = threading.Event()
    worker = RestorationPreviewWorker("unused.mp4", _metadata(), on_stopped=stopped.set)
    worker.start()

    worker.close()

    assert stopped.wait(timeout=30)


def _queue_jobs(app, tmp_path, statuses) -> list:
    from jasna.gui.models import JobStatus

    for index, status in enumerate(statuses):
        app._queue_panel.add_jobs([JobItem(path=tmp_path / f"clip-{index}.mp4")])
        job = app._queue_panel.get_jobs()[index]
        if status is JobStatus.COMPLETED:
            job.output_path = tmp_path / f"clip-{index}_restored.mp4"
        app._queue_panel.update_job_status(job.id, status)
    return app._queue_panel.get_jobs()


@pytest.fixture
def start_calls(jasna_app, monkeypatch) -> dict[str, list]:
    """What Start validated and handed to the processor; neither validation nor processing runs."""
    calls: dict[str, list] = {"validated": [], "started": []}
    preflight = EnginePreflightResult(requirements=())

    def validate(settings, jobs, **_kwargs):
        calls["validated"].append(list(jobs))
        return []

    monkeypatch.setattr("jasna.gui.validation.validate_gui_start", validate)
    monkeypatch.setattr("jasna.gui.engine_preflight.run_engine_preflight", lambda _settings: preflight)
    monkeypatch.setattr(jasna_app._processor, "start", lambda jobs, *_args: calls["started"].append(jobs))
    return calls


def test_app_disables_start_while_preview_gpu_is_busy(jasna_app, tmp_path) -> None:
    from jasna.gui.models import JobStatus

    _queue_jobs(jasna_app, tmp_path, [JobStatus.PENDING])
    assert jasna_app._control_bar._start_btn.cget("state") == "normal"

    jasna_app._set_preview_gpu_busy("preview", True)
    jasna_app._main_thread._run_pending()

    assert jasna_app._control_bar._start_btn.cget("state") == "disabled"


@pytest.mark.parametrize("busy", ["preview", "processor"])
def test_app_start_is_ignored_while_the_gpu_is_busy(jasna_app, start_calls, monkeypatch, tmp_path, busy) -> None:
    from jasna.gui.models import JobStatus

    _queue_jobs(jasna_app, tmp_path, [JobStatus.PENDING])
    if busy == "preview":
        jasna_app._set_preview_gpu_busy("preview", True)
    else:
        monkeypatch.setattr(jasna_app._processor, "is_running", lambda: True)

    jasna_app._on_start()

    assert start_calls == {"validated": [], "started": []}


@pytest.mark.parametrize("stopped", [False, True])
@pytest.mark.parametrize("pending", [False, True])
def test_app_prepares_queue_or_continues_after_pause(jasna_app, start_calls, tmp_path, stopped, pending) -> None:
    from jasna.gui.models import JobStatus

    finished, skipped, paused = _queue_jobs(
        jasna_app, tmp_path, [JobStatus.COMPLETED, JobStatus.SKIPPED, JobStatus.PENDING if pending else JobStatus.COMPLETED]
    )
    if stopped:
        jasna_app._processor.stop()

    jasna_app._on_start()

    continuing = stopped and pending
    assert start_calls["validated"] == [[paused] if continuing else [finished, skipped, paused]]
    assert start_calls["started"] == [jasna_app._queue_panel.job_queue]
    if continuing:
        assert (finished.status, skipped.status) == (JobStatus.COMPLETED, JobStatus.SKIPPED)
        assert finished.output_path == tmp_path / "clip-0_restored.mp4"
    else:
        assert all(job.status is JobStatus.PENDING for job in (finished, skipped, paused))
