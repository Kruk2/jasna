from __future__ import annotations

import queue
import threading
from pathlib import Path
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.gui import restoration_preview
from jasna.gui.models import AppSettings
from jasna.gui.restoration_preview import (
    RestorationPreviewWorker,
    _Cancel,
    _Request,
    _CenterFrameCollector,
    _PlaybackFrameCollector,
    _Stop,
    playback_window,
    preview_window,
)
from jasna.media.probe import VideoMetadata


def _metadata(*, fps=30.0, duration=10.0, time_base=Fraction(1, 90000), start_pts=0) -> VideoMetadata:
    return VideoMetadata(
        video_file="video.mp4",
        num_frames=round(duration * fps),
        video_fps=fps,
        average_fps=fps,
        video_fps_exact=Fraction(fps).limit_denominator(1001),
        codec_name="h264",
        duration=duration,
        video_width=1920,
        video_height=1080,
        time_base=time_base,
        start_pts=start_pts,
        color_space=AvColorspace.ITU709,
        color_range=AvColorRange.MPEG,
        is_10bit=False,
    )


def test_preview_window_centers_on_requested_time() -> None:
    metadata = _metadata()
    window = preview_window(metadata, 5.0, 90)

    assert window.seek_ts == pytest.approx(5.0 - 1.5)
    assert window.end_pts == round((5.0 + 1.5) * 90000)
    assert window.center_pts == round(5.0 * 90000)


def test_preview_window_clamps_at_video_start() -> None:
    metadata = _metadata()
    window = preview_window(metadata, 0.2, 90)

    assert window.seek_ts == 0.0
    assert window.end_pts == round(3.0 * 90000)
    assert window.center_pts == round(0.2 * 90000)


def test_preview_window_clamps_at_video_end() -> None:
    metadata = _metadata(duration=10.0)
    window = preview_window(metadata, 9.8, 90)

    assert window.seek_ts == pytest.approx(7.0)
    assert window.end_pts == round(10.0 * 90000)
    assert window.center_pts == round(9.8 * 90000)


def test_preview_window_offsets_by_start_pts() -> None:
    metadata = _metadata(start_pts=1000)
    window = preview_window(metadata, 5.0, 90)

    assert window.end_pts == 1000 + round(6.5 * 90000)
    assert window.center_pts == 1000 + round(5.0 * 90000)


def test_playback_window_starts_at_requested_time() -> None:
    metadata = _metadata()
    window = playback_window(metadata, 5.0, 90)

    assert window.seek_ts == pytest.approx(5.0)
    assert window.center_pts == round(5.0 * 90000)
    assert window.end_pts == round(8.0 * 90000)


def test_playback_window_clamps_at_video_end() -> None:
    metadata = _metadata()
    window = playback_window(metadata, 9.0, 90)

    assert window.seek_ts == pytest.approx(9.0)
    assert window.end_pts == round(10.0 * 90000)


def _frame(value: int) -> torch.Tensor:
    return torch.full((3, 4, 4), value, dtype=torch.uint8)


def test_collector_keeps_frame_closest_to_center_and_cancels() -> None:
    cancel = threading.Event()
    collector = _CenterFrameCollector(100, cancel)

    collector.write(_frame(1), 40)
    collector.write(_frame(2), 90)
    assert not collector.done
    collector.write(_frame(3), 130)

    assert collector.done
    assert cancel.is_set()
    assert collector._best_pts == 90
    collector.write(_frame(4), 160)
    assert collector._best_pts == 90


def test_collector_prefers_first_frame_past_center_when_closer() -> None:
    cancel = threading.Event()
    collector = _CenterFrameCollector(100, cancel)

    collector.write(_frame(1), 40)
    collector.write(_frame(2), 105)

    assert collector.done
    assert collector._best_pts == 105


def test_collector_exposes_nearest_frame_when_center_is_past_eof() -> None:
    collector = _CenterFrameCollector(100, threading.Event())

    collector.write(_frame(1), 40)
    collector.write(_frame(2), 90)

    assert not collector.done
    assert collector.has_result
    assert collector._best_pts == 90
    assert collector.result_image((10, 10)).getpixel((0, 0)) == (2, 2, 2)


def test_collector_result_image_caps_size() -> None:
    cancel = threading.Event()
    collector = _CenterFrameCollector(0, cancel)
    collector.write(torch.zeros((3, 1080, 1920), dtype=torch.uint8), 0)

    image = collector.result_image((960, 540))

    assert image.width <= 960
    assert image.height <= 540


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
    worker = RestorationPreviewWorker("unused.mp4", _metadata())

    first_generation = worker.request(1.0, AppSettings(), projection="auto")
    second_generation = worker.request(2.0, AppSettings(), projection="fisheye")

    command = worker._commands.get_nowait()
    assert command.center_seconds == 2.0
    assert command.projection == "fisheye"
    assert command.generation == second_generation
    assert second_generation == first_generation + 1
    with pytest.raises(queue.Empty):
        worker._commands.get_nowait()
    worker.close()


def test_restoration_worker_marks_playback_requests() -> None:
    worker = RestorationPreviewWorker("unused.mp4", _metadata())

    worker.request(2.0, AppSettings(), projection="raw", playback=True)

    command = worker._commands.get_nowait()
    assert command.playback is True
    assert command.projection == "raw"
    worker.close()


def test_preview_pass_forwards_scene_detection(monkeypatch) -> None:
    from jasna import pipeline_threads

    class ImmediateThread:
        def __init__(self, *, target, args=(), kwargs=None, **_options):
            self._run = lambda: target(*args, **(kwargs or {}))

        def start(self):
            self._run()

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

    decode_detect = MagicMock()
    monkeypatch.setattr(pipeline_threads, "decode_detect_loop", decode_detect)
    monkeypatch.setattr(pipeline_threads, "primary_restore_loop", MagicMock())
    monkeypatch.setattr(pipeline_threads, "secondary_restore_loop", MagicMock())
    monkeypatch.setattr(pipeline_threads, "blend_encode_loop", MagicMock())
    monkeypatch.setattr(restoration_preview.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(pipeline_threads, "VramOffloader", MagicMock())
    worker = RestorationPreviewWorker("video.mp4", _metadata())
    settings = AppSettings(scene_detection=False)
    worker.request(2.0, settings, projection="raw")
    command = worker._commands.get_nowait()
    pipeline = SimpleNamespace(
        input_video=Path("video.mp4"),
        decode_backend="auto",
        device=torch.device("cpu"),
        restoration_pipeline=SimpleNamespace(secondary_num_workers=1),
        max_clip_size=settings.max_clip_size,
        batch_size=settings.batch_size,
        temporal_overlap=settings.temporal_overlap,
        max_detection_gap=settings.max_detection_gap,
        min_detection_duration=settings.min_detection_duration,
        scene_detection=settings.scene_detection,
        vr_projection="auto",
        vr_projector=None,
        job_detection_model=MagicMock(),
        vr_resolution=SimpleNamespace(resolved="off"),
        stereo=None,
        configure_vr=lambda metadata: None,
    )

    worker._run_preview_pass(command, pipeline)

    assert pipeline.vr_projection == "raw"
    assert decode_detect.call_args.args[1].scene_detection is False
    worker.close()


def _preview_pipeline() -> SimpleNamespace:
    return SimpleNamespace(
        vr_projection="auto",
        configure_vr=lambda metadata: None,
        vr_resolution=SimpleNamespace(resolved="off"),
        stereo=None,
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize(
    ("action", "queued"),
    [
        (lambda worker: worker.request(2.0, AppSettings(), projection="auto"), _Request),
        (lambda worker: worker.cancel(), _Cancel),
        (lambda worker: worker.close(), _Stop),
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
    worker = RestorationPreviewWorker("unused.mp4", _metadata())
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
    worker.join(timeout=2)

    assert stopped.is_set()


def _queue_jobs(app, tmp_path, statuses) -> list:
    from jasna.gui.models import JobStatus

    for index, status in enumerate(statuses):
        app._queue_panel.add_job(tmp_path / f"clip-{index}.mp4")
        job = app._queue_panel.get_jobs()[index]
        if status is JobStatus.COMPLETED:
            job.output_path = tmp_path / f"clip-{index}_restored.mp4"
        app._queue_panel.update_job_status(job.id, status)
    return app._queue_panel.get_jobs()


@pytest.fixture
def start_calls(jasna_app, monkeypatch) -> dict[str, list]:
    """What Start validated and handed to the processor; neither validation nor processing runs."""
    calls: dict[str, list] = {"validated": [], "started": []}
    preflight = SimpleNamespace(missing=(), should_warn_first_run_slow=False)

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

    jasna_app._set_preview_gpu_busy(True)
    jasna_app._main_thread._run_pending()

    assert jasna_app._control_bar._start_btn.cget("state") == "disabled"


@pytest.mark.parametrize("busy", ["preview", "processor"])
def test_app_start_is_ignored_while_the_gpu_is_busy(jasna_app, start_calls, monkeypatch, tmp_path, busy) -> None:
    from jasna.gui.models import JobStatus

    _queue_jobs(jasna_app, tmp_path, [JobStatus.PENDING])
    if busy == "preview":
        jasna_app._set_preview_gpu_busy(True)
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
    assert start_calls["started"] == [jasna_app._queue_panel.get_jobs_ref()]
    if continuing:
        assert (finished.status, skipped.status) == (JobStatus.COMPLETED, JobStatus.SKIPPED)
        assert finished.output_path == tmp_path / "clip-0_restored.mp4"
    else:
        assert all(job.status is JobStatus.PENDING for job in (finished, skipped, paused))
