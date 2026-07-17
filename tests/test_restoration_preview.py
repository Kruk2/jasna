from __future__ import annotations

import queue
import threading
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.gui.models import AppSettings
from jasna.gui.app import JasnaApp
from jasna.gui.restoration_preview import (
    RestoredClipFrame,
    RestorationPreviewWorker,
    _Cancel,
    _CenterFrameCollector,
    _PlaybackFrameCollector,
    _Stop,
    playback_window,
    preview_window,
)
from jasna.gui.segment_editor import SegmentEditor
from jasna.media import VideoMetadata


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


def test_restoration_worker_coalesces_pending_requests() -> None:
    worker = RestorationPreviewWorker("unused.mp4", _metadata())

    first_generation = worker.request(1.0, AppSettings())
    second_generation = worker.request(2.0, AppSettings())

    command = worker._commands.get_nowait()
    assert command.center_seconds == 2.0
    assert command.generation == second_generation
    assert second_generation == first_generation + 1
    with pytest.raises(queue.Empty):
        worker._commands.get_nowait()
    worker.close()


def test_restoration_worker_marks_playback_requests() -> None:
    worker = RestorationPreviewWorker("unused.mp4", _metadata())

    worker.request(2.0, AppSettings(), playback=True)

    assert worker._commands.get_nowait().playback is True
    worker.close()


def test_restoration_worker_cancels_before_queueing_replacement() -> None:
    worker = RestorationPreviewWorker("unused.mp4", _metadata())
    order = []
    worker._cancel_active_pass = MagicMock(side_effect=lambda: order.append("cancel"))
    worker._replace_command = MagicMock(side_effect=lambda *_args, **_kwargs: order.append("queue"))

    worker.request(1.0, AppSettings())

    assert order == ["cancel", "queue"]


def test_restoration_worker_request_cancels_active_pass() -> None:
    worker = RestorationPreviewWorker("unused.mp4", _metadata())
    active = threading.Event()
    worker._active_cancel = active

    worker.request(1.0, AppSettings())

    assert active.is_set()
    worker.close()


def test_restoration_worker_close_cancels_active_pass() -> None:
    worker = RestorationPreviewWorker("unused.mp4", _metadata())
    active = threading.Event()
    worker._active_cancel = active

    worker.close()

    assert active.is_set()
    assert isinstance(worker._commands.get_nowait(), _Stop)


def test_restoration_worker_cancel_stops_pass_without_closing_worker() -> None:
    worker = RestorationPreviewWorker("unused.mp4", _metadata())
    active = threading.Event()
    worker._active_cancel = active

    worker.cancel()

    assert active.is_set()
    assert not worker._closed.is_set()
    assert isinstance(worker._commands.get_nowait(), _Cancel)
    worker.close()


def test_restoration_worker_reports_completed_teardown() -> None:
    stopped = threading.Event()
    worker = RestorationPreviewWorker("unused.mp4", _metadata(), on_stopped=stopped.set)
    worker.start()

    worker.close()
    worker.join(timeout=2)

    assert stopped.is_set()


def test_segment_editor_requests_restored_playback_when_no_buffer_exists() -> None:
    editor = SegmentEditor.__new__(SegmentEditor)
    editor._state = SimpleNamespace(duration=10.0, fps=30.0)
    editor._current = 2.0
    editor._restore_active = True
    editor._restore_play_pending = False
    editor._restored_clip = ()
    editor._playing = False
    editor._request_restoration_playback = MagicMock()

    editor._toggle_play()

    editor._request_restoration_playback.assert_called_once_with(2.0)


def test_segment_editor_continues_with_next_restored_playback_window() -> None:
    editor = SegmentEditor.__new__(SegmentEditor)
    editor._state = SimpleNamespace(duration=10.0, fps=30.0)
    editor._current = 2.0
    editor._playing = True
    editor._restore_active = True
    editor._restored_clip = (
        RestoredClipFrame(1.0, MagicMock()),
        RestoredClipFrame(2.0, MagicMock()),
    )
    editor._closed = threading.Event()
    editor._set_playing = MagicMock()
    editor._request_restoration_playback = MagicMock()

    editor._request_next_frame()

    editor._set_playing.assert_called_once_with(False)
    editor._request_restoration_playback.assert_called_once_with(pytest.approx(2.0 + 1 / 30))


def test_app_disables_queue_start_while_preview_gpu_is_tearing_down() -> None:
    app = JasnaApp.__new__(JasnaApp)
    app._preview_gpu_busy = True
    app._queue_panel = MagicMock()
    app._queue_panel.get_jobs.return_value = [object()]
    app._control_bar = MagicMock()

    app._update_start_button_state()

    assert app._control_bar.set_start_enabled.call_args.args[0] is False


def test_app_start_guard_blocks_preview_gpu_overlap() -> None:
    app = JasnaApp.__new__(JasnaApp)
    app._preview_gpu_busy = True
    app._queue_panel = MagicMock()

    app._on_start()

    app._queue_panel.get_jobs.assert_not_called()
