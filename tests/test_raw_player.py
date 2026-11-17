from __future__ import annotations

import subprocess
import threading
from contextlib import nullcontext
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from factories import FixedDetectorSession, make_pipeline, session_config, video_metadata
from jasna.gui import raw_player
from jasna.gui.raw_player import (
    PlayerFailed,
    PlayerFrame,
    PlayerStatus,
    RawFrameWriter,
    RawPlayerWorker,
    SoftwareClock,
    TimestampFrameBuffer,
    VlcAudioClock,
    run_raw_restoration_pass,
)
from jasna.gui.models import AppSettings
from jasna.restorer.restoration_pipeline import RestorationPipeline

METADATA = video_metadata(
    fps=60,
    num_frames=600,
    video_file="video.mp4",
    video_width=1920,
    video_height=1080,
    codec_name="h264",
    time_base=Fraction(1, 1000),
    start_pts=100,
)


def _player_frame(seconds: float, generation: int = 1) -> PlayerFrame:
    return PlayerFrame(seconds, raw_player.Image.new("RGB", (2, 2)), generation)


@pytest.mark.parametrize(
    ("max_seconds", "max_frames"), [(20, 1), (0.1, 60)], ids=["frame-cap", "time-cap"]
)
def test_timestamp_buffer_caps_and_releases_a_cancelled_producer(max_seconds, max_frames) -> None:
    frame_buffer = TimestampFrameBuffer(max_seconds=max_seconds, max_frames=max_frames)
    cancel = threading.Event()
    frame_buffer.reset(1)
    assert frame_buffer.put(_player_frame(0.0), cancel)
    result: list[bool] = []
    producer = threading.Thread(
        target=lambda: result.append(frame_buffer.put(_player_frame(0.1), cancel))
    )

    producer.start()
    cancel.set()
    producer.join()

    assert result == [False]
    assert frame_buffer.pop_due(1.0).seconds == 0.0
    assert frame_buffer.empty()


def test_timestamp_buffer_discards_old_generation() -> None:
    frame_buffer = TimestampFrameBuffer()
    frame_buffer.reset(2)

    assert not frame_buffer.put(_player_frame(0.0, generation=1), threading.Event())
    assert frame_buffer.empty()


def test_timestamp_buffer_returns_newest_due_frame() -> None:
    frame_buffer = TimestampFrameBuffer()
    frame_buffer.reset(1)
    cancel = threading.Event()
    for seconds in (1.0, 1.03, 1.06, 1.2):
        assert frame_buffer.put(_player_frame(seconds), cancel)

    due = frame_buffer.pop_due(1.05, tolerance=0.02)

    assert due is not None
    assert due.seconds == pytest.approx(1.06)
    assert frame_buffer.peek().seconds == pytest.approx(1.2)


def test_timestamp_buffer_reports_seconds_ahead_of_playhead() -> None:
    frame_buffer = TimestampFrameBuffer()
    frame_buffer.reset(1)
    cancel = threading.Event()
    frame_buffer.put(_player_frame(4.0), cancel)
    frame_buffer.put(_player_frame(5.5), cancel)

    assert frame_buffer.buffered_ahead(3.0) == pytest.approx(2.5)
    assert frame_buffer.buffered_ahead(6.0) == 0.0


def test_timestamp_buffer_preroll_accepts_short_eof_tail() -> None:
    frame_buffer = TimestampFrameBuffer()
    frame_buffer.reset(3)
    frame_buffer.put(_player_frame(9.9, generation=3), threading.Event())

    assert not frame_buffer.ready(0.5)
    frame_buffer.mark_eof(3)
    assert frame_buffer.ready(0.5)


def test_raw_writer_downsizes_before_building_image_and_caps_display_rate() -> None:
    frame_buffer = TimestampFrameBuffer()
    frame_buffer.reset(4)
    writer = RawFrameWriter(
        METADATA,
        frame_buffer,
        threading.Event(),
        4,
        (80, 40),
    )
    frame = torch.full((3, 100, 200), 20, dtype=torch.uint8)

    writer.write(frame, 100)
    writer.write(frame, 116)
    writer.write(frame, 134)

    assert frame_buffer.peek().image.size == (80, 40)
    assert [frame_buffer.pop_due(0.0).seconds, frame_buffer.pop_due(1.0).seconds] == [0.0, pytest.approx(0.034)]
    assert frame_buffer.empty()


def _worker(settings: AppSettings, frame_buffer: TimestampFrameBuffer | None = None, on_stopped=lambda: None):
    return RawPlayerWorker(
        "video.mp4",
        METADATA,
        settings,
        TimestampFrameBuffer() if frame_buffer is None else frame_buffer,
        max_size=(640, 360),
        on_stopped=on_stopped,
    )


def test_raw_worker_coalesces_seeks_and_invalidates_buffer() -> None:
    frame_buffer = TimestampFrameBuffer()
    worker = _worker(AppSettings(), frame_buffer)

    first = worker.play_from(1.0)
    second = worker.play_from(4.0)
    command = worker._commands.get_nowait()

    assert second == first + 1
    assert command.seconds == pytest.approx(4.0)
    assert command.generation == second
    assert not frame_buffer.put(_player_frame(4.0, generation=first), threading.Event())
    assert frame_buffer.put(_player_frame(4.0, generation=second), threading.Event())
    worker.close()


def test_raw_worker_reports_stop_after_close() -> None:
    stopped = threading.Event()
    worker = _worker(AppSettings(), on_stopped=stopped.set)
    worker.start()

    worker.close()

    assert stopped.wait(timeout=30)


def test_raw_worker_reload_plays_the_new_settings_from_the_new_position() -> None:
    worker = _worker(AppSettings(secondary_restoration="none"))
    changed = AppSettings(secondary_restoration="rtx-super-res")

    generation = worker.reload_from(changed, 4.0)
    command = worker._commands.get_nowait()

    assert command.settings == changed
    assert command.seconds == pytest.approx(4.0)
    assert command.generation == generation
    worker.close()


class _ClosableRestorer:
    """Primary restorer that only records being closed."""

    device = torch.device("cpu")
    input_dtype = torch.float32

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def player_sessions(monkeypatch):
    """Real CPU sessions in place of the ones ``RawPlayerWorker`` loads, with a fake restorer and
    no detector; returns them, the threads that built them and the devices whose memory was released."""
    from jasna.gui import video_session

    built = {"sessions": [], "threads": [], "released": []}

    def build_session(*_args, **_kwargs):
        built["threads"].append(threading.get_ident())
        built["sessions"].append(
            FixedDetectorSession(
                device=torch.device("cpu"),
                restoration_pipeline=RestorationPipeline(_ClosableRestorer()),
                log_callback=None,
                detector=MagicMock(),
            )
        )
        return built["sessions"][-1]

    monkeypatch.setattr(video_session, "build_video_session", build_session)
    monkeypatch.setattr(video_session, "video_session_config", lambda *_args, **_kwargs: session_config(device="cpu"))
    monkeypatch.setattr(video_session, "release_session_memory", built["released"].append)
    return built


def test_raw_worker_rebuilds_changed_settings_on_same_owner_thread(monkeypatch, player_sessions) -> None:
    initial = AppSettings(secondary_restoration="none")
    changed = AppSettings(secondary_restoration="rtx-super-res")
    stopped = threading.Event()
    worker = _worker(initial, on_stopped=stopped.set)
    pipelines = []
    first_pass_started = threading.Event()
    release_first_pass = threading.Event()
    second_pass_started = threading.Event()

    def run_pass(pipeline, _metadata, _writer, _cancel_event, *, seek_seconds):
        pipelines.append(pipeline)
        if len(pipelines) == 1:
            first_pass_started.set()
            assert release_first_pass.wait(30)
        else:
            second_pass_started.set()

    monkeypatch.setattr(raw_player, "run_raw_restoration_pass", run_pass)

    worker.start()
    worker.play_from(0)
    assert first_pass_started.wait(30)
    worker.reload_from(changed, 3.0)
    release_first_pass.set()
    assert second_pass_started.wait(30)
    worker.close()

    assert stopped.wait(30)
    first, second = player_sessions["sessions"]
    assert len(set(player_sessions["threads"])) == 1
    assert [pipeline.restoration_pipeline for pipeline in pipelines] == [
        first.restoration_pipeline, second.restoration_pipeline
    ]
    assert first.restoration_pipeline.restorer.closed
    assert player_sessions["released"] == [first.device, second.device]


def test_raw_worker_releases_session_when_pipeline_construction_fails(monkeypatch, player_sessions) -> None:
    from jasna import session_factory

    monkeypatch.setattr(session_factory, "build_pipeline", MagicMock(side_effect=RuntimeError("pipeline failed")))
    stopped = threading.Event()
    worker = _worker(AppSettings(), on_stopped=stopped.set)

    worker.start()
    generation = worker.play_from(0)
    assert worker.events.get(timeout=30) == PlayerStatus("loading_models", generation)
    assert worker.events.get(timeout=30) == PlayerFailed("pipeline failed", generation)
    worker.close()

    assert stopped.wait(30)
    (session,) = player_sessions["sessions"]
    assert session.restoration_pipeline.restorer.closed
    assert player_sessions["released"] == [session.device]


class _AsyncSecondary:
    """Satisfies ``AsyncSecondaryRestorer``, as Topaz Video AI does."""

    name = "async"
    num_workers = 1
    has_pending = False

    def push_clip(self, frames_256, *, keep_start, keep_end):
        return 0

    def pop_completed(self):
        return []

    def flush_pending(self, target_seqs=None):
        return False

    def flush_all(self):
        pass

    def close(self):
        pass


def test_raw_pass_refuses_an_async_secondary_restorer() -> None:
    pipeline = make_pipeline(restoration_pipeline=RestorationPipeline(_ClosableRestorer(), secondary_restorer=_AsyncSecondary()))

    with pytest.raises(ValueError, match="Topaz Video AI"):
        run_raw_restoration_pass(pipeline, METADATA, MagicMock(), threading.Event(), seek_seconds=1.0)


@pytest.mark.parametrize(("seek_seconds", "seek_ts"), [(0.0, None), (2.5, 2.5)])
def test_raw_pass_seeks_only_past_the_start(monkeypatch, seek_seconds, seek_ts) -> None:
    from jasna import pipeline_threads

    seeks = []
    monkeypatch.setattr(pipeline_threads, "run_restoration_pass", lambda *_args, seek_ts, **_kwargs: seeks.append(seek_ts))

    run_raw_restoration_pass(make_pipeline(), METADATA, MagicMock(), threading.Event(), seek_seconds=seek_seconds)

    assert seeks == [seek_ts]


@pytest.mark.parametrize("cancelled", [False, True])
def test_raw_pass_raises_a_worker_error_unless_cancelled(monkeypatch, cancelled) -> None:
    from jasna import pipeline_threads

    cancel = threading.Event()
    if cancelled:
        cancel.set()
    monkeypatch.setattr(pipeline_threads, "run_restoration_pass", lambda *_args, **_kwargs: RuntimeError("decode failed"))

    with nullcontext() if cancelled else pytest.raises(RuntimeError, match="decode failed"):
        run_raw_restoration_pass(make_pipeline(), METADATA, MagicMock(), cancel, seek_seconds=1.0)


def test_software_clock_freezes_and_resumes(monkeypatch) -> None:
    now = [10.0]
    monkeypatch.setattr(raw_player.time, "monotonic", lambda: now[0])
    clock = SoftwareClock()
    clock.seek(3.0)
    clock.play()
    now[0] = 11.25

    assert clock.seconds() == pytest.approx(4.25)
    clock.pause()
    now[0] = 20.0
    assert clock.seconds() == pytest.approx(4.25)
    clock.play()
    now[0] = 20.5
    assert clock.seconds() == pytest.approx(4.75)


@pytest.mark.parametrize(("stdout", "expected"), [("1\n", True), ("", False)])
def test_source_has_audio_reads_the_audio_streams_ffprobe_lists(monkeypatch, stdout: str, expected: bool) -> None:
    monkeypatch.setattr(raw_player, "resolve_executable", lambda name: "/tools/ffprobe")
    monkeypatch.setattr(
        raw_player.subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout, "")
    )

    assert raw_player.source_has_audio("video.mp4") is expected


def test_source_has_audio_reports_ffprobe_failure(monkeypatch) -> None:
    monkeypatch.setattr(raw_player, "resolve_executable", lambda name: "/tools/ffprobe")
    monkeypatch.setattr(
        raw_player.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1, "", "invalid media"),
    )

    with pytest.raises(OSError, match="ffprobe audio probe failed: invalid media"):
        raw_player.source_has_audio("broken.mp4")


def _probe(path) -> tuple:
    probed: list[tuple] = []
    done = threading.Event()
    raw_player.probe_video(path, lambda *result: (probed.append(result), done.set()))
    assert done.wait(5)
    return probed[0]


def _fail(error: Exception):
    def raise_error(_path):
        raise error

    return raise_error


def test_probe_video_reports_the_metadata_and_audio_of_a_readable_video(monkeypatch) -> None:
    monkeypatch.setattr(raw_player, "get_video_meta_data", lambda _path: METADATA)
    monkeypatch.setattr(raw_player, "source_has_audio", lambda _path: True)

    assert _probe(Path("video.mp4")) == (METADATA, True, "")


def test_probe_video_passes_on_an_audio_probe_failure_for_playback(monkeypatch) -> None:
    failure = OSError("ffprobe audio probe failed")
    monkeypatch.setattr(raw_player, "get_video_meta_data", lambda _path: METADATA)
    monkeypatch.setattr(raw_player, "source_has_audio", _fail(failure))

    assert _probe(Path("video.mp4")) == (METADATA, failure, "")


def test_probe_video_reports_an_unreadable_video(monkeypatch) -> None:
    monkeypatch.setattr(raw_player, "get_video_meta_data", _fail(ValueError("no video stream")))

    assert _probe(Path("broken.mp4")) == (None, False, "no video stream")


def test_vlc_audio_clock_uses_audio_only_player(monkeypatch) -> None:
    calls: list[tuple] = []
    now = [10.0]
    monkeypatch.setattr(raw_player.time, "monotonic", lambda: now[0])

    class FakeMedia:
        def add_option(self, option):
            calls.append(("option", option))

        def release(self):
            calls.append(("media_release",))

    class FakePlayer:
        def set_media(self, media):
            calls.append(("media", media))

        def play(self):
            calls.append(("play",))
            return 0

        def set_pause(self, paused):
            calls.append(("pause", paused))

        def set_time(self, milliseconds):
            calls.append(("time", milliseconds))

        def get_time(self):
            return 2500

        def is_playing(self):
            return True

        def audio_set_volume(self, volume):
            calls.append(("volume", volume))
            return 0

        def audio_set_mute(self, muted):
            calls.append(("mute", muted))

        def stop(self):
            calls.append(("stop",))

        def release(self):
            calls.append(("player_release",))

    class FakeInstance:
        def __init__(self, *args):
            calls.append(("instance", *args))

        def media_new(self, path):
            calls.append(("path", path))
            return FakeMedia()

        def media_player_new(self):
            return FakePlayer()

        def release(self):
            calls.append(("instance_release",))

    monkeypatch.setattr(raw_player, "_configure_bundled_vlc", lambda: None)
    monkeypatch.setattr(
        raw_player.importlib,
        "import_module",
        lambda name: SimpleNamespace(Instance=FakeInstance),
    )
    clock = VlcAudioClock("video.mp4")

    clock.seek(1.5)
    clock.set_volume(120)
    clock.set_muted(False)
    assert ("volume", 100) not in calls
    assert ("mute", False) not in calls
    clock.play()
    now[0] = 10.25
    assert clock.seconds() == pytest.approx(1.75)
    assert ("volume", 100) in calls
    assert ("mute", False) in calls
    clock.pause()
    now[0] = 20.0
    assert clock.seconds() == pytest.approx(1.75)
    clock.play()
    now[0] = 20.5
    assert clock.seconds() == pytest.approx(2.25)
    clock.set_muted(True)
    clock.close()

    assert ("instance", "--no-video", "--no-spu", "--quiet") in calls
    assert ("option", ":no-video") in calls
    assert calls.count(("time", 1500)) == 1
    assert ("pause", 1) in calls
    assert ("mute", True) in calls
