from __future__ import annotations

import threading
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from av.video.reformatter import ColorRange as AvColorRange
from av.video.reformatter import Colorspace as AvColorspace

from jasna.gui import raw_player
from jasna.gui.raw_player import (
    PlayerFrame,
    RawFrameWriter,
    RawPlayerWorker,
    SoftwareClock,
    TimestampFrameBuffer,
    VlcAudioClock,
    run_raw_restoration_pass,
)
from jasna.gui.models import AppSettings
from jasna.media import VideoMetadata


def _metadata() -> VideoMetadata:
    return VideoMetadata(
        video_file="video.mp4",
        video_height=1080,
        video_width=1920,
        video_fps=60.0,
        average_fps=60.0,
        video_fps_exact=Fraction(60, 1),
        codec_name="h264",
        duration=10.0,
        time_base=Fraction(1, 1000),
        start_pts=100,
        color_space=AvColorspace.ITU709,
        color_range=AvColorRange.MPEG,
        num_frames=600,
        is_10bit=False,
    )


def _player_frame(seconds: float, generation: int = 1) -> PlayerFrame:
    return PlayerFrame(seconds, raw_player.Image.new("RGB", (2, 2)), generation)


def test_timestamp_buffer_caps_frames_and_wakes_cancelled_producer() -> None:
    frame_buffer = TimestampFrameBuffer(max_seconds=20, max_frames=1)
    cancel = threading.Event()
    frame_buffer.reset(1)
    assert frame_buffer.put(_player_frame(0.0), cancel)
    result: list[bool] = []
    producer = threading.Thread(
        target=lambda: result.append(frame_buffer.put(_player_frame(0.1), cancel))
    )

    producer.start()
    cancel.set()
    with frame_buffer._condition:
        frame_buffer._condition.notify_all()
    producer.join(timeout=1)

    assert result == [False]
    assert len(frame_buffer) == 1


def test_timestamp_buffer_caps_buffered_time() -> None:
    frame_buffer = TimestampFrameBuffer(max_seconds=0.1, max_frames=60)
    cancel = threading.Event()
    frame_buffer.reset(1)
    assert frame_buffer.put(_player_frame(0.0), cancel)
    result: list[bool] = []
    producer = threading.Thread(
        target=lambda: result.append(frame_buffer.put(_player_frame(0.1), cancel))
    )

    producer.start()
    cancel.set()
    with frame_buffer._condition:
        frame_buffer._condition.notify_all()
    producer.join(timeout=1)

    assert result == [False]
    assert len(frame_buffer) == 1


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


def test_timestamp_buffer_preroll_accepts_short_eof_tail() -> None:
    frame_buffer = TimestampFrameBuffer()
    frame_buffer.reset(3)
    frame_buffer.put(_player_frame(9.9, generation=3), threading.Event())

    assert not frame_buffer.ready(0.5)
    frame_buffer.mark_eof(3)
    assert frame_buffer.ready(0.5)


def test_default_player_buffer_absorbs_restoration_bursts() -> None:
    frame_buffer = TimestampFrameBuffer()

    assert raw_player.PREROLL_SECONDS >= 1.0
    assert frame_buffer._max_seconds >= raw_player.PREROLL_SECONDS * 2
    assert frame_buffer._max_frames >= raw_player.DISPLAY_FPS * frame_buffer._max_seconds


def test_gpu_frame_image_casts_byte_before_bilinear_resize(monkeypatch) -> None:
    import torch.nn.functional as functional

    dtypes = []
    interpolate = functional.interpolate

    def checked_interpolate(frame, *args, **kwargs):
        dtypes.append(frame.dtype)
        return interpolate(frame, *args, **kwargs)

    monkeypatch.setattr(functional, "interpolate", checked_interpolate)
    frame = torch.arange(3 * 4 * 8, dtype=torch.uint8).reshape(3, 4, 8)

    image = raw_player._gpu_frame_image(frame, (4, 4))

    assert image.size == (4, 2)
    assert dtypes == [torch.float32]


def test_gpu_frame_image_upscales_to_fill_player_surface() -> None:
    frame = torch.full((3, 4, 8), 20, dtype=torch.uint8)

    image = raw_player._gpu_frame_image(frame, (16, 8), exact_size=True)

    assert image.size == (16, 8)


def test_gpu_frame_image_converts_anamorphic_frame_to_display_shape() -> None:
    frame = torch.full((3, 576, 720), 20, dtype=torch.uint8)

    image = raw_player._gpu_frame_image(frame, (768, 576), exact_size=True)

    assert image.size == (768, 576)


def test_raw_writer_downsizes_before_building_image_and_caps_display_rate() -> None:
    frame_buffer = TimestampFrameBuffer()
    frame_buffer.reset(4)
    writer = RawFrameWriter(
        _metadata(),
        frame_buffer,
        threading.Event(),
        4,
        (80, 40),
    )
    frame = torch.full((3, 100, 200), 20, dtype=torch.uint8)

    writer.write(frame, 100)
    writer.write(frame, 116)
    writer.write(frame, 134)

    assert len(frame_buffer) == 2
    assert frame_buffer.peek().image.size == (80, 40)


def test_raw_worker_coalesces_seeks_and_invalidates_buffer() -> None:
    frame_buffer = TimestampFrameBuffer()
    settings = AppSettings()
    worker = RawPlayerWorker(
        "video.mp4",
        _metadata(),
        settings,
        frame_buffer,
        max_size=(640, 360),
    )

    first = worker.play_from(1.0)
    second = worker.play_from(4.0)
    command = worker._commands.get_nowait()

    assert second == first + 1
    assert command.seconds == pytest.approx(4.0)
    assert command.generation == second
    assert command.settings is settings
    assert frame_buffer.generation == second
    worker.close()


def test_raw_worker_reload_reuses_worker_with_new_settings() -> None:
    frame_buffer = TimestampFrameBuffer()
    initial = AppSettings(secondary_restoration="none")
    changed = AppSettings(secondary_restoration="rtx-super-res")
    worker = RawPlayerWorker(
        "video.mp4",
        _metadata(),
        initial,
        frame_buffer,
        max_size=(640, 360),
    )

    generation = worker.reload_from(changed, 4.0)
    command = worker._commands.get_nowait()

    assert not worker._closed.is_set()
    assert worker.settings is changed
    assert command.settings is changed
    assert command.seconds == pytest.approx(4.0)
    assert command.generation == generation
    worker.close()


def test_raw_worker_rebuilds_changed_settings_on_same_owner_thread(
    monkeypatch,
) -> None:
    from jasna.gui import video_session

    initial = AppSettings(secondary_restoration="none")
    changed = AppSettings(secondary_restoration="rtx-super-res")
    worker = RawPlayerWorker(
        "video.mp4",
        _metadata(),
        initial,
        TimestampFrameBuffer(),
        max_size=(640, 360),
    )
    build_threads: list[int] = []
    pipelines: list[SimpleNamespace] = []
    sessions: list[SimpleNamespace] = []
    first_pass_started = threading.Event()
    release_first_pass = threading.Event()
    second_pass_started = threading.Event()

    def build_pipeline(settings):
        build_threads.append(threading.get_ident())
        session = SimpleNamespace(
            device=torch.device("cpu"),
            close=MagicMock(),
        )
        pipeline = SimpleNamespace(close=MagicMock())
        sessions.append(session)
        pipelines.append(pipeline)
        return session, pipeline

    def run_pass(command, _pipeline, _session):
        if command.settings is initial:
            first_pass_started.set()
            assert release_first_pass.wait(2)
        else:
            second_pass_started.set()
        return False

    release_memory = MagicMock()
    monkeypatch.setattr(worker, "_build_pipeline", build_pipeline)
    monkeypatch.setattr(worker, "_run_pass", run_pass)
    monkeypatch.setattr(video_session, "release_session_memory", release_memory)

    worker.start()
    worker.play_from(0)
    assert first_pass_started.wait(2)
    worker.reload_from(changed, 3.0)
    release_first_pass.set()
    assert second_pass_started.wait(2)
    worker.close()
    worker.join(2)

    assert not worker.is_alive()
    assert len(set(build_threads)) == 1
    assert len(pipelines) == 2
    pipelines[0].close.assert_called_once_with()
    sessions[0].close.assert_called_once_with()
    release_memory.assert_any_call(sessions[0].device)


def test_raw_worker_releases_session_when_pipeline_construction_fails(
    monkeypatch,
) -> None:
    from jasna.gui import video_session
    from jasna import session_factory

    session = SimpleNamespace(
        device=torch.device("cpu"),
        close=MagicMock(),
    )
    release = MagicMock()
    monkeypatch.setattr(video_session, "build_video_session", lambda *_args, **_kwargs: session)
    monkeypatch.setattr(video_session, "video_session_config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(video_session, "release_session_memory", release)
    monkeypatch.setattr(
        session_factory,
        "build_pipeline",
        MagicMock(side_effect=RuntimeError("pipeline failed")),
    )
    worker = RawPlayerWorker(
        "video.mp4",
        _metadata(),
        AppSettings(),
        TimestampFrameBuffer(),
        max_size=(640, 360),
    )

    with pytest.raises(RuntimeError, match="pipeline failed"):
        worker._build_pipeline()

    session.close.assert_called_once_with()
    release.assert_called_once_with(session.device)


def test_raw_pass_forwards_seek_and_player_writer(monkeypatch) -> None:
    from jasna import pipeline_threads, vram_offloader

    class ImmediateThread:
        def __init__(self, *, target, **_kwargs):
            self._target = target

        def start(self):
            self._target()

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

    decode = MagicMock()
    blend = MagicMock()
    monkeypatch.setattr(pipeline_threads, "decode_detect_loop", decode)
    monkeypatch.setattr(pipeline_threads, "primary_restore_loop", MagicMock())
    monkeypatch.setattr(pipeline_threads, "secondary_restore_loop", MagicMock())
    monkeypatch.setattr(pipeline_threads, "blend_encode_loop", blend)
    monkeypatch.setattr(raw_player.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(vram_offloader, "VramOffloader", MagicMock())
    pipeline = SimpleNamespace(
        input_video=Path("video.mp4"),
        device=torch.device("cpu"),
        restoration_pipeline=SimpleNamespace(
            secondary_num_workers=1,
            secondary_restorer=None,
        ),
        max_clip_size=30,
        batch_size=2,
        temporal_overlap=4,
        max_detection_gap=2,
        min_detection_duration=2,
        enable_crossfade=True,
        scene_detection=False,
        _vr_projector=None,
        _job_detection_model=MagicMock(),
        _vr_resolution=SimpleNamespace(resolved="off"),
    )
    writer = MagicMock()

    run_raw_restoration_pass(
        pipeline,
        _metadata(),
        writer,
        threading.Event(),
        seek_seconds=2.5,
    )

    assert decode.call_args.kwargs["seek_ts"] == pytest.approx(2.5)
    assert decode.call_args.kwargs["scene_detection"] is False
    assert blend.call_args.kwargs["frame_writer"] is writer


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


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [("1\n", True), ("", False)],
)
def test_source_has_audio_uses_ffprobe_process(
    monkeypatch,
    stdout: str,
    expected: bool,
) -> None:
    calls: list[tuple[list[str], dict]] = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(raw_player, "resolve_executable", lambda name: "/tools/ffprobe")
    monkeypatch.setattr(raw_player.subprocess, "run", run)

    assert raw_player.source_has_audio("video.mp4") is expected
    assert calls == [
        (
            [
                "/tools/ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                "video.mp4",
            ],
            {
                "stdout": raw_player.subprocess.PIPE,
                "stderr": raw_player.subprocess.PIPE,
                "text": True,
            },
        )
    ]


def test_source_has_audio_reports_ffprobe_failure(monkeypatch) -> None:
    monkeypatch.setattr(raw_player, "resolve_executable", lambda name: "/tools/ffprobe")
    monkeypatch.setattr(
        raw_player.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="invalid media",
        ),
    )

    with pytest.raises(OSError, match="ffprobe audio probe failed: invalid media"):
        raw_player.source_has_audio("broken.mp4")


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
