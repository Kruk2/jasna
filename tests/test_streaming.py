from __future__ import annotations

import io
import json
import re
import subprocess
import threading
import urllib.error
import urllib.request
from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch

import av
import pytest
import torch

from factories import video_metadata
from jasna.os_utils import resolve_executable
from jasna.streaming import SEGMENT_WAIT_SECONDS, HlsStreamingServer, SegmentTimeline, _generate_vod_playlist
from jasna.streaming_encoder import StreamingEncodeError, StreamingEncoder, _browser_audio_args
from jasna.streaming_pipeline import _StreamingFrameWriter

pytestmark = pytest.mark.usefixtures("nvidia_vendor")


def _metadata(duration: float = 60.0, **overrides):
    return video_metadata(fps=30, num_frames=int(duration * 30), duration=duration, **overrides)


def _finalize_segments(segments_dir: Path, *numbers: int) -> None:
    names = [f"seg_{n:05d}.ts" for n in numbers]
    for name in names:
        (segments_dir / name).write_bytes(b"complete")
    # Renamed into place like ffmpeg does, so a waiting request never reads a half-written playlist.
    staged = segments_dir / "_hls_internal.m3u8.tmp"
    staged.write_text("".join(f"#EXTINF:4.000,\n{name}\n" for name in names))
    staged.replace(segments_dir / "_hls_internal.m3u8")


def _server(tmp_path=None, segment_duration: float = 4.0, **kwargs) -> HlsStreamingServer:
    server = HlsStreamingServer(segment_duration=segment_duration, port=0, **kwargs)
    server.segments_dir = tmp_path
    return server


# ---------------------------------------------------------------------------
# Playlist and server state
# ---------------------------------------------------------------------------

def _playlist_durations(fps, frame_count: int, segment_duration: float = 4.0) -> list[float]:
    timeline = SegmentTimeline.for_video(video_metadata(fps=fps, num_frames=frame_count), segment_duration)
    text, count = _generate_vod_playlist(timeline)
    durations = [float(d) for d in re.findall(r"#EXTINF:([\d.]+),\nseg_\d+\.ts", text)]
    assert count == len(durations)
    return durations


@pytest.mark.parametrize(
    ("fps", "frame_count", "count", "first", "last"),
    [
        (Fraction(30000, 1001), 12000, 100, 4.004, 4.004),
        (Fraction(24000, 1001), 9600, 100, 4.004, 4.004),
        (Fraction(60000, 1001), 24000, 100, 4.004, 4.004),
        (Fraction(25), 10050, 101, 4.0, 2.0),
        (Fraction(30), 12030, 101, 4.0, 1.0),
    ],
)
def test_playlist_lists_the_whole_gop_segments_the_encoder_cuts(fps, frame_count, count, first, last):
    durations = _playlist_durations(fps, frame_count)

    assert (len(durations), durations[0], durations[-1]) == (count, first, last)


def test_vod_playlist_lists_every_segment_with_a_short_last_one():
    timeline = SegmentTimeline.for_video(video_metadata(fps=30, num_frames=300), 4.0)
    text, _ = _generate_vod_playlist(timeline)

    assert text.startswith("#EXTM3U")
    assert "#EXT-X-PLAYLIST-TYPE:VOD" in text and text.rstrip().endswith("#EXT-X-ENDLIST")
    assert re.findall(r"#EXTINF:([\d.]+),\n(seg_\d+\.ts)", text) == [
        ("4.000", "seg_00000.ts"), ("4.000", "seg_00001.ts"), ("2.000", "seg_00002.ts"),
    ]


@pytest.mark.parametrize(
    "fps", [Fraction(30000, 1001), Fraction(24000, 1001), Fraction(60000, 1001), Fraction(25), Fraction(30)]
)
@pytest.mark.parametrize("segment", [0, 1, 37, 99])
def test_a_segment_start_time_maps_back_to_that_segment(fps, segment):
    timeline = SegmentTimeline.for_video(video_metadata(fps=fps, num_frames=100_000), 4.0)
    start = timeline.start_time(segment)

    assert timeline.segment_at(start) == segment
    assert timeline.segment_at(start + timeline.duration(segment) - 0.01) == segment


def test_loading_and_unloading_a_video():
    server = _server()
    assert not server.is_loaded

    server.load_video(_metadata(duration=20.0))
    segments_dir = server.segments_dir
    assert server.is_loaded and server.segment_count == 5 and segments_dir.is_dir()

    server.unload_video()
    assert not server.is_loaded and server.segment_count == 0

    server.stop()
    assert not segments_dir.exists()


def test_playlist_being_replaced_reads_as_not_ready(tmp_path, monkeypatch):
    server = _server(tmp_path, segment_duration=2.0)
    _finalize_segments(tmp_path, 0)

    def _replaced_by_ffmpeg(path, *args, **kwargs):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(Path, "read_text", _replaced_by_ffmpeg)
    assert server.segment_is_ready(tmp_path / "seg_00000.ts") is False


def test_seek_request_is_consumed_once_and_the_last_one_wins():
    server = _server()
    assert server.consume_seek() is None

    for segment in (1, 3, 2):
        server.request_seek(segment)
    assert server.seek_requested.is_set()

    assert server.consume_seek() == 2
    assert not server.seek_requested.is_set()
    assert server.consume_seek() is None


def test_seek_to_active_pass_is_ignored():
    server = _server()
    server.request_seek(5)
    assert server.consume_seek_for_pass(5) is None
    assert not server.seek_requested.is_set()


def test_evicted_pass_start_is_not_ignored(tmp_path):
    server = _server(tmp_path)
    server.reset_demand(0)
    (tmp_path / "seg_00000.ts").write_bytes(b"data")
    server.update_production(18)
    server.notify_segment_requested(18)
    server.request_seek(0)

    assert server.consume_seek_for_pass(0) == 0


def test_stale_playback_cannot_request_seek():
    server = _server()
    server.request_seek(10, new_playback=True)
    assert not server.request_seek(1, expected_epoch=0)
    assert server.seek_target_segment == 10


def test_selected_video_is_handed_to_the_waiting_loop():
    server = _server()
    server.select_video(Path("test.mp4"), 0.0)

    assert server.wait_for_video() == Path("test.mp4")
    assert not server.video_change.is_set()


@pytest.mark.parametrize("stop", ["select-another", "stop-current"])
def test_stopping_a_loaded_video_signals_a_video_change(stop):
    server = _server()
    server.load_video(_metadata(duration=20.0))

    if stop == "select-another":
        server.select_video(Path("other.mp4"), 0.0)
    else:
        server.stop_current()

    assert server.video_change.is_set()
    assert not server.is_loaded
    server.stop()


def test_load_video_resets_seek_state():
    server = _server()
    server.request_seek(5)
    server.video_change.set()

    server.load_video(_metadata(duration=20.0))

    assert not server.seek_requested.is_set()
    assert not server.video_change.is_set()
    server.stop()


def test_first_segment_is_ready_once_the_loaded_video_produces_one():
    server = _server()
    assert not server.wait_until_first_segment(timeout=0)
    server.load_video(_metadata(duration=20.0))
    assert not server.wait_until_first_segment(timeout=0)

    server.update_production(0)
    assert server.wait_until_first_segment(timeout=0)

    server.select_video(Path("new.mp4"), 0.0)
    assert not server.wait_until_first_segment(timeout=0)
    server.stop()


def test_outgoing_pass_does_not_mark_the_next_video_ready():
    server = _server()
    server.load_video(_metadata(duration=20.0))
    server.update_production(3)
    server.select_video(Path("next.mp4"), 0.0)
    server.update_production(4)
    assert not server.wait_until_first_segment(timeout=0)
    server.load_video(_metadata(duration=20.0))
    server.update_production(0)
    assert server.wait_until_first_segment(timeout=0)
    server.stop()


# ---------------------------------------------------------------------------
# Demand flow control
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("start_segment", "requested", "finalized", "writer_at", "ahead"),
    [
        (0, None, (), 50, False),
        (0, 0, (0,), 2, False),
        (0, 0, (0,), 5, True),
        (0, 3, (0, 3), 5, False),
        (0, 134, (130, 131, 132, 133), 138, False),
        (100, None, (), 102, False),
    ],
    ids=[
        "no-player-yet", "within-buffer", "too-far-ahead", "player-caught-up",
        "requested-segment-not-finalized", "fresh-pass-start",
    ],
)
def test_writer_pauses_only_when_far_ahead_of_a_finalized_player_request(
    tmp_path, start_segment, requested, finalized, writer_at, ahead
):
    server = _server(tmp_path, segment_duration=2.0)
    server.reset_demand(start_segment)
    _finalize_segments(tmp_path, *finalized)
    if requested is not None:
        (tmp_path / f"seg_{requested:05d}.ts").touch()
        server.notify_segment_requested(requested)

    assert server._writer_is_ahead_of_player(writer_at, 3) is ahead


@pytest.mark.parametrize("release", ["player-request", "cancel"])
def test_a_paused_writer_resumes_on_a_player_request_or_cancel(tmp_path, release):
    server = _server(tmp_path)
    _finalize_segments(tmp_path, 0, 3)
    server.notify_segment_requested(0)
    cancel = threading.Event()
    done = threading.Event()

    def writer():
        server.wait_for_demand(current_segment=5, max_ahead=3, cancel_event=cancel)
        done.set()

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    if release == "cancel":
        cancel.set()
    else:
        server.notify_segment_requested(3)

    assert done.wait(timeout=2.0)
    thread.join(timeout=2.0)


def test_forward_seek_triggered_only_when_far_ahead():
    server = _server()
    server.reset_demand(start_segment=0)
    server.update_production(3)
    assert not server.needs_seek(5)
    assert server.needs_seek(50)

    server.update_production(10)
    assert not server.needs_seek(14)


def test_backward_seek_always_triggers():
    server = _server()
    server.reset_demand(start_segment=50)
    server.update_production(55)
    assert server.needs_seek(10)


def test_evicted_segment_triggers_seek(tmp_path):
    server = _server(tmp_path)
    server.reset_demand()
    server.update_production(18)
    (tmp_path / "seg_00008.ts").write_bytes(b"data")
    server.notify_segment_requested(18)
    assert server.needs_seek(8)


def test_late_request_does_not_restart_segments_not_yet_written(tmp_path):
    server = _server(tmp_path)
    server.reset_demand(10)
    server.update_production(11)
    (tmp_path / "seg_00010.ts").write_bytes(b"data")
    server.notify_segment_requested(20)
    assert server.needs_seek(10)
    assert not server.needs_seek(12)


@pytest.mark.parametrize(("written", "player_at", "kept"), [(15, 12, range(6, 15)), (5, 3, range(5))])
def test_old_segments_are_evicted_behind_the_player(tmp_path, written, player_at, kept):
    server = _server(tmp_path, max_segments_ahead=3)
    for index in range(written):
        (tmp_path / f"seg_{index:05d}.ts").write_bytes(b"data")

    server.notify_segment_requested(player_at)

    assert sorted(path.name for path in tmp_path.glob("seg_*.ts")) == [f"seg_{index:05d}.ts" for index in kept]


# ---------------------------------------------------------------------------
# Streaming encoder
# ---------------------------------------------------------------------------

def _ffmpeg_process():
    process = MagicMock()
    process.stdin.closed = False
    process.stderr = io.BytesIO(b"")
    process.returncode = 0
    process.wait.return_value = 0
    return process


def _encoder(
    tmp_path, *, segment_duration=4.0, source_video="nonexistent.mp4", device=None, lut_path=None, **metadata
) -> StreamingEncoder:
    return StreamingEncoder(
        segments_dir=tmp_path,
        segment_duration=segment_duration,
        metadata=_metadata(duration=20.0, **metadata),
        source_video=source_video,
        device=device,
        lut_path=lut_path,
    )


def _ffmpeg_command(tmp_path, *, start_number=0, audio_codec=None, **encoder_options) -> list[str]:
    """The ffmpeg command an encoder starts with; ``audio_codec`` gives the source an audio track."""
    if audio_codec is not None:
        encoder_options["source_video"] = "existing.mp4"
    with (
        patch("jasna.streaming_encoder.subprocess.Popen", return_value=_ffmpeg_process()) as popen,
        patch("jasna.streaming_encoder.os.path.isfile", return_value=audio_codec is not None),
        patch("jasna.streaming_encoder.source_audio_codec", return_value=audio_codec or None),
    ):
        encoder = _encoder(tmp_path, **encoder_options)
        encoder.start(start_number=start_number)
        encoder.stop()
    return popen.call_args.args[0]


def _option(command: list[str], name: str) -> str | None:
    return command[command.index(name) + 1] if name in command else None


def test_ffmpeg_encodes_with_nvenc_on_the_chosen_gpu_with_keyframes_only_on_gop_boundaries(tmp_path):
    command = _ffmpeg_command(tmp_path, device=torch.device("cuda:2"))

    assert (_option(command, "-c:v"), _option(command, "-gpu"), _option(command, "-no-scenecut")) == ("h264_nvenc", "2", "1")
    assert _option(_ffmpeg_command(tmp_path, device=torch.device("cuda")), "-gpu") == "0"


@pytest.mark.parametrize(("sar", "filter"), [(Fraction(8, 9), "setsar=8/9"), (Fraction(1, 1), None)])
def test_ffmpeg_keeps_anamorphic_pixels(tmp_path, sar, filter):
    assert _option(_ffmpeg_command(tmp_path, sample_aspect_ratio=sar), "-vf") == filter


@pytest.mark.parametrize(("start_number", "offset"), [(0, None), (5, "20.000")])
def test_ffmpeg_numbers_and_timestamps_segments_from_the_start_segment(tmp_path, start_number, offset):
    command = _ffmpeg_command(tmp_path, start_number=start_number)

    assert _option(command, "-start_number") == str(start_number)
    assert _option(command, "-output_ts_offset") == offset


@pytest.mark.parametrize(
    ("source_codec", "expected_audio_args"),
    [
        ("aac", ["-c:a:0", "copy"]),
        ("mp3", ["-c:a:0", "copy"]),
        ("ac3", ["-c:a:0", "aac", "-b:a:0", "256k"]),
        ("eac3", ["-c:a:0", "aac", "-b:a:0", "256k"]),
        ("flac", ["-c:a:0", "aac", "-b:a:0", "256k"]),
        ("pcm_s16le", ["-c:a:0", "aac", "-b:a:0", "256k"]),
    ],
)
def test_audio_copied_only_when_browsers_play_it(tmp_path, source_codec, expected_audio_args):
    command = _ffmpeg_command(tmp_path, start_number=3, audio_codec=source_codec)

    audio_start = command.index(expected_audio_args[0])
    assert command[audio_start:audio_start + len(expected_audio_args)] == expected_audio_args
    assert _option(command, "-i") == "existing.mp4"
    assert float(_option(command, "-ss")) == pytest.approx(12.0)


@pytest.mark.parametrize("audio_codec", [None, ""], ids=["no-source", "source-without-audio"])
def test_no_audio_args_without_source_audio(tmp_path, audio_codec):
    assert not any(arg.startswith("-c:a") for arg in _ffmpeg_command(tmp_path, audio_codec=audio_codec))


@pytest.mark.parametrize(
    ("fps", "duration", "frames"),
    [
        (Fraction(30000, 1001), 2.0, 60),
        (Fraction(30000, 1001), 4.0, 120),
        (Fraction(24000, 1001), 2.0, 48),
        (Fraction(25), 2.0, 50),
        (Fraction(121, 5), 2.0, 49),
    ],
)
def test_every_segment_is_one_gop(tmp_path, fps, duration, frames):
    command = _ffmpeg_command(tmp_path, segment_duration=duration, video_fps_exact=fps)
    assert _option(command, "-g") == str(frames)
    assert _encoder(tmp_path, segment_duration=duration, video_fps_exact=fps).frames_per_segment == frames


@pytest.mark.parametrize("fps", [Fraction(30000, 1001), Fraction(24000, 1001), Fraction(25)])
def test_a_pass_starts_where_the_playlist_puts_its_first_segment(tmp_path, fps):
    command = _ffmpeg_command(tmp_path, start_number=10, audio_codec="aac", video_fps_exact=fps)
    listed_start = sum(_playlist_durations(fps, 100_000)[:10])

    assert float(_option(command, "-ss")) == pytest.approx(listed_start, abs=1e-3)
    assert float(_option(command, "-output_ts_offset")) == pytest.approx(listed_start, abs=1e-3)


def test_writer_counts_segments_like_the_muxer(tmp_path):
    encoder = _encoder(tmp_path, segment_duration=2.0, video_fps_exact=Fraction(30000, 1001))
    server = MagicMock()

    _StreamingFrameWriter(encoder, server, start_segment=20).after_write(590, threading.Event())

    server.update_production.assert_called_once_with(29)


@pytest.mark.parametrize("layout", ["chw", "hwc"])
def test_frames_reach_ffmpeg_as_packed_rgb(tmp_path, layout):
    process = _ffmpeg_process()
    frame = torch.arange(3 * 4 * 4, dtype=torch.uint8).reshape(3, 4, 4)
    with patch("jasna.streaming_encoder.subprocess.Popen", return_value=process):
        encoder = _encoder(tmp_path)
        encoder.start(start_number=0)
        encoder.write_frame(frame if layout == "chw" else frame.permute(1, 2, 0), pts=0)
        encoder.stop()

    process.stdin.write.assert_called_once_with(frame.permute(1, 2, 0).contiguous().numpy().tobytes())


def test_color_lut_is_applied_to_streamed_frames(tmp_path):
    lut = tmp_path / "invert.cube"
    lut.write_text("LUT_1D_SIZE 2\n1 1 1\n0 0 0\n")
    process = _ffmpeg_process()
    frame = torch.tensor([0, 255], dtype=torch.uint8).repeat(3, 1, 1)
    with patch("jasna.streaming_encoder.subprocess.Popen", return_value=process):
        encoder = _encoder(tmp_path, lut_path=str(lut))
        encoder.start(start_number=0)
        encoder.write_frame(frame, pts=0)
        encoder.stop()

    process.stdin.write.assert_called_once_with(bytes([255, 255, 255, 0, 0, 0]))


def test_write_before_start_is_dropped(tmp_path):
    with patch("jasna.streaming_encoder.subprocess.Popen") as popen:
        _encoder(tmp_path).write_frame(torch.zeros(3, 4, 4, dtype=torch.uint8), pts=0)
    popen.assert_not_called()


def test_start_rejects_active_encoder(tmp_path):
    with patch("jasna.streaming_encoder.subprocess.Popen", return_value=_ffmpeg_process()):
        encoder = _encoder(tmp_path)
        encoder.start(start_number=0)
        with pytest.raises(RuntimeError, match="already started"):
            encoder.start(start_number=1)
        encoder.stop()


@pytest.mark.parametrize(
    "failure",
    [None, "stdin-close-error", "exit-timeout", "nonzero-exit"],
)
def test_a_stopped_encoder_can_start_again(tmp_path, failure):
    process = _ffmpeg_process()
    if failure == "stdin-close-error":
        process.stdin.close.side_effect = OSError("broken")
    elif failure == "exit-timeout":
        timeouts = [subprocess.TimeoutExpired("ffmpeg", 10)]

        def wait(timeout=None):
            if timeouts:
                raise timeouts.pop()
            return 0

        process.wait.side_effect = wait
    elif failure == "nonzero-exit":
        process.returncode = 1
    with patch("jasna.streaming_encoder.subprocess.Popen", return_value=process) as popen:
        encoder = _encoder(tmp_path)
        encoder.start(start_number=0)
        encoder.stop()
        encoder.start(start_number=1)
        encoder.stop()

    assert popen.call_count == 2
    assert process.kill.called is (failure == "exit-timeout")


def test_flush_and_restart_drops_old_segments_and_restarts_at_the_seek_target(tmp_path):
    for name in ("seg_00000.ts", "seg_00001.ts", "_hls_internal.m3u8", "other.txt"):
        (tmp_path / name).write_text("data")
    events = []
    old = _ffmpeg_process()
    old.kill.side_effect = lambda: events.append("kill")
    old.wait.side_effect = lambda: events.append("wait")
    old.stdin.close.side_effect = lambda: events.append("close")
    with patch("jasna.streaming_encoder.subprocess.Popen", side_effect=[old, _ffmpeg_process()]) as popen:
        encoder = _encoder(tmp_path)
        encoder.start(start_number=0)
        encoder.flush_and_restart(start_number=5)
        encoder.stop()

    assert _option(popen.call_args.args[0], "-start_number") == "5"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["other.txt"]
    # ffmpeg blocked on a full pipe never sees its stdin close, so it is killed first.
    assert events == ["kill", "wait", "close"]


def test_restart_waits_for_writer_before_closing_pipe(monkeypatch, tmp_path):
    class RacingStdin:
        closed = False
        write_entered = threading.Event()
        release_write = threading.Event()

        def write(self, _item):
            self.write_entered.set()
            self.release_write.wait(timeout=2.0)
            if self.closed:
                raise ValueError("write to closed file")

        def close(self):
            self.closed = True

    old, new = _ffmpeg_process(), _ffmpeg_process()
    old.stdin = RacingStdin()
    uncaught = []
    monkeypatch.setattr(threading, "excepthook", lambda args: uncaught.append(args.exc_value))

    with patch("jasna.streaming_encoder.subprocess.Popen", side_effect=[old, new]):
        encoder = _encoder(tmp_path)
        encoder.start(start_number=0)
        encoder.write_frame(torch.zeros(3, 4, 4, dtype=torch.uint8), pts=0)
        assert old.stdin.write_entered.wait(timeout=2.0)
        threading.Timer(0.05, old.stdin.release_write.set).start()
        encoder.flush_and_restart(start_number=5)
        encoder.write_frame(torch.zeros(3, 4, 4, dtype=torch.uint8), pts=1)
        encoder.stop()

    assert uncaught == []
    assert not encoder.failed
    new.stdin.write.assert_called_once()


def _flood(encoder) -> list[BaseException]:
    """Write more frames than the queue holds from another thread; returns what it raised."""
    raised = []

    def write():
        try:
            for pts in range(40):
                encoder.write_frame(torch.zeros(3, 4, 4, dtype=torch.uint8), pts=pts)
        except StreamingEncodeError as error:
            raised.append(error)

    thread = threading.Thread(target=write, daemon=True)
    thread.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "write_frame blocked after the writer died"
    return raised


@pytest.mark.parametrize("pipe_error", [BrokenPipeError(), ValueError("write to closed file")])
def test_a_dead_ffmpeg_pipe_fails_the_writes_instead_of_blocking(monkeypatch, tmp_path, pipe_error):
    process = _ffmpeg_process()
    process.stdin.write.side_effect = pipe_error
    uncaught = []
    monkeypatch.setattr(threading, "excepthook", lambda args: uncaught.append(args.exc_value))

    with patch("jasna.streaming_encoder.subprocess.Popen", return_value=process):
        encoder = _encoder(tmp_path)
        encoder.start(start_number=0)
        raised = _flood(encoder)
        encoder.stop()

    assert [str(error) for error in raised] == ["Streaming writer failed"]
    assert encoder.failed
    assert uncaught == []


def test_restart_clears_a_writer_failure(tmp_path):
    process = _ffmpeg_process()
    process.stdin.write.side_effect = BrokenPipeError()
    with patch("jasna.streaming_encoder.subprocess.Popen", return_value=process):
        encoder = _encoder(tmp_path)
        encoder.start(start_number=0)
        assert _flood(encoder)

        process.stdin.write.side_effect = None
        encoder.flush_and_restart(start_number=1)
        assert not encoder.failed
        encoder.write_frame(torch.zeros(3, 4, 4, dtype=torch.uint8), pts=0)
        encoder.stop()

    assert not encoder.failed


def test_reencoded_surround_audio_plays_from_any_segment(tmp_path):
    ffmpeg = resolve_executable("ffmpeg")
    source = tmp_path / "source.mka"
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "sine=d=7,aformat=channel_layouts=5.1(side)", "-c:a", "ac3", str(source)],
        check=True,
    )
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(source), "-map", "0:a:0",
         *_browser_audio_args(str(source)),
         "-f", "hls", "-hls_time", "2", "-hls_segment_type", "mpegts",
         "-hls_segment_filename", str(tmp_path / "seg_%05d.ts"), "-hls_list_size", "0",
         str(tmp_path / "_hls_internal.m3u8")],
        check=True,
    )
    with av.open(str(tmp_path / "seg_00002.ts")) as segment:
        frame = next(segment.decode(audio=0))
    assert frame.layout.name == "5.1"


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

def _http(port: int, path: str, method: str = "GET", body: dict | None = None) -> tuple[int, dict | str, dict]:
    """Status, body (decoded JSON when it is JSON) and lowercase headers of one request.

    The server listens on IPv4 only; on Windows a "localhost" client first tries ::1 for ~2 s.
    """
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", method=method, data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        response = urllib.request.urlopen(request, timeout=SEGMENT_WAIT_SECONDS + 5)
        status = response.status
    except urllib.error.HTTPError as error:
        response, status = error, error.code
    with response:
        raw = response.read().decode()
        headers = {key.lower(): value for key, value in response.headers.items()}
    try:
        return status, json.loads(raw), headers
    except json.JSONDecodeError:
        return status, raw, headers


@pytest.fixture
def serve():
    """Start an HLS server on a free port: ``serve(video=..., metadata=...)`` -> (server, port)."""
    started = []

    def start(*, segment_duration=4.0, video: Path | None = None, metadata=None):
        server = HlsStreamingServer(segment_duration=segment_duration, port=0)
        if video is not None:
            server.select_video(video, 0.0)
        if metadata is not None:
            server.load_video(metadata)
        server.start()
        started.append(server)
        return server, server._httpd.server_address[1]

    yield start
    for server in started:
        server.stop()


def _on_segment_request(monkeypatch, server, segment: int) -> threading.Event:
    """An event set once the player has asked for ``segment``."""
    requested = threading.Event()
    notify = server.notify_segment_requested

    def record(index):
        notify(index)
        if index == segment:
            requested.set()

    monkeypatch.setattr(server, "notify_segment_requested", record)
    return requested


def _get_in_background(port: int, path: str) -> tuple[threading.Thread, list]:
    result = []
    thread = threading.Thread(target=lambda: result.append(_http(port, path)), daemon=True)
    thread.start()
    return thread, result


@pytest.fixture
def video_file(tmp_path) -> Path:
    path = tmp_path / "test.mp4"
    path.write_bytes(b"fake")
    return path


@pytest.mark.parametrize("path", ["/", "/status"])
def test_responses_allow_any_origin(serve, path):
    _, port = serve()
    status, _, headers = _http(port, path)

    assert status == 200
    assert headers["access-control-allow-origin"] == "*"
    assert "GET" in headers["access-control-allow-methods"]


def test_options_preflight(serve):
    _, port = serve()
    status, _, headers = _http(port, "/open", "OPTIONS")

    assert status == 204
    assert headers["access-control-allow-origin"] == "*"
    assert "Content-Type" in headers["access-control-allow-headers"]


@pytest.mark.parametrize(
    ("query", "fps", "first_segment"),
    [
        ("?start=40.0", Fraction(30), 10),
        ("", Fraction(30), 0),
        ("?start=0", Fraction(30), 0),
        ("?start=40.04", Fraction(30000, 1001), 10),
        ("?start=40.0", Fraction(30000, 1001), 9),
    ],
)
def test_manifest_starts_at_the_requested_time(serve, query, fps, first_segment):
    _, port = serve(metadata=video_metadata(fps=fps, num_frames=3600))

    status, body, _ = _http(port, f"/stream.m3u8{query}")

    assert status == 200
    assert f"#EXT-X-MEDIA-SEQUENCE:{first_segment}" in body
    assert re.findall(r"seg_\d+\.ts\?epoch=0", body)[0] == f"seg_{first_segment:05d}.ts?epoch=0"


def test_old_playback_cannot_restart_new_seek(serve, video_file):
    server, port = serve(video=video_file, metadata=_metadata(duration=120.0))

    assert "seg_00000.ts?epoch=1" in _http(port, "/stream.m3u8")[1]
    assert _http(port, "/open", "POST", {"path": str(video_file), "start": 40.0})[0] == 200
    assert "seg_00010.ts?epoch=2" in _http(port, "/stream.m3u8?start=40")[1]
    assert server.consume_seek() == 10
    assert _http(port, "/seg_00001.ts?epoch=1")[0] == 404
    assert not server.seek_requested.is_set()


def test_segment_without_epoch_is_served_for_proxies(serve):
    server, port = serve(metadata=_metadata(duration=20.0))
    server.reset_demand()
    _finalize_segments(server.segments_dir, 0)

    assert _http(port, "/seg_00000.ts")[0:2] == (200, "complete")


def test_segment_without_epoch_never_restarts_open_seek(serve):
    server, port = serve(metadata=_metadata(duration=200.0))
    server.request_seek(10, new_playback=True)
    server.consume_seek()
    server.reset_demand(10)
    server.update_production(10)
    _finalize_segments(server.segments_dir, 10)

    assert _http(port, "/seg_00002.ts")[0] == 404
    assert _http(port, "/seg_00040.ts")[0] == 404
    assert not server.seek_requested.is_set()
    assert _http(port, "/seg_00010.ts")[0:2] == (200, "complete")


def test_evicted_segment_restarts_and_serves(serve):
    server, port = serve(metadata=_metadata(duration=120.0))
    server.reset_demand()
    server.update_production(18)
    (server.segments_dir / "seg_00008.ts").write_bytes(b"old")
    server.notify_segment_requested(18)

    thread, result = _get_in_background(port, "/seg_00008.ts?epoch=0")
    assert server.seek_requested.wait(timeout=2)
    assert server.seek_target_segment == 8
    server.reset_demand(8)
    _finalize_segments(server.segments_dir, 8)
    thread.join(timeout=5)

    assert result[0][0:2] == (200, "complete")


def test_wait_ends_when_pass_moves_away(serve, monkeypatch):
    server, port = serve(metadata=_metadata(duration=800.0))
    server.reset_demand(10)
    server.update_production(10)
    requested = _on_segment_request(monkeypatch, server, 11)

    thread, result = _get_in_background(port, "/seg_00011.ts?epoch=0")
    assert requested.wait(timeout=5)
    server.reset_demand(100)
    thread.join(timeout=5)

    assert [status for status, _, _ in result] == [404]


@pytest.mark.parametrize("pass_finishes", [False, True], ids=["finalized", "pass-finished"])
def test_segment_is_served_once_the_playlist_lists_it(serve, monkeypatch, pass_finishes):
    server, port = serve(metadata=_metadata(duration=20.0))
    server.reset_demand()
    server.update_production(4)
    segment = server.segments_dir / "seg_00004.ts"
    segment.write_bytes(b"partial")
    requested = _on_segment_request(monkeypatch, server, 4)

    thread, result = _get_in_background(port, "/seg_00004.ts?epoch=0")
    assert requested.wait(timeout=5)
    _finalize_segments(server.segments_dir, 4)
    if pass_finishes:
        server.mark_finished()
    thread.join(timeout=5)

    assert result[0][0:2] == (200, "complete")
    assert result[0][2]["content-length"] == "8"


def test_open_seek_to_a_nearby_segment_serves_it_from_the_new_pass(serve, monkeypatch, video_file):
    server, port = serve(segment_duration=2.0, video=video_file, metadata=_metadata(duration=200.0))
    server.reset_demand(20)
    server.update_production(22)
    requested = _on_segment_request(monkeypatch, server, 25)

    assert _http(port, "/open", "POST", {"path": str(video_file), "start": 50.001})[0] == 200
    thread, result = _get_in_background(port, "/seg_00025.ts?epoch=2")
    assert requested.wait(timeout=5)
    assert server.consume_seek_for_pass(20) == 25
    server.reset_demand(25)
    server.notify_segment_requested(25)
    _finalize_segments(server.segments_dir, 25)
    thread.join(timeout=5)

    assert result[0][0:2] == (200, "complete")


def test_player_waits_longer_for_first_byte_than_server_long_poll(serve):
    _, port = serve()
    page = _http(port, "/")[1]

    assert int(re.search(r"maxTimeToFirstByteMs:(\d+)", page).group(1)) > SEGMENT_WAIT_SECONDS * 1000


def test_status_reports_the_streamed_video(serve):
    server, port = serve()
    assert _http(port, "/status")[0:2] == (200, {"streaming": False})

    server.select_video(Path("/some/video.mp4"), 0.0)
    server.load_video(_metadata(duration=20.0))

    assert _http(port, "/status")[1] == {"streaming": True, "path": str(Path("/some/video.mp4"))}


def test_delete_open_stops_the_stream(serve):
    server, port = serve(metadata=_metadata(duration=20.0))

    assert _http(port, "/open", "DELETE")[0:2] == (200, {"status": "stopped"})
    assert not server.is_loaded


@pytest.mark.parametrize("path", ["/stop", "/api/stop", "/api/load"])
def test_legacy_aliases_are_gone(serve, path):
    _, port = serve()
    assert _http(port, path, "POST")[0] == 404


@pytest.mark.parametrize(
    ("body", "error"),
    [({"start": "nan"}, "Invalid start time"), ({"path": "/nonexistent/file.mp4"}, None)],
    ids=["non-finite-start", "missing-file"],
)
def test_open_rejects_bad_requests(serve, video_file, body, error):
    _, port = serve()
    status, response, _ = _http(port, "/open", "POST", {"path": str(video_file), **body})

    assert status == 400
    assert response["error"] == error if error else "error" in response


@pytest.mark.parametrize(("first_segment", "status"), [(True, 200), (False, 500)])
def test_opening_a_new_video_waits_for_its_first_segment(serve, video_file, first_segment, status):
    server, port = serve()
    with patch.object(server, "wait_until_first_segment", return_value=first_segment):
        response_status, body, _ = _http(port, "/open", "POST", {"path": str(video_file), "start": 40.0})

    assert response_status == status
    if first_segment:
        assert body["status"] == "ready"
        server.load_video(video_metadata(fps=Fraction(30000, 1001), num_frames=3600))
        assert server.initial_start_segment == 9
    else:
        assert "error" in body


def test_open_blocks_until_the_first_segment_is_produced(serve, video_file, monkeypatch):
    server, port = serve()
    waiting = threading.Event()
    wait = server.wait_until_first_segment

    def wait_with_signal(timeout=30.0):
        waiting.set()
        return wait(timeout=timeout)

    monkeypatch.setattr(server, "wait_until_first_segment", wait_with_signal)
    result = []
    thread = threading.Thread(target=lambda: result.append(_http(port, "/open", "POST", {"path": str(video_file)})), daemon=True)
    thread.start()
    assert waiting.wait(timeout=5)
    assert thread.is_alive()

    server.load_video(_metadata(duration=20.0))
    server.update_production(0)
    thread.join(timeout=10)

    assert result[0][0:2] == (200, {"status": "ready"})


@pytest.mark.parametrize(("start", "seek_to"), [(40.0, 10), (None, None)])
def test_opening_the_playing_video_again_seeks_without_reloading(serve, video_file, start, seek_to):
    server, port = serve(video=video_file, metadata=_metadata(duration=120.0))
    body = {"path": str(video_file)} | ({"start": start} if start is not None else {})

    status, response, _ = _http(port, "/open", "POST", body)

    assert (status, response["status"]) == (200, "ready")
    assert server.is_loaded
    assert server.seek_requested.is_set() is (seek_to is not None)
    if seek_to is not None:
        assert server.seek_target_segment == seek_to
