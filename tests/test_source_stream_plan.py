from __future__ import annotations

import subprocess
from pathlib import Path

import av
import pytest

from jasna.media.container_utils import source_stream_plan
from jasna.os_utils import resolve_executable

from factories import write_unknown_codec_audio_source

SINE = ("-f", "lavfi", "-i", "sine=duration=1")


def _source(directory: Path, name: str, *args: str) -> Path:
    (directory / "subtitle.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
    (directory / "font.txt").write_bytes(b"font payload")
    (directory / "chapters.ffmeta").write_text(
        ";FFMETADATA1\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=1000\ntitle=Opening\n", encoding="utf-8"
    )
    subprocess.run(
        [
            resolve_executable("ffmpeg"), "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=12:duration=1",
            *args,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", name,
        ],
        check=True,
        cwd=directory,
    )
    return directory / name


def _plan(source: Path, output_format: str, output_suffix: str) -> list[tuple[str, str, str | None]]:
    with av.open(str(source)) as container:
        plan = source_stream_plan(
            container,
            video_index=container.streams.video[0].index,
            output_format=output_format,
            output_suffix=output_suffix,
        )
        return [(action.stream.type, action.kind, action.codec) for action in plan]


@pytest.mark.parametrize(
    ("name", "args", "output_format", "output_suffix", "expected"),
    [
        ("aac.mp4", (*SINE, "-c:a", "aac"), "mp4", ".mp4", [("audio", "copy_adts_to_asc", None)]),
        ("aac.mp4", (*SINE, "-c:a", "aac"), "matroska", ".mkv", [("audio", "copy", None)]),
        ("adts.ts", (*SINE, "-c:a", "aac"), "mp4", ".mp4", [("audio", "copy", None)]),
        ("vorbis.mkv", (*SINE, "-c:a", "libvorbis"), "mp4", ".mp4", [("audio", "encode", "aac")]),
        ("vorbis.mkv", (*SINE, "-c:a", "libvorbis"), "matroska", ".mkv", [("audio", "copy", None)]),
        ("opus.mkv", (*SINE, "-c:a", "libopus"), "mov", ".mov", [("audio", "encode", "aac")]),
        ("opus.mkv", (*SINE, "-c:a", "libopus"), "mp4", ".mp4", [("audio", "copy", None)]),
        ("wma.avi", (*SINE, "-c:a", "wmav2"), "matroska", ".mkv", [("audio", "encode", "aac")]),
        ("pcm.mkv", (*SINE, "-c:a", "pcm_s16le"), "mp4", ".mp4", [("audio", "copy", None)]),
        ("srt.mkv", ("-i", "subtitle.srt", "-c:s", "srt"), "mp4", ".mp4", [("subtitle", "encode", "mov_text")]),
        ("srt.mkv", ("-i", "subtitle.srt", "-c:s", "srt"), "matroska", ".mkv", [("subtitle", "copy", None)]),
        # MPEG-TS cannot list the codecs it takes; a copy is still what PyAV accepts.
        ("srt.mkv", ("-i", "subtitle.srt", "-c:s", "srt"), "mpegts", ".ts", [("subtitle", "copy", None)]),
        ("font.mkv", ("-attach", "font.txt", "-metadata:s:t:0", "mimetype=text/plain"), "matroska", ".mkv",
         [("attachment", "copy", None)]),
        ("font.mkv", ("-attach", "font.txt", "-metadata:s:t:0", "mimetype=text/plain"), "mp4", ".mp4",
         [("attachment", "skip", None)]),
        ("timecode.mp4", (*SINE, "-c:a", "aac", "-timecode", "00:00:00:00"), "mp4", ".mp4",
         [("audio", "copy_adts_to_asc", None), ("data", "skip", None)]),
    ],
)
def test_source_stream_plan(tmp_path, name, args, output_format, output_suffix, expected):
    assert _plan(_source(tmp_path, name, *args), output_format, output_suffix) == expected


def test_audio_without_codec_is_skipped(tmp_path):
    source = write_unknown_codec_audio_source(tmp_path / "source.mkv")

    assert _plan(source, "mp4", ".mp4") == [("audio", "copy_adts_to_asc", None), ("audio", "skip", None)]


def test_mov_chapter_track_is_left_to_the_muxer(tmp_path):
    source = _source(tmp_path, "chapters.mp4", "-f", "ffmetadata", "-i", "chapters.ffmeta", "-map_chapters", "1")
    with av.open(str(source)) as container:
        assert [stream.type for stream in container.streams] == ["video", "data"]

    assert _plan(source, "mp4", ".mp4") == []
