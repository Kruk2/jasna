from __future__ import annotations

import subprocess
from pathlib import Path

import av
import numpy as np
import pytest

from jasna.media import get_video_meta_data
from jasna.media.splice import (
    SpliceSpan,
    concatenate_fragments,
    create_copy_fragment,
    mux_final_output,
    normalize_fragment,
    probe_keyframes,
    resolve_smart_encoder_settings,
)
from jasna.os_utils import resolve_executable, subprocess_no_window_kwargs


def _ffmpeg(*args: str) -> None:
    completed = subprocess.run(
        [resolve_executable("ffmpeg"), "-hide_banner", "-y", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
        check=False,
        **subprocess_no_window_kwargs(),
    )
    if completed.returncode != 0:
        pytest.fail(completed.stderr)


def test_h264_probe_resolves_source_compatible_smart_settings(tmp_path: Path) -> None:
    source = tmp_path / "source-h264-main.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=12:duration=3",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "main",
        "-g", "12",
        "-keyint_min", "12",
        "-bf", "3",
        "-x264-params", "b-pyramid=none:scenecut=0",
        str(source),
    )

    metadata = get_video_meta_data(str(source))
    index = probe_keyframes(source, metadata)
    settings = resolve_smart_encoder_settings(
        "h264",
        metadata,
        index,
        {"cq": 22, "profile": "high", "g": 250, "bf": 4, "b_ref_mode": "middle"},
    )

    assert metadata.profile == "Main"
    assert index.max_b_frames == 3
    assert index.uses_b_references is False
    assert settings == {
        "cq": 22,
        "profile": "main",
        "g": 12,
        "bf": 3,
        "b_ref_mode": "disabled",
    }


@pytest.mark.parametrize(
    ("codec", "encoder", "source_options", "render_options"),
    [
        ("h264", "libx264", ["-g", "12", "-keyint_min", "12", "-sc_threshold", "0"], ["-g", "12", "-bf", "3", "-flags", "+cgop"]),
        ("hevc", "libx265", ["-x265-params", "keyint=12:min-keyint=12:scenecut=0:open-gop=0"], ["-x265-params", "keyint=12:bframes=4:open-gop=0"]),
        ("av1", "libsvtav1", ["-preset", "10", "-g", "12", "-svtav1-params", "scd=0"], ["-preset", "10", "-g", "12"]),
    ],
)
def test_mixed_encoder_splice_decodes_with_exact_duration_and_audio(
    tmp_path: Path,
    codec: str,
    encoder: str,
    source_options: list[str],
    render_options: list[str],
) -> None:
    source = tmp_path / f"source-{codec}.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=12:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=3",
        "-c:v", encoder,
        "-pix_fmt", "yuv420p",
        *source_options,
        "-c:a", "aac",
        str(source),
    )
    metadata = get_video_meta_data(str(source))
    index = probe_keyframes(source, metadata)
    assert len(index.pts) >= 3

    raw_parts = [tmp_path / f"raw-{i}.nut" for i in range(3)]
    create_copy_fragment(source, SpliceSpan("copy", index.start_pts, index.pts[1]), index, raw_parts[0])
    create_copy_fragment(source, SpliceSpan("copy", index.pts[2], index.end_pts), index, raw_parts[2])
    _ffmpeg(
        "-ss", "1",
        "-i", str(source),
        "-t", "1",
        "-map", "0:v:0",
        "-an",
        "-vf", "hue=s=0",
        "-c:v", encoder,
        *render_options,
        "-f", "nut",
        str(raw_parts[1]),
    )

    suffix = ".ts" if codec in {"h264", "hevc"} else ".mkv"
    fragments = []
    for part_index, raw in enumerate(raw_parts):
        normalized = tmp_path / f"part-{part_index}{suffix}"
        normalize_fragment(raw, normalized, codec=codec)
        fragments.append((normalized, 1.0))
    assembled = tmp_path / f"assembled{suffix}"
    concatenate_fragments(
        fragments,
        manifest=tmp_path / "parts.ffconcat",
        destination=assembled,
        codec=codec,
    )
    output = tmp_path / f"output-{codec}.mp4"
    mux_final_output(assembled, source, output, codec=codec)

    with av.open(str(output)) as container:
        assert len(container.streams.video) == 1
        assert len(container.streams.audio) == 1
        assert container.streams.video[0].codec_context.name in {codec, "libdav1d"}
        output_frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        assert len(output_frames) == 36
        assert float(container.duration / av.time_base) == pytest.approx(3.0, abs=0.01)
    with av.open(str(source)) as container:
        source_frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    for frame_index in [*range(12), *range(24, 36)]:
        assert np.array_equal(output_frames[frame_index], source_frames[frame_index])


@pytest.mark.parametrize(
    ("suffix", "expected_subtitle_codec", "expected_attachments"),
    [(".mkv", "srt", 1), (".mp4", "mov_text", 0)],
)
def test_final_mux_preserves_compatible_source_structure(
    tmp_path: Path,
    suffix: str,
    expected_subtitle_codec: str,
    expected_attachments: int,
) -> None:
    subtitle = tmp_path / "subtitle.srt"
    subtitle.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nOpening subtitle\n",
        encoding="utf-8",
    )
    attachment = tmp_path / "font.txt"
    attachment.write_bytes(b"font payload")
    ffmetadata = tmp_path / "chapters.ffmeta"
    ffmetadata.write_text(
        ";FFMETADATA1\n"
        "title=Source title\n"
        "[CHAPTER]\n"
        "TIMEBASE=1/1000\n"
        "START=0\n"
        "END=1000\n"
        "title=Opening\n"
        "[CHAPTER]\n"
        "TIMEBASE=1/1000\n"
        "START=1000\n"
        "END=2000\n"
        "title=Main\n",
        encoding="utf-8",
    )
    source = tmp_path / "source.mkv"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=12:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=2",
        "-f", "ffmetadata", "-i", str(ffmetadata),
        "-i", str(subtitle),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-map", "3:s:0",
        "-map_metadata", "2",
        "-map_chapters", "2",
        "-metadata:s:v:0", "language=jpn",
        "-metadata:s:s:0", "language=pol",
        "-metadata:s:s:0", "title=Signs",
        "-disposition:s:0", "default+forced",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-c:s", "srt",
        "-attach", str(attachment),
        "-metadata:s:t:0", "mimetype=text/plain",
        str(source),
    )
    assembled = tmp_path / "assembled.mkv"
    _ffmpeg(
        "-i", str(source),
        "-map", "0:v:0",
        "-an",
        "-c:v", "copy",
        str(assembled),
    )
    output = tmp_path / f"output{suffix}"

    mux_final_output(assembled, source, output, codec="h264")

    with av.open(str(output)) as container:
        assert container.metadata["title"] == "Source title"
        assert [chapter["metadata"]["title"] for chapter in container.chapters()] == [
            "Opening",
            "Main",
        ]
        assert container.streams.video[0].metadata["language"] == "jpn"
        assert len(container.streams.audio) == 1
        assert len(container.streams.subtitles) == 1
        assert len(container.streams.attachments) == expected_attachments
        output_subtitle = container.streams.subtitles[0]
        assert output_subtitle.codec_context.name == expected_subtitle_codec
        assert output_subtitle.metadata["language"] == "pol"
        assert (
            output_subtitle.metadata.get("title")
            or output_subtitle.metadata.get("name")
        ) == "Signs"
        assert output_subtitle.disposition.default
        assert output_subtitle.disposition.forced
        subtitle_text = b"".join(
            getattr(rect, "ass", b"") + getattr(rect, "text", b"")
            for packet in container.demux(output_subtitle)
            if packet.size
            for rect in packet.decode()
        )
        assert b"Opening subtitle" in subtitle_text
        if expected_attachments:
            output_attachment = container.streams.attachments[0]
            assert output_attachment.name == "font.txt"
            assert output_attachment.mimetype == "text/plain"
            assert output_attachment.data == b"font payload"


def test_final_mux_rebuilds_mp4_chapter_carrier_only_once(tmp_path: Path) -> None:
    ffmetadata = tmp_path / "chapters.ffmeta"
    ffmetadata.write_text(
        ";FFMETADATA1\n"
        "[CHAPTER]\n"
        "TIMEBASE=1/1000\n"
        "START=0\n"
        "END=1000\n"
        "title=Opening\n"
        "[CHAPTER]\n"
        "TIMEBASE=1/1000\n"
        "START=1000\n"
        "END=2000\n"
        "title=Main\n",
        encoding="utf-8",
    )
    source = tmp_path / "source.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=12:duration=2",
        "-f", "ffmetadata", "-i", str(ffmetadata),
        "-map", "0:v:0",
        "-map_chapters", "1",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(source),
    )
    assembled = tmp_path / "assembled.mp4"
    _ffmpeg(
        "-i", str(source),
        "-map", "0:v:0",
        "-c:v", "copy",
        str(assembled),
    )
    output = tmp_path / "output.mp4"

    mux_final_output(assembled, source, output, codec="h264")

    with av.open(str(output)) as container:
        assert [chapter["metadata"]["title"] for chapter in container.chapters()] == [
            "Opening",
            "Main",
        ]
        assert len(container.streams.data) == 1
