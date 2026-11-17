from __future__ import annotations

import json
import os
import struct
from dataclasses import replace
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import av
import numpy as np
import pytest

from jasna.accelerator import AcceleratorVendor
from jasna.media.audio_utils import source_audio_codec
from jasna.media.container_utils import output_creation_time
from jasna.media.probe import get_video_meta_data
from jasna.media.splice import (
    SpliceSpan,
    build_splice_plan,
    concatenate_fragments,
    create_copy_fragment,
    mux_final_output,
    normalize_fragment,
    probe_keyframes,
    resolve_smart_encoder_settings,
)
from jasna.os_utils import resolve_executable, subprocess_no_window_kwargs
from jasna.segments import SegmentRange

from factories import (
    creation_times,
    write_dated_source,
    write_double_adts_aac_source,
    write_hrd_timing_sei_source,
    write_variable_frame_rate_source,
    write_non_utf8_metadata_source,
    write_unknown_codec_audio_source,
)

COMMENT = '{"jasna":"test","schema":1}'


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


def _shorten_mp4_video_duration(path: Path, ticks: int) -> None:
    """Lower the video track's mdhd duration, leaving every sample in place."""
    data = bytearray(path.read_bytes())

    def boxes(start: int, end: int):
        while start + 8 <= end:
            size, kind = struct.unpack_from(">I4s", data, start)
            yield start, size, kind
            start += size

    def child(start: int, size: int, kind: bytes) -> int:
        return next(at for at, _, k in boxes(start + 8, start + size) if k == kind)

    moov, moov_size = next((at, size) for at, size, kind in boxes(0, len(data)) if kind == b"moov")
    for trak, trak_size, kind in boxes(moov + 8, moov + moov_size):
        if kind != b"trak":
            continue
        mdia = child(trak, trak_size, b"mdia")
        mdia_size = struct.unpack_from(">I", data, mdia)[0]
        hdlr = child(mdia, mdia_size, b"hdlr")
        if data[hdlr + 16 : hdlr + 20] != b"vide":
            continue
        mdhd = child(mdia, mdia_size, b"mdhd")
        field, offset = (">Q", 32) if data[mdhd + 8] == 1 else (">I", 24)
        (duration,) = struct.unpack_from(field, data, mdhd + offset)
        struct.pack_into(field, data, mdhd + offset, duration - ticks)
        path.write_bytes(bytes(data))
        return
    raise AssertionError(f"{path} has no video track")


def test_copy_span_keeps_frames_past_the_recorded_video_duration(tmp_path: Path) -> None:
    # Some MP4s, e.g. lossless cuts of B-frame streams, record a video duration that
    # ends before the last frame. The tail copy must still carry that frame: the
    # B-frames shown before it reference it.
    source = tmp_path / "short-duration.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=12:duration=3",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-g", "12",
        "-keyint_min", "12",
        "-bf", "3",
        "-x264-params", "scenecut=0",
        str(source),
    )
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        frame_ticks = round(1 / (stream.average_rate * stream.time_base))
    _shorten_mp4_video_duration(source, frame_ticks)
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        last_pts = max(packet.pts for packet in container.demux(stream) if packet.pts is not None)
        assert stream.start_time + stream.duration <= last_pts
    with av.open(str(source)) as container:
        source_frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    assert len(source_frames) == 36

    metadata = get_video_meta_data(str(source))
    index = probe_keyframes(source, metadata)
    assert index.end_pts > last_pts

    copied = tmp_path / "tail.nut"
    create_copy_fragment(source, SpliceSpan("copy", index.pts[-1], index.end_pts), index, copied, codec="h264")
    with av.open(str(copied)) as container:
        tail_frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    assert len(tail_frames) == 12
    for frame_index, frame in enumerate(tail_frames):
        assert np.array_equal(frame, source_frames[24 + frame_index])


def _fragment_packets(path: Path) -> list[tuple[int, int | None, bytes]]:
    with av.open(str(path)) as container:
        return [
            (packet.pts, packet.dts, bytes(packet))
            for packet in container.demux(container.streams.video[0])
            if packet.size
        ]


def test_copy_span_stops_reading_at_its_end_with_the_same_packets(tmp_path: Path) -> None:
    source = tmp_path / "b-frames.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=12:duration=4",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-g", "12", "-keyint_min", "12", "-bf", "3", "-x264-params", "scenecut=0",
        str(source),
    )
    index = probe_keyframes(source, get_video_meta_data(str(source)))
    assert index.monotonic_dts
    span = SpliceSpan("copy", index.pts[1], index.pts[2])

    stopped = tmp_path / "stopped.nut"
    create_copy_fragment(source, span, index, stopped, codec="h264")
    read_to_end = tmp_path / "read-to-end.nut"
    create_copy_fragment(source, span, replace(index, monotonic_dts=False), read_to_end, codec="h264")

    packets = _fragment_packets(stopped)
    assert len(packets) == 12
    assert packets == _fragment_packets(read_to_end)


def test_copy_span_that_cannot_start_on_its_keyframe_fails(tmp_path: Path) -> None:
    source = tmp_path / "b-frames.ts"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=12:duration=2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-g", "12", "-keyint_min", "12", "-bf", "3", "-x264-params", "scenecut=0",
        str(source),
    )
    index = probe_keyframes(source, get_video_meta_data(str(source)))
    # The second frame is a B-frame: in decode order the P-frame shown after it comes first.
    b_frame = index.pts[0] + round(1 / (12 * index.time_base))
    index = replace(index, pts=tuple(sorted((*index.pts, b_frame))))

    with pytest.raises(RuntimeError, match="from its keyframe"):
        create_copy_fragment(source, SpliceSpan("copy", b_frame, index.end_pts), index, tmp_path / "copy.nut", codec="h264")


def _decoded_frames(path: Path) -> list[tuple[float, np.ndarray]]:
    """(seconds past the video start, RGB picture) per decoded frame."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return [
            (float((frame.pts - stream.start_time) * stream.time_base), frame.to_ndarray(format="rgb24"))
            for frame in container.decode(stream)
        ]


def _audio_packet_count(path: Path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for packet in container.demux(audio=0) if packet.size)


@pytest.mark.parametrize("container", ["ts", "mp4"])
def test_copy_spans_keep_every_frame_and_the_audio(tmp_path: Path, container: str) -> None:
    # MPEG-TS seeks by decode time: a copy that seeks to its keyframe's pts lands after
    # the keyframe's packet when B-frames delay its pts, and loses the whole GOP.
    source = tmp_path / f"source.{container}"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=12:duration=6",
        "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=6",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-g", "12", "-keyint_min", "12", "-bf", "3", "-x264-params", "scenecut=0",
        "-c:a", "aac",
        str(source),
    )
    index = probe_keyframes(source, get_video_meta_data(str(source)))
    assert len(index.pts) == 6
    # Two GOPs per span: a copy that loses its first GOP still decodes, just shorter.
    bounds = (index.start_pts, index.pts[2], index.pts[4], index.end_pts)
    fragments = []
    for span_index, (start, end) in enumerate(zip(bounds, bounds[1:])):
        raw = tmp_path / f"raw-{span_index}.nut"
        create_copy_fragment(source, SpliceSpan("copy", start, end), index, raw, codec="h264")
        normalized = tmp_path / f"part-{span_index}.ts"
        normalize_fragment(raw, normalized, codec="h264")
        fragments.append((normalized, float((end - start) * index.time_base)))
    assembled = tmp_path / "assembled.ts"
    concatenate_fragments(fragments, manifest=tmp_path / "parts.ffconcat", destination=assembled, codec="h264")
    output = tmp_path / "output.mkv"

    mux_final_output(assembled, source, output, codec="h264", comment=COMMENT)

    source_frames, output_frames = _decoded_frames(source), _decoded_frames(output)
    assert [seconds for seconds, _ in output_frames] == pytest.approx(
        [seconds for seconds, _ in source_frames], abs=0.002
    )
    assert all(np.array_equal(out, src) for (_, out), (_, src) in zip(output_frames, source_frames))
    assert _audio_packet_count(output) == _audio_packet_count(source)
    remuxed = tmp_path / "remuxed.mkv"
    _ffmpeg("-i", str(source), "-c", "copy", str(remuxed))
    assert _video_delay_after_audio(output) == pytest.approx(_video_delay_after_audio(remuxed), abs=0.002)



_LATE_VIDEO_ENCODERS = [
    ("h264", "libx264", ["-g", "12", "-keyint_min", "12", "-sc_threshold", "0"]),
    ("av1", "libsvtav1", ["-preset", "10", "-g", "12", "-svtav1-params", "scd=0"]),
]


def _late_video_source(tmp_path: Path, codec: str, encoder: str, options: list[str]) -> Path:
    """MKV whose video starts 0.25 s after its audio, as MKV keeps an audio priming delay."""
    source = tmp_path / f"late-video-{codec}.mkv"
    _ffmpeg(
        "-itsoffset", "0.25",
        "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=12:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=3",
        "-c:v", encoder,
        "-pix_fmt", "yuv420p",
        *options,
        "-c:a", "libopus",
        str(source),
    )
    return source


def _video_delay_after_audio(path: Path) -> float:
    with av.open(str(path)) as container:
        video, audio = container.streams.video[0], container.streams.audio[0]
        return float(video.start_time * video.time_base - audio.start_time * audio.time_base)


@pytest.mark.parametrize(("codec", "encoder", "options"), _LATE_VIDEO_ENCODERS)
def test_copy_span_of_video_starting_after_audio_holds_only_its_frames(
    tmp_path: Path, codec: str, encoder: str, options: list[str]
) -> None:
    source = _late_video_source(tmp_path, codec, encoder, options)
    with av.open(str(source)) as container:
        assert container.streams.video[0].start_time * container.streams.video[0].time_base == pytest.approx(0.25)
        source_frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    metadata = get_video_meta_data(str(source))
    index = probe_keyframes(source, metadata)
    assert len(index.pts) == 3

    for first_frame, (start, end) in ((0, (index.start_pts, index.pts[1])), (12, (index.pts[1], index.pts[2]))):
        copied = tmp_path / f"from-{first_frame}.nut"
        create_copy_fragment(source, SpliceSpan("copy", start, end), index, copied, codec=codec)
        with av.open(str(copied)) as container:
            copied_frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        assert len(copied_frames) == 12
        for frame_index, frame in enumerate(copied_frames):
            assert np.array_equal(frame, source_frames[first_frame + frame_index])


@pytest.mark.parametrize("suffix", [".mkv", ".mp4"])
@pytest.mark.parametrize(("codec", "encoder", "options"), _LATE_VIDEO_ENCODERS)
def test_final_mux_keeps_the_video_start_after_the_audio(
    tmp_path: Path, codec: str, encoder: str, options: list[str], suffix: str
) -> None:
    source = _late_video_source(tmp_path, codec, encoder, options)
    source_delay = _video_delay_after_audio(source)
    assert source_delay > 0.2
    index = probe_keyframes(source, get_video_meta_data(str(source)))
    bounds = (*index.pts[1:], index.end_pts)
    fragments = []
    for span_index, (start, end) in enumerate(zip((index.start_pts, *index.pts[1:]), bounds)):
        span = SpliceSpan("copy", start, end)
        raw = tmp_path / f"raw-{span_index}.nut"
        create_copy_fragment(source, span, index, raw, codec=codec)
        normalized = tmp_path / f"part-{span_index}{'.ts' if codec == 'h264' else '.mkv'}"
        normalize_fragment(raw, normalized, codec=codec)
        fragments.append((normalized, float((end - start) * index.time_base)))
    assembled = tmp_path / f"assembled{'.ts' if codec == 'h264' else '.mkv'}"
    concatenate_fragments(fragments, manifest=tmp_path / "parts.ffconcat", destination=assembled, codec=codec)
    output = tmp_path / f"output{suffix}"

    mux_final_output(assembled, source, output, codec=codec, comment=COMMENT)

    assert _video_delay_after_audio(output) == pytest.approx(source_delay, abs=0.02)


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
        vendor=AcceleratorVendor.NVIDIA,
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
        ("hevc", "libx265", ["-x265-params", "keyint=12:min-keyint=12:scenecut=0:open-gop=0"], ["-x265-params", "keyint=12:bframes=4:open-gop=0:ref=5"]),
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
    create_copy_fragment(
        source, SpliceSpan("copy", index.start_pts, index.pts[1]), index, raw_parts[0], codec=metadata.codec_name
    )
    create_copy_fragment(
        source, SpliceSpan("copy", index.pts[2], index.end_pts), index, raw_parts[2], codec=metadata.codec_name
    )
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
    mux_final_output(assembled, source, output, codec=codec, comment=COMMENT)

    with av.open(str(output)) as container:
        assert len(container.streams.video) == 1
        assert len(container.streams.audio) == 1
        assert container.streams.video[0].codec_context.name in {codec, "libdav1d"}
        if codec == "hevc":
            assert container.streams.video[0].codec_tag == "hev1"
        output_frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        assert len(output_frames) == 36
        assert float(container.duration / av.time_base) == pytest.approx(3.0, abs=0.01)
    with av.open(str(source)) as container:
        source_frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    for frame_index in [*range(12), *range(24, 36)]:
        assert np.array_equal(output_frames[frame_index], source_frames[frame_index])
    with av.open(str(raw_parts[1])) as container:
        render_frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    for frame_index in range(12):
        assert np.array_equal(output_frames[12 + frame_index], render_frames[frame_index])


@pytest.mark.parametrize("write_source", [write_hrd_timing_sei_source, write_variable_frame_rate_source])
@pytest.mark.parametrize(
    "selections",
    [
        [SegmentRange(1.2, 1.8)],
        [SegmentRange(1.1, 1.4), SegmentRange(3.1, 3.5)],
    ],
)
def test_splice_keeps_source_timestamps(tmp_path: Path, selections: list[SegmentRange], write_source) -> None:
    source = write_source(tmp_path / "source.mp4")
    metadata = get_video_meta_data(str(source))
    index = probe_keyframes(source, metadata)
    plan = build_splice_plan(selections, index, duration=metadata.duration)
    assert [span.kind for span in plan.spans][:2] == ["copy", "render"]

    fragments = []
    for span_index, span in enumerate(plan.spans):
        raw = tmp_path / f"raw-{span_index}.nut"
        duration = float((span.end_pts - span.start_pts) * index.time_base)
        if span.kind == "copy":
            create_copy_fragment(source, span, index, raw, codec="h264")
        else:
            _ffmpeg(
                "-ss", f"{index.seconds_for_pts(span.start_pts):.6f}",
                "-i", str(source),
                "-t", f"{duration:.6f}",
                "-map", "0:v:0",
                "-an",
                "-vf", "hue=s=0",
                "-c:v", "libx264", "-profile:v", "main",
                "-g", "30", "-bf", "3", "-flags", "+cgop", "-b:v", "500k",
                "-x264-params", "nal-hrd=vbr:vbv-maxrate=1000:vbv-bufsize=2000",
                "-f", "nut",
                str(raw),
            )
        normalized = tmp_path / f"part-{span_index}.ts"
        normalize_fragment(raw, normalized, codec="h264")
        fragments.append((normalized, duration))
    assembled = tmp_path / "assembled.ts"
    concatenate_fragments(fragments, manifest=tmp_path / "parts.ffconcat", destination=assembled, codec="h264")
    output = tmp_path / "output.mp4"
    mux_final_output(assembled, source, output, codec="h264", comment=COMMENT)

    def shown_seconds(path: Path) -> list[float]:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            return sorted(
                round(float((packet.pts - stream.start_time) * stream.time_base), 4)
                for packet in container.demux(stream)
                if packet.size
            )

    assert shown_seconds(output) == shown_seconds(source)


@pytest.mark.parametrize("selection", [SegmentRange(0.0, 0.25), SegmentRange(0.9, 1.1)])
def test_mid_gop_cut_splices_to_the_shown_frames_and_duration(tmp_path: Path, selection: SegmentRange) -> None:
    # A `-c copy` cut at 1.5 s keeps the GOP from 1.0 s as a hidden lead-in, so the
    # shown video starts half a GOP before the first keyframe at or after its start.
    full = tmp_path / "full.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=12:duration=4.5",
        "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=4.5",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-g", "12", "-keyint_min", "12", "-sc_threshold", "0", "-bf", "3",
        "-c:a", "aac",
        str(full),
    )
    source = tmp_path / "cut.mp4"
    _ffmpeg("-ss", "1.5", "-i", str(full), "-c", "copy", str(source))
    metadata = get_video_meta_data(str(source))
    index = probe_keyframes(source, metadata)
    assert index.pts[0] < index.start_pts
    plan = build_splice_plan([selection], index, duration=metadata.duration)

    fragments = []
    for span_index, span in enumerate(plan.spans):
        raw = tmp_path / f"raw-{span_index}.nut"
        if span.kind == "copy":
            assert span.start_pts in index.pts
            create_copy_fragment(source, span, index, raw, codec="h264")
        else:
            _ffmpeg(
                "-ss", f"{index.seconds_for_pts(span.start_pts):.6f}",
                "-i", str(source),
                "-t", f"{float((span.end_pts - span.start_pts) * index.time_base):.6f}",
                "-map", "0:v:0",
                "-an",
                "-c:v", "libx264",
                "-g", "12", "-bf", "3", "-flags", "+cgop",
                "-f", "nut",
                str(raw),
            )
        normalized = tmp_path / f"part-{span_index}.ts"
        normalize_fragment(raw, normalized, codec="h264")
        fragments.append((normalized, float((span.end_pts - span.start_pts) * index.time_base)))
    assembled = tmp_path / "assembled.ts"
    concatenate_fragments(fragments, manifest=tmp_path / "parts.ffconcat", destination=assembled, codec="h264")
    output = tmp_path / "output.mp4"
    mux_final_output(assembled, source, output, codec="h264", comment=COMMENT)

    with av.open(str(source)) as container:
        source_frames = sum(1 for _ in container.decode(video=0))
    with av.open(str(output)) as container:
        assert sum(1 for _ in container.decode(video=0)) == source_frames
        video, audio = container.streams.video[0], container.streams.audio[0]
        video_seconds = float(video.duration * video.time_base)
        assert video_seconds == pytest.approx(source_frames / 12, abs=0.01)
        assert float(audio.start_time * audio.time_base) == pytest.approx(
            float(video.start_time * video.time_base), abs=0.05
        )


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
        "comment=Source comment\n"
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
        "-metadata:s:t:0", f"filename={attachment.name}",
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

    mux_final_output(assembled, source, output, codec="h264", comment=COMMENT)

    with av.open(str(output)) as container:
        assert container.metadata["title"] == "Source title"
        assert [
            json.loads(value) for key, value in container.metadata.items() if key.lower() == "comment"
        ] == [{**json.loads(COMMENT), "source_comment": "Source comment"}]
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

    mux_final_output(assembled, source, output, codec="h264", comment=COMMENT)

    with av.open(str(output)) as container:
        assert [chapter["metadata"]["title"] for chapter in container.chapters()] == [
            "Opening",
            "Main",
        ]
        assert len(container.streams.data) == 1


def test_final_mux_skips_mp4_timecode_stream_without_codec(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=12:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=1",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-timecode", "00:00:00:00",
        str(source),
    )
    with av.open(str(source)) as container:
        assert container.streams.data[0].codec_context is None

    assembled = tmp_path / "assembled.mp4"
    _ffmpeg("-i", str(source), "-map", "0:v:0", "-c:v", "copy", str(assembled))
    output = tmp_path / "output.mp4"

    mux_final_output(assembled, source, output, codec="h264", comment=COMMENT)

    with av.open(str(output)) as container:
        assert len(container.streams.video) == 1
        assert len(container.streams.audio) == 1
        assert container.streams.video[0].metadata["timecode"] == "00:00:00:00"


@pytest.mark.parametrize("suffix", [".mkv", ".mp4"])
def test_final_mux_skips_audio_stream_without_codec(tmp_path: Path, suffix: str) -> None:
    source = write_unknown_codec_audio_source(tmp_path / "source.mkv")
    with av.open(str(source)) as container:
        assert [stream.codec_context is None for stream in container.streams.audio] == [False, True]
    assembled = tmp_path / "assembled.mkv"
    _ffmpeg("-i", str(source), "-map", "0:v:0", "-c:v", "copy", str(assembled))
    output = tmp_path / f"output{suffix}"

    mux_final_output(assembled, source, output, codec="h264", comment=COMMENT)

    with av.open(str(output)) as container:
        assert [stream.codec_context.name for stream in container.streams.audio] == ["aac"]


def test_final_mux_copies_transport_stream_aac_into_mp4(tmp_path: Path) -> None:
    source = tmp_path / "source.ts"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=12:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=1",
        "-c:v", "libx264", "-c:a", "aac", str(source),
    )
    assembled = tmp_path / "assembled.ts"
    _ffmpeg("-i", str(source), "-map", "0:v:0", "-c:v", "copy", str(assembled))
    output = tmp_path / "output.mp4"

    mux_final_output(assembled, source, output, codec="h264", comment=COMMENT)

    with av.open(str(output)) as container:
        audio = container.streams.audio[0]
        assert audio.codec_context.name == "aac"
        assert sum(frame.samples for frame in container.decode(audio)) / audio.sample_rate == pytest.approx(1.0, abs=0.1)


def test_final_mux_copies_double_adts_aac_into_mp4(tmp_path: Path) -> None:
    source = write_double_adts_aac_source(tmp_path)
    assembled = tmp_path / "assembled.mp4"
    _ffmpeg("-i", str(source), "-map", "0:v:0", "-c:v", "copy", str(assembled))
    output = tmp_path / "output.mp4"

    mux_final_output(assembled, source, output, codec="h264", comment=COMMENT)

    with av.open(str(output)) as container:
        assert sum(1 for _ in container.decode(audio=0)) > 1


def test_output_creation_time_is_utc_in_ffmpeg_form() -> None:
    evening_at_utc_plus_one = datetime(2026, 11, 15, 19, 0, 0, 120, tzinfo=timezone(timedelta(hours=1)))

    assert output_creation_time(evening_at_utc_plus_one) == "2026-11-15T18:00:00.000120Z"


@pytest.mark.parametrize(("suffix", "dated_tags"), [(".mp4", 3), (".mkv", 1)])
def test_final_mux_claims_the_processing_time_not_the_sources(tmp_path: Path, suffix: str, dated_tags: int) -> None:
    source = write_dated_source(tmp_path / f"source{suffix}")
    assembled = tmp_path / "assembled.ts"
    _ffmpeg("-i", str(source), "-map", "0:v:0", "-c:v", "copy", "-bsf:v", "h264_mp4toannexb", str(assembled))
    output = tmp_path / f"output{suffix}"

    started = datetime.now(timezone.utc).replace(microsecond=0)
    mux_final_output(assembled, source, output, codec="h264", comment=COMMENT)
    finished = datetime.now(timezone.utc)

    claimed = creation_times(output)
    assert len(claimed) == dated_tags
    assert all(started <= moment <= finished for moment in claimed)


def test_source_with_non_utf8_tags_probes_and_splices(tmp_path: Path) -> None:
    source = write_non_utf8_metadata_source(tmp_path / "bad-tags.mp4")

    metadata = get_video_meta_data(str(source))
    assert source_audio_codec(str(source)) == "aac"
    index = probe_keyframes(source, metadata)
    copied = tmp_path / "copy.nut"
    create_copy_fragment(source, SpliceSpan("copy", index.start_pts, index.pts[1]), index, copied, codec="h264")
    normalized = tmp_path / "copy.ts"
    normalize_fragment(copied, normalized, codec="h264")
    output = tmp_path / "output.mp4"
    mux_final_output(normalized, source, output, codec="h264", comment=COMMENT)

    with av.open(str(output), metadata_errors="replace") as container:
        assert container.metadata["title"] == "JASNA\ufffdTAG"
        assert container.streams.audio


def test_restored_fragment_with_long_multibyte_title_splices_and_keeps_title(tmp_path: Path) -> None:
    title = "長いタイトル" * 20
    assert len(title) < 255 < len(title.encode("utf-8"))
    source = tmp_path / "long-title.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=12:duration=1",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-metadata", f"title={title}",
        str(source),
    )
    restored = tmp_path / "restored.nut"
    _ffmpeg("-i", str(source), "-map", "0:v:0", "-c:v", "libx264", "-f", "nut", str(restored))
    with av.open(str(restored)) as container:
        assert container.metadata["title"] == title

    normalized = tmp_path / "restored.ts"
    normalize_fragment(restored, normalized, codec="h264")
    output = tmp_path / "output.mp4"
    mux_final_output(normalized, source, output, codec="h264", comment=COMMENT)

    with av.open(str(output)) as container:
        assert container.metadata["title"] == title


def _packet_count(path: Path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for packet in container.demux(container.streams.video[0]) if packet.size)


def test_concatenation_keeps_a_fragment_of_a_few_tiny_frames(tmp_path: Path) -> None:
    """#733: a two-frame black tail made a TS too small for ffmpeg to recognize, and concat skipped it."""
    sources = {
        "picture": "testsrc2=size=160x96:rate=12:duration=1",
        "black-tail": "color=c=black:size=160x96:rate=12:duration=0.1",
    }
    fragments = []
    expected = 0
    for name, graph in sources.items():
        raw = tmp_path / f"{name}.mp4"
        _ffmpeg("-f", "lavfi", "-i", graph, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(raw))
        expected += _packet_count(raw)
        normalized = tmp_path / f"{name}.ts"
        normalize_fragment(raw, normalized, codec="h264")
        fragments.append((normalized, float(get_video_meta_data(str(raw)).duration)))
    assembled = tmp_path / "assembled.ts"

    concatenate_fragments(fragments, manifest=tmp_path / "parts.ffconcat", destination=assembled, codec="h264")

    assert _packet_count(assembled) == expected


def test_concatenation_fails_when_a_fragment_cannot_be_read(tmp_path: Path) -> None:
    unreadable = tmp_path / "broken.ts"
    unreadable.write_bytes(b"not a video")

    with pytest.raises(RuntimeError, match="Impossible to open"):
        concatenate_fragments(
            [(unreadable, 1.0)], manifest=tmp_path / "parts.ffconcat", destination=tmp_path / "out.ts", codec="h264"
        )


def test_ffmpeg_failure_naming_a_non_utf8_path_reports_its_error(tmp_path: Path) -> None:
    unreadable = tmp_path / os.fsdecode(b"broken\x90.nut")
    unreadable.write_bytes(b"not a video")

    with pytest.raises(RuntimeError, match="Failed to normalize smart-render fragment"):
        normalize_fragment(unreadable, tmp_path / "out.ts", codec="h264")
