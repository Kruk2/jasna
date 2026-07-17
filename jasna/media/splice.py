from __future__ import annotations

import bisect
import logging
import os
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av

from jasna.media import VideoMetadata, resolve_video_start_pts
from jasna.media.audio_utils import needs_audio_reencode
from jasna.os_utils import find_executable, resolve_executable, subprocess_no_window_kwargs
from jasna.segments import SegmentRange, normalize_segments

log = logging.getLogger(__name__)

SUPPORTED_SMART_CODECS = frozenset({"h264", "hevc", "av1"})
SUPPORTED_SMART_OUTPUTS = frozenset({".mp4", ".mov", ".mkv"})


class SmartRenderCompatibilityError(ValueError):
    def __init__(self, message: str, *, reason: str = "generic") -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class KeyframeIndex:
    pts: tuple[int, ...]
    time_base: Fraction
    start_pts: int
    end_pts: int

    def seconds_for_pts(self, pts: int) -> float:
        return float((int(pts) - self.start_pts) * self.time_base)


@dataclass(frozen=True)
class SpliceSpan:
    kind: str  # "copy" or "render"
    start_pts: int
    end_pts: int
    effect_ranges: tuple[tuple[int, int], ...] = ()

    @property
    def is_render(self) -> bool:
        return self.kind == "render"


@dataclass(frozen=True)
class SplicePlan:
    index: KeyframeIndex
    spans: tuple[SpliceSpan, ...]
    segments: tuple[SegmentRange, ...]

    @property
    def render_spans(self) -> tuple[SpliceSpan, ...]:
        return tuple(span for span in self.spans if span.is_render)


def _canonical_codec(name: str) -> str:
    value = str(name).lower()
    if value in {"h265", "h.265"}:
        return "hevc"
    if value in {"avc", "h.264"}:
        return "h264"
    if value == "av01":
        return "av1"
    return value


def validate_smart_render(
    metadata: VideoMetadata,
    *,
    output_path: str | Path,
    codec: str,
    retarget_high_fps: bool = False,
) -> str:
    input_codec = _canonical_codec(metadata.codec_name)
    output_codec = _canonical_codec(codec)
    if input_codec not in SUPPORTED_SMART_CODECS:
        raise SmartRenderCompatibilityError(
            f"Smart rendering does not support input codec {metadata.codec_name!r}; "
            "supported codecs are H.264, HEVC, and AV1"
        )
    if output_codec != input_codec:
        raise SmartRenderCompatibilityError(
            f"Smart rendering requires the output codec to match the input codec "
            f"({input_codec}); selected {output_codec}"
        )
    suffix = Path(output_path).suffix.lower()
    if suffix not in SUPPORTED_SMART_OUTPUTS:
        raise SmartRenderCompatibilityError(
            f"Smart rendering supports MP4, MOV, and MKV output, not {suffix or 'an extensionless file'}"
        )
    if retarget_high_fps:
        raise SmartRenderCompatibilityError(
            "Smart rendering cannot be combined with frame-rate retargeting"
        )
    if find_executable("ffmpeg") is None:
        raise SmartRenderCompatibilityError("Smart rendering requires ffmpeg")

    pixel_format = str(getattr(metadata, "pixel_format", "") or "").lower()
    supported_formats = {"", "yuv420p", "yuvj420p", "nv12", "yuv420p10le", "p010le"}
    if pixel_format not in supported_formats:
        raise SmartRenderCompatibilityError(
            f"Smart rendering requires 4:2:0 input; pixel format {pixel_format!r} is unsupported"
        )
    field_order = str(getattr(metadata, "field_order", "") or "").lower()
    if field_order not in {"", "unknown", "progressive"}:
        raise SmartRenderCompatibilityError("Smart rendering currently requires progressive video")
    if input_codec == "h264" and metadata.is_10bit:
        raise SmartRenderCompatibilityError("10-bit H.264 smart rendering is not supported by this NVENC path")
    if metadata.average_fps > 0 and metadata.video_fps > 0:
        relative_delta = abs(metadata.average_fps - metadata.video_fps) / metadata.video_fps
        if relative_delta > 0.001:
            raise SmartRenderCompatibilityError("Smart rendering currently requires constant-frame-rate video")
    return input_codec


def probe_keyframes(path: str | Path, metadata: VideoMetadata) -> KeyframeIndex:
    keyframes: list[int] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        codec = _canonical_codec(metadata.codec_name)
        extradata = bytes(stream.codec_context.extradata or b"")
        length_size = 4
        length_prefixed = codec == "h264" and len(extradata) > 4 and extradata[0] == 1
        if length_prefixed:
            length_size = (extradata[4] & 0x03) + 1
        elif codec == "hevc" and len(extradata) > 21 and extradata[0] == 1:
            length_prefixed = True
            length_size = (extradata[21] & 0x03) + 1
        time_base = Fraction(stream.time_base)
        start_pts = resolve_video_start_pts(stream.start_time, metadata.start_pts)
        for packet in container.demux(stream):
            if (
                packet.pts is not None
                and packet.is_keyframe
                and _is_safe_random_access_packet(
                    bytes(packet),
                    codec,
                    length_size,
                    length_prefixed=length_prefixed,
                )
            ):
                keyframes.append(int(packet.pts))
        stream_duration = stream.duration

    if not keyframes:
        raise SmartRenderCompatibilityError("No random-access video keyframes were found")
    keyframes = sorted(set(keyframes))
    if stream_duration is not None:
        end_pts = start_pts + int(stream_duration)
    else:
        end_pts = start_pts + round(float(metadata.duration) / time_base)
    if end_pts <= keyframes[-1]:
        end_pts = keyframes[-1] + max(1, round(1 / (metadata.video_fps * time_base)))
    return KeyframeIndex(tuple(keyframes), time_base, start_pts, int(end_pts))


def _nal_unit_types(
    data: bytes,
    *,
    codec: str,
    length_size: int,
    length_prefixed: bool,
) -> tuple[int, ...]:
    units: list[bytes] = []
    if not length_prefixed:
        starts: list[tuple[int, int]] = []
        i = 0
        while i + 3 <= len(data):
            if data[i:i + 4] == b"\x00\x00\x00\x01":
                starts.append((i, 4))
                i += 4
            elif data[i:i + 3] == b"\x00\x00\x01":
                starts.append((i, 3))
                i += 3
            else:
                i += 1
        for index, (start, prefix) in enumerate(starts):
            end = starts[index + 1][0] if index + 1 < len(starts) else len(data)
            if start + prefix < end:
                units.append(data[start + prefix:end])
    else:
        offset = 0
        while offset + length_size <= len(data):
            size = int.from_bytes(data[offset:offset + length_size], "big")
            offset += length_size
            if size <= 0 or offset + size > len(data):
                break
            units.append(data[offset:offset + size])
            offset += size
    if codec == "h264":
        return tuple(unit[0] & 0x1F for unit in units if unit)
    return tuple((unit[0] >> 1) & 0x3F for unit in units if unit)


def _is_safe_random_access_packet(
    data: bytes,
    codec: str,
    length_size: int,
    *,
    length_prefixed: bool,
) -> bool:
    if codec == "av1":
        return True
    nal_types = _nal_unit_types(
        data,
        codec=codec,
        length_size=length_size,
        length_prefixed=length_prefixed,
    )
    if codec == "h264":
        return 5 in nal_types
    return any(16 <= nal_type <= 20 for nal_type in nal_types)


def build_splice_plan(
    segments: tuple[SegmentRange, ...] | list[SegmentRange],
    index: KeyframeIndex,
    *,
    duration: float,
) -> SplicePlan:
    normalized = normalize_segments(segments, duration=duration)
    if not normalized:
        raise ValueError("smart rendering requires at least one segment")

    expanded: list[tuple[int, int, list[tuple[int, int]]]] = []
    for segment in normalized:
        effect_start = index.start_pts + round(segment.start / index.time_base)
        effect_end = index.start_pts + round(segment.end / index.time_base)
        if effect_end <= effect_start:
            raise SmartRenderCompatibilityError(
                "A selected range is shorter than one video timestamp interval",
                reason="range_too_short",
            )
        left_index = bisect.bisect_right(index.pts, effect_start) - 1
        if left_index < 0:
            raise SmartRenderCompatibilityError(
                "The first selected segment begins before the first random-access keyframe",
                reason="before_first_keyframe",
            )
        render_start = index.pts[left_index]
        right_index = bisect.bisect_left(index.pts, effect_end)
        render_end = index.pts[right_index] if right_index < len(index.pts) else index.end_pts
        if render_end <= render_start:
            render_end = index.end_pts
        expanded.append((render_start, render_end, [(effect_start, effect_end)]))

    merged: list[tuple[int, int, list[tuple[int, int]]]] = []
    for start, end, effects in expanded:
        if merged and start <= merged[-1][1]:
            old_start, old_end, old_effects = merged[-1]
            merged[-1] = (old_start, max(old_end, end), old_effects + effects)
        else:
            merged.append((start, end, effects))

    spans: list[SpliceSpan] = []
    cursor = index.start_pts
    for start, end, effects in merged:
        if cursor < start:
            spans.append(SpliceSpan("copy", cursor, start))
        spans.append(SpliceSpan("render", start, end, tuple(effects)))
        cursor = end
    if cursor < index.end_pts:
        spans.append(SpliceSpan("copy", cursor, index.end_pts))
    if (
        len(spans) == 1
        and spans[0].is_render
        and sum(segment.duration for segment in normalized) < duration - float(index.time_base)
    ):
        raise SmartRenderCompatibilityError(
            "The selected ranges have no usable safe video cut points, so they would "
            "require re-encoding the entire video",
            reason="whole_video_reencode",
        )
    return SplicePlan(index=index, spans=tuple(spans), segments=normalized)


def _run_ffmpeg(args: list[str], *, purpose: str) -> None:
    command = [resolve_executable("ffmpeg"), "-hide_banner", "-y", "-loglevel", "error", *args]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        **subprocess_no_window_kwargs(),
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "unknown ffmpeg error").strip()
        raise RuntimeError(f"Failed to {purpose}: {detail}")


def _seconds(value: float) -> str:
    return f"{max(0.0, float(value)):.9f}"


def create_copy_fragment(
    source: Path,
    span: SpliceSpan,
    index: KeyframeIndex,
    destination: Path,
    *,
    codec: str | None = None,
) -> None:
    if destination.exists():
        destination.unlink()
    if _canonical_codec(codec or _codec_name(source)) == "av1":
        start = index.seconds_for_pts(span.start_pts)
        duration = float((span.end_pts - span.start_pts) * index.time_base)
        args: list[str] = []
        if start > 0:
            args += ["-ss", _seconds(start)]
        args += [
            "-i", str(source),
            "-t", _seconds(duration),
            "-map", "0:v:0",
            "-an",
            "-c:v", "copy",
            "-avoid_negative_ts", "make_zero",
            "-f", "nut",
            str(destination),
        ]
        _run_ffmpeg(args, purpose=f"copy AV1 smart-render span at {start:.3f}s")
        return
    with av.open(str(source)) as src, av.open(str(destination), "w", format="nut") as dst:
        in_stream = src.streams.video[0]
        out_stream = dst.add_stream_from_template(in_stream)
        src.seek(span.start_pts, stream=in_stream, backward=True)
        for packet in src.demux(in_stream):
            if packet.pts is None or not (span.start_pts <= packet.pts < span.end_pts):
                continue
            packet.pts -= span.start_pts
            if packet.dts is not None:
                packet.dts -= span.start_pts
            packet.stream = out_stream
            dst.mux(packet)


def _codec_name(path: Path) -> str:
    with av.open(str(path)) as container:
        return container.streams.video[0].codec_context.codec.canonical_name


def normalize_fragment(source: Path, destination: Path, *, codec: str) -> None:
    bitstream_filter = {
        "h264": "h264_mp4toannexb,dump_extra=freq=keyframe",
        "hevc": "hevc_mp4toannexb,dump_extra=freq=keyframe",
        "av1": "av1_metadata=td=insert,dump_extra=freq=keyframe",
    }[codec]
    muxer = "mpegts" if codec in {"h264", "hevc"} else "matroska"
    container_args = ["-muxdelay", "0"] if muxer == "mpegts" else ["-avoid_negative_ts", "make_zero"]
    _run_ffmpeg(
        [
            "-i", str(source),
            "-map", "0:v:0",
            "-an",
            "-c:v", "copy",
            "-bsf:v", bitstream_filter,
            *container_args,
            "-f", muxer,
            str(destination),
        ],
        purpose=f"normalize smart-render fragment {source.name}",
    )


def _quote_concat_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def concatenate_fragments(
    fragments: list[tuple[Path, float]],
    *,
    manifest: Path,
    destination: Path,
    codec: str,
) -> None:
    lines = ["ffconcat version 1.0"]
    for path, duration in fragments:
        lines.append(f"file '{_quote_concat_path(path)}'")
        lines.append(f"duration {_seconds(duration)}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    muxer = "mpegts" if codec in {"h264", "hevc"} else "matroska"
    muxer_args = ["-muxdelay", "0"] if muxer == "mpegts" else []
    _run_ffmpeg(
        [
            "-f", "concat",
            "-safe", "0",
            "-i", str(manifest),
            "-map", "0:v:0",
            "-an",
            "-c:v", "copy",
            *muxer_args,
            "-f", muxer,
            str(destination),
        ],
        purpose="concatenate smart-render video fragments",
    )


def mux_final_output(
    video: Path,
    source: Path,
    destination: Path,
    *,
    codec: str,
) -> None:
    temporary = destination.with_name(f".{destination.stem}.smart-render{destination.suffix}")
    args = [
        "-i", str(video),
        "-i", str(source),
        "-map", "0:v:0",
        "-map", "1:a?",
        "-map_metadata", "1",
        "-map_metadata:s:v:0", "1:s:v:0",
        "-c:v", "copy",
    ]
    with av.open(str(source)) as container:
        audio_streams = list(container.streams.audio)
        for output_index, stream in enumerate(audio_streams):
            name = stream.codec_context.name
            if needs_audio_reencode(name, destination.suffix):
                args += [f"-c:a:{output_index}", "aac", f"-b:a:{output_index}", "256k"]
            else:
                args += [f"-c:a:{output_index}", "copy"]
    if destination.suffix.lower() in {".mp4", ".mov"}:
        tag = {"h264": "avc3", "hevc": "hev1", "av1": "av01"}[codec]
        args += ["-tag:v", tag, "-movflags", "+faststart"]
    args.append(str(temporary))
    try:
        _run_ffmpeg(args, purpose=f"mux smart-render output {destination.name}")
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            log.warning("Could not remove temporary output %s", temporary)
