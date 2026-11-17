from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from typing import Literal

import av
from av.codec.codec import Properties

from jasna.media.audio_utils import needs_audio_reencode

logger = logging.getLogger(__name__)

_MOV_FORMATS = frozenset({"mov", "mp4", "m4a", "3gp", "3g2", "mj2"})
MOV_SUFFIXES = frozenset({".mp4", ".mov"})
CREATION_TIME = "creation_time"


@dataclass(frozen=True)
class StreamAction:
    """What an output does with one source stream: ``copy`` it (``copy_adts_to_asc``: through
    FFmpeg's ``aac_adtstoasc`` filter), ``encode`` it to ``codec``, or ``skip`` it for ``reason``."""

    stream: av.stream.Stream
    kind: Literal["copy", "copy_adts_to_asc", "encode", "skip"]
    codec: str | None = None
    reason: str | None = None


def source_stream_plan(
    source: av.container.InputContainer,
    *,
    video_index: int,
    output_format: str,
    output_suffix: str,
) -> list[StreamAction]:
    """The action for every stream of ``source`` an ``output_format`` output carries besides the
    video at ``video_index``. A MOV chapter track is left out: the muxer rebuilds it from the chapters."""
    source_formats = set(source.format.name.split(","))
    chapters = source.chapters()
    # A copy needs what PyAV's add_stream_from_template accepts, which is wider than
    # supported_codecs for muxers that cannot say (MPEG-TS), so ask a throwaway output.
    with av.open(BytesIO(), "w", format=output_format) as probe:
        return [
            _stream_action(stream, probe, output_suffix)
            for stream in source.streams
            if stream.index != video_index
            and not is_mov_chapter_stream(stream, source_formats=source_formats, chapters=chapters)
        ]


def _stream_action(
    stream: av.stream.Stream,
    probe: av.container.OutputContainer,
    output_suffix: str,
) -> StreamAction:
    output_formats = set(probe.format.name.split(","))
    if stream.type == "attachment" and "matroska" not in output_formats:
        return StreamAction(stream, "skip", reason=f"{output_suffix} output does not support attachments")
    if stream.codec_context is None and stream.type != "attachment":
        return StreamAction(stream, "skip", reason="it has no copyable codec")
    if stream.type == "audio" and needs_audio_reencode(
        stream.codec_context.name, output_suffix, probe.supported_codecs
    ):
        return StreamAction(stream, "encode", codec="aac")
    try:
        probe.add_stream_from_template(stream, opaque=True)
    except ValueError as exc:
        transcode_codec = subtitle_transcode_codec(
            stream.codec_context.name,
            output_formats=output_formats,
            supported_codecs=probe.supported_codecs,
        )
        if stream.type != "subtitle" or transcode_codec is None:
            return StreamAction(stream, "skip", reason=str(exc))
        return StreamAction(stream, "encode", codec=transcode_codec)
    if (
        stream.type == "audio"
        and stream.codec_context.name == "aac"
        and output_formats & {"mp4", "mov"}
        and stream.codec_context.extradata
    ):
        return StreamAction(stream, "copy_adts_to_asc")
    return StreamAction(stream, "copy")


def log_stream_plan(plan: list[StreamAction], output_suffix: str) -> None:
    for action in plan:
        if action.kind == "skip":
            logger.warning("Skipping %s stream %s: %s", action.stream.type, action.stream.index, action.reason)
        elif action.kind == "encode":
            logger.info(
                "re-encoding %s %s -> %s for %s",
                action.stream.type,
                action.stream.codec_context.name,
                action.codec,
                output_suffix,
            )


def is_mov_chapter_stream(
    stream,
    *,
    source_formats: Collection[str],
    chapters: Collection[object],
) -> bool:
    if (
        not chapters
        or not set(source_formats) & _MOV_FORMATS
        or stream.type != "data"
        or stream.codec_context is not None
        or getattr(stream, "name", None) != "bin_data"
    ):
        return False
    handler = stream.metadata.get("handler_name", "").casefold()
    return "subtitle" in handler or "chapter" in handler


def output_creation_time(processed_at: datetime) -> str:
    """The ``creation_time`` an output claims: when Jasna wrote it, not when the source was
    made, so restored files sort by processing date. FFmpeg's form: UTC ISO 8601, microseconds, Z."""
    return processed_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def with_output_creation_time(source_tags: Mapping[str, str], creation_time: str) -> dict[str, str]:
    """``source_tags`` copied to an output stream, a ``creation_time`` among them replaced."""
    return {key: creation_time if key == CREATION_TIME else value for key, value in source_tags.items()}


def subtitle_transcode_codec(
    codec_name: str | None,
    *,
    output_formats: Collection[str],
    supported_codecs: Collection[str],
) -> str | None:
    if not codec_name or codec_name in supported_codecs:
        return None
    try:
        codec = av.Codec(codec_name, "r")
    except ValueError:
        return None
    if not codec.properties & Properties.TEXT_SUB.value:
        return None
    if "webm" in output_formats:
        candidate = "webvtt"
    elif set(output_formats) & {"mp4", "mov"}:
        candidate = "mov_text"
    elif "matroska" in output_formats:
        candidate = "ass"
    else:
        return None
    return candidate if candidate in supported_codecs else None


def demux_video(container, stream):
    # ASF video packets carry only a decode timestamp. VC-1 Advanced cannot
    # signal whether it uses B-frames, so FFmpeg assumes a one-frame delay and
    # fills each pts with the next packet's dts: video would start one frame
    # late. Windows Media encoders write each frame's own time, as VALI reads it.
    restamp = container.format.name == "asf" and stream.codec_context.name == "vc1"
    for packet in container.demux(stream):
        if restamp:
            packet.pts = packet.dts
        yield packet


def keyframe_seek_pts(container, stream, target_pts: int) -> int:
    """The timestamp to seek ``stream`` to so demuxing starts at a keyframe shown at or before
    ``target_pts``. Most demuxers land a seek on such a keyframe. MPEG-TS seeks by decode time
    and takes every packet for a keyframe, so its seek lands mid-GOP, past a keyframe whose
    B-frames delay its pts; there, step back until a keyframe shows up and seek to its decode time."""
    container.seek(target_pts, stream=stream, backward=True)
    landing = next(demux_video(container, stream), None)
    if landing is None or landing.is_keyframe:
        return target_pts
    step = round(1 / stream.time_base)
    while True:
        seek_pts = target_pts - step
        container.seek(seek_pts, stream=stream, backward=True)
        keyframe_dts = None
        landing_dts = None
        for packet in demux_video(container, stream):
            if packet.dts is None:
                continue
            if landing_dts is None:
                landing_dts = packet.dts
            if packet.dts > target_pts:
                break
            if packet.is_keyframe and packet.pts is not None and packet.pts <= target_pts:
                keyframe_dts = packet.dts
        if keyframe_dts is not None:
            return keyframe_dts
        # A seek before the first packet lands on it: the whole start was searched.
        if landing_dts is None or landing_dts > seek_pts:
            return seek_pts
        step *= 2


def idle_decoder_threads(codec_context: av.CodecContext) -> None:
    """Let the decoder's worker threads finish before its context is freed. PyAV frees a context
    while holding the GIL; with FFmpeg logging on (``log_ffmpeg_errors``) a worker that logs at
    that moment waits for the GIL, and the free waits for the worker."""
    if codec_context.is_open:
        codec_context.flush_buffers()
