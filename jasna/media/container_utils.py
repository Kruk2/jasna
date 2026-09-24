from __future__ import annotations

from collections.abc import Collection

import av
from av.codec.codec import Properties


_MOV_FORMATS = frozenset({"mov", "mp4", "m4a", "3gp", "3g2", "mj2"})
MOV_SUFFIXES = frozenset({".mp4", ".mov"})


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
