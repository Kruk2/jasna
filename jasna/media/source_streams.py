"""Everything an encoded output carries over from its source besides the video."""
from __future__ import annotations

import logging
import math
from collections import deque
from fractions import Fraction
from pathlib import Path

import av

from jasna.media.audio_utils import needs_audio_reencode
from jasna.media.container_utils import (
    CREATION_TIME,
    is_mov_chapter_stream,
    subtitle_transcode_codec,
    with_output_creation_time,
)
from jasna.settings_tag import with_source_comment

logger = logging.getLogger(__name__)


def _normalized_audio_layout(layout: av.AudioLayout) -> av.AudioLayout:
    channels = layout.nb_channels
    if layout.name == f"{channels} channels" and channels <= 8:
        return av.AudioLayout(f"{channels}c")
    return layout


class SourceStreamCopier:
    """Copies the source's container metadata, chapters and non-video streams into ``dst``.

    Audio, subtitle and data packets are muxed by ``pump`` up to the time the caller's video
    has reached, and the rest by ``drain``. A smart-render fragment gets only the container
    metadata: the final splice brings its chapters and other streams."""

    def __init__(
        self,
        src: av.container.InputContainer,
        dst: av.container.OutputContainer,
        output_path: Path,
        *,
        video_index: int,
        creation_time: str,
        comment: str | None,
        smart_fragment: bool,
    ):
        """``comment`` replaces the source's comment tag, which it keeps as ``source_comment``."""
        self._src = src
        self._dst = dst
        self._output_path = output_path
        self._smart_fragment = smart_fragment
        self._chapters = ()
        self._copy_metadata(creation_time, comment)
        self._setup_streams(video_index, creation_time)

    def _copy_metadata(self, creation_time: str, comment: str | None) -> None:
        self._dst.metadata.update(self._src.metadata)
        self._dst.metadata[CREATION_TIME] = creation_time
        if comment is not None:
            for key in [key for key in self._dst.metadata if key.lower() == "comment"]:
                del self._dst.metadata[key]
            self._dst.metadata["comment"] = with_source_comment(comment, self._src.metadata)
            logger.info("%s settings: %s", self._output_path.name, self._dst.metadata["comment"])
        if not self._smart_fragment:
            self._chapters = self._src.chapters()
            self._dst.set_chapters(self._chapters)

    def _setup_streams(self, video_index: int, creation_time: str) -> None:
        self._pipes: dict[int, tuple[str, object, object]] = {}
        self._backlog: deque = deque()
        self._packets = None
        self._last_dts: dict[int, tuple[int, Fraction]] = {}
        self._warned_dts: set[int] = set()
        self._next_audio_pts: dict[int, int] = {}
        if self._smart_fragment:
            return

        packet_streams = []
        output_formats = set(self._dst.format.name.split(","))
        source_formats = set(self._src.format.name.split(","))
        for in_stream in self._src.streams:
            if in_stream.index == video_index:
                continue
            if is_mov_chapter_stream(
                in_stream,
                source_formats=source_formats,
                chapters=self._chapters,
            ):
                continue
            if in_stream.type == "attachment" and "matroska" not in output_formats:
                logger.warning(
                    "Skipping attachment stream %s: %s output does not support attachments",
                    in_stream.index,
                    self._output_path.suffix,
                )
                continue
            if in_stream.codec_context is None and in_stream.type != "attachment":
                logger.warning(
                    "Skipping %s stream %s: it has no copyable codec",
                    in_stream.type,
                    in_stream.index,
                )
                continue

            source_audio_layout = (
                in_stream.codec_context.layout
                if in_stream.type == "audio"
                else None
            )
            audio_layout = (
                _normalized_audio_layout(source_audio_layout)
                if source_audio_layout is not None
                else None
            )
            if in_stream.type == "audio" and needs_audio_reencode(
                in_stream.codec_context.name,
                self._output_path.suffix,
                self._dst.supported_codecs,
            ):
                logger.info(
                    "re-encoding audio %s -> aac for %s",
                    in_stream.codec_context.name,
                    self._output_path.suffix,
                )
                out_stream = self._dst.add_stream(
                    "aac", rate=in_stream.codec_context.sample_rate
                )
                out_stream.codec_context.layout = audio_layout
                out_stream.bit_rate = 256_000
                processor = av.AudioResampler(
                    format="fltp",
                    layout=audio_layout,
                    rate=in_stream.codec_context.sample_rate,
                )
                kind = "transcode"
            else:
                try:
                    out_stream = self._dst.add_stream_from_template(
                        in_stream, opaque=True
                    )
                except ValueError as exc:
                    transcode_codec = subtitle_transcode_codec(
                        in_stream.codec_context.name,
                        output_formats=output_formats,
                        supported_codecs=self._dst.supported_codecs,
                    )
                    if in_stream.type != "subtitle" or transcode_codec is None:
                        logger.warning(
                            "Skipping %s stream %s: %s",
                            in_stream.type,
                            in_stream.index,
                            exc,
                        )
                        continue
                    logger.info(
                        "re-encoding subtitle %s -> %s for %s",
                        in_stream.codec_context.name,
                        transcode_codec,
                        self._output_path.suffix,
                    )
                    out_stream = self._dst.add_stream(transcode_codec)
                    subtitle_time_base = in_stream.time_base or Fraction(1, 1_000)
                    out_stream.time_base = subtitle_time_base
                    out_stream.codec_context.time_base = subtitle_time_base
                    out_stream.codec_context.subtitle_header = b""
                    processor = None
                    kind = "subtitle_transcode"
                else:
                    if (
                        audio_layout is not None
                        and audio_layout is not source_audio_layout
                    ):
                        out_stream.codec_context.layout = audio_layout
                    processor = self._aac_copy_filter(
                        in_stream, out_stream, output_formats
                    )
                    kind = "copy"

            out_stream.metadata.update(with_output_creation_time(in_stream.metadata, creation_time))
            out_stream.disposition = in_stream.disposition
            if in_stream.type == "attachment":
                continue
            self._pipes[in_stream.index] = (kind, out_stream, processor)
            packet_streams.append(in_stream)

        if packet_streams:
            self._packets = self._src.demux(packet_streams)

    @staticmethod
    def _aac_copy_filter(in_stream, out_stream, output_formats):
        if (
            in_stream.type == "audio"
            and in_stream.codec_context.name == "aac"
            and output_formats & {"mp4", "mov"}
            and out_stream.codec_context.extradata
        ):
            return av.bitstream.BitStreamFilterContext("aac_adtstoasc", in_stream)
        return None

    def _produce_packets(self, in_packet) -> list:
        kind, out_stream, processor = self._pipes[in_packet.stream.index]
        if kind == "copy":
            return self._copy_packets(in_packet, out_stream, processor)
        if kind == "subtitle_transcode":
            if in_packet.size == 0:
                return []
            subtitle = in_packet.stream.codec_context.decode2(in_packet)
            if subtitle is None:
                return []
            packet = out_stream.codec_context.encode_subtitle(subtitle)
            output_time_base = (
                out_stream.time_base
                or out_stream.codec_context.time_base
                or in_packet.time_base
                or Fraction(1, 1_000)
            )

            def rescale(value):
                if value is None or in_packet.time_base is None:
                    return value
                return round(value * in_packet.time_base / output_time_base)

            packet.stream = out_stream
            packet.pts = rescale(in_packet.pts)
            packet.dts = rescale(
                in_packet.dts if in_packet.dts is not None else in_packet.pts
            )
            packet.duration = rescale(in_packet.duration)
            packet.time_base = output_time_base
            return [packet]
        out_packets = []
        sample_time_base = Fraction(1, out_stream.codec_context.sample_rate)
        for aframe in in_packet.decode():
            aframe.pts = self._audio_frame_pts(
                aframe, in_packet.stream.index, sample_time_base
            )
            aframe.time_base = sample_time_base
            for rframe in processor.resample(aframe):
                out_packets.extend(out_stream.encode(rframe))
        return out_packets

    def _audio_frame_pts(
        self, aframe, stream_index: int, sample_time_base: Fraction
    ) -> int | None:
        """A decoder that splits one packet into several frames stamps only the first;
        the rest follow on from the frame before."""
        if aframe.pts is None:
            pts = self._next_audio_pts.get(stream_index)
        else:
            pts = round(aframe.pts * aframe.time_base / sample_time_base)
        if pts is not None:
            self._next_audio_pts[stream_index] = pts + aframe.samples
        return pts

    @staticmethod
    def _copy_packets(in_packet, out_stream, bitstream_filter):
        if in_packet.size == 0:
            return []
        if bitstream_filter is None:
            packets = [in_packet]
        else:
            packets = bitstream_filter.filter(in_packet)
        for packet in packets:
            packet.stream = out_stream
        return packets

    def pump(self, upto_seconds: float | None) -> None:
        """Mux the source packets up to ``upto_seconds``, or all of them for ``None``."""
        if self._packets is None:
            return
        while True:
            if self._backlog:
                packet = self._backlog[0]
                ts = packet.dts if packet.dts is not None else packet.pts
                if (
                    upto_seconds is not None
                    and ts is not None
                    and packet.time_base is not None
                    and float(ts * packet.time_base) > upto_seconds
                ):
                    return
                self._backlog.popleft()
                self._mux(packet)
                continue
            in_packet = next(self._packets, None)
            if in_packet is None:
                self._packets = None
                return
            self._backlog.extend(self._produce_packets(in_packet))

    def drain(self) -> None:
        """Mux the rest of the source and flush the audio re-encoders."""
        self.pump(None)
        for kind, out_stream, processor in self._pipes.values():
            if kind != "transcode":
                continue
            packets = []
            for rframe in processor.resample(None):
                packets.extend(out_stream.encode(rframe))
            packets.extend(out_stream.encode(None))
            for packet in packets:
                self._mux(packet)

    def _mux(self, packet):
        if packet.dts is not None:
            last = self._last_dts.get(packet.stream.index)
            if last is not None:
                time_base = Fraction(packet.time_base or packet.stream.time_base)
                minimum_time = (last[0] + 1) * last[1]
                if packet.dts * time_base < minimum_time:
                    first_nudge = packet.stream.index not in self._warned_dts
                    self._warned_dts.add(packet.stream.index)
                    logger.log(
                        logging.WARNING if first_nudge else logging.DEBUG,
                        "Source DTS %s at %s overlaps last muxed DTS %s at %s in output stream %s; nudging forward",
                        packet.dts,
                        time_base,
                        last[0],
                        last[1],
                        packet.stream.index,
                    )
                    packet.dts = math.ceil(minimum_time / time_base)
                    if packet.pts is not None and packet.pts < packet.dts:
                        packet.pts = packet.dts
        self._dst.mux(packet)
        if packet.dts is not None:
            self._last_dts[packet.stream.index] = (
                packet.dts,
                Fraction(packet.time_base or packet.stream.time_base),
            )
