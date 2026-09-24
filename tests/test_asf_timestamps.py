"""ASF stores only decode timestamps; FFmpeg filled VC-1 pts from the next
packet, starting video one frame late."""
from __future__ import annotations

import io
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.media.container_utils import demux_video
from jasna.media.probe import VideoMetadata
from jasna.media.video_decoder import VideoReader

FRAME_MS = 40
FRAME_COUNT = 12
# Minimal 16x16 VC-1 Advanced elementary stream: sequence header, entry point,
# then I and P pictures that FFmpeg's VC-1 decoder accepts.
_VC1_HEADERS = bytes.fromhex("0000010fca0000700708800000010e4002")
_VC1_I_FRAME = bytes.fromhex("0000010dc52bb8569d")
_VC1_P_FRAME = bytes.fromhex("0000010d361251dcc9")


def _write_vc1_asf(path: Path) -> None:
    elementary = _VC1_HEADERS + _VC1_I_FRAME + _VC1_P_FRAME * (FRAME_COUNT - 1)
    with av.open(io.BytesIO(elementary), format="vc1") as src, av.open(str(path), "w", format="asf") as dst:
        stream = dst.add_stream_from_template(src.streams.video[0], opaque=True)
        packets = (packet for packet in src.demux() if packet.size)
        for index, packet in enumerate(packets):
            packet.time_base = Fraction(1, 1000)
            packet.pts = packet.dts = index * FRAME_MS
            packet.stream = stream
            dst.mux(packet)


def _write_mpeg4_b_frame_asf(path: Path) -> None:
    with av.open(str(path), "w", format="asf") as container:
        stream = container.add_stream("mpeg4", rate=25, options={"bf": "2"})
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        for index in range(FRAME_COUNT):
            data = np.full((96, 64), (index * 20) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(data, format="yuv420p")
            frame.pts = index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def _decoded_pts(path: Path, packets) -> list[int]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return [frame.pts for packet in packets(container, stream) for frame in packet.decode()]


def _plain_demux(container, stream):
    return container.demux(stream)


def test_vc1_asf_frames_keep_their_own_timestamps(tmp_path):
    path = tmp_path / "vc1.wmv"
    _write_vc1_asf(path)

    assert _decoded_pts(path, demux_video) == [i * FRAME_MS for i in range(FRAME_COUNT)]


def test_b_frame_asf_from_other_codecs_keeps_ffmpeg_timestamps(tmp_path):
    path = tmp_path / "mpeg4.asf"
    _write_mpeg4_b_frame_asf(path)

    assert _decoded_pts(path, demux_video) == _decoded_pts(path, _plain_demux)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_video_reader_starts_vc1_asf_at_first_timestamp(tmp_path, monkeypatch):
    monkeypatch.setenv("JASNA_DECODE_BACKEND", "pyav-sw")
    path = tmp_path / "vc1.wmv"
    _write_vc1_asf(path)
    metadata = VideoMetadata(
        video_file=str(path),
        video_height=16,
        video_width=16,
        video_fps=25.0,
        average_fps=25.0,
        video_fps_exact=Fraction(25),
        codec_name="vc1",
        duration=FRAME_COUNT / 25,
        time_base=Fraction(1, 1000),
        start_pts=0,
        color_range=AvColorRange.MPEG,
        color_space=AvColorspace.ITU601,
        num_frames=FRAME_COUNT,
        is_10bit=False,
    )
    with VideoReader(str(path), 4, torch.device("cuda:0"), metadata) as reader:
        pts = [p for _, batch_pts in reader.frames() for p in batch_pts]

    assert pts == [i * FRAME_MS for i in range(FRAME_COUNT)]
