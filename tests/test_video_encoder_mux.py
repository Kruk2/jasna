"""Behavioral tests for the PyAV-based NvidiaVideoEncoder muxing: color tags,
audio copy vs aac fallback, metadata/disposition, faststart, pts passthrough.
Requires a CUDA GPU and an ffmpeg binary for fixture generation."""
from __future__ import annotations

import subprocess
from fractions import Fraction
from pathlib import Path

import av
import pytest
import torch

from jasna.media import get_video_meta_data
from jasna.media.audio_utils import needs_audio_reencode
from jasna.media.video_decoder import NvidiaVideoReader
from jasna.media.video_encoder import NvidiaVideoEncoder
from jasna.os_utils import resolve_executable

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEVICE = torch.device("cuda:0")


def _make_source(tmp_path: Path, name: str, acodec: str | None = "aac", extra: list[str] | None = None) -> Path:
    out = tmp_path / name
    cmd = [
        resolve_executable("ffmpeg"), "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=256x256:rate=12:duration=2",
    ]
    if acodec:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", acodec]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    cmd += extra or []
    cmd.append(str(out))
    subprocess.run(cmd, check=True)
    return out


def _transcode(src: Path, dst: Path) -> None:
    metadata = get_video_meta_data(str(src))
    with (
        NvidiaVideoReader(str(src), batch_size=4, device=DEVICE, metadata=metadata) as reader,
        NvidiaVideoEncoder(str(dst), device=DEVICE, metadata=metadata, codec="hevc", encoder_settings={}) as encoder,
    ):
        for frames, pts_list in reader.frames():
            for i, pts in enumerate(pts_list):
                encoder.encode(frames[i], pts)


def test_color_tags_and_frame_count(tmp_path):
    src = _make_source(tmp_path, "src.mp4")
    dst = tmp_path / "out.mp4"
    _transcode(src, dst)

    with av.open(str(dst)) as c:
        v = c.streams.video[0]
        assert v.codec_context.name == "hevc"
        assert int(v.codec_context.color_range) == 1  # tv/mpeg
        assert int(v.codec_context.colorspace) == 1  # bt709
        assert int(v.codec_context.color_primaries) == 1
        assert int(v.codec_context.color_trc) == 1
        n = sum(1 for _ in c.decode(v))
        assert n == 24


def test_audio_copy_when_compatible(tmp_path):
    src = _make_source(tmp_path, "src.mp4", acodec="aac")
    dst = tmp_path / "out.mp4"
    _transcode(src, dst)

    with av.open(str(dst)) as c:
        assert len(c.streams.audio) == 1
        assert c.streams.audio[0].codec_context.name == "aac"
        assert float(c.streams.audio[0].duration * c.streams.audio[0].time_base) == pytest.approx(2.0, abs=0.15)


def test_audio_reencoded_when_incompatible(tmp_path):
    assert needs_audio_reencode("vorbis", ".mp4")
    src = _make_source(tmp_path, "src.mkv", acodec="libvorbis")
    dst = tmp_path / "out.mp4"
    _transcode(src, dst)

    with av.open(str(dst)) as c:
        a = c.streams.audio[0]
        assert a.codec_context.name == "aac"
        assert float(a.duration * a.time_base) == pytest.approx(2.0, abs=0.2)


def test_container_metadata_and_audio_language_copied(tmp_path):
    src = _make_source(
        tmp_path, "src.mp4",
        extra=["-metadata", "title=jasna-test", "-metadata:s:a:0", "language=pol"],
    )
    dst = tmp_path / "out.mp4"
    _transcode(src, dst)

    with av.open(str(dst)) as c:
        assert c.metadata.get("title") == "jasna-test"
        assert c.streams.audio[0].metadata.get("language") == "pol"


def test_faststart_moov_before_mdat(tmp_path):
    src = _make_source(tmp_path, "src.mp4")
    dst = tmp_path / "out.mp4"
    _transcode(src, dst)

    data = dst.read_bytes()
    assert data.index(b"moov") < data.index(b"mdat")


def test_video_pts_deltas_passthrough(tmp_path):
    src = _make_source(tmp_path, "src.mp4")
    dst = tmp_path / "out.mp4"
    _transcode(src, dst)

    def _pts(path):
        with av.open(str(path)) as c:
            v = c.streams.video[0]
            values = sorted(p.pts for p in c.demux(v) if p.pts is not None)
            return [(p - values[0], v.time_base) for p in values]

    src_pts = _pts(src)
    dst_pts = _pts(dst)
    assert len(src_pts) == len(dst_pts)
    src_seconds = [float(p * tb) for p, tb in src_pts]
    dst_seconds = [float(p * tb) for p, tb in dst_pts]
    assert src_seconds == pytest.approx(dst_seconds, abs=1e-6)


def test_av_skew_preserved(tmp_path):
    src = _make_source(tmp_path, "src.mp4")
    dst = tmp_path / "out.mp4"
    _transcode(src, dst)

    def _skew(path):
        with av.open(str(path)) as c:
            v, a = c.streams.video[0], c.streams.audio[0]
            v_start = float((v.start_time or 0) * v.time_base)
            a_start = float((a.start_time or 0) * a.time_base)
            return v_start - a_start

    assert _skew(dst) == pytest.approx(_skew(src), abs=0.05)


def test_zero_frame_job_closes_cleanly(tmp_path):
    src = _make_source(tmp_path, "src.mp4")
    metadata = get_video_meta_data(str(src))
    dst = tmp_path / "out.mp4"

    with NvidiaVideoEncoder(str(dst), device=DEVICE, metadata=metadata, codec="hevc", encoder_settings={}):
        pass

    # constructing without entering must not leak containers or threads
    NvidiaVideoEncoder(str(dst), device=DEVICE, metadata=metadata, codec="hevc", encoder_settings={})


def test_mkv_output_supported(tmp_path):
    src = _make_source(tmp_path, "src.mp4")
    dst = tmp_path / "out.mkv"
    _transcode(src, dst)

    with av.open(str(dst)) as c:
        assert c.streams.video[0].codec_context.name == "hevc"
        assert c.streams.video[0].average_rate == Fraction(12, 1)
        assert c.streams.audio[0].codec_context.name == "aac"
