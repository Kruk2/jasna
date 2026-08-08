"""Behavioral tests for the PyAV-based NvidiaVideoEncoder muxing: color tags,
audio copy vs aac fallback, metadata/disposition, faststart, pts passthrough.
Requires a CUDA GPU and an ffmpeg binary for fixture generation."""
from __future__ import annotations

import os
import subprocess
import sys
import wave
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
import torch

from jasna.media import get_video_meta_data
from jasna.media.audio_utils import needs_audio_reencode
from jasna.media.splice import probe_keyframes
from jasna.media.video_decoder import NvidiaVideoReader
from jasna.media.video_encoder import NvidiaVideoEncoder
from jasna.os_utils import resolve_executable

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEVICE = torch.device("cuda:0")


_EXPECTED_STREAM = {
    "hevc": ("hevc", "Main 10", "yuv420p10le"),
    "h264": ("h264", "High", "yuv420p"),
    "av1": ("av1", "Main", "yuv420p10le"),
}


def _av1_probe(tmp_path_factory) -> str | None:
    """One minimal av1_nvenc open; returns the failure text when AV1 NVENC is missing."""
    import av as _av

    tmp = tmp_path_factory.mktemp("av1probe")
    src = _make_source(tmp, "probe_src.mp4", acodec=None)
    metadata = get_video_meta_data(str(src))
    frame = torch.zeros((3, 256, 256), dtype=torch.uint8, device=DEVICE)
    try:
        with NvidiaVideoEncoder(
            str(tmp / "probe.mp4"), device=DEVICE, metadata=metadata, codec="av1", encoder_settings={}
        ) as enc:
            for i in range(8):
                enc.encode(frame.clone(), i * 512)
    except RuntimeError as exc:
        if "Failed to open av1 encoder (av1_nvenc)" in str(exc):
            return str(exc)
        raise
    return None


@pytest.fixture(scope="session")
def av1_capability(tmp_path_factory):
    failure = _av1_probe(tmp_path_factory)
    if failure is not None:
        pytest.skip(f"AV1 NVENC unavailable on this GPU: {failure}")


@pytest.fixture
def require_codec(request):
    def _require(codec: str):
        if codec == "av1":
            request.getfixturevalue("av1_capability")

    return _require


def _make_source(
    tmp_path: Path,
    name: str,
    acodec: str | None = "aac",
    extra: list[str] | None = None,
    *,
    rate: str = "12",
    duration: float = 2,
) -> Path:
    out = tmp_path / name
    cmd = [
        resolve_executable("ffmpeg"), "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc2=size=256x256:rate={rate}:duration={duration}",
    ]
    if acodec:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}", "-c:a", acodec]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    cmd += extra or []
    cmd.append(str(out))
    subprocess.run(cmd, check=True)
    return out


def _make_count_only_stereo_pcm_source(tmp_path: Path) -> Path:
    duration = 0.5
    sample_rate = 48_000
    audio = tmp_path / "count-only-stereo.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(int(duration * sample_rate) * 2 * 2))

    source = tmp_path / "count-only-stereo.mkv"
    subprocess.run(
        [
            resolve_executable("ffmpeg"),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=256x256:rate=12:duration={duration}",
            "-guess_layout_max",
            "0",
            "-i",
            str(audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            str(source),
        ],
        check=True,
    )
    return source


def _make_rich_source(tmp_path: Path) -> Path:
    base = _make_source(tmp_path, "rich-base.mkv")
    subtitle = tmp_path / "rich-subtitle.srt"
    subtitle.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nOpening subtitle\n",
        encoding="utf-8",
    )
    attachment = tmp_path / "rich-font.txt"
    attachment.write_bytes(b"font payload")
    ffmetadata = tmp_path / "rich-chapters.ffmeta"
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
    source = tmp_path / "rich-source.mkv"
    cmd = [
        resolve_executable("ffmpeg"), "-y", "-loglevel", "error",
        "-i", str(base),
        "-f", "ffmetadata", "-i", str(ffmetadata),
        "-i", str(subtitle),
        "-map", "0:v:0",
        "-map", "0:a:0",
        "-map", "2:s:0",
        "-map_metadata", "1",
        "-map_chapters", "1",
        "-metadata:s:v:0", "language=jpn",
        "-metadata:s:s:0", "language=pol",
        "-metadata:s:s:0", "title=Signs",
        "-disposition:s:0", "default+forced",
        "-c", "copy",
        "-c:s", "srt",
        "-attach", str(attachment),
        "-metadata:s:t:0", "mimetype=text/plain",
        str(source),
    ]
    subprocess.run(cmd, check=True)
    return source


def _transcode(src: Path, dst: Path, codec: str = "hevc") -> None:
    metadata = get_video_meta_data(str(src))
    with (
        NvidiaVideoReader(str(src), batch_size=4, device=DEVICE, metadata=metadata) as reader,
        NvidiaVideoEncoder(str(dst), device=DEVICE, metadata=metadata, codec=codec, encoder_settings={}) as encoder,
    ):
        for frames, pts_list in reader.frames():
            for i, pts in enumerate(pts_list):
                encoder.encode(frames[i], pts)


def _run_unaligned_pitch_probe(src: str, dst: str, codec: str, width: int) -> None:
    metadata = replace(
        get_video_meta_data(src),
        video_width=width,
        video_height=480,
        num_frames=12,
    )
    frame = _gradient_frame(0, 480, width)
    with NvidiaVideoEncoder(
        dst,
        device=DEVICE,
        metadata=metadata,
        codec=codec,
        encoder_settings={},
        mux_audio=False,
    ) as encoder:
        for index in range(12):
            encoder.encode(frame.clone(), index * 512)


@pytest.mark.parametrize("width", [852, 854, 860])
@pytest.mark.parametrize("codec", ["hevc", "h264", "av1"])
def test_unaligned_pitch_encodes_in_isolated_process(
    tmp_path,
    codec,
    width,
    require_codec,
):
    require_codec(codec)
    src = _make_source(tmp_path, "pitch-src.mp4", acodec=None)
    dst = tmp_path / f"pitch-{codec}-{width}.mp4"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "pitch-probe",
        str(src),
        str(dst),
        codec,
        str(width),
    ]
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    with av.open(str(dst)) as container:
        video = container.streams.video[0]
        assert video.width == width
        assert video.height == 480
        frames = list(container.decode(video))
        assert len(frames) == 12
        rgb = frames[0].to_ndarray(format="rgb24")
        assert rgb[:, -width // 4 :, 0].mean() > rgb[:, : width // 4, 0].mean() + 100
        assert rgb[-120:, :, 1].mean() > rgb[:120, :, 1].mean() + 100


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


def test_smart_fragment_keeps_periodic_random_access_points(tmp_path):
    src = _make_source(tmp_path, "smart-src.mp4", acodec=None, duration=3)
    metadata = get_video_meta_data(str(src))
    dst = tmp_path / "smart-part.nut"

    with (
        NvidiaVideoReader(str(src), batch_size=4, device=DEVICE, metadata=metadata) as reader,
        NvidiaVideoEncoder(
            str(dst),
            device=DEVICE,
            metadata=metadata,
            codec="hevc",
            encoder_settings={"g": "12"},
            smart_fragment=True,
            mux_audio=False,
        ) as encoder,
    ):
        for frames, pts_list in reader.frames():
            for index, pts in enumerate(pts_list):
                encoder.encode(frames[index], pts)

    output_metadata = get_video_meta_data(str(dst))
    keyframes = probe_keyframes(dst, output_metadata)

    assert len(keyframes.pts) >= 3
    with av.open(str(dst)) as container:
        stream = container.streams.video[0]
        assert any(
            packet.pts is not None
            and packet.dts is not None
            and packet.pts != packet.dts
            for packet in container.demux(stream)
        )


@pytest.mark.parametrize(
    ("source_rate", "target_rate", "expected_frames"),
    [
        ("60", Fraction(30, 1), 60),
        ("60000/1001", Fraction(30_000, 1_001), 60),
    ],
)
def test_half_rate_output_preserves_duration_and_audio_sync(
    tmp_path,
    source_rate,
    target_rate,
    expected_frames,
):
    src = _make_source(tmp_path, "src.mp4", rate=source_rate)
    metadata = get_video_meta_data(str(src))
    dst = tmp_path / "out.mp4"

    with (
        NvidiaVideoReader(
            str(src),
            batch_size=8,
            device=DEVICE,
            metadata=metadata,
            frame_stride=2,
        ) as reader,
        NvidiaVideoEncoder(
            str(dst),
            device=DEVICE,
            metadata=metadata,
            codec="hevc",
            encoder_settings={},
            output_fps=target_rate,
        ) as encoder,
    ):
        for frames, pts_list in reader.frames():
            for i, pts in enumerate(pts_list):
                encoder.encode(frames[i], pts)

    with av.open(str(dst)) as container:
        video = container.streams.video[0]
        audio = container.streams.audio[0]
        frames = list(container.decode(video))
        assert video.average_rate == target_rate
        assert len(frames) == expected_frames
        assert float(video.duration * video.time_base) == pytest.approx(2.0, abs=0.05)
        assert float(audio.duration * audio.time_base) == pytest.approx(2.0, abs=0.05)
        frame_seconds = [float(frame.pts * frame.time_base) for frame in frames]
        assert frame_seconds[1] - frame_seconds[0] == pytest.approx(
            float(1 / target_rate),
            abs=float(video.time_base),
        )

    def stream_start_skew(path: Path) -> float:
        with av.open(str(path)) as container:
            video = container.streams.video[0]
            audio = container.streams.audio[0]
            video_start = float((video.start_time or 0) * video.time_base)
            audio_start = float((audio.start_time or 0) * audio.time_base)
            return video_start - audio_start

    assert stream_start_skew(dst) == pytest.approx(stream_start_skew(src), abs=0.05)


def test_anamorphic_sar_preserved(tmp_path):
    src = _make_source(tmp_path, "src.mp4", extra=["-vf", "setsar=8/9"])
    metadata = get_video_meta_data(str(src))
    assert metadata.sample_aspect_ratio == Fraction(8, 9)

    dst = tmp_path / "out.mp4"
    _transcode(src, dst)

    with av.open(str(dst)) as c:
        assert c.streams.video[0].sample_aspect_ratio == Fraction(8, 9)


def test_square_pixels_keep_default_sar(tmp_path):
    src = _make_source(tmp_path, "src.mp4")
    dst = tmp_path / "out.mp4"
    _transcode(src, dst)

    with av.open(str(dst)) as c:
        sar = c.streams.video[0].sample_aspect_ratio
        assert sar is None or sar == Fraction(1, 1)


def test_audio_copy_when_compatible(tmp_path):
    src = _make_source(tmp_path, "src.mp4", acodec="aac")
    dst = tmp_path / "out.mp4"
    _transcode(src, dst)

    with av.open(str(dst)) as c:
        assert len(c.streams.audio) == 1
        assert c.streams.audio[0].codec_context.name == "aac"
        assert float(c.streams.audio[0].duration * c.streams.audio[0].time_base) == pytest.approx(2.0, abs=0.15)


def test_count_only_stereo_pcm_copies_to_mp4_with_explicit_layout(tmp_path):
    src = _make_count_only_stereo_pcm_source(tmp_path)
    dst = tmp_path / "out.mp4"

    with av.open(str(src)) as container:
        audio = container.streams.audio[0]
        assert audio.codec_context.layout.name == "2 channels"
        source_samples = np.concatenate(
            [frame.to_ndarray() for frame in container.decode(audio)], axis=1
        )

    _transcode(src, dst, codec="h264")

    with av.open(str(dst)) as container:
        audio = container.streams.audio[0]
        assert audio.codec_context.name == "pcm_s16le"
        assert audio.codec_context.layout.name == "stereo"
        output_samples = np.concatenate(
            [frame.to_ndarray() for frame in container.decode(audio)], axis=1
        )

    assert np.array_equal(output_samples, source_samples)
    data = dst.read_bytes()
    assert data.index(b"moov") < data.index(b"mdat")


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


@pytest.mark.parametrize(
    ("suffix", "expected_subtitle_codec", "expected_attachments"),
    [(".mkv", "srt", 1), (".mp4", "mov_text", 0)],
)
def test_container_structure_preserved(
    tmp_path,
    suffix,
    expected_subtitle_codec,
    expected_attachments,
):
    src = _make_rich_source(tmp_path)
    dst = tmp_path / f"rich-output{suffix}"

    _transcode(src, dst)

    with av.open(str(dst)) as container:
        assert container.metadata["title"] == "Source title"
        assert [chapter["metadata"]["title"] for chapter in container.chapters()] == [
            "Opening",
            "Main",
        ]
        assert container.streams.video[0].metadata["language"] == "jpn"
        assert len(container.streams.audio) == 1
        assert len(container.streams.subtitles) == 1
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
        assert len(container.streams.attachments) == expected_attachments
        if expected_attachments:
            output_attachment = container.streams.attachments[0]
            assert output_attachment.name == "rich-font.txt"
            assert output_attachment.mimetype == "text/plain"
            assert output_attachment.data == b"font payload"


def test_faststart_moov_before_mdat(tmp_path):
    src = _make_source(tmp_path, "src.mp4")
    dst = tmp_path / "out.mp4"
    _transcode(src, dst)

    data = dst.read_bytes()
    assert data.index(b"moov") < data.index(b"mdat")
    assert b"moof" not in data


def _transcode_fragmented(src: Path, dst: Path, gop: int) -> None:
    metadata = get_video_meta_data(str(src))
    with (
        NvidiaVideoReader(str(src), batch_size=4, device=DEVICE, metadata=metadata) as reader,
        NvidiaVideoEncoder(
            str(dst),
            device=DEVICE,
            metadata=metadata,
            codec="hevc",
            encoder_settings={"g": str(gop)},
            fmp4=True,
        ) as encoder,
    ):
        for frames, pts_list in reader.frames():
            for i, pts in enumerate(pts_list):
                encoder.encode(frames[i], pts)


def _box_offsets(data: bytes, box: bytes) -> list[int]:
    offsets = []
    start = data.find(box)
    while start != -1:
        offsets.append(start - 4)
        start = data.find(box, start + 1)
    return offsets


def test_fmp4_writes_moov_up_front_then_fragments(tmp_path):
    src = _make_source(tmp_path, "src.mp4")
    dst = tmp_path / "out.mp4"
    _transcode_fragmented(src, dst, gop=8)

    data = dst.read_bytes()
    assert data.index(b"moov") < data.index(b"moof")
    assert len(_box_offsets(data, b"moof")) > 1


def test_fmp4_truncated_output_still_decodes(tmp_path):
    src = _make_source(tmp_path, "src.mp4")
    dst = tmp_path / "out.mp4"
    _transcode_fragmented(src, dst, gop=8)

    data = dst.read_bytes()
    partial = tmp_path / "partial.mp4"
    partial.write_bytes(data[: _box_offsets(data, b"moof")[-1]])

    with av.open(str(partial)) as container:
        decoded = sum(1 for _ in container.decode(video=0))
    assert decoded > 0


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


def _gradient_frame(i: int, h: int, w: int) -> torch.Tensor:
    row = torch.linspace(0, 255, w, device=DEVICE)
    col = torch.linspace(0, 255, h, device=DEVICE).unsqueeze(1)
    frame = torch.empty((3, h, w), device=DEVICE)
    frame[0] = row.expand(h, w)
    frame[1] = col.expand(h, w)
    frame[2] = float((i * 9) % 256)
    return frame.to(torch.uint8)


_VFR_PTS_STEPS = [512, 512, 700, 300, 512, 1024, 256, 512, 512, 900, 400, 512]


def _encode_synthetic_vfr(tmp_path: Path, codec: str, suffix: str) -> tuple[Path, list[int], Fraction]:
    src = _make_source(tmp_path, "vfr_meta_src.mp4", acodec=None)
    metadata = get_video_meta_data(str(src))
    h, w = metadata.video_height, metadata.video_width
    dst = tmp_path / f"vfr_{codec}{suffix}"
    pts_list = []
    pts = 0
    for step in _VFR_PTS_STEPS + _VFR_PTS_STEPS:
        pts_list.append(pts)
        pts += step
    with NvidiaVideoEncoder(str(dst), device=DEVICE, metadata=metadata, codec=codec, encoder_settings={}) as enc:
        assert enc._cuda_ctx.cuda_stream == enc.stream.cuda_stream
        for i, p in enumerate(pts_list):
            enc.encode(_gradient_frame(i, h, w), p)
    return dst, pts_list, metadata.time_base


@pytest.mark.parametrize("suffix", [".mp4", ".mkv"])
@pytest.mark.parametrize("codec", ["hevc", "h264", "av1"])
def test_codec_smoke_matrix(tmp_path, codec, suffix, require_codec):
    require_codec(codec)
    dst, in_pts, src_tb = _encode_synthetic_vfr(tmp_path, codec, suffix)
    name, profile, pix_fmt = _EXPECTED_STREAM[codec]

    with av.open(str(dst)) as c:
        v = c.streams.video[0]
        ctx = v.codec_context
        assert ctx.codec.canonical_name == name
        assert ctx.pix_fmt == pix_fmt
        assert int(ctx.color_range) == 1  # tv/mpeg
        assert int(ctx.colorspace) == 1  # bt709
        assert int(ctx.color_primaries) == 1
        assert int(ctx.color_trc) == 1
        assert ctx.profile == profile

        frames = [f for p in c.demux(v) for f in p.decode()]
        assert len(frames) == len(in_pts)

        out_pts = sorted(f.pts for f in frames)
        out_seconds = [float((p - out_pts[0]) * v.time_base) for p in out_pts]
        in_seconds = [float((p - in_pts[0]) * src_tb) for p in in_pts]
        # The container may quantize pts to its own time base (1 ms for MKV).
        tolerance = float(v.time_base) + 1e-6
        for a, b in zip(out_seconds, in_seconds):
            assert a == pytest.approx(b, abs=tolerance)

    # bf=4/b_ref_mode=middle defaults must yield B-frames: reordering shows up
    # as pts != dts on at least one packet. AV1 hides reordering behind
    # show_existing_frame, so its packets stay in presentation order.
    if codec != "av1":
        with av.open(str(dst)) as c:
            v = c.streams.video[0]
            assert any(
                p.pts is not None and p.dts is not None and p.pts != p.dts
                for p in c.demux(v)
            )

    if suffix == ".mp4":
        data = dst.read_bytes()
        assert data.index(b"moov") < data.index(b"mdat")


@pytest.mark.parametrize("codec", ["hevc", "h264", "av1"])
def test_codec_round_trip_pixels(tmp_path, codec, require_codec):
    require_codec(codec)
    dst, in_pts, _ = _encode_synthetic_vfr(tmp_path, codec, ".mp4")
    metadata = get_video_meta_data(str(dst))
    src = _make_source(tmp_path, "rt_ref_src.mp4", acodec=None)
    ref_meta = get_video_meta_data(str(src))
    h, w = ref_meta.video_height, ref_meta.video_width

    with NvidiaVideoReader(str(dst), batch_size=8, device=DEVICE, metadata=metadata) as reader:
        batch, _ = next(iter(reader.frames()))
    decoded = batch[0].float()
    expected = _gradient_frame(0, h, w).float()
    assert (decoded - expected).abs().mean().item() < 4.0


@pytest.mark.parametrize("codec", ["h264", "av1"])
def test_shared_mux_path_for_new_codecs(tmp_path, codec, require_codec):
    require_codec(codec)
    src = _make_source(
        tmp_path, "src.mp4",
        extra=["-metadata", "title=jasna-test", "-metadata:s:a:0", "language=pol"],
    )
    dst = tmp_path / "out.mp4"
    _transcode(src, dst, codec=codec)

    with av.open(str(dst)) as c:
        assert c.streams.video[0].codec_context.codec.canonical_name == _EXPECTED_STREAM[codec][0]
        a = c.streams.audio[0]
        assert a.codec_context.name == "aac"  # compatible aac is copied
        assert float(a.duration * a.time_base) == pytest.approx(2.0, abs=0.15)
        assert a.metadata.get("language") == "pol"
        assert c.metadata.get("title") == "jasna-test"
        n = sum(1 for _ in c.decode(c.streams.video[0]))
        assert n == 24  # delayed video/audio packets drained

    data = dst.read_bytes()
    assert data.index(b"moov") < data.index(b"mdat")


@pytest.mark.parametrize("codec", ["h264", "av1"])
def test_audio_transcode_for_new_codecs(tmp_path, codec, require_codec):
    require_codec(codec)
    src = _make_source(tmp_path, "src.mkv", acodec="libvorbis")
    dst = tmp_path / "out.mp4"
    _transcode(src, dst, codec=codec)

    with av.open(str(dst)) as c:
        a = c.streams.audio[0]
        assert a.codec_context.name == "aac"
        assert float(a.duration * a.time_base) == pytest.approx(2.0, abs=0.2)


if __name__ == "__main__" and len(sys.argv) == 6 and sys.argv[1] == "pitch-probe":
    _run_unaligned_pitch_probe(
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
        int(sys.argv[5]),
    )
