"""Software decode fallback: FFV1/non-NVDEC inputs must decode on CPU, normalize
to NV12/P010, upload YUV, and return the unchanged CUDA uint8 RGB batch contract.
Requires a CUDA GPU; fixtures are generated through PyAV (no ffmpeg CLI)."""
from __future__ import annotations

import logging
from pathlib import Path

import av
import numpy as np
import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from factories import video_metadata, yuv_to_rgb
from jasna.media.video_decoder import VideoReader
from jasna.media.yuv_to_rgb import YuvToRgbConverter

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
nvdec_only = pytest.mark.skipif(torch.version.hip is not None, reason="NVDEC selection; AMD always decodes to CPU frames and uploads them")

DEVICE = torch.device("cuda:0")


def _write_video(
    path: Path,
    codec: str,
    pix_fmt: str,
    frames: list[av.VideoFrame],
    rate: int = 12,
    options: dict | None = None,
) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=rate, options=options or {})
        stream.width = frames[0].width
        stream.height = frames[0].height
        stream.pix_fmt = pix_fmt
        for i, frame in enumerate(frames):
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def _solid_yuv420p_frame(w: int, h: int, y: int, u: int, v: int) -> av.VideoFrame:
    data = np.concatenate(
        [
            np.full((h, w), y, dtype=np.uint8).reshape(-1),
            np.full((h // 2, w // 2), u, dtype=np.uint8).reshape(-1),
            np.full((h // 2, w // 2), v, dtype=np.uint8).reshape(-1),
        ]
    ).reshape(h * 3 // 2, w)
    return av.VideoFrame.from_ndarray(data, format="yuv420p")


def _solid_yuv420p10_frame(w: int, h: int, y: int, u: int, v: int) -> av.VideoFrame:
    frame = av.VideoFrame(w, h, "yuv420p10le")
    for plane, value, rows in (
        (frame.planes[0], y, h),
        (frame.planes[1], u, h // 2),
        (frame.planes[2], v, h // 2),
    ):
        width = plane.line_size // 2
        plane.update(np.full((rows, width), value, dtype="<u2").tobytes())
    return frame


def _metadata(path: Path, w: int, h: int, n: int, *, codec_name: str = "ffv1", **overrides):
    return video_metadata(
        fps=12, num_frames=n, video_file=str(path), video_width=w, video_height=h, codec_name=codec_name, **overrides
    )


@pytest.fixture
def ffv1_video(tmp_path):
    w, h, n = 128, 96, 20
    frames = [_solid_yuv420p_frame(w, h, 120, 90, 200) for _ in range(n)]
    path = tmp_path / "soft.mkv"
    _write_video(path, "ffv1", "yuv420p", frames)
    return path, _metadata(path, w, h, n)


def _reference_rgb(y_val, u_val, v_val, w, h, color_space, full_range, ten_bit):
    shift = 64 if ten_bit else 1
    dtype = torch.uint16 if ten_bit else torch.uint8
    y = torch.full((h, w), y_val * shift, dtype=torch.int32).to(dtype)
    uv = torch.empty((h // 2, w // 2, 2), dtype=torch.int32)
    uv[..., 0] = u_val * shift
    uv[..., 1] = v_val * shift
    converter = YuvToRgbConverter(h, w, color_space, full_range, ten_bit, torch.device("cpu"))
    return yuv_to_rgb(converter, y, uv.to(dtype))


def test_ffv1_returns_cuda_uint8_rgb_batches(ffv1_video):
    path, metadata = ffv1_video
    with VideoReader(str(path), batch_size=6, device=DEVICE, metadata=metadata) as reader:
        all_pts = []
        for batch, pts in reader.frames():
            assert batch.is_cuda
            assert batch.dtype == torch.uint8
            assert batch.shape == (len(pts), 3, 96, 128)
            all_pts.extend(pts)
    assert all_pts == sorted(all_pts)
    assert len(all_pts) == 20


def test_pts_and_frame_count_match_direct_decode(ffv1_video):
    path, metadata = ffv1_video
    with av.open(str(path)) as c:
        direct_pts = [f.pts for p in c.demux(c.streams.video[0]) for f in p.decode()]
    with VideoReader(str(path), batch_size=4, device=DEVICE, metadata=metadata) as reader:
        reader_pts = [p for _, pts in reader.frames() for p in pts]
    assert reader_pts == direct_pts


@pytest.mark.parametrize(
    ("width", "height", "ten_bit", "yuv", "color_space", "color_range"),
    [
        (64, 64, False, (140, 100, 180), AvColorspace.ITU601, AvColorRange.MPEG),
        (64, 64, False, (140, 100, 180), AvColorspace.ITU601, AvColorRange.JPEG),
        (64, 64, False, (140, 100, 180), AvColorspace.ITU709, AvColorRange.MPEG),
        (64, 64, False, (140, 100, 180), AvColorspace.ITU709, AvColorRange.JPEG),
        (64, 64, False, (140, 100, 180), AvColorspace.BT2020, AvColorRange.MPEG),
        (64, 64, False, (140, 100, 180), AvColorspace.BT2020, AvColorRange.JPEG),
        # 100-px rows are narrower than the 32/64-byte swscale pitch alignment.
        (100, 50, False, (120, 90, 200), AvColorspace.ITU709, AvColorRange.MPEG),
        (64, 64, True, (700, 400, 600), AvColorspace.ITU709, AvColorRange.MPEG),
    ],
    ids=["bt601-limited", "bt601-full", "bt709-limited", "bt709-full", "bt2020-limited", "bt2020-full",
         "pitch-padding", "10-bit"],
)
def test_software_decode_matches_the_reference_conversion(tmp_path, width, height, ten_bit, yuv, color_space, color_range):
    solid_frame = _solid_yuv420p10_frame if ten_bit else _solid_yuv420p_frame
    frames = [solid_frame(width, height, *yuv) for _ in range(4)]
    path = tmp_path / "solid.mkv"
    _write_video(path, "ffv1", "yuv420p10le" if ten_bit else "yuv420p", frames)
    metadata = _metadata(path, width, height, 4, is_10bit=ten_bit, color_space=color_space, color_range=color_range)
    full_range = color_range == AvColorRange.JPEG
    expected = _reference_rgb(*yuv, width, height, color_space, full_range, ten_bit)
    with VideoReader(str(path), batch_size=4, device=DEVICE, metadata=metadata) as reader:
        batch, _ = next(iter(reader.frames()))
    assert batch.dtype == torch.uint8
    diff = (batch[0].cpu().float() - expected.float()).abs().max().item()
    assert diff <= 1.0, f"maxdiff {diff}"


def test_seek_into_software_video_lands_at_or_after_target(ffv1_video):
    path, metadata = ffv1_video
    seek_ts = 10 / 12.0
    with av.open(str(path)) as c:
        stream_tb = c.streams.video[0].time_base
        all_pts = [f.pts for p in c.demux(c.streams.video[0]) for f in p.decode()]
    target = round(seek_ts / stream_tb)
    expected_first = min(p for p in all_pts if p >= target)
    with VideoReader(str(path), batch_size=4, device=DEVICE, metadata=metadata) as reader:
        _, pts = next(iter(reader.frames(seek_ts=seek_ts)))
    assert pts[0] == expected_first


@nvdec_only
def test_software_selection_logged_once(ffv1_video, caplog):
    path, metadata = ffv1_video
    with caplog.at_level(logging.WARNING, logger="jasna.media.video_decoder"):
        with VideoReader(str(path), batch_size=4, device=DEVICE, metadata=metadata) as reader:
            for _ in reader.frames():
                pass
    fallback_logs = [r for r in caplog.records if "software decoding" in r.getMessage()]
    assert len(fallback_logs) == 1


def test_h264_444_with_b_frames_falls_back_and_keeps_all_frames(tmp_path):
    # NVDEC has no H.264 4:4:4 support; the codec advertises a CUDA config and
    # falls back after init. B-frames force delayed EOF frames through the
    # software loop.
    w, h, n = 128, 96, 24
    frames = []
    for i in range(n):
        data = np.full((h * 3 // 2, w), 128, dtype=np.uint8)
        data[:h] = (i * 10) % 256
        frames.append(av.VideoFrame.from_ndarray(data, format="yuv420p").reformat(format="yuv444p"))
    path = tmp_path / "h264_444.mp4"
    _write_video(path, "libx264", "yuv444p", frames, options={"bf": "2", "g": "12"})
    metadata = _metadata(path, w, h, n, codec_name="h264")
    with VideoReader(str(path), batch_size=5, device=DEVICE, metadata=metadata) as reader:
        all_pts = [p for _, pts in reader.frames() for p in pts]
    assert len(all_pts) == n
    assert all_pts == sorted(all_pts)


@nvdec_only
def test_hardware_input_never_enters_software_path(tmp_path, monkeypatch):
    w, h, n = 128, 96, 24
    frames = [_solid_yuv420p_frame(w, h, 120, 90, 200) for _ in range(n)]
    path = tmp_path / "hw.mp4"
    _write_video(path, "libx264", "yuv420p", frames)
    metadata = _metadata(path, w, h, n, codec_name="h264")

    def _boom(self, decoded, group):
        raise AssertionError("hardware-decodable input entered the software path")

    monkeypatch.setattr(VideoReader, "_frames_software", _boom)
    with VideoReader(str(path), batch_size=6, device=DEVICE, metadata=metadata) as reader:
        total = sum(len(pts) for _, pts in reader.frames())
    assert total == n


def _has_av1_software_encoder() -> bool:
    try:
        av.codec.Codec("libsvtav1", "w")
        return True
    except ValueError:
        return False


def _gpu_supports_av1_nvdec() -> bool:
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        return False
    major, minor = torch.cuda.get_device_capability(DEVICE)
    return major >= 10 or (major == 8 and minor in {6, 7, 9})


@pytest.mark.skipif(not _has_av1_software_encoder(), reason="libsvtav1 unavailable")
@pytest.mark.skipif(not _gpu_supports_av1_nvdec(), reason="GPU has no AV1 NVDEC")
def test_av1_decodes_on_nvdec_not_software(tmp_path, monkeypatch):
    # PyAV's default AV1 decoder (libdav1d) has no NVDEC hwaccel config, so
    # av.open decodes AV1 in software. _setup_nvdec_decoder swaps in the native
    # av1 decoder, which does, keeping AV1 on the hardware path. Dimensions stay
    # above NVDEC's AV1 minimum coded size (~128px) so the GPU path is exercised.
    w, h, n = 256, 144, 12
    frames = [_solid_yuv420p_frame(w, h, 120, 90, 200) for _ in range(n)]
    path = tmp_path / "sample.mp4"
    _write_video(path, "libsvtav1", "yuv420p", frames, options={"preset": "8", "crf": "40"})
    metadata = _metadata(path, w, h, n, codec_name="av1")

    def _boom(self, decoded, group):
        raise AssertionError("AV1 input entered the software path instead of NVDEC")

    monkeypatch.setattr(VideoReader, "_frames_software", _boom)
    with VideoReader(str(path), batch_size=6, device=DEVICE, metadata=metadata) as reader:
        batches = [(batch, pts) for batch, pts in reader.frames()]
    assert batches
    assert all(batch.is_cuda and batch.dtype == torch.uint8 for batch, _ in batches)
    assert sum(len(pts) for _, pts in batches) == n


@nvdec_only
@pytest.mark.skipif(not _has_av1_software_encoder(), reason="libsvtav1 unavailable")
def test_subminimum_av1_uses_software_decoder(tmp_path):
    w, h, n = 128, 96, 4
    frames = [_solid_yuv420p_frame(w, h, 120, 90, 200) for _ in range(n)]
    path = tmp_path / "tiny.mp4"
    _write_video(path, "libsvtav1", "yuv420p", frames, options={"preset": "8", "crf": "40"})
    metadata = _metadata(path, w, h, n, codec_name="av1")

    with VideoReader(str(path), batch_size=4, device=DEVICE, metadata=metadata) as reader:
        assert reader._decoder_ctx is None
        total = sum(len(pts) for _, pts in reader.frames())
    assert total == n
