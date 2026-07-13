"""Software decode fallback: FFV1/non-NVDEC inputs must decode on CPU, normalize
to NV12/P010, upload YUV, and return the unchanged CUDA uint8 RGB batch contract.
Requires a CUDA GPU; fixtures are generated through PyAV (no ffmpeg CLI)."""
from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
import torch
from av.codec.hwaccel import HWAccel
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.media import VideoMetadata
from jasna.media.video_decoder import NvidiaVideoReader
from jasna.media.yuv_to_rgb import YuvToRgbConverter

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

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


def _metadata(
    path: Path,
    w: int,
    h: int,
    n: int,
    *,
    codec_name: str = "ffv1",
    is_10bit: bool = False,
    color_space=AvColorspace.ITU709,
    color_range=AvColorRange.MPEG,
) -> VideoMetadata:
    return VideoMetadata(
        video_file=str(path),
        video_height=h,
        video_width=w,
        video_fps=12.0,
        average_fps=12.0,
        video_fps_exact=Fraction(12, 1),
        codec_name=codec_name,
        duration=n / 12.0,
        time_base=Fraction(1, 12),
        start_pts=0,
        color_range=color_range,
        color_space=color_space,
        num_frames=n,
        is_10bit=is_10bit,
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
    return converter.convert(y, uv.to(dtype))


def test_fallback_off_rejects_software_only_codec(ffv1_video):
    path, _ = ffv1_video
    torch.cuda.current_stream(DEVICE)
    hwaccel = HWAccel("cuda", device="0", allow_software_fallback=False, is_hw_owned=True)
    hwaccel.options["primary_ctx"] = "0"
    hwaccel.options["current_ctx"] = "1"
    with pytest.raises(RuntimeError, match="no stream is compatible"):
        av.open(str(path), hwaccel=hwaccel)


def test_ffv1_returns_cuda_uint8_rgb_batches(ffv1_video):
    path, metadata = ffv1_video
    with NvidiaVideoReader(str(path), batch_size=6, device=DEVICE, metadata=metadata) as reader:
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
    with NvidiaVideoReader(str(path), batch_size=4, device=DEVICE, metadata=metadata) as reader:
        reader_pts = [p for _, pts in reader.frames() for p in pts]
    assert reader_pts == direct_pts


def test_software_pixel_values_match_reference(ffv1_video):
    path, metadata = ffv1_video
    expected = _reference_rgb(120, 90, 200, 128, 96, AvColorspace.ITU709, False, False)
    with NvidiaVideoReader(str(path), batch_size=4, device=DEVICE, metadata=metadata) as reader:
        batch, _ = next(iter(reader.frames()))
    diff = (batch[0].cpu().float() - expected.float()).abs().max().item()
    assert diff <= 1.0, f"maxdiff {diff}"


@pytest.mark.parametrize(
    ("color_space", "color_range"),
    [
        (AvColorspace.ITU601, AvColorRange.MPEG),
        (AvColorspace.ITU601, AvColorRange.JPEG),
        (AvColorspace.ITU709, AvColorRange.MPEG),
        (AvColorspace.ITU709, AvColorRange.JPEG),
        (AvColorspace.BT2020, AvColorRange.MPEG),
        (AvColorspace.BT2020, AvColorRange.JPEG),
    ],
)
def test_software_matrix_and_range_pixel_reference(tmp_path, color_space, color_range):
    w, h = 64, 64
    frames = [_solid_yuv420p_frame(w, h, 140, 100, 180) for _ in range(4)]
    path = tmp_path / "colors.mkv"
    _write_video(path, "ffv1", "yuv420p", frames)
    metadata = _metadata(path, w, h, 4, color_space=color_space, color_range=color_range)
    full = color_range == AvColorRange.JPEG
    expected = _reference_rgb(140, 100, 180, w, h, color_space, full, False)
    with NvidiaVideoReader(str(path), batch_size=4, device=DEVICE, metadata=metadata) as reader:
        batch, _ = next(iter(reader.frames()))
    diff = (batch[0].cpu().float() - expected.float()).abs().max().item()
    assert diff <= 1.0, f"maxdiff {diff}"


def test_10bit_software_source_normalizes_to_p010(tmp_path):
    w, h = 64, 64
    frames = [_solid_yuv420p10_frame(w, h, 700, 400, 600) for _ in range(4)]
    path = tmp_path / "soft10.mkv"
    _write_video(path, "ffv1", "yuv420p10le", frames)
    metadata = _metadata(path, w, h, 4, is_10bit=True)
    expected = _reference_rgb(700, 400, 600, w, h, AvColorspace.ITU709, False, True)
    with NvidiaVideoReader(str(path), batch_size=4, device=DEVICE, metadata=metadata) as reader:
        batch, _ = next(iter(reader.frames()))
    assert batch.dtype == torch.uint8
    diff = (batch[0].cpu().float() - expected.float()).abs().max().item()
    assert diff <= 1.0, f"maxdiff {diff}"


def test_pitched_plane_padding_is_discarded(tmp_path):
    # 100-px rows are narrower than typical 32/64-byte swscale pitch alignment,
    # so the CPU views must slice off the padding columns.
    w, h = 100, 50
    frames = [_solid_yuv420p_frame(w, h, 120, 90, 200) for _ in range(4)]
    path = tmp_path / "pitch.mkv"
    _write_video(path, "ffv1", "yuv420p", frames)
    metadata = _metadata(path, w, h, 4)
    expected = _reference_rgb(120, 90, 200, w, h, AvColorspace.ITU709, False, False)
    with NvidiaVideoReader(str(path), batch_size=4, device=DEVICE, metadata=metadata) as reader:
        batch, _ = next(iter(reader.frames()))
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
    with NvidiaVideoReader(str(path), batch_size=4, device=DEVICE, metadata=metadata) as reader:
        _, pts = next(iter(reader.frames(seek_ts=seek_ts)))
    assert pts[0] == expected_first


def test_software_selection_logged_once(ffv1_video, caplog):
    path, metadata = ffv1_video
    with caplog.at_level(logging.WARNING, logger="jasna.media.video_decoder"):
        with NvidiaVideoReader(str(path), batch_size=4, device=DEVICE, metadata=metadata) as reader:
            for _ in reader.frames():
                pass
    fallback_logs = [r for r in caplog.records if "software" in r.getMessage()]
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
    with NvidiaVideoReader(str(path), batch_size=5, device=DEVICE, metadata=metadata) as reader:
        all_pts = [p for _, pts in reader.frames() for p in pts]
    assert len(all_pts) == n
    assert all_pts == sorted(all_pts)


def test_hardware_input_never_enters_software_path(tmp_path, monkeypatch):
    w, h, n = 128, 96, 24
    frames = [_solid_yuv420p_frame(w, h, 120, 90, 200) for _ in range(n)]
    path = tmp_path / "hw.mp4"
    _write_video(path, "libx264", "yuv420p", frames)
    metadata = _metadata(path, w, h, n, codec_name="h264")

    def _boom(self, decoded, group):
        raise AssertionError("hardware-decodable input entered the software path")

    monkeypatch.setattr(NvidiaVideoReader, "_frames_software", _boom)
    with NvidiaVideoReader(str(path), batch_size=6, device=DEVICE, metadata=metadata) as reader:
        total = sum(len(pts) for _, pts in reader.frames())
    assert total == n
