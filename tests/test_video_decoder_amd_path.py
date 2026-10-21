"""Structural contracts for the AMD software-decode color path (issue #252).

These tests do not prove bit-perfect ROCm correctness under full pipeline load
(see Phase 0 / Phase 4). They lock vendor segregation: batch device YUV +
current_stream + sync-before-yield on AMD; single staging + private stream on
NVIDIA software fallback.
"""
from __future__ import annotations

from contextlib import nullcontext
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock

import av
import numpy as np
import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

import jasna.media.video_decoder as module
from jasna.accelerator import AcceleratorVendor
from jasna.media.probe import VideoMetadata

H, W = 8, 8
BATCH = 3


def _metadata() -> VideoMetadata:
    return VideoMetadata(
        video_file="soft.mkv",
        video_height=H,
        video_width=W,
        video_fps=12.0,
        average_fps=12.0,
        video_fps_exact=Fraction(12, 1),
        codec_name="ffv1",
        duration=1.0,
        time_base=Fraction(1, 12),
        start_pts=0,
        color_range=AvColorRange.MPEG,
        color_space=AvColorspace.ITU709,
        num_frames=BATCH,
        is_10bit=False,
    )


class _Plane:
    """Minimal plane stand-in for torch.frombuffer + line_size."""

    def __init__(self, rows: int, cols: int):
        self.line_size = cols
        self._buf = bytearray(rows * cols)

    def __buffer__(self, flags):
        return memoryview(self._buf)


def _fake_normalized_frame():
    y = _Plane(H, W)
    uv = _Plane(H // 2, W)  # NV12 interleaved UV rows
    return SimpleNamespace(planes=(y, uv))


def _cpu_group(n: int = BATCH) -> list:
    """Lightweight av frames; reformatter is mocked so content is unused."""
    frames = []
    for i in range(n):
        data = np.zeros((H * 3 // 2, W), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(data, format="yuv420p")
        frame.pts = i
        frames.append(frame)
    return frames


def _reader(vendor: AcceleratorVendor) -> module.VideoReader:
    reader = module.VideoReader(
        "soft.mkv",
        BATCH,
        torch.device("cpu"),
        _metadata(),
    )
    reader.vendor = vendor
    reader.height = H
    reader.width = W
    reader._full_range = False
    return reader


def _run_software_batch(
    monkeypatch,
    vendor: AcceleratorVendor,
    *,
    events: list,
    luma_ptrs: list,
):
    reader = _reader(vendor)

    def convert_into(y, uv, out):
        events.append("convert")
        luma_ptrs.append(y.data_ptr())

    monkeypatch.setattr(
        module,
        "YuvToRgbConverter",
        lambda *args, **kwargs: SimpleNamespace(convert_into=convert_into),
    )
    monkeypatch.setattr(
        module,
        "VideoReformatter",
        lambda: SimpleNamespace(reformat=lambda *a, **k: _fake_normalized_frame()),
    )

    stream = MagicMock(name=f"{vendor.value}_stream")
    stream.synchronize.side_effect = lambda: events.append("sync")

    current_calls: list = []
    new_calls: list = []

    def current_stream(_device):
        current_calls.append(True)
        return stream

    def new_stream(_device):
        new_calls.append(True)
        return stream

    monkeypatch.setattr(module, "current_stream", current_stream)
    monkeypatch.setattr(module, "new_stream", new_stream)
    monkeypatch.setattr(module, "stream_context", lambda _s: nullcontext())
    monkeypatch.setattr(reader, "_read_group", lambda _decoded: [])

    # Avoid pin_memory hard requirements on exotic hosts.
    real_empty = torch.empty

    def empty_no_pin(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty_no_pin)

    group = _cpu_group(BATCH)
    batches = list(reader._frames_software(None, group))
    return batches, stream, current_calls, new_calls


def test_amd_software_path_uses_batch_device_yuv_and_current_stream(monkeypatch):
    events: list = []
    luma_ptrs: list = []
    batches, stream, current_calls, new_calls = _run_software_batch(
        monkeypatch,
        AcceleratorVendor.AMD,
        events=events,
        luma_ptrs=luma_ptrs,
    )

    assert len(batches) == 1
    batch, pts = batches[0]
    assert batch.shape == (BATCH, 3, H, W)
    assert pts == list(range(BATCH))

    # Distinct device planes per frame index (not one overwritten staging buffer).
    assert len(luma_ptrs) == BATCH
    assert len(set(luma_ptrs)) == BATCH

    assert current_calls, "AMD software path must use current_stream"
    assert not new_calls, "AMD software path must not open a private new_stream"

    # Sync before yield: all converts complete, then one synchronize, then yield.
    assert events == ["convert"] * BATCH + ["sync"]
    stream.synchronize.assert_called_once_with()


def test_nvidia_software_path_reuses_single_staging_on_private_stream(monkeypatch):
    events: list = []
    luma_ptrs: list = []
    batches, stream, current_calls, new_calls = _run_software_batch(
        monkeypatch,
        AcceleratorVendor.NVIDIA,
        events=events,
        luma_ptrs=luma_ptrs,
    )

    assert len(batches) == 1
    assert len(luma_ptrs) == BATCH
    # Same staging object reused for every frame index.
    assert len(set(luma_ptrs)) == 1

    assert new_calls, "NVIDIA software fallback must use a private new_stream"
    assert current_calls == [True], "NVIDIA software fallback reads current_stream only to wait on it"
    stream.wait_stream.assert_called_once_with(stream)
    assert events == ["convert"] * BATCH + ["sync"]
    stream.synchronize.assert_called_once_with()


def test_amf_decoder_has_no_ffmpeg_hwaccel(monkeypatch):
    create = MagicMock()
    monkeypatch.setattr(module.av, "CodecContext", SimpleNamespace(create=create))
    reader = _reader(AcceleratorVendor.AMD)
    source = SimpleNamespace(
        name="hevc", extradata=b"", width=W, height=H, framerate=None, sample_aspect_ratio=None
    )

    reader._setup_amf_decoder(source)

    create.assert_called_once_with("hevc_amf", "r")


def _write_hevc_main10(path, frame_count: int) -> None:
    rng = np.random.default_rng(0)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx265", rate=24)
        stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p10le"
        stream.options = {"x265-params": "log-level=error"}
        for _ in range(frame_count):
            data = rng.integers(64, 940, (240 * 3 // 2, 320), dtype=np.uint16)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(data, format="yuv420p10le")))
        container.mux(stream.encode(None))


def _decode_all(path, backend: str, monkeypatch) -> tuple[torch.Tensor, list[int], bool]:
    from jasna.media.probe import get_video_meta_data

    monkeypatch.setenv("JASNA_DECODE_BACKEND", backend)
    batches, pts = [], []
    with module.VideoReader(str(path), 4, torch.device("cuda:0"), get_video_meta_data(str(path))) as reader:
        for batch, batch_pts in reader.frames():
            batches.append(batch.cpu())
            pts += batch_pts
        uses_amf = reader._decoder_ctx is not None
    return torch.cat(batches), pts, uses_amf


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.version.hip and "hevc_amf" in av.codecs_available),
    reason="needs an AMD GPU with the AMF decoder",
)
def test_amf_decodes_hevc_main10_like_software(tmp_path, monkeypatch):
    path = tmp_path / "main10.mp4"
    _write_hevc_main10(path, 12)

    amf_frames, amf_pts, uses_amf = _decode_all(path, "auto", monkeypatch)
    sw_frames, sw_pts, _ = _decode_all(path, "pyav-sw", monkeypatch)

    assert uses_amf
    assert amf_pts == sw_pts
    assert torch.equal(amf_frames, sw_frames)
