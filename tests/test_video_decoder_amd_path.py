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
import torch
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

import jasna.media.video_decoder as module
from jasna.accelerator import AcceleratorVendor
from jasna.media import VideoMetadata

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


def _reader(vendor: AcceleratorVendor) -> module.NvidiaVideoReader:
    reader = module.NvidiaVideoReader(
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
    assert not current_calls, "NVIDIA software fallback must not switch to current_stream"
    assert events == ["convert"] * BATCH + ["sync"]
    stream.synchronize.assert_called_once_with()
