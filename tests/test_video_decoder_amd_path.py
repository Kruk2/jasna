"""Structural contracts for the AMD software-decode color path (issue #252).

These tests do not prove bit-perfect ROCm correctness under full pipeline load.
They lock vendor segregation: batch device YUV + a private stream that owns the
batch + sync-before-yield + decoding on a prefetch thread on AMD; single staging +
private stream waiting on the caller's stream on NVIDIA software fallback.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager, nullcontext
from dataclasses import replace
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

    def converter(*args, **kwargs):
        events.append("initialize converter")
        return SimpleNamespace(convert_into=convert_into)

    monkeypatch.setattr(
        module,
        "YuvToRgbConverter",
        converter,
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

    @contextmanager
    def stream_context(_stream):
        events.append("enter")
        yield
        events.append("exit")

    monkeypatch.setattr(module, "current_stream", current_stream)
    monkeypatch.setattr(module, "new_stream", new_stream)
    monkeypatch.setattr(module, "stream_context", stream_context)
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


def test_amd_software_path_owns_its_batch_on_a_private_stream(monkeypatch):
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

    assert new_calls == [True], "AMD software path must convert on a private stream"
    assert not current_calls, "AMD conversion must not wait for the caller's stream"
    stream.wait_stream.assert_not_called()

    # Converter scratch, planes and batch belong to the private stream, with
    # one synchronize after conversion before yield.
    assert events == (
        ["enter", "initialize converter", "exit"]
        + ["enter", "exit"] * 2
        + ["enter"] + ["convert"] * BATCH + ["exit", "sync"]
    )
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
    assert events == (
        ["enter", "initialize converter", "exit", "enter"]
        + ["convert"] * BATCH + ["exit", "sync"]
    )
    stream.synchronize.assert_called_once_with()


class _Batch:
    def __init__(self, index: int):
        self.index = index
        self.recorded_on: list = []

    def record_stream(self, stream) -> None:
        self.recorded_on.append(stream)


def _prefetch_env(monkeypatch) -> list:
    consumer_streams: list = []
    monkeypatch.setattr(module, "set_device", lambda _device: None)

    def current_stream(_device):
        stream = (threading.current_thread().name, len(consumer_streams))
        consumer_streams.append(stream)
        return stream

    monkeypatch.setattr(module, "current_stream", current_stream)
    return consumer_streams


def test_prefetcher_decodes_on_its_own_thread_and_records_the_consumer_stream(monkeypatch):
    consumer_streams = _prefetch_env(monkeypatch)
    producer_threads: list = []

    def batches():
        for i in range(3):
            producer_threads.append(threading.current_thread().name)
            yield _Batch(i), [i]

    prefetcher = module._BatchPrefetcher(batches(), torch.device("cpu"))
    received = list(prefetcher)
    prefetcher.close()

    assert [batch.index for batch, _ in received] == [0, 1, 2]
    assert [pts for _, pts in received] == [[0], [1], [2]]
    assert set(producer_threads) == {"DecodePrefetch"}
    assert [batch.recorded_on for batch, _ in received] == [[stream] for stream in consumer_streams]
    assert {name for name, _ in consumer_streams} == {threading.current_thread().name}


def test_prefetcher_reraises_the_decode_error_in_the_consumer(monkeypatch):
    _prefetch_env(monkeypatch)

    def batches():
        yield _Batch(0), [0]
        raise module.VideoDecodeError("corrupt")

    prefetcher = module._BatchPrefetcher(batches(), torch.device("cpu"))
    iterator = iter(prefetcher)
    next(iterator)
    with pytest.raises(module.VideoDecodeError, match="corrupt"):
        next(iterator)
    prefetcher.close()


def test_prefetcher_close_stops_decoding_and_closes_the_source(monkeypatch):
    _prefetch_env(monkeypatch)
    produced: list = []
    closed = threading.Event()

    def batches():
        try:
            for i in range(1000):
                produced.append(i)
                yield _Batch(i), [i]
        finally:
            closed.set()

    prefetcher = module._BatchPrefetcher(batches(), torch.device("cpu"))
    next(iter(prefetcher))
    prefetcher.close()

    assert closed.is_set()
    assert not prefetcher._thread.is_alive()
    assert len(produced) < 10


def test_amd_reader_exit_stops_the_prefetcher_before_closing_the_container(monkeypatch):
    _prefetch_env(monkeypatch)
    reader = _reader(AcceleratorVendor.AMD)
    order: list = []
    decoding = threading.Event()

    def decoded_batches(_seek_ts):
        try:
            for i in range(1000):
                decoding.set()
                yield _Batch(i), [i]
        finally:
            order.append("decode stopped")

    monkeypatch.setattr(reader, "_decoded_batches", decoded_batches)
    reader.container = SimpleNamespace(close=lambda: order.append("container closed"))
    reader._raw_stream = None

    frames = reader.frames()
    next(frames)
    assert decoding.wait(1)
    reader.__exit__(None, None, None)

    assert order == ["decode stopped", "container closed"]
    frames.close()


def test_nvidia_reader_decodes_on_the_calling_thread(monkeypatch):
    reader = _reader(AcceleratorVendor.NVIDIA)
    threads: list = []

    def decoded_batches(_seek_ts):
        threads.append(threading.current_thread().name)
        yield "batch", [0]

    monkeypatch.setattr(reader, "_decoded_batches", decoded_batches)

    assert list(reader.frames()) == [("batch", [0])]
    assert threads == [threading.current_thread().name]


def test_amf_decoder_has_no_ffmpeg_hwaccel(monkeypatch):
    create = MagicMock()
    monkeypatch.setattr(module.av, "CodecContext", SimpleNamespace(create=create))
    reader = _reader(AcceleratorVendor.AMD)
    reader.metadata = replace(reader.metadata, codec_name="hevc")
    source = SimpleNamespace(
        name="hevc", extradata=b"", width=W, height=H, framerate=None, sample_aspect_ratio=None
    )

    reader._setup_amf_decoder(source)

    create.assert_called_once_with("hevc_amf", "r")


def test_amf_decoder_follows_the_stream_codec_not_the_software_decoder(monkeypatch):
    create = MagicMock()
    monkeypatch.setattr(module.av, "CodecContext", SimpleNamespace(create=create))
    reader = _reader(AcceleratorVendor.AMD)
    reader.metadata = replace(reader.metadata, codec_name="av1")
    source = SimpleNamespace(
        name="libdav1d", extradata=b"", width=W, height=H, framerate=None, sample_aspect_ratio=None
    )

    reader._setup_amf_decoder(source)

    create.assert_called_once_with("av1_amf", "r")


def _write_clip(path, encoder: str, pix_fmt: str, options: dict[str, str], frame_count: int) -> None:
    rng = np.random.default_rng(0)
    low, high, dtype = (64, 940, np.uint16) if pix_fmt.endswith("10le") else (16, 235, np.uint8)
    with av.open(str(path), "w") as container:
        stream = container.add_stream(encoder, rate=24)
        stream.width, stream.height, stream.pix_fmt = 320, 240, pix_fmt
        stream.options = options
        for _ in range(frame_count):
            data = rng.integers(low, high, (240 * 3 // 2, 320), dtype=dtype)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(data, format=pix_fmt)))
        container.mux(stream.encode(None))


def _decode_all(path, backend: str, monkeypatch) -> tuple[torch.Tensor, list[int], str | None]:
    from jasna.media.probe import get_video_meta_data

    monkeypatch.setenv("JASNA_DECODE_BACKEND", backend)
    batches, pts = [], []
    with module.VideoReader(str(path), 4, torch.device("cuda:0"), get_video_meta_data(str(path))) as reader:
        for batch, batch_pts in reader.frames():
            batches.append(batch.cpu())
            pts += batch_pts
        decoder = reader._decoder_ctx.name if reader._decoder_ctx is not None else None
    return torch.cat(batches), pts, decoder


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.version.hip and "hevc_amf" in av.codecs_available),
    reason="needs an AMD GPU with the AMF decoder",
)
@pytest.mark.parametrize(
    ("encoder", "pix_fmt", "options", "amf_decoder"),
    [
        ("libx265", "yuv420p10le", {"x265-params": "log-level=error"}, "hevc_amf"),
        ("libsvtav1", "yuv420p10le", {}, "av1_amf"),
        ("libsvtav1", "yuv420p", {}, "av1_amf"),
    ],
)
def test_amf_decodes_like_software(tmp_path, monkeypatch, encoder, pix_fmt, options, amf_decoder):
    path = tmp_path / "clip.mp4"
    _write_clip(path, encoder, pix_fmt, options, 12)

    amf_frames, amf_pts, decoder = _decode_all(path, "auto", monkeypatch)
    sw_frames, sw_pts, _ = _decode_all(path, "pyav-sw", monkeypatch)

    assert decoder == amf_decoder
    assert amf_pts == sw_pts
    assert torch.equal(amf_frames, sw_frames)
