"""AMD decoding: batches decode on a prefetch thread, and AMF decodes like software."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import av
import numpy as np
import pytest
import torch

import jasna.media.video_decoder as module
from factories import video_metadata


def _reader() -> module.VideoReader:
    return module.VideoReader("soft.mkv", 3, torch.device("cuda:0"), video_metadata(num_frames=3))


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


def test_amd_reader_exit_stops_the_prefetcher_before_closing_the_container(monkeypatch, amd_gpu):
    amd_gpu()
    _prefetch_env(monkeypatch)
    reader = _reader()
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


@pytest.mark.usefixtures("nvidia_vendor")
def test_nvidia_reader_decodes_on_the_calling_thread(monkeypatch):
    reader = _reader()
    threads: list = []

    def decoded_batches(_seek_ts):
        threads.append(threading.current_thread().name)
        yield "batch", [0]

    monkeypatch.setattr(reader, "_decoded_batches", decoded_batches)

    assert list(reader.frames()) == [("batch", [0])]
    assert threads == [threading.current_thread().name]


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


def _decode_all(path, backend: str) -> tuple[torch.Tensor, list[int], str | None]:
    from jasna.media.probe import get_video_meta_data

    batches, pts = [], []
    metadata = get_video_meta_data(str(path))
    with module.VideoReader(str(path), 4, torch.device("cuda:0"), metadata, decode_backend=backend) as reader:
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
def test_amf_decodes_like_software(tmp_path, encoder, pix_fmt, options, amf_decoder):
    path = tmp_path / "clip.mp4"
    _write_clip(path, encoder, pix_fmt, options, 12)

    amf_frames, amf_pts, decoder = _decode_all(path, "auto")
    sw_frames, sw_pts, _ = _decode_all(path, "pyav-sw")

    assert decoder == amf_decoder
    assert amf_pts == sw_pts
    assert torch.equal(amf_frames, sw_frames)
