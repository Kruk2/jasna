"""AMD decoding: batches decode on a prefetch thread, and AMF decodes like software."""
from __future__ import annotations

import itertools
import threading

import av
import numpy as np
import pytest
import torch

import jasna.media.video_decoder as module
from factories import write_dated_source
from jasna.media.probe import get_video_meta_data


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

    prefetcher = module.BatchPrefetcher(batches(), torch.device("cuda:0"))
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

    prefetcher = module.BatchPrefetcher(batches(), torch.device("cuda:0"))
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

    prefetcher = module.BatchPrefetcher(batches(), torch.device("cuda:0"))
    next(iter(prefetcher))
    prefetcher.close()

    assert closed.is_set()
    assert not prefetcher._thread.is_alive()
    assert len(produced) < 10


def test_amd_reader_exit_stops_the_prefetcher_before_closing_the_container(monkeypatch, amd_gpu, tmp_path):
    amd_gpu()
    _prefetch_env(monkeypatch)
    source = write_dated_source(tmp_path / "source.mp4")
    reader = module.VideoReader(str(source), 3, torch.device("cuda:0"), get_video_meta_data(str(source)))
    packets_readable_when_decoding_stopped: list = []

    def decoded_batches(source, _seek_ts):
        try:
            for i in range(1000):
                yield _Batch(i), [i]
        finally:
            packets = itertools.islice(source.container.demux(video=0), 1)
            packets_readable_when_decoding_stopped.append(len(list(packets)))

    monkeypatch.setattr(module._PyAvFrameSource, "frames", decoded_batches)
    reader.__enter__()
    frames = reader.frames()
    next(frames)
    reader.__exit__(None, None, None)
    frames.close()

    assert packets_readable_when_decoding_stopped == [1]


@pytest.mark.usefixtures("nvidia_vendor")
def test_nvidia_reader_decodes_on_the_calling_thread(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "current_stream", lambda _device: None)
    source = write_dated_source(tmp_path / "source.mp4")
    threads: list = []

    def decoded_batches(_source, _seek_ts):
        threads.append(threading.current_thread().name)
        yield "batch", [0]

    monkeypatch.setattr(module._PyAvFrameSource, "frames", decoded_batches)
    with module.VideoReader(
        str(source), 3, torch.device("cuda:0"), get_video_meta_data(str(source)), decode_backend="pyav-sw"
    ) as reader:
        batches = list(reader.frames())

    assert batches == [("batch", [0])]
    assert threads == [threading.current_thread().name]


class _AmfDecoder:
    """An AMF decoder context that rejects what PyAV 18 rejects on a decoder: assigning a
    time_base, or None for a rational."""

    def __init__(self, name: str):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "opened", False)

    def __setattr__(self, name, value):
        if name == "time_base":
            raise RuntimeError("Cannot access 'time_base' as a decoder")
        if name in {"framerate", "sample_aspect_ratio"} and value is None:
            raise AttributeError("'NoneType' object has no attribute 'numerator'")
        object.__setattr__(self, name, value)

    def open(self, strict=False):
        object.__setattr__(self, "opened", True)


class _AmfCodecContext:
    @staticmethod
    def create(name: str, mode: str) -> _AmfDecoder:
        return _AmfDecoder(name)


def _write_h264_without_aspect_ratio(path):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=12)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        for _ in range(3):
            frame = av.VideoFrame.from_ndarray(np.zeros((96, 64), dtype=np.uint8), format="yuv420p")
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    return path


@pytest.mark.parametrize("write_source", [write_dated_source, _write_h264_without_aspect_ratio])
def test_amd_source_opens_the_amf_decoder_of_the_stream_codec(monkeypatch, tmp_path, write_source):
    monkeypatch.setattr(module.av, "CodecContext", _AmfCodecContext)
    path = str(write_source(tmp_path / "source.mp4"))
    source = module._PyAvFrameSource(
        path, 3, torch.device("cuda:0"), get_video_meta_data(path), 1, module.AcceleratorVendor.AMD, False
    )
    decoder = source._decoder_ctx
    source.close(error_in_flight=False)

    assert (decoder.name, decoder.opened) == ("h264_amf", True)
    assert decoder.sample_aspect_ratio == 1
    assert decoder.framerate == 12


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
        decoder_ctx = reader._source._decoder_ctx
        decoder = decoder_ctx.name if decoder_ctx is not None else None
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
