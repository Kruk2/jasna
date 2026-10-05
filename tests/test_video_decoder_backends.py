from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

import jasna.media.video_decoder as module
from jasna.accelerator import AcceleratorVendor
from jasna.media.probe import VideoMetadata

TEST_CLIP = Path("assets/test_clip1_1080p.mp4")


@pytest.fixture(autouse=True)
def _clear_decode_backend_env(monkeypatch) -> None:
    monkeypatch.delenv(module.DECODE_BACKEND_ENV, raising=False)


def _metadata(codec_name: str = "h264", is_10bit: bool = False) -> VideoMetadata:
    return VideoMetadata(
        video_file="input.mp4",
        video_height=16,
        video_width=16,
        video_fps=30.0,
        average_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        codec_name=codec_name,
        duration=1.0,
        time_base=Fraction(1, 30),
        start_pts=0,
        color_range=AvColorRange.MPEG,
        color_space=AvColorspace.ITU709,
        num_frames=30,
        is_10bit=is_10bit,
    )


def _reader(
    monkeypatch,
    vendor: AcceleratorVendor,
    metadata: VideoMetadata | None = None,
) -> module.VideoReader:
    monkeypatch.setattr(module, "vendor_for_device", lambda _device: vendor)
    monkeypatch.setattr(module, "current_stream", lambda _device: None)
    return module.VideoReader(
        "input.mp4",
        4,
        torch.device("cuda:0"),
        metadata or _metadata(),
    )


def _fake_container(is_hwaccel: bool, codec_name: str = "h264") -> MagicMock:
    ctx = SimpleNamespace(
        is_hwaccel=is_hwaccel,
        width=16,
        height=16,
        color_range=0,
        thread_count=0,
        thread_type=None,
        name=codec_name,
        extradata=None,
    )
    container = MagicMock()
    container.streams.video = [SimpleNamespace(codec_context=ctx)]
    return container


def test_auto_backend_falls_back_to_pyav_when_vali_fails(monkeypatch, caplog) -> None:
    monkeypatch.setenv("JASNA_DECODE_BACKEND", "auto")

    def broken_vali(*args, **kwargs):
        raise module.VideoDecodeError("vali is broken")

    monkeypatch.setattr(module, "_ValiFrameSource", broken_vali)
    monkeypatch.setattr(module.av, "open", MagicMock(return_value=_fake_container(True)))
    reader = _reader(monkeypatch, AcceleratorVendor.NVIDIA)
    with caplog.at_level("WARNING"):
        reader.__enter__()
    assert reader._vali_source is None
    assert "falling back to PyAV" in caplog.text
    assert "hwaccel" in module.av.open.call_args.kwargs


@pytest.mark.parametrize("backend", module._DECODE_BACKENDS)
def test_decode_backend_env_override(monkeypatch, backend: str) -> None:
    monkeypatch.setenv("JASNA_DECODE_BACKEND", "auto")
    monkeypatch.setenv(module.DECODE_BACKEND_ENV, backend)
    assert module._decode_backend() == backend


def test_decode_backend_defaults_to_auto(monkeypatch) -> None:
    monkeypatch.setenv("JASNA_DECODE_BACKEND", "auto")
    assert module._decode_backend() == "auto"


def test_windows_resident_backend_requires_pipeline_coordinator(
    monkeypatch,
) -> None:
    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module, "current_stream", lambda _device: None)
    av_open = MagicMock(side_effect=AssertionError("must fail before PyAV open"))
    monkeypatch.setattr(module.av, "open", av_open)
    reader = module.VideoReader(
        "input.mp4",
        1,
        torch.device("cuda:0"),
        _metadata("hevc", False),
        decode_backend=module.WINDOWS_D3D11_HIP_RESIDENT_BACKEND,
    )

    with pytest.raises(module.VideoDecodeError, match="coordinator"):
        reader.__enter__()
    av_open.assert_not_called()


def test_forced_vali_backend_raises_on_failure(monkeypatch) -> None:
    monkeypatch.setenv("JASNA_DECODE_BACKEND", "vali")

    def broken_vali(*args, **kwargs):
        raise RuntimeError("no decoder")

    monkeypatch.setattr(module, "_ValiFrameSource", broken_vali)
    reader = _reader(monkeypatch, AcceleratorVendor.NVIDIA)
    with pytest.raises(module.VideoDecodeError, match="no decoder"):
        reader.__enter__()


def test_forced_vali_backend_requires_nvidia(monkeypatch) -> None:
    monkeypatch.setenv("JASNA_DECODE_BACKEND", "vali")
    reader = _reader(monkeypatch, AcceleratorVendor.AMD)
    with pytest.raises(module.VideoDecodeError, match="NVIDIA"):
        reader.__enter__()


def test_auto_backend_skips_vali_on_amd(monkeypatch) -> None:
    monkeypatch.setenv("JASNA_DECODE_BACKEND", "auto")
    vali_factory = MagicMock()
    monkeypatch.setattr(module, "_ValiFrameSource", vali_factory)
    monkeypatch.setattr(module.av, "open", MagicMock(return_value=_fake_container(False)))
    reader = _reader(monkeypatch, AcceleratorVendor.AMD)
    amf_setup = MagicMock()
    monkeypatch.setattr(reader, "_setup_amf_decoder", amf_setup)
    reader.__enter__()
    vali_factory.assert_not_called()
    amf_setup.assert_called_once()


@pytest.mark.parametrize(
    ("codec_name", "is_10bit", "log_marker", "thread_count", "thread_type"),
    [
        ("hevc", True, "HEVC Main10/P010", 1, "SLICE"),
        ("av1", False, "AV1 PyAV AMF", 0, "AUTO"),
        ("av1", True, "AV1 PyAV AMF", 0, "AUTO"),
    ],
)
def test_auto_windows_amd_problem_formats_use_controlled_software_decode(
    monkeypatch,
    caplog,
    codec_name: str,
    is_10bit: bool,
    log_marker: str,
    thread_count: int,
    thread_type: str,
) -> None:
    monkeypatch.setattr(module, "DECODE_BACKEND", "auto")
    monkeypatch.setattr(module.sys, "platform", "win32")
    container = _fake_container(False, codec_name)
    monkeypatch.setattr(module.av, "open", MagicMock(return_value=container))
    reader = _reader(
        monkeypatch,
        AcceleratorVendor.AMD,
        _metadata(codec_name, is_10bit),
    )
    amf_setup = MagicMock()
    monkeypatch.setattr(reader, "_setup_amf_decoder", amf_setup)

    with caplog.at_level("WARNING"):
        reader.__enter__()

    amf_setup.assert_not_called()
    assert reader._software_only is True
    assert module.av.open.call_args.kwargs == {}
    context = container.streams.video[0].codec_context
    assert context.thread_count == thread_count
    assert context.thread_type == thread_type
    assert log_marker in caplog.text
    assert "ROCm" in caplog.text


@pytest.mark.parametrize("codec_name", ("h264", "hevc"))
def test_auto_windows_amd_8bit_formats_keep_amf_host_route(
    monkeypatch,
    codec_name: str,
) -> None:
    monkeypatch.setattr(module, "DECODE_BACKEND", "auto")
    monkeypatch.setattr(module.sys, "platform", "win32")
    container = _fake_container(False, codec_name)
    monkeypatch.setattr(module.av, "open", MagicMock(return_value=container))
    reader = _reader(
        monkeypatch,
        AcceleratorVendor.AMD,
        _metadata(codec_name, is_10bit=False),
    )
    amf_setup = MagicMock()
    monkeypatch.setattr(reader, "_setup_amf_decoder", amf_setup)

    reader.__enter__()

    amf_setup.assert_called_once_with(container.streams.video[0].codec_context)
    assert reader._software_only is False


@pytest.mark.parametrize(
    ("platform", "vendor", "codec_name", "is_10bit"),
    [
        ("linux", AcceleratorVendor.AMD, "hevc", True),
        ("darwin", AcceleratorVendor.AMD, "av1", True),
        ("win32", AcceleratorVendor.NVIDIA, "hevc", True),
    ],
)
def test_windows_amd_software_policy_does_not_match_other_routes(
    monkeypatch,
    platform: str,
    vendor: AcceleratorVendor,
    codec_name: str,
    is_10bit: bool,
) -> None:
    monkeypatch.setattr(module.sys, "platform", platform)
    assert not module._requires_windows_amd_software_decode(
        _metadata(codec_name, is_10bit),
        vendor,
    )


@pytest.mark.parametrize(
    ("codec_name", "is_10bit"),
    (("hevc", True), ("av1", False), ("av1", True)),
)
def test_explicit_pyav_hw_bypasses_windows_amd_software_policy(
    monkeypatch,
    codec_name: str,
    is_10bit: bool,
) -> None:
    monkeypatch.setenv(module.DECODE_BACKEND_ENV, "pyav-hw")
    monkeypatch.setattr(module.sys, "platform", "win32")
    container = _fake_container(False, codec_name)
    monkeypatch.setattr(module.av, "open", MagicMock(return_value=container))
    reader = _reader(
        monkeypatch,
        AcceleratorVendor.AMD,
        _metadata(codec_name, is_10bit),
    )
    amf_setup = MagicMock()
    monkeypatch.setattr(reader, "_setup_amf_decoder", amf_setup)

    reader.__enter__()

    amf_setup.assert_called_once_with(container.streams.video[0].codec_context)
    assert reader._software_only is False


def test_auto_windows_amd_software_open_failure_is_terminal(monkeypatch) -> None:
    monkeypatch.setattr(module, "DECODE_BACKEND", "auto")
    monkeypatch.setattr(module.sys, "platform", "win32")
    reader = _reader(
        monkeypatch,
        AcceleratorVendor.AMD,
        _metadata("av1", is_10bit=True),
    )
    open_pyav = MagicMock(side_effect=module.VideoDecodeError("software decode unavailable"))
    monkeypatch.setattr(reader, "_open_pyav", open_pyav)

    with pytest.raises(module.VideoDecodeError, match="software decode unavailable"):
        reader.__enter__()

    open_pyav.assert_called_once_with(software_only=True)


def test_auto_linux_amd_av1_outside_native_gate_pyav_open_failure_is_terminal(
    monkeypatch,
) -> None:
    monkeypatch.setattr(module, "DECODE_BACKEND", "auto")
    monkeypatch.setattr(module.sys, "platform", "linux")
    metadata = _metadata("av1")
    metadata.profile = "Professional"
    metadata.pixel_format = "yuv420p"
    reader = _reader(monkeypatch, AcceleratorVendor.AMD, metadata)
    open_pyav = MagicMock(side_effect=module.VideoDecodeError("unsupported AV1 stream"))
    monkeypatch.setattr(reader, "_open_pyav", open_pyav)

    with pytest.raises(module.VideoDecodeError, match="unsupported AV1 stream"):
        reader.__enter__()

    open_pyav.assert_called_once_with(software_only=False)


def test_auto_pyav_decode_failure_is_terminal_without_backend_retry(monkeypatch) -> None:
    reader = _reader(monkeypatch, AcceleratorVendor.AMD, _metadata("av1"))
    decode = MagicMock(side_effect=module.VideoDecodeError("decode failed"))
    monkeypatch.setattr(reader, "_frames_pyav", decode)

    with pytest.raises(module.VideoDecodeError, match="decode failed"):
        list(reader.frames())

    decode.assert_called_once_with(None)


def test_removed_rocdecode_backend_fails_closed_as_unknown(monkeypatch) -> None:
    monkeypatch.setenv(module.DECODE_BACKEND_ENV, "rocdecode")
    reader = _reader(monkeypatch, AcceleratorVendor.AMD, _metadata("av1"))

    with pytest.raises(ValueError, match="Unknown decode backend 'rocdecode'"):
        reader.__enter__()


def test_pyav_sw_backend_skips_hwaccel_and_amf(monkeypatch) -> None:
    monkeypatch.setenv("JASNA_DECODE_BACKEND", "pyav-sw")
    vali_factory = MagicMock()
    monkeypatch.setattr(module, "_ValiFrameSource", vali_factory)
    for vendor in (AcceleratorVendor.NVIDIA, AcceleratorVendor.AMD):
        container = _fake_container(False)
        monkeypatch.setattr(module.av, "open", MagicMock(return_value=container))
        reader = _reader(monkeypatch, vendor)
        amf_setup = MagicMock()
        monkeypatch.setattr(reader, "_setup_amf_decoder", amf_setup)
        nvdec_setup = MagicMock()
        monkeypatch.setattr(reader, "_setup_nvdec_decoder", nvdec_setup)
        reader.__enter__()
        vali_factory.assert_not_called()
        amf_setup.assert_not_called()
        nvdec_setup.assert_not_called()
        assert module.av.open.call_args.kwargs == {}
        assert container.streams.video[0].codec_context.thread_type == "AUTO"


@pytest.mark.parametrize(
    ("platform", "vendor", "codec_name", "is_10bit", "thread_count", "thread_type"),
    [
        ("win32", AcceleratorVendor.AMD, "hevc", True, 1, "SLICE"),
        ("win32", AcceleratorVendor.AMD, "hevc", False, 0, "AUTO"),
        ("win32", AcceleratorVendor.AMD, "av1", True, 0, "AUTO"),
        ("win32", AcceleratorVendor.NVIDIA, "hevc", True, 0, "AUTO"),
        ("linux", AcceleratorVendor.AMD, "hevc", True, 0, "AUTO"),
    ],
)
def test_pyav_sw_limits_only_windows_amd_hevc_main10_threads(
    monkeypatch,
    platform: str,
    vendor: AcceleratorVendor,
    codec_name: str,
    is_10bit: bool,
    thread_count: int,
    thread_type: str,
) -> None:
    monkeypatch.setenv(module.DECODE_BACKEND_ENV, "pyav-sw")
    monkeypatch.setattr(module.sys, "platform", platform)
    container = _fake_container(False, codec_name)
    monkeypatch.setattr(module.av, "open", MagicMock(return_value=container))
    reader = _reader(
        monkeypatch,
        vendor,
        _metadata(codec_name, is_10bit),
    )

    reader.__enter__()

    context = container.streams.video[0].codec_context
    assert context.thread_count == thread_count
    assert context.thread_type == thread_type


def test_unknown_backend_raises(monkeypatch) -> None:
    monkeypatch.setenv(module.DECODE_BACKEND_ENV, "cpu-only")
    reader = _reader(monkeypatch, AcceleratorVendor.NVIDIA)
    with pytest.raises(ValueError, match="JASNA_DECODE_BACKEND"):
        reader.__enter__()


def test_reader_decode_backend_override_rejects_unknown_value(monkeypatch) -> None:
    monkeypatch.setattr(module, "vendor_for_device", lambda _device: AcceleratorVendor.NVIDIA)
    monkeypatch.setattr(module, "current_stream", lambda _device: None)
    reader = module.VideoReader(
        "input.mp4",
        4,
        torch.device("cuda:0"),
        _metadata(),
        decode_backend="cpu-only",
    )

    with pytest.raises(ValueError, match="Unknown decode backend 'cpu-only'"):
        reader.__enter__()


def test_reader_decode_backend_override_takes_precedence_over_environment(monkeypatch) -> None:
    monkeypatch.setenv(module.DECODE_BACKEND_ENV, "vali")
    monkeypatch.setattr(module, "vendor_for_device", lambda _device: AcceleratorVendor.NVIDIA)
    monkeypatch.setattr(module, "current_stream", lambda _device: None)
    container = _fake_container(False)
    monkeypatch.setattr(module.av, "open", MagicMock(return_value=container))
    reader = module.VideoReader(
        "input.mp4",
        4,
        torch.device("cuda:0"),
        _metadata(),
        decode_backend="pyav-sw",
    )

    reader.__enter__()

    assert reader._decode_backend == "pyav-sw"
    assert module.av.open.call_args.kwargs == {}
    assert container.streams.video[0].codec_context.thread_type == "AUTO"


class _FakeVali:
    class TaskExecInfo:
        END_OF_STREAM = "END_OF_STREAM"
        FAIL = SimpleNamespace(name="FAIL")

    class PixelFormat:
        NV12 = SimpleNamespace(name="NV12")
        P10 = SimpleNamespace(name="P10")
        P12 = SimpleNamespace(name="P12")

    class PacketData:
        def __init__(self):
            self.pts = None

    class SeekContext:
        def __init__(self, seek_ts):
            self.seek_ts = seek_ts

    class Surface:
        @staticmethod
        def Make(format, width, height, gpu_id):
            plane = SimpleNamespace(Pitch=width, GpuMem=1 << 20, ElemSize=1)
            return SimpleNamespace(Planes=[plane])


def _install_fake_vali(monkeypatch, script, fmt=None):
    """script: list of (success, info, pts, message) consumed per decode call."""
    calls = []

    class PyDecoder:
        def __init__(self, file, opts, gpu_id, stream):
            self.IsAccelerated = True
            self.Format = fmt or _FakeVali.PixelFormat.NV12
            self.Width = 16
            self.Height = 16

        def DecodeSingleSurfaceAsyncDetailed(self, surf, pkt_data, seek_ctx=None):
            calls.append(seek_ctx)
            success, info, pts, message = script.pop(0)
            pkt_data.pts = pts
            return success, SimpleNamespace(info=info, message=message)

    fake = SimpleNamespace(
        PyDecoder=PyDecoder,
        TaskExecInfo=_FakeVali.TaskExecInfo,
        PixelFormat=_FakeVali.PixelFormat,
        PacketData=_FakeVali.PacketData,
        SeekContext=_FakeVali.SeekContext,
        Surface=_FakeVali.Surface,
    )
    monkeypatch.setitem(sys.modules, "python_vali", fake)
    stream = SimpleNamespace(cuda_stream=0, synchronize=lambda: None)
    monkeypatch.setattr(module, "_create_blocking_cuda_stream", lambda _device: (None, stream))
    converter = MagicMock()
    monkeypatch.setattr(module, "YuvToRgbConverter", MagicMock(return_value=converter))
    return calls, converter


def _frame(pts: int, message: str = ""):
    return (True, None, pts, message)


_EOF = (False, _FakeVali.TaskExecInfo.END_OF_STREAM, None, "")


def _vali_source(batch_size: int = 2, frame_stride: int = 1) -> module._ValiFrameSource:
    return module._ValiFrameSource(
        "input.mp4", batch_size, torch.device("cpu"), _metadata(), frame_stride
    )


def test_vali_source_batches_and_final_partial_batch(monkeypatch) -> None:
    script = [_frame(pts) for pts in (0, 512, 1024, 2048, 4096)] + [_EOF]
    _install_fake_vali(monkeypatch, script)
    source = _vali_source(batch_size=2)
    batches = list(source.frames(None))
    assert [pts for _batch, pts in batches] == [[0, 512], [1024, 2048], [4096]]
    assert [tuple(batch.shape) for batch, _pts in batches] == [
        (2, 3, 16, 16),
        (2, 3, 16, 16),
        (1, 3, 16, 16),
    ]


def test_vali_source_frame_stride_skips_conversion(monkeypatch) -> None:
    script = [_frame(pts) for pts in (0, 1, 2, 3, 4, 5)] + [_EOF]
    _calls, converter = _install_fake_vali(monkeypatch, script)
    source = _vali_source(batch_size=4, frame_stride=2)
    batches = list(source.frames(None))
    assert [pts for _batch, pts in batches] == [[0, 2, 4]]
    assert converter.convert_surface_into.call_count == 3


def test_vali_source_logs_recovered_corruption(monkeypatch, caplog) -> None:
    script = [_frame(0), _frame(512, "recovered after 1 corrupt packet(s)"), _EOF]
    _install_fake_vali(monkeypatch, script)
    source = _vali_source(batch_size=4)
    with caplog.at_level("WARNING"):
        batches = list(source.frames(None))
    assert [pts for _batch, pts in batches] == [[0, 512]]
    assert "Recovered video corruption" in caplog.text


def test_vali_source_raises_on_hard_decode_failure(monkeypatch) -> None:
    script = [
        _frame(0),
        (False, _FakeVali.TaskExecInfo.FAIL, None, "too many consecutive corrupt packets"),
    ]
    _install_fake_vali(monkeypatch, script)
    source = _vali_source(batch_size=4)
    with pytest.raises(module.VideoDecodeError, match="too many consecutive corrupt packets"):
        list(source.frames(None))


def test_vali_source_seek_discards_smoke_frame(monkeypatch) -> None:
    script = [_frame(0), _frame(30720), _frame(31232), _EOF]
    calls, _converter = _install_fake_vali(monkeypatch, script)
    source = _vali_source(batch_size=2)
    batches = list(source.frames(2.0))
    assert [pts for _batch, pts in batches] == [[30720, 31232]]
    assert calls[0] is None
    assert isinstance(calls[1], _FakeVali.SeekContext) and calls[1].seek_ts == 2.0
    assert calls[2] is None


def test_vali_source_rejects_unsupported_surface_format(monkeypatch) -> None:
    script = [_frame(0), _EOF]
    _install_fake_vali(monkeypatch, script, fmt=_FakeVali.PixelFormat.P12)
    with pytest.raises(module.VideoDecodeError, match="P12"):
        _vali_source()


def _vali_fork_available() -> bool:
    try:
        import python_vali as vali
    except ImportError:
        return False
    return hasattr(vali.PyDecoder, "DecodeSingleSurfaceAsyncDetailed")


@pytest.mark.skipif(
    not torch.cuda.is_available() or not TEST_CLIP.exists() or not _vali_fork_available(),
    reason="needs a GPU, the test clip and the python_vali fork",
)
def test_vali_backend_matches_pyav_hw_output(monkeypatch) -> None:
    from jasna.media.probe import get_video_meta_data

    metadata = get_video_meta_data(str(TEST_CLIP))
    device = torch.device("cuda", 0)

    def first_batch(backend):
        monkeypatch.setenv("JASNA_DECODE_BACKEND", backend)
        with module.VideoReader(
            str(TEST_CLIP), batch_size=4, device=device, metadata=metadata
        ) as reader:
            batch, pts = next(reader.frames())
            return batch.clone(), pts

    vali_batch, vali_pts = first_batch("vali")
    pyav_batch, pyav_pts = first_batch("pyav-hw")
    assert vali_pts == pyav_pts
    assert torch.equal(vali_batch, pyav_batch)


def test_start_pts_uses_metadata_for_vali_and_stream_for_pyav() -> None:
    import dataclasses

    reader = module.VideoReader.__new__(module.VideoReader)
    reader.metadata = dataclasses.replace(_metadata(), start_pts=1500)
    reader._vali_source = object()
    assert reader.start_pts == 1500

    reader._vali_source = None
    reader.video_stream = SimpleNamespace(start_time=2000)
    assert reader.start_pts == 2000


def test_close_pyav_detaches_all_reader_owned_contexts() -> None:
    reader = module.VideoReader.__new__(module.VideoReader)
    decoder = MagicMock()
    reader._decoder_ctx = decoder
    reader.video_stream = object()
    reader.container = MagicMock()
    container = reader.container

    reader._close_pyav()

    assert reader._decoder_ctx is None
    assert reader.video_stream is None
    assert reader.container is None
    decoder.close.assert_called_once_with()
    container.close.assert_called_once_with()


def test_close_pyav_synchronizes_amf_consumer_before_decoder_teardown() -> None:
    reader = module.VideoReader.__new__(module.VideoReader)
    decoder = MagicMock()
    container = MagicMock()
    consumer_stream = MagicMock()
    reader._decoder_ctx = decoder
    reader.video_stream = object()
    reader.container = container
    reader._amf_interop_enabled = True
    reader._amf_interop_consumer_stream = consumer_stream

    reader._close_pyav()

    consumer_stream.synchronize.assert_called_once_with()
    decoder.close.assert_called_once_with()
    container.close.assert_called_once_with()


def test_cached_amf_interop_syncs_before_cache_and_decoder_teardown() -> None:
    reader = module.VideoReader.__new__(module.VideoReader)
    decoder = MagicMock()
    container = MagicMock()
    consumer_stream = MagicMock()
    order = []
    consumer_stream.synchronize.side_effect = lambda: order.append("sync")
    reader.device = torch.device("cuda:0")
    reader._vali_source = None
    reader._decoder_ctx = decoder
    reader.video_stream = object()
    reader.container = container
    reader._raw_stream = None
    reader._amf_interop_enabled = True
    reader._amf_interop_resource_cache = True
    reader._amf_interop_audit = object()
    reader._amf_interop_consumer_stream = consumer_stream

    def close_backend(*, validate):
        order.append("interop")
        reader._amf_interop_enabled = False
        reader._amf_interop_audit = None
        reader._amf_interop_consumer_stream = None

    reader._close_amf_interop_backend = close_backend
    original_close_pyav = reader._close_pyav

    def close_pyav():
        order.append("decoder")
        original_close_pyav()

    reader._close_pyav = close_pyav

    reader.__exit__(None, None, None)

    assert order == ["sync", "interop", "decoder"]
    decoder.close.assert_called_once_with()
    container.close.assert_called_once_with()


def test_vali_source_seek_with_stride_reanchors_at_seek(monkeypatch) -> None:
    script = [_frame(0)] + [_frame(pts) for pts in (30720, 31232, 31744, 32256)] + [_EOF]
    _install_fake_vali(monkeypatch, script)
    source = _vali_source(batch_size=2, frame_stride=2)
    batches = list(source.frames(2.0))
    assert [pts for _batch, pts in batches] == [[30720, 31744]]
