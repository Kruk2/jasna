from __future__ import annotations

from contextlib import nullcontext
import copy
from fractions import Fraction
from pathlib import Path
import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from jasna.accelerator import AcceleratorVendor
from jasna.media import dual_gop_encoder
from jasna.media.video_encoder import AMF_ENCODER_SPECS, AMF_SMART_FRAGMENT_OPTIONS
from jasna.pipeline import _OfflineFrameWriter, _dual_gop_smart_encoder_settings


def _encoder(**overrides):
    values = {
        "vendor": AcceleratorVendor.AMD,
        "codec": "hevc",
        "_amf_host_zero_copy": True,
        "_target_bit_rate": 12_000_000,
        "smart_fragment": False,
        "fmp4": False,
        "auto_source_rate": True,
        "output_fps": Fraction(60_000, 1_001),
        "spec": SimpleNamespace(ten_bit=True, frame_format="p010le"),
        "metadata": SimpleNamespace(
            codec_name="hevc",
            is_10bit=True,
            pixel_format="yuv420p10le",
            profile="Main 10",
            video_width=8192,
            video_height=4096,
            video_fps_exact=Fraction(60_000, 1_001),
            video_bitrate=12_000_000,
        ),
        "encoder_options": {
            "rc": "vbr_peak",
            "maxrate": "15000000",
            "bufsize": "30000000",
            "preanalysis": "0",
            "g": "250",
            "async_depth": "4",
            "host_zero_copy": "1",
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_dual_gop_is_default_off() -> None:
    assert not dual_gop_encoder.use_dual_gop_writer(
        SimpleNamespace(),
        enabled=False,
    )


def test_dual_gop_accepts_a_supported_linux_amd_shape(monkeypatch) -> None:
    monkeypatch.setattr(dual_gop_encoder.sys, "platform", "linux")

    assert dual_gop_encoder.use_dual_gop_writer(_encoder(), enabled=True)


def test_dual_gop_accepts_smart_render_fragment(monkeypatch) -> None:
    monkeypatch.setattr(dual_gop_encoder.sys, "platform", "linux")

    assert dual_gop_encoder.use_dual_gop_writer(
        _encoder(smart_fragment=True),
        enabled=True,
    )


@pytest.mark.parametrize("smart_fragment", [False, True])
def test_persistent_session_uses_current_shared_encoder_contract(
    tmp_path, smart_fragment: bool
) -> None:
    worker = object.__new__(dual_gop_encoder._EncoderWorker)
    template = _encoder(
        smart_fragment=smart_fragment, spec=AMF_ENCODER_SPECS["hevc"]
    )
    original_options = dict(template.encoder_options)
    worker.template = template
    worker.combined_output = tmp_path / "encoder-0.nut"

    encoder = worker._session_encoder()

    assert encoder is not template
    assert encoder.spec is template.spec
    assert encoder.smart_fragment is True
    assert encoder.mux_audio is False
    assert encoder.fmp4 is False
    assert encoder.pts_origin == 0
    assert encoder.file == str(worker.combined_output)
    assert encoder.output_path == worker.combined_output
    assert encoder.encoder_options == {
        **original_options, **AMF_SMART_FRAGMENT_OPTIONS
    }
    assert template.encoder_options == original_options
    assert template.smart_fragment is smart_fragment
    assert encoder._target_bit_rate == template._target_bit_rate


def test_dual_gop_smart_render_restores_fixed_writer_contract() -> None:
    source_matched = {"g": 300, "bf": 2, "level": "6.2"}

    assert _dual_gop_smart_encoder_settings(
        source_matched,
        enabled=True,
    ) == {"g": 250, "bf": 0, "level": "6.2"}
    assert _dual_gop_smart_encoder_settings(
        source_matched,
        enabled=False,
    ) is source_matched


def test_dual_gop_accepts_bundled_ffprobe_empty_profile_fallback(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dual_gop_encoder.sys, "platform", "linux")
    encoder = _encoder()
    encoder.metadata.profile = ""

    assert dual_gop_encoder.use_dual_gop_writer(encoder, enabled=True)


def test_dual_gop_forced_experiment_accepts_main_nv12(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dual_gop_encoder.sys, "platform", "linux")
    monkeypatch.setenv("JASNA_AMF_HOST_ZERO_COPY", "1")
    encoder = _encoder(
        output_fps=Fraction(30, 1),
        spec=SimpleNamespace(ten_bit=False, frame_format="nv12"),
        metadata=SimpleNamespace(
            codec_name="hevc",
            is_10bit=False,
            pixel_format="yuv420p",
            profile="Main",
            video_width=3840,
            video_height=2160,
            video_fps_exact=Fraction(30, 1),
            video_bitrate=8_000_000,
        ),
    )

    assert dual_gop_encoder.use_dual_gop_writer(encoder, enabled=True)


def test_dual_gop_automatically_accepts_5k_main_nv12(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dual_gop_encoder.sys, "platform", "linux")
    encoder = _encoder(
        spec=SimpleNamespace(ten_bit=False, frame_format="nv12"),
        metadata=SimpleNamespace(
            codec_name="hevc",
            is_10bit=False,
            pixel_format="yuv420p",
            profile="Main",
            video_width=5760,
            video_height=2880,
            video_fps_exact=Fraction(60_000, 1_001),
            video_bitrate=18_000_000,
        ),
    )

    assert dual_gop_encoder.use_dual_gop_writer(encoder, enabled=True)


def test_dual_gop_automatically_accepts_5k_main10_p010(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dual_gop_encoder.sys, "platform", "linux")
    encoder = _encoder()
    encoder.metadata.video_width = 5760
    encoder.metadata.video_height = 2880

    assert dual_gop_encoder.use_dual_gop_writer(encoder, enabled=True)
    assert dual_gop_encoder.amd_dual_gop_metadata_eligible(encoder.metadata)


def test_dual_gop_automatically_accepts_4k_main10_p010(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dual_gop_encoder.sys, "platform", "linux")
    encoder = _encoder()
    encoder.metadata.video_width = 3840
    encoder.metadata.video_height = 2160

    assert dual_gop_encoder.use_dual_gop_writer(encoder, enabled=True)
    assert dual_gop_encoder.amd_dual_gop_metadata_eligible(encoder.metadata)


def test_dual_gop_automatically_accepts_4k_vr_main10_p010(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dual_gop_encoder.sys, "platform", "linux")
    encoder = _encoder()
    encoder.metadata.video_width = 4096
    encoder.metadata.video_height = 2048

    assert dual_gop_encoder.use_dual_gop_writer(encoder, enabled=True)
    assert dual_gop_encoder.amd_dual_gop_metadata_eligible(encoder.metadata)


def test_dual_gop_keeps_4k_main8_nv12_on_single_session(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dual_gop_encoder.sys, "platform", "linux")
    encoder = _encoder(
        spec=SimpleNamespace(ten_bit=False, frame_format="nv12"),
        metadata=SimpleNamespace(
            codec_name="hevc",
            is_10bit=False,
            pixel_format="yuv420p",
            profile="Main",
            video_width=3840,
            video_height=2160,
            video_fps_exact=Fraction(60_000, 1_001),
            video_bitrate=12_000_000,
        ),
    )

    with pytest.raises(RuntimeError, match="validated output geometry"):
        dual_gop_encoder.use_dual_gop_writer(encoder, enabled=True)


def test_dual_gop_full_encode_uses_hevc_output_not_source_codec(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dual_gop_encoder.sys, "platform", "linux")
    encoder = _encoder(
        spec=SimpleNamespace(ten_bit=False, frame_format="nv12"),
        metadata=SimpleNamespace(
            codec_name="h264",
            is_10bit=False,
            pixel_format="yuv420p",
            profile="High",
            video_width=5760,
            video_height=2880,
            video_fps_exact=Fraction(60_000, 1_001),
            video_bitrate=18_000_000,
        ),
    )

    assert dual_gop_encoder.use_dual_gop_writer(encoder, enabled=True)
    assert dual_gop_encoder.amd_dual_gop_metadata_eligible(encoder.metadata)


@pytest.mark.parametrize(
    ("source_codec", "source_profile"),
    [("av1", "Main"), ("hevc", "High")],
)
def test_dual_gop_smart_render_requires_compatible_hevc_source(
    monkeypatch,
    source_codec,
    source_profile,
) -> None:
    monkeypatch.setattr(dual_gop_encoder.sys, "platform", "linux")
    encoder = copy.deepcopy(_encoder())
    encoder.smart_fragment = True
    encoder.metadata.codec_name = source_codec
    encoder.metadata.profile = source_profile

    with pytest.raises(RuntimeError, match="HEVC-compatible Smart Render source"):
        dual_gop_encoder.use_dual_gop_writer(encoder, enabled=True)


def test_dual_gop_nv12_prepare_keeps_uint8_layout(monkeypatch) -> None:
    height = 4
    width = 6
    packed = torch.arange((height + height // 2) * width, dtype=torch.uint8).reshape(
        height + height // 2, width
    )
    template = SimpleNamespace(
        metadata=SimpleNamespace(video_height=height, video_width=width),
        spec=SimpleNamespace(ten_bit=False, frame_format="nv12"),
        stream=SimpleNamespace(synchronize=MagicMock()),
        _lut_applier=None,
        _to_yuv=MagicMock(return_value=packed),
    )
    writer = object.__new__(dual_gop_encoder.AmdDualGopFrameWriter)
    writer.template = template
    writer.failed = threading.Event()
    writer.host_pool = dual_gop_encoder._PinnedHostFramePool(
        shape=(height + height // 2, width),
        dtype=torch.uint8,
    )
    real_empty = torch.empty
    input_frame = real_empty(0)

    def host_empty(shape, *, dtype, pin_memory):
        assert pin_memory is True
        return real_empty(shape, dtype=dtype)

    captured = {}

    def from_dlpack(planes, *, format):
        captured["planes"] = [plane.clone() for plane in planes]
        captured["format"] = format
        return "frame"

    monkeypatch.setattr(dual_gop_encoder, "stream_context", lambda _stream: nullcontext())
    monkeypatch.setattr(dual_gop_encoder.torch, "empty", host_empty)
    monkeypatch.setattr(dual_gop_encoder.av.VideoFrame, "from_dlpack", from_dlpack)

    frame, host_yuv = writer._prepare(input_frame, apply_lut=False)

    assert frame == "frame"
    assert host_yuv.dtype == torch.uint8
    assert captured["format"] == "nv12"
    assert captured["planes"][0].dtype is torch.uint8
    assert torch.equal(captured["planes"][0], packed[:height])
    assert torch.equal(captured["planes"][1], packed[height:])


def test_dual_gop_pinned_pool_reuses_released_allocation(monkeypatch) -> None:
    allocations = []

    def host_empty(shape, *, dtype, pin_memory):
        assert pin_memory is True
        value = torch.zeros(shape, dtype=dtype)
        allocations.append(value)
        return value

    monkeypatch.setattr(dual_gop_encoder.torch, "empty", host_empty)
    pool = dual_gop_encoder._PinnedHostFramePool(
        shape=(6, 8),
        dtype=torch.uint8,
    )
    failed = threading.Event()

    first = pool.acquire(failed)
    pool.release(first)
    second = pool.acquire(failed)

    assert second is first
    assert allocations == [first]
    assert pool.allocated == 1
    assert pool.in_use == 1
    assert pool.peak_in_use == 1
    pool.release(second)
    assert pool.in_use == 0


def test_dual_gop_pinned_pool_close_drops_idle_owners(monkeypatch) -> None:
    allocations = []

    def host_empty(shape, *, dtype, pin_memory):
        value = torch.zeros(shape, dtype=dtype)
        allocations.append(value)
        return value

    monkeypatch.setattr(dual_gop_encoder.torch, "empty", host_empty)
    pool = dual_gop_encoder._PinnedHostFramePool(
        shape=(6, 8),
        dtype=torch.uint8,
    )
    failed = threading.Event()
    first = pool.acquire(failed)
    pool.release(first)

    pool.close()

    assert pool.closed is True
    assert pool.allocated == 0
    assert pool.in_use == 0
    with pytest.raises(RuntimeError, match="pool is closed"):
        pool.acquire(failed)


def test_dual_gop_pinned_pool_close_rejects_inflight_owner(monkeypatch) -> None:
    monkeypatch.setattr(
        dual_gop_encoder.torch,
        "empty",
        lambda shape, *, dtype, pin_memory: torch.zeros(shape, dtype=dtype),
    )
    pool = dual_gop_encoder._PinnedHostFramePool(
        shape=(6, 8),
        dtype=torch.uint8,
    )
    owner = pool.acquire(threading.Event())

    with pytest.raises(RuntimeError, match="still in use"):
        pool.close()
    pool.release(owner)
    pool.close()


def test_dual_gop_worker_recycles_host_frame_by_packet_pts() -> None:
    worker = object.__new__(dual_gop_encoder._EncoderWorker)
    worker.worker_index = 0
    worker.host_pool = MagicMock()
    host_yuv = object()
    worker.pending_host_yuv = {123: host_yuv}

    worker._release_packet_host_frame(SimpleNamespace(pts=123))

    worker.host_pool.release.assert_called_once_with(host_yuv)
    assert worker.pending_host_yuv == {}


def test_dual_gop_worker_rejects_packet_without_matching_host_frame() -> None:
    worker = object.__new__(dual_gop_encoder._EncoderWorker)
    worker.worker_index = 1
    worker.host_pool = MagicMock()
    worker.pending_host_yuv = {}

    with pytest.raises(RuntimeError, match="unexpected packet PTS 456"):
        worker._release_packet_host_frame(SimpleNamespace(pts=456))

    worker.host_pool.release.assert_not_called()


def test_dual_gop_worker_skips_split_after_abort(monkeypatch) -> None:
    class _FakeEncoder:
        def __init__(self) -> None:
            self.out_stream = SimpleNamespace(
                codec_context=SimpleNamespace(open=MagicMock())
            )

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    worker = object.__new__(dual_gop_encoder._EncoderWorker)
    worker.worker_index = 0
    worker.template = SimpleNamespace(device=torch.device("cpu"))
    worker.failed = threading.Event()
    worker.failed.set()
    worker.ready = threading.Event()
    worker.error = None
    worker.items = queue.Queue()
    worker.pending_host_yuv = {}
    worker._session_encoder = lambda: _FakeEncoder()
    worker._run = MagicMock()
    worker._release_pending_host_frames = MagicMock()
    worker._split_groups = MagicMock()

    monkeypatch.setattr(dual_gop_encoder, "set_device", lambda _device: None)

    worker.run()

    worker._run.assert_called_once()
    worker._split_groups.assert_not_called()


@pytest.mark.parametrize(
    ("change", "missing"),
    [
        ({"vendor": AcceleratorVendor.NVIDIA}, "AMD"),
        ({"codec": "h264"}, "HEVC output"),
        ({"_amf_host_zero_copy": False}, "host-native AMF input"),
        ({"fmp4": True}, "non-fMP4 output"),
        ({"auto_source_rate": False}, "automatic source-rate VBR Peak"),
        ({"output_fps": Fraction(30, 1)}, "source frame rate"),
    ],
)
def test_dual_gop_rejects_unproven_runtime_shapes(
    monkeypatch,
    change,
    missing,
) -> None:
    monkeypatch.setattr(dual_gop_encoder.sys, "platform", "linux")
    encoder = _encoder()
    for name, value in change.items():
        setattr(encoder, name, value)

    with pytest.raises(RuntimeError, match=missing):
        dual_gop_encoder.use_dual_gop_writer(encoder, enabled=True)


@pytest.mark.parametrize(
    ("mutate", "missing"),
    [
        (
            lambda e: setattr(e.metadata, "pixel_format", "yuv422p10le"),
            "NV12/P010-compatible 4:2:0 input",
        ),
        (
            lambda e: setattr(e.metadata, "is_10bit", False),
            "NV12/P010 output matching the source bit depth",
        ),
        (
            lambda e: setattr(e.metadata, "video_width", 1920),
            "validated output geometry",
        ),
        (lambda e: e.encoder_options.__setitem__("g", "240"), "fixed GOP 250"),
        (lambda e: e.encoder_options.__setitem__("bf", "2"), "no B-frames"),
        (
            lambda e: e.encoder_options.__setitem__("rc", "cqp"),
            "automatic source-rate VBR Peak",
        ),
        (
            lambda e: e.encoder_options.__setitem__("async_depth", "8"),
            "fixed host async depth 4",
        ),
    ],
)
def test_dual_gop_rejects_unproven_metadata_or_encoder_options(
    monkeypatch,
    mutate,
    missing,
) -> None:
    monkeypatch.setattr(dual_gop_encoder.sys, "platform", "linux")
    encoder = copy.deepcopy(_encoder())
    mutate(encoder)

    with pytest.raises(RuntimeError, match=missing):
        dual_gop_encoder.use_dual_gop_writer(encoder, enabled=True)


def test_dual_gop_queue_and_gop_are_fixed_to_validated_values() -> None:
    assert dual_gop_encoder.AMD_DUAL_GOP_QUEUE_DEPTH == 8
    assert dual_gop_encoder.AMD_DUAL_GOP_SIZE == 250
    assert dual_gop_encoder.AMD_DUAL_GOP_AMF_ASYNC_DEPTH == 4
    assert dual_gop_encoder.AMD_DUAL_GOP_PINNED_POOL_SIZE == 25


def test_dual_gop_source_preflight_rejects_missing_bitrate() -> None:
    metadata = copy.deepcopy(_encoder().metadata)
    metadata.video_bitrate = 0

    with pytest.raises(ValueError, match="positive source video bitrate"):
        dual_gop_encoder.validate_amd_dual_gop_source(metadata)


def test_dual_gop_empty_second_worker_finishes_without_fragments(tmp_path) -> None:
    worker = object.__new__(dual_gop_encoder._EncoderWorker)
    worker.worker_index = 1
    worker.groups = []
    worker.fragments = []
    worker.combined_output = tmp_path / "encoder-1.nut"
    worker.combined_output.write_bytes(b"empty container")

    worker._split_groups()

    assert worker.fragments == []
    assert not worker.combined_output.exists()


def test_dual_gop_worker_splits_nut_directly_to_timestamp_reset_ts(
    monkeypatch, tmp_path
) -> None:
    worker = object.__new__(dual_gop_encoder._EncoderWorker)
    worker.worker_index = 0
    worker.combined_output = tmp_path / "encoder-0.nut"
    worker.combined_output.write_bytes(b"combined")
    worker.fragments = []
    worker.groups = [
        (
            dual_gop_encoder._StartGroup(
                0,
                tmp_path / "gop-0.ts",
                100,
            ),
            2,
            101,
        ),
        (
            dual_gop_encoder._StartGroup(
                2,
                tmp_path / "gop-2.ts",
                200,
            ),
            1,
            200,
        ),
    ]
    in_stream = object()
    packets = [
        SimpleNamespace(size=1, pts=10, dts=9, is_keyframe=True, stream=None),
        SimpleNamespace(size=1, pts=11, dts=10, is_keyframe=False, stream=None),
        SimpleNamespace(size=1, pts=20, dts=19, is_keyframe=True, stream=None),
    ]
    destinations = []

    class Container:
        def __init__(self, *, source=False, options=None):
            self.streams = SimpleNamespace(video=[in_stream])
            self.source = source
            self.options = options
            self.muxed = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def demux(self, _stream):
            return list(packets)

        def add_stream_from_template(self, _stream):
            return object()

        def mux(self, packet):
            self.muxed.append((packet.pts, packet.dts, packet.is_keyframe))

    source = Container(source=True)

    def open_container(path, mode="r", **kwargs):
        if mode == "r":
            assert Path(path) == worker.combined_output
            return source
        destination = Container(options=kwargs)
        destinations.append(destination)
        return destination

    monkeypatch.setattr(dual_gop_encoder.av, "open", open_container)

    worker._split_groups()

    assert [destination.muxed for destination in destinations] == [
        [(0, 0, True), (1, 1, False)],
        [(0, 0, True)],
    ]
    assert [destination.options for destination in destinations] == [
        {
            "format": "mpegts",
            "container_options": {"mpegts_copyts": "1", "muxdelay": "0"},
        },
        {
            "format": "mpegts",
            "container_options": {"mpegts_copyts": "1", "muxdelay": "0"},
        },
    ]
    assert [fragment.index for fragment in worker.fragments] == [0, 2]
    assert not worker.combined_output.exists()


def test_dual_gop_abort_skips_worker_that_was_never_started() -> None:
    started = MagicMock(ident=17)
    started.is_alive.return_value = False
    unstarted = MagicMock(ident=None)
    writer = object.__new__(dual_gop_encoder.AmdDualGopFrameWriter)
    writer.workers = [started, unstarted]

    writer._abort_workers()

    started.abort.assert_called_once_with()
    started.join.assert_called_once_with(timeout=30)
    unstarted.abort.assert_called_once_with()
    unstarted.join.assert_not_called()


def test_dual_gop_abort_defers_shared_teardown_while_worker_is_alive(
    monkeypatch,
    tmp_path,
) -> None:
    worker = MagicMock(ident=17)
    worker.is_alive.return_value = True
    worker.worker_index = 0
    writer = object.__new__(dual_gop_encoder.AmdDualGopFrameWriter)
    writer.closed = False
    writer.workers = [worker]
    writer.work_dir = tmp_path / "dual-work"
    writer.host_pool = MagicMock()
    writer.template = SimpleNamespace(
        _packed=object(),
        _cas_luma=object(),
        _converter=object(),
        _lut_applier=object(),
        _cas=object(),
    )
    trim = MagicMock()
    monkeypatch.setattr(writer, "_trim_host_allocator", trim)

    writer.abort()

    assert writer.template._converter is not None
    writer.host_pool.close.assert_not_called()
    trim.assert_not_called()


def test_dual_gop_mux_durations_follow_adjacent_source_pts() -> None:
    fragments = [
        dual_gop_encoder._CompletedGroup(0, Path("gop-0.ts"), 100, 109),
        dual_gop_encoder._CompletedGroup(1, Path("gop-1.ts"), 115, 117),
    ]

    paths = dual_gop_encoder._mux_paths_with_source_durations(
        fragments,
        time_base=Fraction(1, 100),
        frame_step=Fraction(1, 1),
    )

    assert paths == [(Path("gop-0.ts"), 0.15), (Path("gop-1.ts"), 0.03)]


def test_dual_gop_mux_durations_reject_missing_fragment_index() -> None:
    fragments = [
        dual_gop_encoder._CompletedGroup(1, Path("gop-1.ts"), 100, 109),
    ]

    with pytest.raises(RuntimeError, match="order is incomplete"):
        dual_gop_encoder._mux_paths_with_source_durations(
            fragments,
            time_base=Fraction(1, 100),
            frame_step=Fraction(1, 1),
        )


@pytest.mark.parametrize("smart_fragment", [False, True])
def test_dual_gop_close_uses_the_matching_assembly_layer(
    monkeypatch,
    tmp_path,
    smart_fragment,
) -> None:
    fragment = tmp_path / "gop-000000.ts"
    fragment.write_bytes(b"fragment")
    worker = MagicMock(
        fragments=[dual_gop_encoder._CompletedGroup(0, fragment, 100, 109)]
    )
    worker.is_alive.return_value = False
    writer = object.__new__(dual_gop_encoder.AmdDualGopFrameWriter)
    writer.closed = False
    writer.frame_count = 10
    writer.group_index = 0
    writer.group_worker = None
    writer.workers = [worker]
    writer.work_dir = tmp_path / "dual-work"
    writer.work_dir.mkdir()
    writer.template = _encoder(smart_fragment=smart_fragment)
    writer.template.metadata.time_base = Fraction(1, 100)
    writer.template.metadata.video_file = str(tmp_path / "source.mp4")
    writer.template.output_path = tmp_path / "output.nut"
    writer.template._packed = object()
    writer.template._cas_luma = object()
    writer.template._converter = object()
    writer.template._lut_applier = object()
    writer.template._cas = object()
    writer.host_pool = SimpleNamespace(
        in_use=0,
        allocated=3,
        capacity=dual_gop_encoder.AMD_DUAL_GOP_PINNED_POOL_SIZE,
        peak_in_use=2,
    )
    monkeypatch.setattr(writer, "_put", MagicMock())
    monkeypatch.setattr(writer, "_errors", lambda: [])
    monkeypatch.setattr(
        dual_gop_encoder,
        "validate_hevc_fragment_parameter_sets",
        MagicMock(),
    )
    concatenate = MagicMock()
    final_mux = MagicMock()
    monkeypatch.setattr(dual_gop_encoder, "concatenate_fragments", concatenate)
    monkeypatch.setattr(dual_gop_encoder, "mux_fragments_final_output", final_mux)

    writer.close()

    if smart_fragment:
        concatenate.assert_called_once()
        final_mux.assert_not_called()
    else:
        final_mux.assert_called_once()
        concatenate.assert_not_called()


class _FakeDualWriter:
    instances: list["_FakeDualWriter"] = []

    def __init__(self, encoder) -> None:
        self.encoder = encoder
        self.writes = []
        self.closed = False
        self.aborted = False
        self.instances.append(self)

    def write(self, frame, pts, *, apply_lut=True) -> None:
        self.writes.append((frame, pts, apply_lut))

    def close(self) -> None:
        self.closed = True

    def abort(self) -> None:
        self.aborted = True


def test_offline_writer_routes_close_to_dual_writer(monkeypatch) -> None:
    _FakeDualWriter.instances.clear()
    monkeypatch.setattr(dual_gop_encoder, "use_dual_gop_writer", lambda *a, **k: True)
    monkeypatch.setattr(dual_gop_encoder, "AmdDualGopFrameWriter", _FakeDualWriter)
    heartbeat = [None]
    encoder = SimpleNamespace()

    writer = _OfflineFrameWriter(
        encoder,
        heartbeat,
        amd_dual_gop_encode=True,
    )
    writer.write("frame", 17, apply_lut=False)
    writer.close()

    dual = _FakeDualWriter.instances[-1]
    assert dual.writes == [("frame", 17, False)]
    assert dual.closed is True
    assert dual.aborted is False
    assert heartbeat[0] > 0


def test_offline_writer_routes_failure_to_dual_abort(monkeypatch) -> None:
    _FakeDualWriter.instances.clear()
    monkeypatch.setattr(dual_gop_encoder, "use_dual_gop_writer", lambda *a, **k: True)
    monkeypatch.setattr(dual_gop_encoder, "AmdDualGopFrameWriter", _FakeDualWriter)
    writer = _OfflineFrameWriter(
        SimpleNamespace(),
        [0.0],
        amd_dual_gop_encode=True,
    )

    writer.close(abort=True)

    dual = _FakeDualWriter.instances[-1]
    assert dual.aborted is True
    assert dual.closed is False
