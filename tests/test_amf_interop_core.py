"""Focused contracts for the explicit Linux AMD AMF interop core."""

from __future__ import annotations

import gc
import json
import logging
import weakref
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

import jasna.media.video_decoder as module
from jasna.accelerator import AcceleratorVendor
from jasna.media import VideoMetadata


def _metadata(
    *,
    codec: str = "h264",
    profile: str = "high",
    pixel_format: str = "nv12",
    is_10bit: bool = False,
    width: int = 16,
    height: int = 16,
) -> VideoMetadata:
    return VideoMetadata(
        video_file="input.mp4",
        video_height=height,
        video_width=width,
        video_fps=30.0,
        average_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        codec_name=codec,
        duration=1.0,
        time_base=Fraction(1, 30),
        start_pts=0,
        color_range=AvColorRange.MPEG,
        color_space=AvColorspace.ITU709,
        num_frames=30,
        is_10bit=is_10bit,
        profile=profile,
        pixel_format=pixel_format,
    )


def _reader(
    metadata: VideoMetadata,
    batch_size: int,
    vendor: AcceleratorVendor = AcceleratorVendor.AMD,
) -> module.NvidiaVideoReader:
    reader = module.NvidiaVideoReader(
        "input.mp4",
        batch_size,
        torch.device("cpu"),
        metadata,
    )
    reader.vendor = vendor
    return reader


@pytest.fixture(autouse=True)
def _clear_interop_env(monkeypatch):
    monkeypatch.delenv(module.DECODE_BACKEND_ENV, raising=False)
    monkeypatch.delenv(module.AMF_INTEROP_RESOURCE_CACHE_ENV, raising=False)
    monkeypatch.delenv(module.AMF_INTEROP_DECODE_COPY_STREAM_ENV, raising=False)
    monkeypatch.delenv(module.AMF_INTEROP_SURFACE_POOL_SIZE_ENV, raising=False)


def test_backend_enumeration_and_environment_override(monkeypatch) -> None:
    assert "amf-interop" in module._DECODE_BACKENDS
    monkeypatch.setenv(module.DECODE_BACKEND_ENV, "amf-interop")
    assert module._decode_backend() == "amf-interop"


@pytest.mark.parametrize(
    ("codec", "profile", "pixel_format", "is_10bit"),
    [
        ("h264", "main", "nv12", False),
        ("h264", "high", "nv12", False),
        ("hevc", "main", "yuv420p", False),
        ("hevc", "main 10", "p010le", True),
        ("av1", "main", "yuv420p", False),
        ("av1", "main", "nv12", False),
        ("av1", "main", "yuv420p10le", True),
        ("av1", "main", "p010le", True),
    ],
)
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_linux_amd_scope_accepts_only_supported_format_and_batch_matrix(
    monkeypatch,
    codec: str,
    profile: str,
    pixel_format: str,
    is_10bit: bool,
    batch_size: int,
) -> None:
    monkeypatch.setattr(module.sys, "platform", "linux")
    reader = _reader(
        _metadata(
            codec=codec,
            profile=profile,
            pixel_format=pixel_format,
            is_10bit=is_10bit,
        ),
        batch_size,
    )
    reader._validate_amf_interop_scope()


@pytest.mark.parametrize(
    ("platform", "vendor", "metadata"),
    [
        ("win32", AcceleratorVendor.AMD, _metadata()),
        ("linux", AcceleratorVendor.NVIDIA, _metadata()),
        (
            "linux",
            AcceleratorVendor.AMD,
            _metadata(codec="vp9", profile="profile 0", pixel_format="yuv420p"),
        ),
    ],
)
def test_windows_nvidia_and_unsupported_codec_are_rejected(
    monkeypatch,
    platform: str,
    vendor: AcceleratorVendor,
    metadata: VideoMetadata,
) -> None:
    monkeypatch.setattr(module.sys, "platform", platform)
    reader = _reader(metadata, 4, vendor)
    with pytest.raises(module.VideoDecodeError, match="cannot satisfy"):
        reader._validate_amf_interop_scope()


@pytest.mark.parametrize(
    ("platform", "vendor", "metadata", "eligible"),
    [
        ("linux", AcceleratorVendor.AMD, _metadata(), True),
        ("win32", AcceleratorVendor.AMD, _metadata(), False),
        ("linux", AcceleratorVendor.NVIDIA, _metadata(), False),
        (
            "linux",
            AcceleratorVendor.AMD,
            _metadata(codec="vp9", profile="profile 0", pixel_format="yuv420p"),
            False,
        ),
    ],
)
def test_auto_amf_interop_eligibility_is_the_shared_platform_and_format_gate(
    monkeypatch,
    platform: str,
    vendor: AcceleratorVendor,
    metadata: VideoMetadata,
    eligible: bool,
) -> None:
    monkeypatch.setattr(module.sys, "platform", platform)

    assert module.auto_amf_interop_eligible(metadata, vendor) is eligible


@pytest.mark.parametrize(
    "metadata",
    [
        _metadata(codec="av1", profile="professional", pixel_format="yuv420p"),
        _metadata(codec="av1", profile="main", pixel_format="", is_10bit=False),
        _metadata(codec="av1", profile="main", pixel_format="yuv420p12le", is_10bit=False),
        _metadata(codec="av1", profile="main", pixel_format="yuv420p", is_10bit=True),
        _metadata(codec="av1", profile="main", pixel_format="p010le", is_10bit=False),
    ],
)
def test_av1_out_of_scope_profiles_depths_and_formats_are_rejected(
    monkeypatch,
    metadata: VideoMetadata,
) -> None:
    monkeypatch.setattr(module.sys, "platform", "linux")
    with pytest.raises(module.VideoDecodeError, match="cannot satisfy"):
        _reader(metadata, 4)._validate_amf_interop_scope()


@pytest.mark.parametrize("batch_size", [0, 3, 5, 16])
def test_unapproved_batch_sizes_are_rejected(monkeypatch, batch_size: int) -> None:
    monkeypatch.setattr(module.sys, "platform", "linux")
    reader = _reader(_metadata(), batch_size)
    with pytest.raises(module.VideoDecodeError, match="batch_size"):
        reader._validate_amf_interop_scope()


@pytest.mark.parametrize(
    "metadata",
    [
        _metadata(codec="h264", profile="high", pixel_format="yuv420p"),
        _metadata(codec="hevc", profile="main", pixel_format="yuv420p"),
        _metadata(
            codec="hevc",
            profile="main 10",
            pixel_format="p010le",
            is_10bit=True,
        ),
    ],
)
def test_auto_linux_amd_selects_private_deferred_amf_interop(
    monkeypatch,
    metadata: VideoMetadata,
) -> None:
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module, "current_stream", lambda _device: None)
    monkeypatch.setattr(module, "vendor_for_device", lambda _device: AcceleratorVendor.AMD)
    reader = module.NvidiaVideoReader("input.mp4", 4, torch.device("cpu"), metadata)
    native_open = MagicMock()
    monkeypatch.setattr(reader, "_open_amf_interop_backend", native_open)

    reader.__enter__()

    assert reader._decode_backend == "auto"
    native_open.assert_called_once_with(decode_copy_stream="private-deferred")


@pytest.mark.parametrize(
    "metadata",
    [
        _metadata(codec="av1", profile="main", pixel_format="yuv420p"),
        _metadata(
            codec="av1",
            profile="main",
            pixel_format="p010le",
            is_10bit=True,
        ),
    ],
)
def test_auto_linux_amd_selects_av1_stable_resource_cache(
    monkeypatch,
    metadata: VideoMetadata,
) -> None:
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module, "current_stream", lambda _device: None)
    monkeypatch.setattr(module, "vendor_for_device", lambda _device: AcceleratorVendor.AMD)
    reader = module.NvidiaVideoReader("input.mp4", 4, torch.device("cpu"), metadata)
    native_open = MagicMock()
    monkeypatch.setattr(reader, "_open_amf_interop_backend", native_open)

    reader.__enter__()

    native_open.assert_called_once_with(
        decode_copy_stream="null",
        resource_cache=True,
    )


@pytest.mark.parametrize(
    "metadata",
    [
        _metadata(codec="av1", profile="professional", pixel_format="yuv420p"),
        _metadata(codec="h264", profile="baseline", pixel_format="yuv420p"),
        _metadata(codec="vp9", profile="profile 0", pixel_format="yuv420p"),
    ],
)
def test_auto_linux_amd_leaves_noneligible_formats_on_existing_order(
    monkeypatch,
    metadata: VideoMetadata,
) -> None:
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module, "current_stream", lambda _device: None)
    monkeypatch.setattr(module, "vendor_for_device", lambda _device: AcceleratorVendor.AMD)
    reader = module.NvidiaVideoReader("input.mp4", 4, torch.device("cpu"), metadata)
    native_open = MagicMock()
    pyav_open = MagicMock()
    monkeypatch.setattr(reader, "_open_amf_interop_backend", native_open)
    monkeypatch.setattr(reader, "_open_pyav", pyav_open)

    reader.__enter__()

    native_open.assert_not_called()
    pyav_open.assert_called_once_with(software_only=False)


def test_auto_linux_amd_native_failure_is_terminal(monkeypatch) -> None:
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module, "current_stream", lambda _device: None)
    monkeypatch.setattr(module, "vendor_for_device", lambda _device: AcceleratorVendor.AMD)
    reader = module.NvidiaVideoReader("input.mp4", 4, torch.device("cpu"), _metadata())
    native_open = MagicMock(side_effect=module.VideoDecodeError("native bridge failed"))
    pyav_open = MagicMock()
    monkeypatch.setattr(reader, "_open_amf_interop_backend", native_open)
    monkeypatch.setattr(reader, "_open_pyav", pyav_open)

    with pytest.raises(module.VideoDecodeError, match="native bridge failed"):
        reader.__enter__()

    pyav_open.assert_not_called()


def test_missing_bridge_fails_closed(monkeypatch) -> None:
    def missing(_name):
        raise ImportError("no ABI-matched extension")

    monkeypatch.setattr(module.importlib, "import_module", missing)
    with pytest.raises(module.VideoDecodeError, match="ABI-matched"):
        module._load_amf_interop_bridge()


def test_bridge_missing_entrypoint_fails_closed(monkeypatch) -> None:
    incomplete = SimpleNamespace(
        inspect_amf_surface=lambda _frame: {},
        copy_amf_surface_to_hip=lambda *_args: {},
        get_transport_stats=lambda: {},
        reset_transport_stats=lambda: None,
    )
    monkeypatch.setattr(module.importlib, "import_module", lambda _name: incomplete)
    with pytest.raises(module.VideoDecodeError, match="AmfVulkanHipInteropSession"):
        module._load_amf_interop_bridge()


def test_resource_cache_defaults_off_and_true_is_explicit(monkeypatch) -> None:
    assert module._amf_interop_resource_cache_enabled() is False
    monkeypatch.setenv(module.AMF_INTEROP_RESOURCE_CACHE_ENV, "0")
    assert module._amf_interop_resource_cache_enabled() is False
    monkeypatch.setenv(module.AMF_INTEROP_RESOURCE_CACHE_ENV, "true")
    assert module._amf_interop_resource_cache_enabled() is True


def test_decode_copy_stream_defaults_to_null_and_deferred_is_explicit(monkeypatch) -> None:
    assert module._amf_interop_decode_copy_stream() == "null"
    monkeypatch.setenv(
        module.AMF_INTEROP_DECODE_COPY_STREAM_ENV,
        "private-deferred",
    )
    assert module._amf_interop_decode_copy_stream() == "private-deferred"
    monkeypatch.setenv(module.AMF_INTEROP_DECODE_COPY_STREAM_ENV, "private")
    with pytest.raises(ValueError, match="expected 'null' or 'private-deferred'"):
        module._amf_interop_decode_copy_stream()


def test_surface_pool_candidate_is_explicit_and_rejects_known_starvation_values(
    monkeypatch,
) -> None:
    assert module._amf_interop_surface_pool_size() is None
    monkeypatch.setenv(module.AMF_INTEROP_SURFACE_POOL_SIZE_ENV, "16")
    assert module._amf_interop_surface_pool_size() == 16
    monkeypatch.setenv(module.AMF_INTEROP_SURFACE_POOL_SIZE_ENV, "8")
    with pytest.raises(ValueError, match="known to starve"):
        module._amf_interop_surface_pool_size()


@pytest.mark.parametrize(
    ("profile", "pixel_format"),
    [
        ("main 10", "p010le"),
        ("main 10", "yuv420p10le"),
        ("", "p010le"),
    ],
)
def test_surface_pool16_auto_scope_accepts_only_validated_8k_b4(
    monkeypatch,
    profile: str,
    pixel_format: str,
) -> None:
    monkeypatch.setattr(module.sys, "platform", "linux")
    metadata = _metadata(
        codec="hevc",
        profile=profile,
        pixel_format=pixel_format,
        is_10bit=True,
        width=8192,
        height=4096,
    )

    assert module._amf_interop_surface_pool_size(
        metadata,
        batch_size=4,
        vendor=AcceleratorVendor.AMD,
    ) == 16


@pytest.mark.parametrize(
    ("platform", "vendor", "batch_size", "metadata"),
    [
        (
            "win32",
            AcceleratorVendor.AMD,
            4,
            _metadata(
                codec="hevc",
                profile="main 10",
                pixel_format="p010le",
                is_10bit=True,
                width=8192,
                height=4096,
            ),
        ),
        (
            "linux",
            AcceleratorVendor.NVIDIA,
            4,
            _metadata(
                codec="hevc",
                profile="main 10",
                pixel_format="p010le",
                is_10bit=True,
                width=8192,
                height=4096,
            ),
        ),
        (
            "linux",
            AcceleratorVendor.AMD,
            8,
            _metadata(
                codec="hevc",
                profile="main 10",
                pixel_format="p010le",
                is_10bit=True,
                width=8192,
                height=4096,
            ),
        ),
        (
            "linux",
            AcceleratorVendor.AMD,
            4,
            _metadata(
                codec="hevc",
                profile="main",
                pixel_format="yuv420p",
                is_10bit=False,
                width=8192,
                height=4096,
            ),
        ),
        (
            "linux",
            AcceleratorVendor.AMD,
            4,
            _metadata(
                codec="hevc",
                profile="main 10",
                pixel_format="p010le",
                is_10bit=True,
                width=4096,
                height=2048,
            ),
        ),
    ],
)
def test_surface_pool16_auto_scope_leaves_other_routes_on_runtime_default(
    monkeypatch,
    platform: str,
    vendor: AcceleratorVendor,
    batch_size: int,
    metadata: VideoMetadata,
) -> None:
    monkeypatch.setattr(module.sys, "platform", platform)

    assert module._amf_interop_surface_pool_size(
        metadata,
        batch_size=batch_size,
        vendor=vendor,
    ) is None


def test_explicit_decoder_owns_amf_surfaces_and_keeps_runtime_pool_default(monkeypatch) -> None:
    reader = _reader(
        _metadata(codec="hevc", profile="main", pixel_format="yuv420p"),
        4,
    )
    decoder = SimpleNamespace(options={}, open=MagicMock())
    hwaccel = MagicMock()
    monkeypatch.setattr(module, "HWAccel", hwaccel)
    monkeypatch.setattr(
        module.av,
        "CodecContext",
        SimpleNamespace(create=MagicMock(return_value=decoder)),
    )
    source = SimpleNamespace(
        name="hevc_amf",
        extradata=b"header",
        width=8192,
        height=4096,
        framerate=None,
        sample_aspect_ratio=None,
        thread_type=None,
    )

    open_lock = MagicMock()
    open_lock.__enter__.return_value = None
    open_lock.__exit__.return_value = False
    monkeypatch.setattr(module, "_AMF_DECODER_OPEN_LOCK", open_lock)

    reader._setup_amf_decoder(source, fail_closed=True)

    assert hwaccel.call_args.kwargs["is_hw_owned"] is True
    assert module.av.CodecContext.create.call_args.args[:2] == ("hevc_amf", "r")
    assert "surface_pool_size" not in decoder.options
    assert "reset_on_keyframe" not in decoder.options
    decoder.open.assert_called_once_with(strict=False)
    open_lock.__enter__.assert_called_once_with()
    open_lock.__exit__.assert_called_once()


def test_explicit_decoder_applies_transaction_surface_pool_candidate(monkeypatch) -> None:
    reader = _reader(
        _metadata(codec="hevc", profile="main 10", pixel_format="p010le", is_10bit=True),
        4,
    )
    decoder = SimpleNamespace(options={}, open=MagicMock())
    monkeypatch.setattr(module, "HWAccel", MagicMock())
    monkeypatch.setattr(
        module.av,
        "CodecContext",
        SimpleNamespace(create=MagicMock(return_value=decoder)),
    )
    monkeypatch.setenv(module.AMF_INTEROP_SURFACE_POOL_SIZE_ENV, "16")
    source = SimpleNamespace(
        name="hevc_amf",
        extradata=b"header",
        width=8192,
        height=4096,
        framerate=None,
        sample_aspect_ratio=None,
        thread_type=None,
    )

    reader._setup_amf_decoder(source, fail_closed=True)

    assert decoder.options["surface_pool_size"] == "16"
    assert "reset_on_keyframe" not in decoder.options
    decoder.open.assert_called_once_with(strict=False)


def test_explicit_decoder_applies_automatic_8k_b4_surface_pool(monkeypatch) -> None:
    monkeypatch.setattr(module.sys, "platform", "linux")
    reader = _reader(
        _metadata(
            codec="hevc",
            profile="main 10",
            pixel_format="yuv420p10le",
            is_10bit=True,
            width=8192,
            height=4096,
        ),
        4,
    )
    decoder = SimpleNamespace(options={}, open=MagicMock())
    monkeypatch.setattr(module, "HWAccel", MagicMock())
    monkeypatch.setattr(
        module.av,
        "CodecContext",
        SimpleNamespace(create=MagicMock(return_value=decoder)),
    )
    source = SimpleNamespace(
        name="hevc_amf",
        extradata=b"header",
        width=8192,
        height=4096,
        framerate=None,
        sample_aspect_ratio=None,
        thread_type=None,
    )

    reader._setup_amf_decoder(source, fail_closed=True)

    assert decoder.options["surface_pool_size"] == "16"
    assert decoder.options["copy_output"] == "1"
    assert decoder.options["reset_on_keyframe"] == "1"
    decoder.open.assert_called_once_with(strict=False)


def test_cache_enable_fails_closed_outside_validated_av1_scope(monkeypatch) -> None:
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setenv(module.AMF_INTEROP_RESOURCE_CACHE_ENV, "1")
    loader = MagicMock(side_effect=AssertionError("cache must fail before load"))
    monkeypatch.setattr(module, "_load_amf_interop_bridge", loader)
    reader = _reader(_metadata(), 1)

    with pytest.raises(module.VideoDecodeError, match="supported only"):
        reader._open_amf_interop_backend()
    loader.assert_not_called()


def test_uploader_rejects_non_native_amf_frame_before_copy(monkeypatch) -> None:
    metadata = _metadata()
    copy = MagicMock()
    uploader = module.AmfInteropUploader(
        file="input.mp4",
        batch_size=1,
        device=torch.device("cpu"),
        metadata=metadata,
        height=16,
        width=16,
        full_range=False,
        audit=SimpleNamespace(copy_to_hip=copy),
    )
    monkeypatch.setattr(module, "YuvToRgbConverter", MagicMock())
    frame = SimpleNamespace(
        format=SimpleNamespace(name="cuda"),
        sw_format=SimpleNamespace(name="nv12"),
        width=16,
        height=16,
        pts=0,
    )

    with pytest.raises(module.VideoDecodeError, match="native AMF Vulkan"):
        list(uploader.frames(iter(()), [frame]))
    copy.assert_not_called()


def test_uploader_releases_old_amf_surfaces_before_reading_next_group(monkeypatch) -> None:
    class _TrackedAmfFrame:
        format = SimpleNamespace(name="amf")
        sw_format = SimpleNamespace(name="nv12")
        width = 16
        height = 16

        def __init__(self, pts: int, released: list[int]) -> None:
            self.pts = pts
            self._released = released

        def __del__(self) -> None:
            self._released.append(self.pts)

    class _Stream:
        def __init__(self) -> None:
            self.synchronize_calls = 0

        def synchronize(self) -> None:
            self.synchronize_calls += 1

    class _NextGroup:
        def __init__(self, prior_frame, stream: _Stream, released: list[int]) -> None:
            self._prior_frame = prior_frame
            self._stream = stream
            self._released = released
            self._calls = 0

        @property
        def calls(self) -> int:
            return self._calls

        def __iter__(self):
            return self

        def __next__(self):
            self._calls += 1
            if self._calls == 1:
                assert self._stream.synchronize_calls == 1
                gc.collect()
                assert self._prior_frame() is None
                assert self._released == [0]
                return _TrackedAmfFrame(1, self._released)
            raise StopIteration

    def _initial_group(released: list[int]):
        frame = _TrackedAmfFrame(0, released)
        return [frame], weakref.ref(frame)

    released: list[int] = []
    group, first_frame = _initial_group(released)
    stream = _Stream()
    decoded = _NextGroup(first_frame, stream, released)
    uploader = module.AmfInteropUploader(
        file="input.mp4",
        batch_size=1,
        device=torch.device("cpu"),
        metadata=_metadata(),
        height=16,
        width=16,
        full_range=False,
        audit=SimpleNamespace(copy_to_hip=lambda *_args: _safe_copy_result()),
    )
    monkeypatch.setattr(module, "current_stream", lambda _device: stream)
    monkeypatch.setattr(module, "YuvToRgbConverter", MagicMock())

    batches = uploader.frames(iter(decoded), group)
    del group
    batch, pts = next(batches)
    assert batch.shape == (1, 3, 16, 16)
    assert pts == [0]
    assert first_frame() is None
    assert released == [0]
    assert decoded.calls == 0
    del batch
    batch, pts = next(batches)
    assert batch.shape == (1, 3, 16, 16)
    assert pts == [1]
    assert decoded.calls == 1
    batches.close()


class _IdentitySession:
    def __init__(self, *, close_error: BaseException | None = None) -> None:
        self.closed = False
        self.close_error = close_error

    def close(self) -> None:
        if self.close_error is not None:
            raise self.close_error
        self.closed = True

    def stats(self) -> dict[str, object]:
        return {
            "cache_entries": 0,
            "closed": self.closed,
        }


class _DeferredIdentitySession(_IdentitySession):
    def __init__(self) -> None:
        super().__init__()
        self.copy_calls = 0

    def copy_amf_surface_to_hip_private_deferred_stream(self, *args):
        self.copy_calls += 1
        return _safe_deferred_copy_result(consumer_stream_handle=args[-1])

    def stats(self) -> dict[str, object]:
        calls = self.copy_calls
        return {
            "cache_entries": 0,
            "closed": self.closed,
            "vulkan_memory_exports": calls,
            "hip_external_memory_imports": calls,
            "hip_mapped_buffer_acquires": calls,
            "hip_mapped_buffer_releases": calls if self.closed else 0,
            "hip_external_memory_destroys": calls if self.closed else 0,
            "decode_private_deferred_source_release_hip_stream_create_calls": (
                1 if calls else 0
            ),
            "decode_private_deferred_source_release_hip_stream_create_failures": 0,
            "decode_private_deferred_source_release_hip_stream_destroy_calls": (
                1 if self.closed and calls else 0
            ),
            "decode_private_deferred_source_release_hip_stream_destroy_failures": 0,
            "decode_private_deferred_source_release_hip_async_copy_calls": calls * 2,
            "decode_private_deferred_source_release_hip_stream_synchronize_calls": 0,
            "decode_private_deferred_source_release_error_stream_synchronize_calls": 0,
            "decode_private_deferred_source_release_hip_event_create_calls": (
                6 if calls else 0
            ),
            "decode_private_deferred_source_release_hip_event_create_failures": 0,
            "decode_private_deferred_source_release_hip_event_record_calls": calls * 2,
            "decode_private_deferred_source_release_hip_event_record_failures": 0,
            "decode_private_deferred_source_release_hip_event_query_calls": max(calls - 1, 0),
            "decode_private_deferred_source_release_hip_event_query_not_ready": 0,
            "decode_private_deferred_source_release_hip_event_synchronize_calls": (
                (
                    max(calls - 3, 0) + min(calls, 3)
                    if self.closed
                    else max(calls - 3, 0)
                )
            ),
            "decode_private_deferred_source_release_hip_event_synchronize_failures": 0,
            "decode_private_deferred_source_release_hip_event_destroy_calls": (
                6 if self.closed and calls else 0
            ),
            "decode_private_deferred_source_release_hip_event_destroy_failures": 0,
            "decode_private_deferred_source_release_device_wait_calls": calls,
            "decode_private_deferred_source_release_device_wait_failures": 0,
            "decode_private_deferred_source_release_source_acquires": calls,
            "decode_private_deferred_source_release_source_releases": (
                calls if self.closed else max(calls - 3, 0)
            ),
            "decode_private_deferred_source_release_forced_drains": max(calls - 3, 0),
            "decode_private_deferred_source_release_close_drains": (
                min(calls, 3) if self.closed else 0
            ),
            "decode_private_deferred_source_release_max_in_flight": min(calls, 3),
            "decode_private_deferred_source_release_failures": 0,
            "last_decode_private_deferred_source_release_hip_stream_handle": (
                91 if calls else 0
            ),
            "decode_private_deferred_source_release_in_flight": (
                0 if self.closed else min(calls, 3)
            ),
        }


class _CacheIdentitySession(_IdentitySession):
    def __init__(self, *, misses: int = 2) -> None:
        super().__init__()
        self.copy_calls = 0
        self.misses = misses

    def copy_amf_surface_to_hip_resource_cache(self, *args):
        self.copy_calls += 1
        miss = self.copy_calls <= self.misses
        return _safe_copy_result(
            cache_hit=not miss,
            cache_miss=miss,
            copy_synchronization="null-stream-cache-retained",
        )

    def stats(self) -> dict[str, object]:
        misses = min(self.copy_calls, self.misses)
        return {
            "cache_entries": misses,
            "cache_hits": self.copy_calls - misses,
            "cache_misses": misses,
            "cache_active_external_imports": 0 if self.closed else misses,
            "cache_active_mappings": 0 if self.closed else misses,
            "cache_raw_handle_identity_changes": 0,
            "cache_stable_identity_raw_handle_changes": 0,
            "cache_fd_export_calls": self.copy_calls,
            "cache_fd_export_failures": 0,
            "cache_fd_stat_calls": self.copy_calls,
            "cache_fd_stat_failures": 0,
            "cache_fd_close_calls": self.copy_calls,
            "cache_fd_close_failures": 0,
            "cache_last_fd_close_errno": 0,
            "cache_fd_ownership_transfers": 0,
            "vulkan_memory_exports": self.copy_calls,
            "hip_external_memory_imports": misses,
            "hip_mapped_buffer_acquires": misses,
            "hip_mapped_buffer_releases": misses if self.closed else 0,
            "hip_external_memory_destroys": misses if self.closed else 0,
            "closed": self.closed,
        }


def _safe_inspection(*, device: int = 41, frames_context: int = 11) -> dict[str, object]:
    return {
        "memory_type": "vulkan",
        "vulkan": {"memory": 31, "device": device},
        "fixed_context": {
            "frames_context": frames_context,
            "amf_context": 21,
            "vulkan_device": device,
        },
    }


def _safe_copy_result(**overrides) -> dict[str, object]:
    result: dict[str, object] = {
        "width": 16,
        "height": 16,
        "bytes_per_sample": 1,
        "vulkan_export_fd_close_calls": 1,
        "vulkan_export_fd_close_result": 0,
        "vulkan_export_fd_close_errno": 0,
        "hip_result": 0,
        "hip_free_result": 0,
        "hip_destroy_result": 0,
        "d2d_plane_copies": 2,
        "decode_source_release_hip_stream_synchronize_calls": 1,
        "decode_source_release_hip_stream_synchronize_result": 0,
        "copy_synchronization": "null-stream-source-release",
        "fixed_context_bound": True,
    }
    result.update(overrides)
    return result


def _safe_deferred_copy_result(
    *,
    consumer_stream_handle: int = 71,
    **overrides,
) -> dict[str, object]:
    result: dict[str, object] = {
        "width": 16,
        "height": 16,
        "bytes_per_sample": 1,
        "vulkan_export_fd_close_calls": 1,
        "vulkan_export_fd_close_result": 0,
        "vulkan_export_fd_close_errno": 0,
        "hip_result": 0,
        "d2d_plane_copies": 2,
        "decode_source_release_hip_stream_synchronize_calls": 0,
        "decode_null_stream_source_release_hip_stream_synchronize_calls": 0,
        "decode_private_deferred_source_release_hip_async_copy_calls": 2,
        "decode_private_deferred_source_release_hip_stream_synchronize_calls": 0,
        "decode_private_deferred_source_release_error_stream_synchronize_calls": 0,
        "decode_private_deferred_source_release_hip_stream_create_calls": 0,
        "decode_private_deferred_source_release_hip_event_create_calls": 0,
        "decode_private_deferred_source_release_hip_event_record_calls": 2,
        "decode_private_deferred_source_release_hip_event_query_calls": 0,
        "decode_private_deferred_source_release_hip_event_query_not_ready": 0,
        "decode_private_deferred_source_release_hip_event_synchronize_calls": 0,
        "decode_private_deferred_source_release_hip_event_destroy_calls": 0,
        "decode_private_deferred_source_release_device_wait_calls": 1,
        "decode_private_deferred_source_release_source_acquires": 1,
        "decode_private_deferred_source_release_source_releases": 0,
        "decode_private_deferred_source_release_forced_drains": 0,
        "consumer_stream_handle": consumer_stream_handle,
        "copy_synchronization": "private-deferred-device-wait",
        "fixed_context_bound": True,
    }
    result.update(overrides)
    return result


def _audit(
    *,
    inspect=None,
    copy=None,
    session=None,
    process_stats=None,
    decode_copy_stream: str = "null",
    resource_cache: bool = False,
):
    return module._AmfInteropTransportAudit(
        inspect_amf_surface=inspect or (lambda _frame: _safe_inspection()),
        copy_amf_surface_to_hip=copy or (lambda *_args: _safe_copy_result()),
        get_transport_stats=process_stats or (lambda: {}),
        identity_session=session or _IdentitySession(),
        device=torch.device("cpu"),
        decode_copy_stream=decode_copy_stream,
        resource_cache=resource_cache,
    )


def test_explicit_open_fails_closed_when_session_copy_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(module.sys, "platform", "linux")
    session = _IdentitySession()
    bridge = SimpleNamespace(
        inspect_amf_surface=lambda _frame: _safe_inspection(),
        copy_amf_surface_to_hip=lambda *_args: _safe_copy_result(),
        get_transport_stats=lambda: {},
        AmfVulkanHipInteropSession=lambda _purpose: session,
    )
    reader = _reader(_metadata(), 1)
    pyav_open = MagicMock()
    monkeypatch.setattr(module, "_load_amf_interop_bridge", lambda: bridge)
    monkeypatch.setattr(reader, "_open_pyav", pyav_open)

    with pytest.raises(module.VideoDecodeError, match="copy_amf_surface_to_hip"):
        reader._open_amf_interop_backend()

    assert session.closed is True
    pyav_open.assert_not_called()


def test_explicit_open_uses_session_bound_copy_not_top_level_bridge(monkeypatch) -> None:
    class _SessionWithBoundCopy(_IdentitySession):
        def __init__(self) -> None:
            super().__init__()
            self.copy_calls = []

        def copy_amf_surface_to_hip(self, *args):
            self.copy_calls.append(args)
            return _safe_copy_result()

    monkeypatch.setattr(module.sys, "platform", "linux")
    session = _SessionWithBoundCopy()
    top_level_copy = MagicMock(side_effect=AssertionError("top-level copy used"))
    bridge = SimpleNamespace(
        inspect_amf_surface=lambda _frame: _safe_inspection(),
        copy_amf_surface_to_hip=top_level_copy,
        get_transport_stats=lambda: {},
        AmfVulkanHipInteropSession=lambda _purpose: session,
    )
    reader = _reader(_metadata(), 1)
    pyav_open = MagicMock()
    monkeypatch.setattr(module, "_load_amf_interop_bridge", lambda: bridge)
    monkeypatch.setattr(reader, "_open_pyav", pyav_open)

    reader._open_amf_interop_backend()
    audit = reader._amf_interop_audit
    assert audit is not None
    frame = object()
    audit.copy_to_hip(frame, 123, 456)
    reader._close_amf_interop_backend()

    pyav_open.assert_called_once_with(software_only=False, amf_interop=True)
    assert session.copy_calls == [(frame, 123, 456, 0)]
    top_level_copy.assert_not_called()
    assert session.closed is True


def test_av1_cache_open_uses_session_cache_and_balances_reader_epoch(monkeypatch) -> None:
    monkeypatch.setattr(module.sys, "platform", "linux")
    session = _CacheIdentitySession(misses=2)
    constructor = MagicMock(return_value=session)
    bridge = SimpleNamespace(
        inspect_amf_surface=lambda _frame: _safe_inspection(),
        copy_amf_surface_to_hip=lambda *_args: _safe_copy_result(),
        get_transport_stats=lambda: {},
        AmfVulkanHipInteropSession=constructor,
    )
    reader = _reader(
        _metadata(codec="av1", profile="main", pixel_format="yuv420p"),
        4,
    )
    monkeypatch.setattr(module, "_load_amf_interop_bridge", lambda: bridge)
    monkeypatch.setattr(reader, "_open_pyav", MagicMock())

    reader._open_amf_interop_backend(resource_cache=True)
    audit = reader._amf_interop_audit
    assert audit is not None
    for index in range(4):
        audit.copy_to_hip(object(), index + 1, 128)
    reader._close_amf_interop_backend(validate=True)

    constructor.assert_called_once_with("decode", resource_cache=True)
    assert session.closed is True
    assert reader.amf_interop_stats is not None
    assert reader.amf_interop_stats["resource_cache_hits"] == 2
    assert reader.amf_interop_stats["resource_cache_misses"] == 2
    assert reader.amf_interop_stats["hip_external_memory_imports"] == 2
    assert reader.amf_interop_stats["hip_external_memory_destroys"] == 2
    assert reader.amf_interop_stats["cache_fd_close_calls"] == 4
    assert reader.amf_interop_stats["cache_fd_close_failures"] == 0
    assert reader.amf_interop_stats["cache_fd_ownership_transfers"] == 0


def test_cache_close_audit_fails_closed_on_native_fd_close_error() -> None:
    session = _CacheIdentitySession(misses=1)
    audit = _audit(
        copy=session.copy_amf_surface_to_hip_resource_cache,
        session=session,
        resource_cache=True,
    )
    audit.copy_to_hip(object(), 1, 128)
    audit.copy_to_hip(object(), 2, 128)
    audit.close()
    original_stats = session.stats

    def failed_close_stats():
        stats = original_stats()
        stats["cache_fd_close_failures"] = 1
        stats["cache_last_fd_close_errno"] = 9
        return stats

    session.stats = failed_close_stats

    with pytest.raises(module.VideoDecodeError, match="resource/lifetime"):
        audit.validate_closed()


def test_reader_close_serializes_validated_transport_audit(caplog) -> None:
    stats = {
        "schema": "jasna.amf.vulkan-hip-transport.v1",
        "copy_to_hip_calls": 120,
        "copy_to_hip_successes": 120,
        "vulkan_export_fd_close_calls": 120,
        "fixed_context_session_closed": True,
    }
    audit = SimpleNamespace(
        close=MagicMock(),
        validate_closed=MagicMock(return_value=stats),
        snapshot=MagicMock(side_effect=AssertionError("partial snapshot used")),
    )
    reader = _reader(_metadata(), 1)
    reader._amf_interop_audit = audit

    with caplog.at_level(logging.DEBUG, logger=module.__name__):
        reader._close_amf_interop_backend(validate=True)

    audit.close.assert_called_once_with()
    audit.validate_closed.assert_called_once_with()
    records = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith(module.AMF_INTEROP_STATS_PREFIX)
    ]
    assert len(records) == 1
    assert next(
        record.levelno for record in caplog.records if record.getMessage() == records[0]
    ) == logging.DEBUG
    assert json.loads(records[0][len(module.AMF_INTEROP_STATS_PREFIX) :]) == stats
    summaries = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("AMF D2D audit:")
    ]
    assert summaries == [
        "AMF D2D audit: copies=120/120, cache hit/miss=0/0, "
        "FD close/fail=120/0, forbidden host/map/staging/D2H/bridge=0/0/0/0/0"
    ]


def test_deferred_open_requires_session_entrypoint_and_dependency_probe(monkeypatch) -> None:
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setenv(
        module.AMF_INTEROP_DECODE_COPY_STREAM_ENV,
        "private-deferred",
    )
    session = _IdentitySession()
    bridge = SimpleNamespace(
        inspect_amf_surface=lambda _frame: _safe_inspection(),
        copy_amf_surface_to_hip=lambda *_args: _safe_copy_result(),
        get_transport_stats=lambda: {},
        AmfVulkanHipInteropSession=lambda _purpose: session,
    )
    reader = _reader(_metadata(), 1)
    monkeypatch.setattr(module, "_load_amf_interop_bridge", lambda: bridge)

    with pytest.raises(module.VideoDecodeError, match="private_deferred_stream"):
        reader._open_amf_interop_backend()

    assert session.closed is True


def test_deferred_open_runs_one_time_dependency_probe(monkeypatch) -> None:
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setenv(
        module.AMF_INTEROP_DECODE_COPY_STREAM_ENV,
        "private-deferred",
    )
    session = _DeferredIdentitySession()
    probe = MagicMock(
        return_value={
            "mode": "private-deferred",
            "consumer_stream_handle": 79,
            "stream_create_calls": 1,
            "stream_synchronize_calls": 0,
            "device_wait_calls": 1,
            "event_create_calls": 2,
            "event_record_calls": 2,
            "event_synchronize_calls": 1,
            "event_destroy_calls": 2,
        }
    )
    bridge = SimpleNamespace(
        inspect_amf_surface=lambda _frame: _safe_inspection(),
        copy_amf_surface_to_hip=lambda *_args: _safe_copy_result(),
        get_transport_stats=lambda: {},
        verify_private_deferred_stream_dependency=probe,
        AmfVulkanHipInteropSession=lambda _purpose: session,
    )
    reader = _reader(_metadata(), 1)
    pyav_open = MagicMock()
    monkeypatch.setattr(module, "_load_amf_interop_bridge", lambda: bridge)
    monkeypatch.setattr(
        module,
        "new_stream",
        lambda _device: SimpleNamespace(cuda_stream=79),
    )
    monkeypatch.setattr(reader, "_open_pyav", pyav_open)

    reader._open_amf_interop_backend()

    assert reader._amf_interop_audit is not None
    assert reader._amf_interop_audit.decode_copy_stream == "private-deferred"
    assert reader._amf_interop_consumer_stream.cuda_stream == 79
    probe.assert_called_once_with(0, 79)
    pyav_open.assert_called_once_with(software_only=False, amf_interop=True)
    reader._close_amf_interop_backend()


@pytest.mark.parametrize(
    "counter",
    [
        "host_frame_transfers",
        "cpu_map_calls",
        "staging_copy_calls",
        "d2h_copy_calls",
        "hip_non_d2d_copy_calls",
        "av_hwframe_transfer_data_calls",
        "failed_bridge_copies",
        "vulkan_export_fd_close_failures",
    ],
)
def test_transport_audit_rejects_host_or_non_d2d_counters(counter: str) -> None:
    audit = _audit(copy=lambda *_args: _safe_copy_result(**{counter: 1}))
    with pytest.raises(module.VideoDecodeError, match="non-native transport"):
        audit.copy_to_hip(object(), 1, 128)
    assert audit.snapshot()["copy_to_hip_failures"] == 1


def test_transport_audit_rejects_bridge_failed_copy() -> None:
    audit = _audit(copy=lambda *_args: _safe_copy_result(hip_result=700))
    with pytest.raises(module.VideoDecodeError, match="did not prove"):
        audit.copy_to_hip(object(), 1, 128)
    assert audit.snapshot()["copy_to_hip_failures"] == 1


@pytest.mark.parametrize(
    "fd_telemetry",
    [
        {"vulkan_export_fd_close_calls": 0},
        {
            "vulkan_export_fd_close_result": -1,
            "vulkan_export_fd_close_errno": 5,
        },
    ],
)
def test_transport_audit_rejects_unbalanced_vulkan_export_fd(
    fd_telemetry: dict[str, int],
) -> None:
    audit = _audit(copy=lambda *_args: _safe_copy_result(**fd_telemetry))
    with pytest.raises(module.VideoDecodeError, match="did not prove"):
        audit.copy_to_hip(object(), 1, 128)
    assert audit.snapshot()["copy_to_hip_failures"] == 1


def test_transport_audit_rejects_copy_without_fixed_context_binding() -> None:
    audit = _audit(copy=lambda *_args: _safe_copy_result(fixed_context_bound=False))
    with pytest.raises(module.VideoDecodeError, match="did not prove"):
        audit.copy_to_hip(object(), 1, 128)
    assert audit.snapshot()["copy_to_hip_failures"] == 1


def test_transport_audit_rejects_failed_copy_counter_from_bridge() -> None:
    audit = _audit(process_stats=lambda: {"failed_bridge_copies": 1})
    with pytest.raises(module.VideoDecodeError, match="non-native transport"):
        audit.copy_to_hip(object(), 1, 128)
    assert audit.snapshot()["copy_to_hip_failures"] == 1


def test_transport_audit_rejects_fixed_context_change() -> None:
    inspections = iter([_safe_inspection(device=41), _safe_inspection(device=42)])
    audit = _audit(inspect=lambda _frame: next(inspections))
    audit.copy_to_hip(object(), 1, 128)
    with pytest.raises(module.VideoDecodeError, match="identity changed"):
        audit.copy_to_hip(object(), 2, 128)


@pytest.mark.parametrize("field", ["frames_context", "amf_context", "vulkan_device"])
def test_transport_audit_rejects_null_fixed_context_identity(field: str) -> None:
    inspection = _safe_inspection()
    inspection["fixed_context"][field] = 0
    audit = _audit(inspect=lambda _frame: inspection)
    with pytest.raises(module.VideoDecodeError, match="non-null AMF context"):
        audit.copy_to_hip(object(), 1, 128)


def test_transport_audit_balances_create_close_and_per_frame_resources() -> None:
    session = _IdentitySession()
    audit = _audit(session=session)
    audit.copy_to_hip(object(), 1, 128)
    audit.copy_to_hip(object(), 2, 128)
    audit.close()
    stats = audit.validate_closed()

    assert session.closed is True
    assert stats["fixed_context_session_create_calls"] == 1
    assert stats["fixed_context_session_close_calls"] == 1
    assert stats["vulkan_memory_exports"] == 2
    assert stats["vulkan_export_fd_close_calls"] == 2
    assert stats["vulkan_export_fd_close_failures"] == 0
    assert stats["hip_external_memory_destroys"] == 2
    assert stats["hip_d2d_plane_copies"] == 4


@pytest.mark.parametrize("is_10bit", [False, True])
def test_batched_amd_converter_is_pixel_exact_across_b4_chunks(is_10bit: bool) -> None:
    height, width, count = 16, 16, 5
    reference = module.YuvToRgbConverter(
        height,
        width,
        AvColorspace.ITU709,
        False,
        is_10bit,
        torch.device("cpu"),
    )
    candidate = module._BatchedAmdYuvConverter(
        reference,
        4,
        torch.device("cpu"),
    )
    dtype = torch.uint16 if is_10bit else torch.uint8
    maximum = 65536 if is_10bit else 256
    packed = torch.randint(
        0,
        maximum,
        (count, height + height // 2, width),
        dtype=dtype,
    )
    expected = torch.empty((count, 3, height, width), dtype=torch.uint8)
    actual = torch.empty_like(expected)

    for index in range(count):
        reference.convert_into(
            packed[index, :height],
            packed[index, height:].view(height // 2, width // 2, 2),
            expected[index],
        )
    candidate.convert_into(packed, actual)

    assert torch.equal(actual, expected)


def test_deferred_audit_requires_a_non_null_consumer_stream() -> None:
    session = _DeferredIdentitySession()
    audit = _audit(
        copy=session.copy_amf_surface_to_hip_private_deferred_stream,
        session=session,
        decode_copy_stream="private-deferred",
    )

    with pytest.raises(module.VideoDecodeError, match="non-null Torch consumer"):
        audit.copy_to_hip(object(), 1, 128, consumer_stream_handle=0)


def test_deferred_audit_requires_event_pool_telemetry() -> None:
    audit = _audit(
        copy=lambda *_args: _safe_deferred_copy_result(
            decode_private_deferred_source_release_hip_event_destroy_calls=1
        ),
        session=_DeferredIdentitySession(),
        decode_copy_stream="private-deferred",
    )

    with pytest.raises(module.VideoDecodeError, match="did not prove"):
        audit.copy_to_hip(object(), 1, 128, consumer_stream_handle=71)


def test_deferred_audit_validates_three_slot_event_pool_at_close() -> None:
    session = _DeferredIdentitySession()
    audit = _audit(
        copy=session.copy_amf_surface_to_hip_private_deferred_stream,
        session=session,
        decode_copy_stream="private-deferred",
    )

    for index in range(5):
        audit.copy_to_hip(object(), index + 1, 128, consumer_stream_handle=71)
    audit.close()
    stats = audit.validate_closed()

    assert stats["decode_private_deferred_source_release_hip_event_create_calls"] == 6
    assert stats["decode_private_deferred_source_release_hip_event_destroy_calls"] == 6
    assert stats["decode_private_deferred_source_release_hip_event_record_calls"] == 10
    assert stats["decode_private_deferred_source_release_device_wait_calls"] == 5
    assert stats["decode_private_deferred_source_release_source_acquires"] == 5
    assert stats["decode_private_deferred_source_release_source_releases"] == 5
    assert stats["decode_private_deferred_source_release_max_in_flight"] == 3
    assert stats["decode_private_deferred_source_release_in_flight"] == 0
    assert stats["decode_private_deferred_source_release_hip_stream_synchronize_calls"] == 0


def test_deferred_copy_uses_verified_nondefault_stream_for_wait_and_conversion(monkeypatch) -> None:
    class _Stream:
        cuda_stream = 73

        def __init__(self) -> None:
            self.synchronize_calls = 0

        def synchronize(self) -> None:
            self.synchronize_calls += 1

    class _DeferredAudit:
        decode_copy_stream = "private-deferred"

        def __init__(self) -> None:
            self.calls = []

        def copy_to_hip(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return _safe_deferred_copy_result(
                consumer_stream_handle=kwargs["consumer_stream_handle"]
            )

    stream = _Stream()
    audit = _DeferredAudit()
    frame = SimpleNamespace(
        format=SimpleNamespace(name="amf"),
        sw_format=SimpleNamespace(name="nv12"),
        width=16,
        height=16,
        pts=0,
    )
    uploader = module.AmfInteropUploader(
        file="input.mp4",
        batch_size=1,
        device=torch.device("cpu"),
        metadata=_metadata(),
        height=16,
        width=16,
        full_range=False,
        audit=audit,
        consumer_stream=stream,
    )
    context = MagicMock()
    monkeypatch.setattr(module, "stream_context", lambda actual: context(actual))
    monkeypatch.setattr(module, "YuvToRgbConverter", MagicMock())

    batches = uploader.frames(iter(()), [frame])
    batch, pts = next(batches)

    assert batch.shape == (1, 3, 16, 16)
    assert pts == [0]
    assert audit.calls[0][1]["consumer_stream_handle"] == 73
    context.assert_called_once_with(stream)
    assert stream.synchronize_calls == 1


def test_reader_exit_keeps_primary_copy_error_when_deferred_teardown_also_fails() -> None:
    reader = _reader(_metadata(), 1)
    reader._close_pyav = MagicMock()
    reader._amf_interop_audit = SimpleNamespace(
        close=MagicMock(side_effect=RuntimeError("deferred close failed"))
    )

    # __exit__ returns normally so the original with-body exception propagates.
    assert reader.__exit__(RuntimeError, RuntimeError("copy failed"), None) is None
    reader._amf_interop_audit.close.assert_called_once()


def test_reader_exit_closes_cache_before_decoder_owned_allocations() -> None:
    reader = _reader(
        _metadata(codec="av1", profile="main", pixel_format="yuv420p"),
        1,
    )
    order = []
    audit = SimpleNamespace(
        close=lambda: order.append("cache"),
        validate_closed=lambda: {},
    )
    reader._amf_interop_audit = audit
    reader._amf_interop_resource_cache = True
    reader._close_pyav = lambda: order.append("decoder")

    reader.__exit__(None, None, None)

    assert order == ["cache", "decoder"]


def test_transport_audit_does_not_swallow_session_close_failure() -> None:
    audit = _audit(session=_IdentitySession(close_error=RuntimeError("native close failed")))
    with pytest.raises(RuntimeError, match="native close failed"):
        audit.close()
    assert audit.snapshot()["fixed_context_session_close_failures"] == 1
