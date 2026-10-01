from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from scripts import probe_amd_native_yuv_roundtrip as probe


def test_parse_order_normalizes_short_aliases() -> None:
    assert probe._parse_order("baseline,native,yuv,rgb") == (
        probe.ARM_BASELINE,
        probe.ARM_NATIVE_YUV,
        probe.ARM_NATIVE_YUV,
        probe.ARM_BASELINE,
    )


@pytest.mark.parametrize("value", ("", "baseline,wat", "wat"))
def test_parse_order_rejects_unknown_or_empty_arms(value: str) -> None:
    with pytest.raises(Exception, match="baseline/native"):
        probe._parse_order(value)


def test_validate_native_frame_accepts_fixed_amf_nv12() -> None:
    frame = SimpleNamespace(
        format=SimpleNamespace(name="amf"),
        sw_format=SimpleNamespace(name="nv12"),
        width=8,
        height=4,
    )

    probe._validate_native_frame(
        frame,
        software_format="nv12",
        width=8,
        height=4,
    )


@pytest.mark.parametrize(
    ("frame_format", "software_format", "width", "height"),
    (
        ("yuv420p", "nv12", 8, 4),
        ("amf", "p010le", 8, 4),
        ("amf", "nv12", 10, 4),
        ("amf", "nv12", 8, 6),
    ),
)
def test_validate_native_frame_rejects_format_or_geometry_changes(
    frame_format: str,
    software_format: str,
    width: int,
    height: int,
) -> None:
    frame = SimpleNamespace(
        format=SimpleNamespace(name=frame_format),
        sw_format=SimpleNamespace(name=software_format),
        width=width,
        height=height,
    )

    with pytest.raises(probe.VideoDecodeError, match="fixed AMF Vulkan NV12"):
        probe._validate_native_frame(
            frame,
            software_format="nv12",
            width=8,
            height=4,
        )


def test_validate_packed_tensor_rejects_wrong_shape_dtype_or_device() -> None:
    with pytest.raises(ValueError, match="shape"):
        probe._validate_packed_tensor(
            torch.empty((5, 8), dtype=torch.uint8),
            width=8,
            height=4,
            dtype=torch.uint8,
        )
    with pytest.raises(TypeError, match="dtype"):
        probe._validate_packed_tensor(
            torch.empty((6, 8), dtype=torch.uint16),
            width=8,
            height=4,
            dtype=torch.uint8,
        )
    with pytest.raises(ValueError, match="must reside on HIP"):
        probe._validate_packed_tensor(
            torch.empty((6, 8), dtype=torch.uint8),
            width=8,
            height=4,
            dtype=torch.uint8,
        )


def test_expected_format_tracks_source_bit_depth() -> None:
    assert probe._expected_format(SimpleNamespace(is_10bit=False)) == (
        "nv12",
        torch.uint8,
    )
    assert probe._expected_format(SimpleNamespace(is_10bit=True)) == (
        "p010le",
        torch.uint16,
    )


def test_native_encoder_is_probe_local_subclass() -> None:
    assert issubclass(probe.PackedYuvVideoEncoder, probe.NvidiaVideoEncoder)
    assert probe.PackedYuvVideoEncoder is not probe.NvidiaVideoEncoder


def test_native_dual_writer_is_probe_local_subclass() -> None:
    assert issubclass(
        probe.PackedYuvDualGopFrameWriter,
        probe.dual_gop_encoder.AmdDualGopFrameWriter,
    )
    assert (
        probe.PackedYuvDualGopFrameWriter
        is not probe.dual_gop_encoder.AmdDualGopFrameWriter
    )


def test_native_dual_prepare_copies_packed_yuv_without_rgb_conversion(
    monkeypatch,
) -> None:
    writer = object.__new__(probe.PackedYuvDualGopFrameWriter)
    writer.failed = threading.Event()
    writer.packed_d2h_seconds = 0.0
    writer.packed_frames = 0
    writer.host_pool = MagicMock()
    host_yuv = MagicMock()
    writer.host_pool.acquire.return_value = host_yuv
    writer.template = SimpleNamespace(
        metadata=SimpleNamespace(video_height=4, video_width=8),
        spec=SimpleNamespace(ten_bit=False, frame_format="nv12"),
        stream=MagicMock(),
    )
    packed = MagicMock()
    validate = MagicMock()
    monkeypatch.setattr(probe, "_validate_packed_tensor", validate)
    from_dlpack = MagicMock(return_value="av-frame")
    monkeypatch.setattr(probe.av.VideoFrame, "from_dlpack", from_dlpack)

    result = writer._prepare(packed, apply_lut=False)

    assert result == ("av-frame", host_yuv)
    validate.assert_called_once_with(
        packed,
        width=8,
        height=4,
        dtype=torch.uint8,
    )
    writer.template.stream.synchronize.assert_called_once_with()
    host_yuv.copy_.assert_called_once_with(packed, non_blocking=False)
    from_dlpack.assert_called_once_with(
        [host_yuv.__getitem__.return_value, host_yuv.__getitem__.return_value],
        format="nv12",
    )
    assert writer.packed_frames == 1
    assert writer.packed_d2h_seconds >= 0.0


def test_build_encoder_enables_exact_dual_gop_contract(tmp_path) -> None:
    encoder_type = MagicMock(return_value="encoder")

    result = probe._build_encoder(
        encoder_type,
        output=tmp_path / "out.mp4",
        metadata="metadata",
        device=torch.device("cpu"),
        pts_origin=123,
        dual_gop=True,
    )

    assert result == "encoder"
    kwargs = encoder_type.call_args.kwargs
    assert kwargs["encoder_settings"] == {
        "g": probe.dual_gop_encoder.AMD_DUAL_GOP_SIZE,
        "bf": 0,
    }
    assert kwargs["auto_source_rate"] is True
    assert kwargs["prefer_amf_host_native"] is True
    assert kwargs["smart_fragment"] is True


def test_reader_transport_validation_requires_balanced_lifecycle() -> None:
    frames = 7
    stats = {
        "copy_to_hip_calls": frames,
        "copy_to_hip_successes": frames,
        "hip_d2d_plane_copies": frames * 2,
        "hip_external_memory_imports": frames,
        "hip_external_memory_destroys": frames,
        "hip_mapped_buffer_acquires": frames,
        "hip_mapped_buffer_releases": frames,
        "vulkan_export_fd_close_calls": frames,
        "fixed_context_session_create_calls": 1,
        "fixed_context_session_close_calls": 1,
        "fixed_context_session_closed": True,
        "copy_to_hip_failures": 0,
        "failed_bridge_copies": 0,
        "vulkan_export_fd_close_failures": 0,
        "fixed_context_session_close_failures": 0,
        "hip_non_d2d_copy_calls": 0,
        "host_frame_transfers": 0,
        "cpu_map_calls": 0,
        "staging_copy_calls": 0,
        "d2h_copy_calls": 0,
        "av_hwframe_transfer_data_calls": 0,
        "transport_reconfigures": 0,
        "transport_restarts": 0,
    }

    assert probe.validate_reader_stats(stats, expected_frames=frames)["passed"]

    stats["hip_external_memory_destroys"] -= 1
    with pytest.raises(RuntimeError, match="hip_external_memory_destroys"):
        probe.validate_reader_stats(stats, expected_frames=frames)
