from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from jasna.accelerator import AcceleratorVendor
from jasna.media import windows_d3d11_hip_resident as module


def _metadata(
    *,
    ten_bit: bool = False,
    width: int = 1920,
    height: int = 1080,
):
    return SimpleNamespace(
        codec_name="hevc",
        is_10bit=ten_bit,
        video_width=width,
        video_height=height,
    )


class _FakeSession:
    def __init__(self, role: str) -> None:
        self.role = role
        self.calls: list[tuple] = []

    def bind_decoder_frame(self, frame):
        self.calls.append(("bind_decoder_frame", frame))
        return {"format": "nv12", "slot_count": 4}

    def bind_encoder_context(self, encoder, decoder):
        self.calls.append(("bind_encoder_context", encoder, decoder))
        return {"shared_context": True, "slot_count": 4}

    def copy_decoded_to_hip(self, frame, pointer, size, stream):
        self.calls.append(("copy_decoded_to_hip", frame, pointer, size, stream))
        return {
            "width": 1920,
            "height": 1080,
            "bytes_per_sample": 1,
            "slot_count": 4,
        }

    def acquire_encoder_frame(self, pointer, size, pts, stream):
        self.calls.append(("acquire_encoder_frame", pointer, size, pts, stream))
        return object(), {"width": 1920, "height": 1080, "slot_count": 4}

    def begin_drain(self):
        self.calls.append(("begin_drain",))

    def close_session(self, timeout_ms):
        self.calls.append(("close_session", timeout_ms))
        return {
            "slot_count": 4,
            "closed": True,
            "prewarm_complete": True,
            "prewarm_slots_completed": 4,
            "root_create_calls": 1,
            "root_final_close_calls": 1,
            "bridge_slots_created": 4,
            "bridge_slots_destroyed": 4,
            "output_slots_created": 4,
            "output_slots_destroyed": 4,
            "free_output_slots": 4,
            "amf_owned_output_slots": 0,
            "retained_decoder_leases": 0,
            "pending_wrapper_owners": 0,
            "raw_d3d_map_calls": 0,
            "raw_h2d_bytes": 0,
            "raw_d2h_bytes": 0,
            "av_hwframe_transfer_calls": 0,
            "fifth_slot_allocation_attempts": 0,
            "hip_device_reset_calls": 0,
            "terminal_failures": 0,
            "teardown_failures": 0,
            "observer_unexpected_callbacks": 0,
            "external_memory_imports": 8,
            "external_memory_destroys": 8,
            "mapped_arrays_created": 8,
            "mapped_arrays_destroyed": 8,
            "hip_surface_objects_created": 8,
            "hip_surface_objects_destroyed": 8,
            "d3d12_fences_created": 1,
            "d3d12_fences_destroyed": 1,
            "d3d11_opened_fences_created": 1,
            "d3d11_opened_fences_destroyed": 1,
            "hip_external_semaphores_imported": 1,
            "hip_external_semaphores_destroyed": 1,
            "decoder_frame_leases_acquired": 1,
            "decoder_frame_leases_released": 1,
            "encoder_wrapper_creates": 1,
            "encoder_wrapper_buffer_releases": 1,
            "observer_leases_acquired": 0,
            "observer_leases_released": 0,
            "shared_fence_handles_created": 1,
            "shared_fence_handles_closed": 1,
        }

    def stats(self):
        return {"slot_count": 4, "free_output_slots": 4}


class _FakeBridge:
    def __init__(self) -> None:
        self.sessions: dict[str, _FakeSession] = {}
        self.keys: list[dict[str, object]] = []

    @staticmethod
    def api_version() -> int:
        return 1

    def create_or_get_process_root(self, device: int, key):
        assert device == 0
        key = dict(key)
        self.keys.append(key)
        return self.sessions.setdefault(key["role"], _FakeSession(key["role"]))


@pytest.fixture
def accepted_platform(monkeypatch):
    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )


def test_explicit_switch_is_off_by_default_and_rejects_typos() -> None:
    assert not module.windows_d3d11_hip_resident_requested({})
    assert not module.windows_d3d11_hip_resident_requested(
        {module.WINDOWS_D3D11_HIP_RESIDENT_ENV: "off"}
    )
    assert module.windows_d3d11_hip_resident_requested(
        {module.WINDOWS_D3D11_HIP_RESIDENT_ENV: "1"}
    )
    with pytest.raises(ValueError, match=module.WINDOWS_D3D11_HIP_RESIDENT_ENV):
        module.windows_d3d11_hip_resident_requested(
            {module.WINDOWS_D3D11_HIP_RESIDENT_ENV: "sometimes"}
        )


def test_two_reader_sessions_are_bounded_and_encoder_uses_secondary_context(
    accepted_platform,
) -> None:
    bridge = _FakeBridge()
    coordinator = module.WindowsD3D11HipResidentCoordinator(
        device=torch.device("cuda:0"),
        metadata=_metadata(),
        batch_size=4,
        output_codec="hevc",
        bridge=bridge,
    )
    primary_decoder = object()
    secondary_decoder = object()
    primary_frame = object()
    secondary_frame = object()
    coordinator.bind_decoder_frame(
        "decode-detect", primary_decoder, primary_frame
    )
    coordinator.bind_decoder_frame(
        "blend-encode", secondary_decoder, secondary_frame
    )
    coordinator.copy_decoded_to_hip(
        "decode-detect", primary_frame, 100, 1920 * 1080 * 3 // 2, 200
    )
    encoder = object()
    coordinator.bind_encoder_context(encoder)
    coordinator.acquire_encoder_frame(
        300,
        1920 * 1080 * 3 // 2,
        7,
        400,
    )
    stats = coordinator.close()

    assert {key["role"] for key in bridge.keys} == {
        "decode-detect",
        "blend-encode",
    }
    assert all(key["slot_count"] == 4 for key in bridge.keys)
    assert all(key["allocation_width"] == 1920 for key in bridge.keys)
    assert all(key["allocation_height"] == 1088 for key in bridge.keys)
    secondary_calls = bridge.sessions["blend-encode"].calls
    assert (
        "bind_encoder_context",
        encoder,
        secondary_decoder,
    ) in secondary_calls
    assert stats["decode-detect"]["raw_h2d_bytes"] == 0
    assert stats["blend-encode"]["raw_d2h_bytes"] == 0
    assert coordinator.close() == stats


def test_encoder_acquire_waits_for_an_observer_released_slot(
    monkeypatch,
    accepted_platform,
) -> None:
    bridge = _FakeBridge()
    coordinator = module.WindowsD3D11HipResidentCoordinator(
        device=torch.device("cuda:0"),
        metadata=_metadata(),
        batch_size=1,
        output_codec="hevc",
        bridge=bridge,
    )
    decoder = object()
    coordinator.bind_decoder_frame("blend-encode", decoder, object())
    coordinator.bind_encoder_context(object())
    session = bridge.sessions["blend-encode"]
    free_slots = iter((0, 0, 1))
    session.stats = lambda: {
        "slot_count": 4,
        "free_output_slots": next(free_slots),
    }
    sleeps: list[float] = []
    monkeypatch.setattr(module.time, "sleep", sleeps.append)

    frame, telemetry = coordinator.acquire_encoder_frame(300, 1024, 7, 400)

    assert frame is not None
    assert telemetry["slot_count"] == 4
    assert sleeps == [
        module.WINDOWS_D3D11_HIP_RESIDENT_SLOT_POLL_SECONDS,
        module.WINDOWS_D3D11_HIP_RESIDENT_SLOT_POLL_SECONDS,
    ]
    assert sum(
        call[0] == "acquire_encoder_frame" for call in session.calls
    ) == 1


def test_encoder_slot_wait_timeout_fails_without_a_fifth_acquire(
    monkeypatch,
    accepted_platform,
) -> None:
    bridge = _FakeBridge()
    coordinator = module.WindowsD3D11HipResidentCoordinator(
        device=torch.device("cuda:0"),
        metadata=_metadata(),
        batch_size=1,
        output_codec="hevc",
        bridge=bridge,
    )
    coordinator.bind_decoder_frame("blend-encode", object(), object())
    coordinator.bind_encoder_context(object())
    session = bridge.sessions["blend-encode"]
    session.stats = lambda: {"slot_count": 4, "free_output_slots": 0}
    monkeypatch.setattr(
        module,
        "WINDOWS_D3D11_HIP_RESIDENT_SLOT_WAIT_SECONDS",
        0.0,
    )

    with pytest.raises(
        module.WindowsD3D11HipResidentError,
        match="Timed out waiting for AMF",
    ):
        coordinator.acquire_encoder_frame(300, 1024, 7, 400)

    assert not any(
        call[0] == "acquire_encoder_frame" for call in session.calls
    )


def test_decoder_context_change_fails_closed(accepted_platform) -> None:
    coordinator = module.WindowsD3D11HipResidentCoordinator(
        device=torch.device("cuda:0"),
        metadata=_metadata(),
        batch_size=1,
        output_codec="hevc",
        bridge=_FakeBridge(),
    )
    coordinator.bind_decoder_frame("blend-encode", object(), object())
    with pytest.raises(
        module.WindowsD3D11HipResidentError,
        match="context changed",
    ):
        coordinator.bind_decoder_frame("blend-encode", object(), object())


def test_unproven_geometry_is_rejected(accepted_platform) -> None:
    metadata = _metadata()
    metadata.video_height = 720
    with pytest.raises(
        module.WindowsD3D11HipResidentError,
        match="1920x1080 and 3840x2160",
    ):
        module.WindowsD3D11HipResidentCoordinator(
            device=torch.device("cuda:0"),
            metadata=metadata,
            batch_size=1,
            output_codec="hevc",
            bridge=_FakeBridge(),
        )


def test_4k_geometry_uses_decoder_allocation_geometry(
    accepted_platform,
) -> None:
    bridge = _FakeBridge()
    coordinator = module.WindowsD3D11HipResidentCoordinator(
        device=torch.device("cuda:0"),
        metadata=_metadata(width=3840, height=2160),
        batch_size=4,
        output_codec="hevc",
        bridge=bridge,
    )

    coordinator.bind_decoder_frame("decode-detect", object(), object())
    coordinator.bind_decoder_frame("blend-encode", object(), object())
    coordinator.close()

    assert len(bridge.keys) == 2
    assert all(key["visible_width"] == 3840 for key in bridge.keys)
    assert all(key["visible_height"] == 2160 for key in bridge.keys)
    assert all(key["allocation_width"] == 3840 for key in bridge.keys)
    assert all(key["allocation_height"] == 2160 for key in bridge.keys)


def test_8k_geometry_is_rejected_after_dual_reader_capacity_gate(
    accepted_platform,
) -> None:
    with pytest.raises(
        module.WindowsD3D11HipResidentError,
        match="single-reader interop gate.*host commit safety floor",
    ):
        module.WindowsD3D11HipResidentCoordinator(
            device=torch.device("cuda:0"),
            metadata=_metadata(width=8192, height=4096),
            batch_size=1,
            output_codec="hevc",
            bridge=_FakeBridge(),
        )


@pytest.mark.parametrize(
    ("platform", "vendor", "ten_bit", "message"),
    [
        ("linux", AcceleratorVendor.AMD, False, "Windows is required"),
        ("win32", AcceleratorVendor.NVIDIA, False, "AMD HIP device"),
        ("win32", AcceleratorVendor.AMD, True, "Main10/P010"),
    ],
)
def test_scope_rejections_are_terminal(
    monkeypatch,
    platform,
    vendor,
    ten_bit,
    message,
) -> None:
    monkeypatch.setattr(module.sys, "platform", platform)
    monkeypatch.setattr(module, "vendor_for_device", lambda _device: vendor)
    with pytest.raises(module.WindowsD3D11HipResidentError, match=message):
        module.WindowsD3D11HipResidentCoordinator(
            device=torch.device("cuda:0"),
            metadata=_metadata(ten_bit=ten_bit),
            batch_size=4,
            output_codec="hevc",
            bridge=_FakeBridge(),
        )
