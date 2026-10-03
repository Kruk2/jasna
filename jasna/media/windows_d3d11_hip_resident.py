"""Explicit Windows AMD D3D11/HIP resident video transport.

This module is deliberately policy-heavy and native-code-light.  It owns the
product-facing lifetime and fail-closed checks around the separately built
``_jasna_amf_d3d11_hip_resident`` extension.  The route is experimental and is
never selected by the ordinary decoder/encoder ``auto`` policies.
"""

from __future__ import annotations

import importlib
import os
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from jasna import runtime_contract
from jasna.accelerator import AcceleratorVendor, vendor_for_device


WINDOWS_D3D11_HIP_RESIDENT_ENV = "JASNA_WINDOWS_D3D11_HIP_RESIDENT"
WINDOWS_D3D11_HIP_RESIDENT_BACKEND = "amf-d3d11-hip-resident"
WINDOWS_D3D11_HIP_RESIDENT_BRIDGE = "_jasna_amf_d3d11_hip_resident"
WINDOWS_D3D11_HIP_RESIDENT_API_VERSION = 1
WINDOWS_D3D11_HIP_RESIDENT_SLOT_COUNT = 4
WINDOWS_D3D11_HIP_RESIDENT_SLOT_WAIT_SECONDS = 5.0
WINDOWS_D3D11_HIP_RESIDENT_SLOT_POLL_SECONDS = 0.001
WINDOWS_D3D11_HIP_RESIDENT_ADMITTED_GEOMETRIES = {
    (1920, 1080): (1920, 1088),
    (3840, 2160): (3840, 2160),
}
WINDOWS_D3D11_HIP_RESIDENT_CAPACITY_REJECTED_GEOMETRIES = {
    (8192, 4096): (
        "8192x4096 passed the single-reader interop gate but the product "
        "dual-reader topology crosses the host commit safety floor"
    ),
}
_READER_ROLES = frozenset({"decode-detect", "blend-encode"})


class WindowsD3D11HipResidentError(RuntimeError):
    """Raised when the explicit resident transport cannot honor its contract."""


def windows_d3d11_hip_resident_requested(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether the explicit full-pipeline experiment was requested."""

    values = os.environ if environ is None else environ
    raw = values.get(WINDOWS_D3D11_HIP_RESIDENT_ENV)
    if raw is None:
        return False
    value = str(raw).strip().casefold()
    if value in {"", "0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Invalid {WINDOWS_D3D11_HIP_RESIDENT_ENV} value {value!r}; "
        "expected 0/1, false/true, no/yes, or off/on"
    )


def _mapping_result(value: object, operation: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise WindowsD3D11HipResidentError(
            f"resident bridge {operation} returned {type(value).__name__}, "
            "expected a telemetry mapping"
        )
    return dict(value)


def _validate_closed_stats(stats: dict[str, object], role: str) -> None:
    required = (
        "slot_count",
        "closed",
        "prewarm_complete",
        "prewarm_slots_completed",
        "root_create_calls",
        "root_final_close_calls",
        "bridge_slots_created",
        "bridge_slots_destroyed",
        "output_slots_created",
        "output_slots_destroyed",
        "free_output_slots",
        "amf_owned_output_slots",
        "retained_decoder_leases",
        "pending_wrapper_owners",
        "raw_d3d_map_calls",
        "raw_h2d_bytes",
        "raw_d2h_bytes",
        "av_hwframe_transfer_calls",
        "fifth_slot_allocation_attempts",
        "hip_device_reset_calls",
        "terminal_failures",
        "teardown_failures",
        "observer_unexpected_callbacks",
        "external_memory_imports",
        "external_memory_destroys",
        "mapped_arrays_created",
        "mapped_arrays_destroyed",
        "hip_surface_objects_created",
        "hip_surface_objects_destroyed",
        "d3d12_fences_created",
        "d3d12_fences_destroyed",
        "d3d11_opened_fences_created",
        "d3d11_opened_fences_destroyed",
        "hip_external_semaphores_imported",
        "hip_external_semaphores_destroyed",
        "decoder_frame_leases_acquired",
        "decoder_frame_leases_released",
        "encoder_wrapper_creates",
        "encoder_wrapper_buffer_releases",
        "observer_leases_acquired",
        "observer_leases_released",
        "shared_fence_handles_created",
        "shared_fence_handles_closed",
    )
    missing = [name for name in required if name not in stats]
    if missing:
        raise WindowsD3D11HipResidentError(
            f"Resident close telemetry for {role} is missing: "
            + ", ".join(missing)
        )
    if int(stats["slot_count"]) != WINDOWS_D3D11_HIP_RESIDENT_SLOT_COUNT:
        raise WindowsD3D11HipResidentError(
            f"Resident close telemetry for {role} does not report four slots"
        )
    expected = {
        "closed": True,
        "prewarm_complete": True,
        "prewarm_slots_completed": WINDOWS_D3D11_HIP_RESIDENT_SLOT_COUNT,
        "root_create_calls": 1,
        "root_final_close_calls": 1,
        "bridge_slots_created": WINDOWS_D3D11_HIP_RESIDENT_SLOT_COUNT,
        "bridge_slots_destroyed": WINDOWS_D3D11_HIP_RESIDENT_SLOT_COUNT,
        "output_slots_created": WINDOWS_D3D11_HIP_RESIDENT_SLOT_COUNT,
        "output_slots_destroyed": WINDOWS_D3D11_HIP_RESIDENT_SLOT_COUNT,
        "free_output_slots": WINDOWS_D3D11_HIP_RESIDENT_SLOT_COUNT,
        "amf_owned_output_slots": 0,
        "retained_decoder_leases": 0,
        "pending_wrapper_owners": 0,
    }
    mismatched = {
        name: {"expected": value, "observed": stats[name]}
        for name, value in expected.items()
        if stats[name] != value
    }
    if mismatched:
        raise WindowsD3D11HipResidentError(
            f"Resident transport for {role} closed outside its fixed ownership "
            f"contract: {mismatched}"
        )
    forbidden = (
        "raw_d3d_map_calls",
        "raw_h2d_bytes",
        "raw_d2h_bytes",
        "av_hwframe_transfer_calls",
        "fifth_slot_allocation_attempts",
        "hip_device_reset_calls",
        "terminal_failures",
        "teardown_failures",
        "observer_unexpected_callbacks",
    )
    nonzero = {name: int(stats[name]) for name in forbidden if int(stats[name])}
    if nonzero:
        raise WindowsD3D11HipResidentError(
            f"Resident transport for {role} used a forbidden path: {nonzero}"
        )
    for created, destroyed in (
        ("bridge_slots_created", "bridge_slots_destroyed"),
        ("output_slots_created", "output_slots_destroyed"),
        ("external_memory_imports", "external_memory_destroys"),
        ("mapped_arrays_created", "mapped_arrays_destroyed"),
        ("hip_surface_objects_created", "hip_surface_objects_destroyed"),
        ("d3d12_fences_created", "d3d12_fences_destroyed"),
        ("d3d11_opened_fences_created", "d3d11_opened_fences_destroyed"),
        (
            "hip_external_semaphores_imported",
            "hip_external_semaphores_destroyed",
        ),
        ("decoder_frame_leases_acquired", "decoder_frame_leases_released"),
        ("encoder_wrapper_creates", "encoder_wrapper_buffer_releases"),
        ("observer_leases_acquired", "observer_leases_released"),
        ("shared_fence_handles_created", "shared_fence_handles_closed"),
    ):
        if int(stats[created]) != int(stats[destroyed]):
            raise WindowsD3D11HipResidentError(
                f"Resident transport for {role} has an unbalanced "
                f"{created}/{destroyed} ledger: {stats}"
            )


def _load_bridge():
    root = runtime_contract.default_runtime_root("win32").resolve()
    manifest = runtime_contract.validate_runtime_layout(root, platform="win32")
    bridge_record = manifest.get("windows_d3d11_hip_resident_bridge")
    if not isinstance(bridge_record, Mapping):
        raise WindowsD3D11HipResidentError(
            "The explicit Windows D3D11/HIP resident route requires a staged "
            "unified runtime with windows_d3d11_hip_resident_bridge metadata"
        )
    bridge_dir = (root / "bridge").resolve()
    bridge_dir_text = str(bridge_dir)
    if bridge_dir_text not in sys.path:
        sys.path.insert(0, bridge_dir_text)
    runtime_contract.activate_runtime_dll_directories(root, platform="win32")
    try:
        bridge = importlib.import_module(WINDOWS_D3D11_HIP_RESIDENT_BRIDGE)
    except (ImportError, OSError, ValueError) as exc:
        raise WindowsD3D11HipResidentError(
            "Cannot import the ABI-matched Windows D3D11/HIP resident bridge "
            f"from {bridge_dir}: {exc}"
        ) from exc

    bridge_file = Path(getattr(bridge, "__file__", "")).resolve()
    if not bridge_file.is_relative_to(bridge_dir):
        raise WindowsD3D11HipResidentError(
            f"Resident bridge loaded outside the unified runtime: {bridge_file}"
        )
    api_version = getattr(bridge, "api_version", None)
    create_root = getattr(bridge, "create_or_get_process_root", None)
    if not callable(api_version) or not callable(create_root):
        raise WindowsD3D11HipResidentError(
            "Resident bridge is missing api_version() or "
            "create_or_get_process_root()"
        )
    observed_api = int(api_version())
    if observed_api != WINDOWS_D3D11_HIP_RESIDENT_API_VERSION:
        raise WindowsD3D11HipResidentError(
            "Resident bridge API mismatch: expected "
            f"{WINDOWS_D3D11_HIP_RESIDENT_API_VERSION}, observed {observed_api}"
        )
    return bridge


class WindowsD3D11HipResidentCoordinator:
    """Coordinate two bounded decoder transports and the secondary writer.

    The product has independent decode/detect and blend/encode readers.  Each
    reader therefore has a separate four-slot native session.  The encoder is
    intentionally bound to the blend/encode reader's exact AMF hardware
    context before its codec opens.  The coordinator, rather than either
    Python reader, owns final draining and teardown.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        metadata: object,
        batch_size: int,
        output_codec: str,
        smart_fragment: bool = False,
        bridge: object | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.metadata = metadata
        self.batch_size = int(batch_size)
        self.output_codec = str(output_codec).casefold()
        self._validate_scope(smart_fragment=bool(smart_fragment))
        self._bridge = _load_bridge() if bridge is None else bridge
        api_version = getattr(self._bridge, "api_version", None)
        create_root = getattr(self._bridge, "create_or_get_process_root", None)
        if not callable(api_version) or not callable(create_root):
            raise WindowsD3D11HipResidentError(
                "Resident bridge is missing api_version() or "
                "create_or_get_process_root()"
            )
        if int(api_version()) != WINDOWS_D3D11_HIP_RESIDENT_API_VERSION:
            raise WindowsD3D11HipResidentError(
                "Resident bridge API version is not accepted by this product"
            )
        self._create_root = create_root
        self._lock = threading.RLock()
        self._sessions: dict[str, Any] = {}
        self._decoder_contexts: dict[str, object] = {}
        self._decoder_bound: set[str] = set()
        self._closed = False
        self._failed = False
        self._final_stats: dict[str, object] | None = None

    def _validate_scope(self, *, smart_fragment: bool) -> None:
        failures: list[str] = []
        if sys.platform != "win32":
            failures.append("Windows is required")
        if vendor_for_device(self.device) is not AcceleratorVendor.AMD:
            failures.append("an AMD HIP device is required")
        codec_name = str(getattr(self.metadata, "codec_name", "")).casefold()
        if codec_name != "hevc" or self.output_codec != "hevc":
            failures.append("HEVC input and HEVC output are required")
        if bool(getattr(self.metadata, "is_10bit", False)):
            failures.append("Main10/P010 is not yet admitted; Main8/NV12 is required")
        width = int(getattr(self.metadata, "video_width", 0))
        height = int(getattr(self.metadata, "video_height", 0))
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            failures.append("positive even frame dimensions are required")
        elif (width, height) not in WINDOWS_D3D11_HIP_RESIDENT_ADMITTED_GEOMETRIES:
            capacity_rejection = (
                WINDOWS_D3D11_HIP_RESIDENT_CAPACITY_REJECTED_GEOMETRIES.get(
                    (width, height)
                )
            )
            failures.append(
                capacity_rejection
                or "the admitted resident geometries are 1920x1080 and "
                "3840x2160"
            )
        if self.batch_size not in {1, 2, 4}:
            failures.append("batch_size must be 1, 2, or 4")
        if smart_fragment:
            failures.append("Smart Render fragments are not supported")
        if failures:
            raise WindowsD3D11HipResidentError(
                "The explicit Windows D3D11/HIP resident route cannot satisfy "
                "this pipeline: " + "; ".join(failures)
            )

    def _ensure_active(self) -> None:
        if self._closed:
            raise WindowsD3D11HipResidentError("Resident coordinator is closed")
        if self._failed:
            raise WindowsD3D11HipResidentError("Resident coordinator is failed")

    def _session(self, role: str):
        if role not in _READER_ROLES:
            raise WindowsD3D11HipResidentError(
                f"Unknown resident reader role {role!r}"
            )
        with self._lock:
            self._ensure_active()
            session = self._sessions.get(role)
            if session is not None:
                return session
            visible_geometry = (
                int(self.metadata.video_width),
                int(self.metadata.video_height),
            )
            allocation_geometry = (
                WINDOWS_D3D11_HIP_RESIDENT_ADMITTED_GEOMETRIES[
                    visible_geometry
                ]
            )
            session_key = {
                "role": role,
                "codec": "hevc",
                "profile": "main",
                "bit_depth": 8,
                "surface_format": "nv12",
                "visible_width": visible_geometry[0],
                "visible_height": visible_geometry[1],
                "allocation_width": allocation_geometry[0],
                "allocation_height": allocation_geometry[1],
                "batch_size": self.batch_size,
                "slot_count": WINDOWS_D3D11_HIP_RESIDENT_SLOT_COUNT,
            }
            try:
                session = self._create_root(
                    int(self.device.index or 0),
                    session_key,
                )
                for method_name in (
                    "bind_decoder_frame",
                    "bind_encoder_context",
                    "copy_decoded_to_hip",
                    "acquire_encoder_frame",
                    "begin_drain",
                    "close_session",
                    "stats",
                ):
                    if not callable(getattr(session, method_name, None)):
                        raise WindowsD3D11HipResidentError(
                            f"Resident session is missing {method_name}()"
                        )
            except BaseException:
                self._failed = True
                raise
            self._sessions[role] = session
            return session

    def bind_decoder_frame(
        self,
        role: str,
        decoder_context: object,
        frame: object,
    ) -> dict[str, object]:
        with self._lock:
            session = self._session(role)
            previous = self._decoder_contexts.get(role)
            if previous is not None and previous is not decoder_context:
                self._failed = True
                raise WindowsD3D11HipResidentError(
                    f"Resident decoder context changed for {role}"
                )
            try:
                result = _mapping_result(
                    session.bind_decoder_frame(frame),
                    "bind_decoder_frame",
                )
            except BaseException:
                self._failed = True
                raise
            self._decoder_contexts[role] = decoder_context
            self._decoder_bound.add(role)
            return result

    def copy_decoded_to_hip(
        self,
        role: str,
        frame: object,
        destination_ptr: int,
        destination_size: int,
        consumer_stream_handle: int,
    ) -> dict[str, object]:
        with self._lock:
            if role not in self._decoder_bound:
                raise WindowsD3D11HipResidentError(
                    f"Resident decoder is not bound for {role}"
                )
            session = self._session(role)
        try:
            return _mapping_result(
                session.copy_decoded_to_hip(
                    frame,
                    int(destination_ptr),
                    int(destination_size),
                    int(consumer_stream_handle),
                ),
                "copy_decoded_to_hip",
            )
        except BaseException:
            with self._lock:
                self._failed = True
            raise

    def bind_encoder_context(
        self,
        encoder_context: object,
        *,
        role: str = "blend-encode",
    ) -> dict[str, object]:
        with self._lock:
            decoder_context = self._decoder_contexts.get(role)
            if decoder_context is None or role not in self._decoder_bound:
                raise WindowsD3D11HipResidentError(
                    "The blend/encode decoder must bind a native AMF frame before "
                    "the resident encoder opens"
                )
            session = self._session(role)
            try:
                return _mapping_result(
                    session.bind_encoder_context(
                        encoder_context,
                        decoder_context,
                    ),
                    "bind_encoder_context",
                )
            except BaseException:
                self._failed = True
                raise

    def acquire_encoder_frame(
        self,
        source_ptr: int,
        source_size: int,
        pts: int,
        producer_stream_handle: int,
        *,
        role: str = "blend-encode",
    ) -> tuple[object, dict[str, object]]:
        with self._lock:
            session = self._session(role)
        try:
            deadline = (
                time.monotonic()
                + WINDOWS_D3D11_HIP_RESIDENT_SLOT_WAIT_SECONDS
            )
            while True:
                stats = _mapping_result(session.stats(), "stats")
                try:
                    free_output_slots = int(stats["free_output_slots"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise WindowsD3D11HipResidentError(
                        "Resident encoder slot telemetry is missing a valid "
                        "free_output_slots count"
                    ) from exc
                if not 0 <= free_output_slots <= WINDOWS_D3D11_HIP_RESIDENT_SLOT_COUNT:
                    raise WindowsD3D11HipResidentError(
                        "Resident encoder slot telemetry is outside the fixed "
                        f"four-slot contract: {free_output_slots}"
                    )
                if free_output_slots:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise WindowsD3D11HipResidentError(
                        "Timed out waiting for AMF to release one of the four "
                        "resident encoder output surfaces"
                    )
                time.sleep(
                    min(
                        WINDOWS_D3D11_HIP_RESIDENT_SLOT_POLL_SECONDS,
                        remaining,
                    )
                )
            value = session.acquire_encoder_frame(
                int(source_ptr),
                int(source_size),
                int(pts),
                int(producer_stream_handle),
            )
            if not isinstance(value, tuple) or len(value) != 2:
                raise WindowsD3D11HipResidentError(
                    "resident acquire_encoder_frame returned an invalid result"
                )
            frame, telemetry = value
            return frame, _mapping_result(telemetry, "acquire_encoder_frame")
        except BaseException:
            with self._lock:
                self._failed = True
            raise

    def begin_drain(self) -> None:
        with self._lock:
            for session in tuple(self._sessions.values()):
                session.begin_drain()

    def close(self, *, timeout_ms: int = 5_000) -> dict[str, object]:
        with self._lock:
            if self._closed:
                return dict(self._final_stats or {})
            sessions = tuple(self._sessions.items())
        results: dict[str, object] = {}
        first_error: BaseException | None = None
        for role, session in sessions:
            try:
                session.begin_drain()
                results[role] = _mapping_result(
                    session.close_session(int(timeout_ms)),
                    f"close_session({role})",
                )
                _validate_closed_stats(results[role], role)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        with self._lock:
            self._final_stats = results
            if first_error is not None:
                self._failed = True
            else:
                self._closed = True
        if first_error is not None:
            raise WindowsD3D11HipResidentError(
                f"Resident transport teardown failed: {first_error}"
            ) from first_error
        return dict(results)
