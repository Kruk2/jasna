"""Exact-device Windows whole-card dedicated-VRAM telemetry.

``WindowsGlobalVramReader`` joins a selected HIP device identity to an
independent instance of the existing DXGI/PDH telemetry reader.  It deliberately
does not discover a HIP runtime, import Torch, or initialize a GPU at module
load.  A GUI parent that cannot safely initialize HIP can instead construct the
reader from an identity obtained from its actual selected live worker.

Identities are intentionally ephemeral: callers must not persist or publish
them across worker restarts.  The parent owns and must close every reader it
creates.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math
import numbers
import operator
import re
import threading
from typing import Any, Callable


_LUID_SIZE = 8
_MAX_DEVICE_INDEX = (1 << 31) - 1
_ADAPTER_MARKER = re.compile(r"luid_0x[0-9a-f]{8}_0x[0-9a-f]{8}_phys_")


def create_windows_hip_vram_reader(device):
    """Construct a pipeline-owned reader inside an already selected HIP worker.

    The caller supplies this as an offloader factory; never call it from the
    GUI parent's recovery path, which must use from_identity instead.
    """
    import sys
    if sys.platform != "win32":
        raise RuntimeError("Windows HIP sampler requires Windows")
    import torch
    if not getattr(torch.version, "hip", None):
        raise RuntimeError("Windows HIP sampler requires an AMD runtime")
    selected = torch.device(device)
    if selected.type != "cuda":
        raise ValueError("Windows HIP sampler requires a CUDA/HIP device")
    index = selected.index if selected.index is not None else torch.cuda.current_device()
    from jasna.media.hip_kernel import hip_runtime
    return WindowsGlobalVramReader(index, hip_runtime())


@dataclass(frozen=True)
class WindowsGpuIdentity:
    """Canonical, per-worker Windows GPU identity for PDH matching.

    This is a shape-validated handoff token, not proof that a future process
    selected the same GPU.  Obtain it from the actual live worker and use it
    only for the corresponding parent-side recovery window.
    """

    adapter_marker: str
    node_index: int

    def __post_init__(self) -> None:
        if type(self.adapter_marker) is not str:
            raise TypeError("adapter_marker must be a string")
        if _ADAPTER_MARKER.fullmatch(self.adapter_marker) is None:
            raise ValueError("adapter_marker must use canonical lower-case LUID syntax")
        if type(self.node_index) is not int:
            raise TypeError("node_index must be a plain int")
        if not 0 <= self.node_index <= 31:
            raise ValueError("node_index must be from 0 through 31")


def _default_telemetry_factory():
    """Lazily construct one independent product Windows AMD telemetry reader."""

    from jasna.gui.system_stats import _WindowsAmdGpuReader

    return _WindowsAmdGpuReader()


def _bind_hip_device_get_luid(hip_library: Any):
    try:
        function = hip_library.hipDeviceGetLuid
    except AttributeError as error:
        raise TypeError("hip_library must expose hipDeviceGetLuid") from error
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint),
        ctypes.c_int,
    ]
    function.restype = ctypes.c_int
    return function


def _hip_identity(hip_library: Any, device_index: int) -> WindowsGpuIdentity:
    if isinstance(device_index, bool):
        raise TypeError("device_index must be an integer, not bool")
    try:
        index = operator.index(device_index)
    except TypeError as error:
        raise TypeError("device_index must be an integer") from error
    if index < 0:
        raise ValueError("device_index must be nonnegative")
    if index > _MAX_DEVICE_INDEX:
        raise ValueError("device_index exceeds the ctypes.c_int range")

    get_luid = _bind_hip_device_get_luid(hip_library)
    luid = ctypes.create_string_buffer(_LUID_SIZE)
    node_mask = ctypes.c_uint(0)
    status = int(get_luid(luid, ctypes.byref(node_mask), index))
    if status != 0:
        raise RuntimeError(f"hipDeviceGetLuid failed with status {status}")

    mask = int(node_mask.value)
    if mask == 0:
        raise ValueError("hipDeviceGetLuid returned a zero device node mask")
    if mask & (mask - 1):
        raise ValueError(
            "hipDeviceGetLuid returned an ambiguous multi-bit device node mask"
        )

    raw_luid = bytes(luid.raw)
    if len(raw_luid) != _LUID_SIZE:
        raise ValueError("hipDeviceGetLuid returned an invalid LUID")
    high = int.from_bytes(raw_luid[4:8], "little")
    low = int.from_bytes(raw_luid[:4], "little")
    return WindowsGpuIdentity(
        adapter_marker=f"luid_0x{high:08x}_0x{low:08x}_phys_",
        node_index=mask.bit_length() - 1,
    )


def _integer_bytes(value: Any, field_name: str, *, positive: bool = False) -> int:
    """Return an exact integer byte count, rejecting lossy numeric values."""

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer byte count")
    try:
        integer = operator.index(value)
    except TypeError:
        if not isinstance(value, numbers.Real):
            raise ValueError(f"{field_name} must be an integer byte count") from None
        try:
            numeric = float(value)
        except (OverflowError, ValueError):
            raise ValueError(f"{field_name} must be finite") from None
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{field_name} must be a finite integer byte count")
        integer = int(value)
    if positive and integer <= 0:
        raise ValueError(f"{field_name} must be positive")
    if integer < 0:
        raise ValueError(f"{field_name} must be nonnegative")
    return int(integer)


def _close_rejected_reader(reader: Any, original_error: BaseException) -> None:
    close = getattr(reader, "close", None)
    if not callable(close):
        raise original_error
    try:
        close()
    except BaseException as close_error:
        raise close_error from original_error
    raise original_error


class WindowsGlobalVramReader:
    """Join one exact Windows GPU identity to one independent PDH reader.

    ``__init__`` preserves the HIP-based construction used by the accepted
    calibration.  ``from_identity`` is the GUI-parent path: it performs no HIP
    access and validates only the supplied canonical identity against a fresh
    PDH/DXGI reader.  The caller must obtain that identity from the actual
    selected live worker before using it in the parent.

    ``telemetry_factory`` is a zero-argument callable returning the product
    reader shape: ``_adapter_marker``, ``_total_vram``, ``_memory_counter``,
    ``read()``, ``_values(counter)``, and ``close()``.
    """

    def __init__(
        self,
        device_index: int,
        hip_library: Any,
        *,
        telemetry_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._initialize(_hip_identity(hip_library, device_index), telemetry_factory)

    @classmethod
    def from_identity(
        cls,
        identity: WindowsGpuIdentity,
        *,
        telemetry_factory: Callable[[], Any] | None = None,
    ) -> WindowsGlobalVramReader:
        """Create a HIP-free, parent-owned reader for one live-worker identity."""

        if not isinstance(identity, WindowsGpuIdentity):
            raise TypeError("identity must be a WindowsGpuIdentity")
        instance = cls.__new__(cls)
        # Reconstructing retains validation even if a caller bypassed frozen
        # dataclass assignment through object-level tricks.
        canonical_identity = WindowsGpuIdentity(
            adapter_marker=identity.adapter_marker,
            node_index=identity.node_index,
        )
        instance._initialize(canonical_identity, telemetry_factory)
        return instance

    def _initialize(
        self,
        identity: WindowsGpuIdentity,
        telemetry_factory: Callable[[], Any] | None,
    ) -> None:
        factory = _default_telemetry_factory if telemetry_factory is None else telemetry_factory
        if not callable(factory):
            raise TypeError("telemetry_factory must be callable")

        reader = factory()
        try:
            reader_marker = getattr(reader, "_adapter_marker")
            if not isinstance(reader_marker, str):
                raise TypeError("telemetry reader has no valid _adapter_marker")
            if reader_marker != identity.adapter_marker:
                raise ValueError(
                    "telemetry adapter marker does not match the supplied GPU identity"
                )
            total = _integer_bytes(
                getattr(reader, "_total_vram"),
                "total VRAM",
                positive=True,
            )
            memory_counter = getattr(reader, "_memory_counter")
            if not callable(getattr(reader, "read", None)):
                raise TypeError("telemetry reader must expose callable read")
            if not callable(getattr(reader, "_values", None)):
                raise TypeError("telemetry reader must expose callable _values")
            if not callable(getattr(reader, "close", None)):
                raise TypeError("telemetry reader must expose callable close")
        except BaseException as error:
            _close_rejected_reader(reader, error)

        self._identity = identity
        self.adapter_marker = identity.adapter_marker
        self.node_index = identity.node_index
        self.total_bytes = total
        self._memory_counter = memory_counter
        self._telemetry_reader = reader
        self._lock = threading.RLock()
        self._closed = False

    @property
    def identity(self) -> WindowsGpuIdentity:
        """The immutable identity used for exact PDH instance matching."""

        return self._identity

    def __call__(self) -> tuple[int, int]:
        with self._lock:
            if self._closed:
                raise RuntimeError("WindowsGlobalVramReader is closed")

            reader = self._telemetry_reader
            refresh = reader.read()
            try:
                refresh_values = tuple(refresh)
            except TypeError as error:
                raise ValueError("telemetry refresh must return a two-item result") from error
            if len(refresh_values) != 2:
                raise ValueError("telemetry refresh must return a two-item result")
            if refresh_values[0] is None and refresh_values[1] is None:
                raise RuntimeError("telemetry refresh failed")

            total = _integer_bytes(
                getattr(reader, "_total_vram"),
                "total VRAM",
                positive=True,
            )
            self.total_bytes = total
            values = reader._values(self._memory_counter)
            expected_name = (
                f"{self.identity.adapter_marker}{self.identity.node_index}".casefold()
            )
            matches = []
            for item in values:
                try:
                    name, sample = item
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "telemetry memory values must be name/value pairs"
                    ) from error
                if isinstance(name, str) and name.casefold() == expected_name:
                    matches.append(sample)
            if len(matches) == 0:
                raise ValueError(f"missing telemetry sample for {expected_name}")
            if len(matches) != 1:
                raise ValueError(f"duplicate telemetry samples for {expected_name}")

            used = _integer_bytes(matches[0], "used VRAM")
            if used > total:
                raise ValueError("used VRAM cannot exceed total VRAM")
            return used, total

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._telemetry_reader.close()
            self._telemetry_reader = None
            self._closed = True
