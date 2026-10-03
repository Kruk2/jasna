from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
import threading
from typing import Sequence

from jasna import os_utils

_DRM_CLASS_PATH = Path("/sys/class/drm")
_AMD_VENDOR_ID = 0x1002
_DXGI_ADAPTER_FLAG_SOFTWARE = 0x2
_PDH_FMT_DOUBLE = 0x00000200
_PDH_MORE_DATA = 0x800007D2
_VALID_PDH_STATUSES = frozenset((0, 1))


class _Guid(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]


class _Luid(ctypes.Structure):
    _fields_ = [
        ("LowPart", ctypes.c_uint32),
        ("HighPart", ctypes.c_int32),
    ]


class _DxgiAdapterDesc1(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", ctypes.c_uint32),
        ("DeviceId", ctypes.c_uint32),
        ("SubSysId", ctypes.c_uint32),
        ("Revision", ctypes.c_uint32),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _Luid),
        ("Flags", ctypes.c_uint32),
    ]


class _PdhValueUnion(ctypes.Union):
    _fields_ = [
        ("longValue", ctypes.c_long),
        ("doubleValue", ctypes.c_double),
        ("largeValue", ctypes.c_longlong),
        ("AnsiStringValue", ctypes.c_char_p),
        ("WideStringValue", ctypes.c_wchar_p),
    ]


class _PdhFmtCounterValue(ctypes.Structure):
    _anonymous_ = ("value",)
    _fields_ = [
        ("CStatus", ctypes.c_uint32),
        ("value", _PdhValueUnion),
    ]


class _PdhFmtCounterValueItem(ctypes.Structure):
    _fields_ = [
        ("szName", ctypes.c_wchar_p),
        ("FmtValue", _PdhFmtCounterValue),
    ]


def _guid(value: str) -> _Guid:
    import uuid

    return _Guid.from_buffer_copy(uuid.UUID(value).bytes_le)


def _com_method(pointer: ctypes.c_void_p, index: int, result_type, *arg_types):
    vtable = ctypes.cast(
        pointer,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)),
    ).contents
    prototype = ctypes.WINFUNCTYPE(
        result_type,
        ctypes.c_void_p,
        *arg_types,
    )
    return prototype(vtable[index])


def _windows_amd_adapter() -> tuple[str, int] | None:
    """Return the PDH LUID marker and dedicated bytes for the primary AMD GPU."""

    if sys.platform != "win32":
        return None
    try:
        dxgi = ctypes.WinDLL("dxgi.dll")
        create_factory = dxgi.CreateDXGIFactory1
        create_factory.argtypes = [
            ctypes.POINTER(_Guid),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        create_factory.restype = ctypes.c_long
        factory = ctypes.c_void_p()
        iid = _guid("770aae78-f26f-4dba-a829-253c83d1b387")
        if create_factory(ctypes.byref(iid), ctypes.byref(factory)) != 0:
            return None
    except (AttributeError, OSError, ValueError):
        return None

    adapters: list[tuple[str, int]] = []
    release_factory = _com_method(factory, 2, ctypes.c_ulong)
    enum_adapters = _com_method(
        factory,
        12,
        ctypes.c_long,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
    )
    try:
        index = 0
        while True:
            adapter = ctypes.c_void_p()
            if enum_adapters(factory, index, ctypes.byref(adapter)) != 0:
                break
            index += 1
            release_adapter = _com_method(adapter, 2, ctypes.c_ulong)
            try:
                desc = _DxgiAdapterDesc1()
                get_desc = _com_method(
                    adapter,
                    10,
                    ctypes.c_long,
                    ctypes.POINTER(_DxgiAdapterDesc1),
                )
                if get_desc(adapter, ctypes.byref(desc)) != 0:
                    continue
                if (
                    int(desc.VendorId) != _AMD_VENDOR_ID
                    or int(desc.Flags) & _DXGI_ADAPTER_FLAG_SOFTWARE
                    or int(desc.DedicatedVideoMemory) <= 0
                ):
                    continue
                high = int(desc.AdapterLuid.HighPart) & 0xFFFFFFFF
                low = int(desc.AdapterLuid.LowPart) & 0xFFFFFFFF
                marker = f"luid_0x{high:08x}_0x{low:08x}_phys_"
                adapters.append((marker, int(desc.DedicatedVideoMemory)))
            finally:
                release_adapter(adapter)
    finally:
        release_factory(factory)
    return max(adapters, key=lambda item: item[1], default=None)


def _windows_gpu_percentages(
    adapter_marker: str,
    total_vram: int,
    engine_values: Sequence[tuple[str, float]],
    memory_values: Sequence[tuple[str, float]],
) -> tuple[int | None, int | None]:
    marker = adapter_marker.casefold()
    engine_totals: dict[str, float] = {}
    for name, value in engine_values:
        lowered = name.casefold()
        if marker not in lowered:
            continue
        match = re.search(r"_phys_\d+_eng_\d+", lowered)
        if match is None:
            continue
        key = match.group(0)
        engine_totals[key] = engine_totals.get(key, 0.0) + max(
            0.0,
            float(value),
        )
    gpu_util = (
        _clamp_pct(max(engine_totals.values()))
        if engine_totals
        else None
    )

    dedicated = [
        max(0.0, float(value))
        for name, value in memory_values
        if marker in name.casefold()
    ]
    vram_util = (
        _clamp_pct((max(dedicated) / float(total_vram)) * 100.0)
        if dedicated and total_vram > 0
        else None
    )
    return gpu_util, vram_util


class _WindowsAmdGpuReader:
    """Low-overhead Windows AMD telemetry using one persistent PDH query."""

    def __init__(self) -> None:
        adapter = _windows_amd_adapter()
        if adapter is None:
            raise RuntimeError("No hardware AMD DXGI adapter is available")
        self._adapter_marker, self._total_vram = adapter
        self._pdh = ctypes.WinDLL("pdh.dll")
        self._query = ctypes.c_void_p()
        self._engine_counter = ctypes.c_void_p()
        self._memory_counter = ctypes.c_void_p()

        self._pdh.PdhOpenQueryW.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._pdh.PdhOpenQueryW.restype = ctypes.c_long
        self._pdh.PdhAddEnglishCounterW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._pdh.PdhAddEnglishCounterW.restype = ctypes.c_long
        self._pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
        self._pdh.PdhCollectQueryData.restype = ctypes.c_long
        self._pdh.PdhCloseQuery.argtypes = [ctypes.c_void_p]
        self._pdh.PdhCloseQuery.restype = ctypes.c_long
        self._pdh.PdhGetFormattedCounterArrayW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_void_p,
        ]
        self._pdh.PdhGetFormattedCounterArrayW.restype = ctypes.c_long

        if self._pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query)) != 0:
            raise RuntimeError("PdhOpenQueryW failed")
        try:
            for path, counter in (
                (r"\GPU Engine(*)\Utilization Percentage", self._engine_counter),
                (r"\GPU Adapter Memory(*)\Dedicated Usage", self._memory_counter),
            ):
                status = self._pdh.PdhAddEnglishCounterW(
                    self._query,
                    path,
                    0,
                    ctypes.byref(counter),
                )
                if status != 0:
                    raise RuntimeError(f"PdhAddEnglishCounterW failed: {status}")
            self._pdh.PdhCollectQueryData(self._query)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _unsigned_status(value: int) -> int:
        return ctypes.c_uint32(value).value

    def _values(self, counter: ctypes.c_void_p) -> list[tuple[str, float]]:
        buffer_size = ctypes.c_uint32(0)
        item_count = ctypes.c_uint32(0)
        status = self._pdh.PdhGetFormattedCounterArrayW(
            counter,
            _PDH_FMT_DOUBLE,
            ctypes.byref(buffer_size),
            ctypes.byref(item_count),
            None,
        )
        if (
            self._unsigned_status(status) != _PDH_MORE_DATA
            or buffer_size.value == 0
        ):
            return []
        buffer = ctypes.create_string_buffer(buffer_size.value)
        status = self._pdh.PdhGetFormattedCounterArrayW(
            counter,
            _PDH_FMT_DOUBLE,
            ctypes.byref(buffer_size),
            ctypes.byref(item_count),
            buffer,
        )
        if status != 0:
            return []
        items = ctypes.cast(
            buffer,
            ctypes.POINTER(_PdhFmtCounterValueItem),
        )
        return [
            (items[index].szName or "", float(items[index].FmtValue.doubleValue))
            for index in range(item_count.value)
            if int(items[index].FmtValue.CStatus) in _VALID_PDH_STATUSES
        ]

    def read(self) -> tuple[int | None, int | None]:
        if not self._query or self._pdh.PdhCollectQueryData(self._query) != 0:
            return None, None
        return _windows_gpu_percentages(
            self._adapter_marker,
            self._total_vram,
            self._values(self._engine_counter),
            self._values(self._memory_counter),
        )

    def close(self) -> None:
        query = getattr(self, "_query", None)
        if query:
            self._pdh.PdhCloseQuery(query)
            self._query = ctypes.c_void_p()


_WINDOWS_GPU_READER: _WindowsAmdGpuReader | None = None
_WINDOWS_GPU_READER_ATTEMPTED = False
_WINDOWS_GPU_READER_LOCK = threading.Lock()


def _read_windows_amd_gpu() -> tuple[int | None, int | None]:
    global _WINDOWS_GPU_READER, _WINDOWS_GPU_READER_ATTEMPTED
    with _WINDOWS_GPU_READER_LOCK:
        if not _WINDOWS_GPU_READER_ATTEMPTED:
            _WINDOWS_GPU_READER_ATTEMPTED = True
            try:
                _WINDOWS_GPU_READER = _WindowsAmdGpuReader()
            except (AttributeError, OSError, RuntimeError, ValueError):
                _WINDOWS_GPU_READER = None
        if _WINDOWS_GPU_READER is None:
            return None, None
        return _WINDOWS_GPU_READER.read()


@dataclass(frozen=True)
class SystemStats:
    gpu_util: int | None
    vram_util: int | None
    ram_util: int
    cpu_util: int


def _clamp_pct(value: float) -> int:
    v = int(round(float(value)))
    if v < 0:
        return 0
    if v > 100:
        return 100
    return v


def _parse_nvidia_smi_csv_line(line: str) -> tuple[int, int]:
    parts = [p.strip() for p in (line or "").split(",")]
    if len(parts) < 3:
        raise ValueError(f"Unexpected nvidia-smi output: {line!r}")
    gpu_util = _clamp_pct(float(parts[0]))
    mem_used = float(parts[1])
    mem_total = float(parts[2])
    if mem_total <= 0:
        raise ValueError(f"Unexpected nvidia-smi total memory: {mem_total!r}")
    vram_util = _clamp_pct((mem_used / mem_total) * 100.0)
    return gpu_util, vram_util


def _read_amd_sysfs() -> tuple[int | None, int | None]:
    for card in sorted(_DRM_CLASS_PATH.glob("card[0-9]*")):
        device = card / "device"
        try:
            if (device / "vendor").read_text(encoding="utf-8").strip().lower() != "0x1002":
                continue
            gpu_path = device / "gpu_busy_percent"
            gpu_util = (
                _clamp_pct(float(gpu_path.read_text(encoding="utf-8").strip()))
                if gpu_path.is_file()
                else None
            )
            used = int((device / "mem_info_vram_used").read_text(encoding="utf-8"))
            total = int((device / "mem_info_vram_total").read_text(encoding="utf-8"))
            vram_util = _clamp_pct((used / total) * 100.0) if total > 0 else None
            return gpu_util, vram_util
        except (OSError, ValueError):
            continue
    return None, None


def read_gpu_vram() -> tuple[int | None, int | None]:
    exe_path = os_utils.find_executable("nvidia-smi")
    if exe_path is None:
        if sys.platform == "win32":
            return _read_windows_amd_gpu()
        return _read_amd_sysfs()

    cmd = [
        exe_path,
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=0.5,
            **os_utils.subprocess_no_window_kwargs(),
        )
    except (subprocess.TimeoutExpired, OSError):
        return None, None

    if completed.returncode != 0:
        return None, None

    lines = (completed.stdout or "").splitlines()
    first = next((ln for ln in lines if ln.strip()), "")
    if first == "":
        return None, None

    try:
        return _parse_nvidia_smi_csv_line(first)
    except ValueError:
        return None, None


def read_cpu_ram() -> tuple[int, int]:
    import psutil
    cpu_util = _clamp_pct(psutil.cpu_percent(interval=None))
    ram_util = _clamp_pct(psutil.virtual_memory().percent)
    return cpu_util, ram_util


def read_system_stats() -> SystemStats:
    cpu_util, ram_util = read_cpu_ram()
    gpu_util, vram_util = read_gpu_vram()
    return SystemStats(
        gpu_util=gpu_util,
        vram_util=vram_util,
        ram_util=ram_util,
        cpu_util=cpu_util,
    )
