from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ctypes
import subprocess
import sys

from jasna import os_utils

_DRM_CLASS_PATH = Path("/sys/class/drm")


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


class _AdlSingleSensorData(ctypes.Structure):
    _fields_ = [("supported", ctypes.c_int), ("value", ctypes.c_int)]


class _AdlPmLogData(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int), ("sensors", _AdlSingleSensorData * 256)]


class _AdlAdapterInfo(ctypes.Structure):
    _fields_ = [
        ("iSize", ctypes.c_int),
        ("iAdapterIndex", ctypes.c_int),
        ("strUDID", ctypes.c_char * 256),
        ("iBusNumber", ctypes.c_int),
        ("iDeviceNumber", ctypes.c_int),
        ("iFunctionNumber", ctypes.c_int),
        ("iVendorID", ctypes.c_int),
        ("strAdapterName", ctypes.c_char * 256),
        ("strDisplayName", ctypes.c_char * 256),
        ("iPresent", ctypes.c_int),
        ("iExist", ctypes.c_int),
        ("strDriverPath", ctypes.c_char * 256),
        ("strDriverPathExt", ctypes.c_char * 256),
        ("strPNPString", ctypes.c_char * 256),
        ("iOSDisplayIndex", ctypes.c_int),
    ]


# PMLog sensor IDs from adl_defines.h (ADL_PMLOG_SENSORS).
_PMLOG_INFO_ACTIVITY_GFX = 19


class _AdlPmLog:
    """AMD Display Library PMLog reader: the same kernel counters AMD Software uses.

    The DLL is loaded once and the discrete adapter is picked by name; PMLog
    sensor 19 is the GPU activity percentage. Everything is read through the
    driver, never through Task Manager's PDH counters.
    """

    def __init__(self) -> None:
        self._dll = ctypes.WinDLL("atiadlxx.dll")
        self._allocated: list[ctypes.Array] = []
        callback_type = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int)

        def _malloc(size: int) -> int:
            buffer = ctypes.create_string_buffer(size)
            self._allocated.append(buffer)
            return ctypes.addressof(buffer)

        self._callback = callback_type(_malloc)
        self._context = ctypes.c_void_p()
        rc = self._dll.ADL2_Main_Control_Create(
            self._callback, 1, ctypes.byref(self._context)
        )
        if rc != 0 or not self._context:
            raise OSError(f"ADL2_Main_Control_Create failed: {rc}")
        self._adapter_index = self._find_discrete_adapter()
        if self._adapter_index is None:
            raise OSError("No AMD discrete adapter found through ADL")

    def _find_discrete_adapter(self) -> int | None:
        count = ctypes.c_int()
        if self._dll.ADL2_Adapter_NumberOfAdapters_Get(
            self._context, ctypes.byref(count)
        ) != 0 or count.value <= 0:
            return None
        infos = (_AdlAdapterInfo * count.value)()
        for info in infos:
            info.iSize = ctypes.sizeof(_AdlAdapterInfo)
        if self._dll.ADL2_Adapter_AdapterInfo_Get(
            self._context, infos, ctypes.sizeof(_AdlAdapterInfo) * count.value
        ) != 0:
            return None

        candidates: list[tuple[int, str]] = []
        for info in infos:
            if info.iVendorID not in (1002, 0x1002) or not info.iExist:
                continue
            name = info.strAdapterName.decode("mbcs", errors="ignore").strip()
            if "graphics" in name.lower() and "rx" not in name.lower():
                continue
            candidates.append((int(info.iAdapterIndex), name))
        if not candidates:
            return None
        # Prefer a discrete Radeon RX part; otherwise take the last AMD adapter.
        for index, name in candidates:
            if "rx" in name.lower() or "radeon" in name.lower():
                return index
        return candidates[-1][0]

    def gpu_util(self) -> int | None:
        data = _AdlPmLogData()
        data.size = ctypes.sizeof(_AdlPmLogData)
        if self._dll.ADL2_New_QueryPMLogData_Get(
            self._context, self._adapter_index, ctypes.byref(data)
        ) != 0:
            return None
        sensor = data.sensors[_PMLOG_INFO_ACTIVITY_GFX]
        if not sensor.supported:
            return None
        return _clamp_pct(sensor.value)


_adl_pmlog: _AdlPmLog | None = None
_adl_failed = False


def _read_amd_adl_util() -> int | None:
    """GPU activity straight from ADL PMLog (AMD Software's kernel counters)."""
    global _adl_pmlog, _adl_failed
    if sys.platform != "win32" or _adl_failed:
        return None
    if _adl_pmlog is None:
        try:
            _adl_pmlog = _AdlPmLog()
        except Exception:
            _adl_failed = True
            return None
    try:
        return _adl_pmlog.gpu_util()
    except Exception:
        return None


def _read_torch_vram_util() -> int | None:
    """Device-wide VRAM usage from the HIP/CUDA runtime (bytes, not PDH)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        if total <= 0:
            return None
        return _clamp_pct((total - free) / total * 100.0)
    except Exception:
        return None


def read_gpu_vram() -> tuple[int | None, int | None]:
    exe_path = os_utils.find_executable("nvidia-smi")
    if exe_path is None:
        # AMD on Windows: ADL PMLog for utilization and the HIP runtime for VRAM.
        # Neither path touches Task Manager's PDH counters, whose "3D engine"
        # figure reads 0 on AMD while the card is clearly working.
        if sys.platform == "win32":
            gpu_util = _read_amd_adl_util()
            if gpu_util is not None:
                return gpu_util, _read_torch_vram_util()
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
