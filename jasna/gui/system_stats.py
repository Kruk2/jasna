from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

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


def read_gpu_vram() -> tuple[int | None, int | None]:
    exe_path = os_utils.find_executable("nvidia-smi")
    if exe_path is None:
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
