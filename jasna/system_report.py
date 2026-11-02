"""PC details that bug reports need, written to the log so users don't have to collect them."""
from __future__ import annotations

import functools
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

import psutil

from jasna import __version__
from jasna.os_utils import subprocess_no_window_kwargs, windows_powershell

log = logging.getLogger(__name__)

_DISPLAY_ADAPTERS_COMMAND = (
    "$ErrorActionPreference = 'Stop'; "
    "Get-CimInstance Win32_VideoController | "
    "ForEach-Object { \"$($_.Name), driver $($_.DriverVersion), status $($_.Status)\" }"
)


def _cpu_name() -> str:
    if sys.platform == "win32":
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as cpuinfo:
        for line in cpuinfo:
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def _run_windows_tool(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, errors="replace", check=False, timeout=15,
            **subprocess_no_window_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        log.warning("Could not run %s: %s", Path(command[0]).name, error)
        return None
    if result.returncode:
        log.warning("%s failed: %s", Path(command[0]).name, result.stderr.strip())
        return None
    return result.stdout


@functools.cache
def log_system_summary() -> None:
    """Log once per process the OS, CPU, RAM and, on Windows, every display adapter with its driver."""
    log.info(
        "System: Jasna %s, %s, %s (%s threads), %.0f GB RAM",
        __version__, platform.platform(), _cpu_name(), os.cpu_count(), psutil.virtual_memory().total / 2**30,
    )
    if sys.platform != "win32":
        return
    adapters = _run_windows_tool([windows_powershell(), "-NoProfile", "-NonInteractive", "-Command", _DISPLAY_ADAPTERS_COMMAND])
    for adapter in (adapters or "").splitlines():
        if adapter.strip():
            log.info("Display adapter: %s", adapter.strip())

