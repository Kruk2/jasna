"""Sample RSS and VRAM (median/peak) of one child process while it runs."""

import statistics
import subprocess
import threading
import time

import psutil


class MemorySampler:
    def __init__(self, pid: int, interval_seconds: float = 0.5) -> None:
        self.pid = pid
        self.interval_seconds = interval_seconds
        self._ram_mb: list[float] = []
        self._vram_mb: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _sample_ram(self) -> None:
        try:
            rss_bytes = psutil.Process(self.pid).memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return
        self._ram_mb.append(rss_bytes / (1024 * 1024))

    def _sample_vram(self) -> None:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
        except (OSError, subprocess.TimeoutExpired):
            return
        for line in out.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 2 and parts[0] == str(self.pid) and parts[1].isdigit():
                self._vram_mb.append(float(parts[1]))
                return

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_ram()
            self._sample_vram()
            self._stop.wait(self.interval_seconds)

    def stop(self) -> dict[str, float]:
        self._stop.set()
        self._thread.join()

        def med_peak(samples: list[float]) -> tuple[float, float]:
            if not samples:
                return 0.0, 0.0
            return statistics.median(samples), max(samples)

        ram_med, ram_peak = med_peak(self._ram_mb)
        vram_med, vram_peak = med_peak(self._vram_mb)
        return {
            "ram_med_mb": ram_med,
            "ram_peak_mb": ram_peak,
            "vram_med_mb": vram_med,
            "vram_peak_mb": vram_peak,
        }
