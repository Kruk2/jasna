"""Asynchronous durable GUI batch logs and parent-safe diagnostics.

The GUI and processing workers only enqueue short strings here.  A dedicated
daemon owns file creation, filesystem writes, flushes, durable syncs, and the
optional Linux telemetry reads so logging cannot change processing behavior.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
import itertools
import os
from pathlib import Path
import sys
import threading
import time
from typing import TextIO
import uuid

from jasna.gui.paths import get_settings_path


RUN_LOG_DIRECTORY_NAME = ".jasna-logs"
RUN_LOG_FALLBACK_DIRECTORY_NAME = "run-logs"
DEFAULT_FLUSH_INTERVAL_SECONDS = 1.0
DEFAULT_SYNC_INTERVAL_SECONDS = 5.0
DEFAULT_TELEMETRY_INTERVAL_SECONDS = 30.0
DEFAULT_QUEUE_CAPACITY = 2_048

_PROC_MEMINFO_PATH = Path("/proc/meminfo")
_PROC_LOADAVG_PATH = Path("/proc/loadavg")
_DRM_CLASS_PATH = Path("/sys/class/drm")


def resolve_run_log_directory(output_folder: str | Path | None) -> Path:
    """Resolve the log directory without touching the filesystem.

    Same-as-input batches can span unrelated source directories, so their logs
    use the per-user Jasna config directory instead of modifying every source
    folder in the queue.
    """
    output_text = str(output_folder or "").strip()
    if output_text:
        return Path(output_text).expanduser() / RUN_LOG_DIRECTORY_NAME
    return get_settings_path().parent / RUN_LOG_FALLBACK_DIRECTORY_NAME


def build_run_log_path(
    output_folder: str | Path | None,
    *,
    now: Callable[[], datetime] = datetime.now,
    pid: int | None = None,
    token: str | None = None,
) -> Path:
    """Build a timestamped, collision-resistant path without opening a file."""
    timestamp = now().strftime("%Y%m%d-%H%M%S-%f")
    process_id = os.getpid() if pid is None else int(pid)
    unique_token = token or uuid.uuid4().hex[:8]
    safe_token = "".join(ch for ch in unique_token if ch.isascii() and ch.isalnum())
    safe_token = safe_token[:16] or "run"
    return resolve_run_log_directory(output_folder) / (
        f"jasna-run-{timestamp}-pid{process_id}-{safe_token}.log"
    )


def _open_run_log_stream(path: Path) -> TextIO:
    """Open a new UTF-8 run log from the writer thread."""
    return path.open("x", encoding="utf-8", buffering=1)


class _BoundedLogQueue:
    """Keep recent events under storms without making producers wait for I/O."""

    def __init__(self, capacity: int) -> None:
        self._capacity = max(1, int(capacity))
        self._items: deque[tuple[int, str, str]] = deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._sequence = itertools.count()
        self._latest_sequence = -1

    def append(self, level: str, message: str) -> bool:
        # Never wait for the writer.  Contention means this event was dropped;
        # its sequence gap lets the writer emit an exact dropped-line marker.
        sequence = next(self._sequence)
        self._latest_sequence = sequence
        if not self._lock.acquire(blocking=False):
            self._wake.set()
            return False
        try:
            if len(self._items) >= self._capacity:
                self._items.popleft()
            self._items.append((sequence, str(level).upper(), str(message)))
        finally:
            self._lock.release()
        self._wake.set()
        return True

    def drain(self) -> list[tuple[int, str, str]]:
        with self._lock:
            items = list(self._items)
            self._items.clear()
        return items

    def wait(self, timeout: float) -> None:
        self._wake.wait(timeout)
        self._wake.clear()

    def is_empty(self) -> bool:
        with self._lock:
            return not self._items

    def latest_sequence(self) -> int:
        return self._latest_sequence

    def wake(self) -> None:
        self._wake.set()


class AsyncRunLog:
    """Fail-open run-log side channel with a bounded daemon writer thread."""

    def __init__(
        self,
        path: Path,
        *,
        on_status: Callable[[str, str], None] | None,
        now: Callable[[], datetime],
        monotonic: Callable[[], float],
        fsync: Callable[[int], None],
        flush_interval_seconds: float,
        sync_interval_seconds: float,
        queue_capacity: int,
        open_stream: Callable[[Path], TextIO],
    ) -> None:
        self.path = path
        self._on_status = on_status
        self._now = now
        self._monotonic = monotonic
        self._fsync = fsync
        self._flush_interval_seconds = max(0.05, float(flush_interval_seconds))
        self._sync_interval_seconds = max(
            self._flush_interval_seconds, float(sync_interval_seconds)
        )
        self._open_stream = open_stream
        self._queue = _BoundedLogQueue(queue_capacity)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._failed = threading.Event()
        self._started = False

    @classmethod
    def start(
        cls,
        output_folder: str | Path | None,
        *,
        on_status: Callable[[str, str], None] | None = None,
        now: Callable[[], datetime] = datetime.now,
        monotonic: Callable[[], float] = time.monotonic,
        fsync: Callable[[int], None] = os.fsync,
        flush_interval_seconds: float = DEFAULT_FLUSH_INTERVAL_SECONDS,
        sync_interval_seconds: float = DEFAULT_SYNC_INTERVAL_SECONDS,
        queue_capacity: int = DEFAULT_QUEUE_CAPACITY,
        pid: int | None = None,
        token: str | None = None,
        open_stream: Callable[[Path], TextIO] = _open_run_log_stream,
    ) -> "AsyncRunLog":
        """Start the writer without opening the path on the caller's thread."""
        path = build_run_log_path(output_folder, now=now, pid=pid, token=token)
        writer = cls(
            path,
            on_status=on_status,
            now=now,
            monotonic=monotonic,
            fsync=fsync,
            flush_interval_seconds=flush_interval_seconds,
            sync_interval_seconds=sync_interval_seconds,
            queue_capacity=queue_capacity,
            open_stream=open_stream,
        )
        writer.enqueue("INFO", "Persistent run log requested")
        try:
            writer._thread = threading.Thread(
                target=writer._run,
                daemon=True,
                name="jasna-run-log-writer",
            )
            writer._thread.start()
            writer._started = True
        except Exception as exc:
            writer._failed.set()
            writer._report("WARNING", f"Run log writer could not start: {exc}")
        return writer

    @property
    def failed(self) -> bool:
        return self._failed.is_set()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def accepts_events(self) -> bool:
        return self._started and not self._failed.is_set() and not self._stop_event.is_set()

    def enqueue(self, level: str, message: str) -> bool:
        """Best-effort non-blocking producer operation; never raises to callers."""
        if not self.accepts_events and self._started:
            return False
        try:
            return self._queue.append(level, message)
        except Exception:
            return False

    def close(self, timeout: float = 1.0) -> None:
        """Ask the writer for one final flush/sync without blocking indefinitely."""
        self.enqueue("INFO", "Persistent run log closing")
        self._stop_event.set()
        self._queue.wake()
        thread = self._thread
        if (
            timeout > 0
            and thread is not None
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=max(0.0, timeout))

    def _run(self) -> None:
        stream: TextIO | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            stream = self._open_stream(self.path)
            self._report("INFO", f"Run log saved: {self.path}")
            self._drain_until_closed(stream)
        except Exception as exc:
            self._failed.set()
            self._report("WARNING", f"Run log unavailable: {self.path}: {exc}")
        finally:
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def _drain_until_closed(self, stream: TextIO) -> None:
        last_flush = self._monotonic()
        last_sync = last_flush
        last_sequence = -1
        has_unflushed_writes = False
        needs_sync = False
        while True:
            events = self._queue.drain()
            wrote_event = False
            if events:
                for sequence, level, message in events:
                    if sequence > last_sequence + 1:
                        dropped = sequence - last_sequence - 1
                        self._write_event(
                            stream,
                            "WARNING",
                            "Run log dropped "
                            f"{dropped} event(s) while its bounded queue was full",
                        )
                        wrote_event = True
                    last_sequence = sequence
                    self._write_event(stream, level, message)
                    wrote_event = True

            current = self._monotonic()
            force_close = self._stop_event.is_set() and self._queue.is_empty()
            if force_close:
                final_sequence = self._queue.latest_sequence()
                if final_sequence > last_sequence:
                    self._write_event(
                        stream,
                        "WARNING",
                        "Run log dropped "
                        f"{final_sequence - last_sequence} event(s) while its bounded queue was full",
                    )
                    wrote_event = True
            if wrote_event:
                has_unflushed_writes = True
                needs_sync = True

            should_flush = force_close or (
                has_unflushed_writes
                and current - last_flush >= self._flush_interval_seconds
            )
            should_sync = force_close or (
                needs_sync and current - last_sync >= self._sync_interval_seconds
            )
            if should_sync:
                # A durable sync must include any buffered lines, even if a
                # previous timed flush already made them visible to readers.
                stream.flush()
                has_unflushed_writes = False
                last_flush = current
                self._fsync(stream.fileno())
                needs_sync = False
                last_sync = current
            elif should_flush:
                stream.flush()
                has_unflushed_writes = False
                last_flush = current
            if force_close:
                return
            self._queue.wait(0.25)

    def _write_event(self, stream: TextIO, level: str, message: str) -> None:
        timestamp = self._now().astimezone().isoformat(timespec="milliseconds")
        for line in message.splitlines() or [""]:
            stream.write(f"{timestamp} {level:<8} {line}\n")

    def _report(self, level: str, message: str) -> None:
        if self._on_status is None:
            return
        try:
            self._on_status(level, message)
        except Exception:
            pass


@dataclass(frozen=True)
class RunTelemetry:
    """Read-only host diagnostics used to investigate a failed batch.

    Linux records physical-memory and AMD sysfs byte counters.  Windows uses
    the existing GUI system-stats provider, whose public contract exposes
    utilization percentages rather than byte counters; unavailable fields are
    kept as ``None`` instead of being estimated.
    """

    load_1: float | None
    load_5: float | None
    load_15: float | None
    memory_used_bytes: int | None
    memory_total_bytes: int | None
    gpu_busy_percent: int | None
    vram_used_bytes: int | None
    vram_total_bytes: int | None
    gpu_edge_celsius: float | None
    gpu_junction_celsius: float | None
    gpu_memory_celsius: float | None
    gpu_power_watts: float | None
    cpu_util_percent: int | None = None
    ram_util_percent: int | None = None
    vram_util_percent: int | None = None
    telemetry_source: str | None = None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _read_int(path: Path) -> int | None:
    text = _read_text(path)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _read_proc_memory(path: Path) -> tuple[int | None, int | None]:
    text = _read_text(path)
    if text is None:
        return None, None
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        amount = value.strip().split()
        if not amount:
            continue
        try:
            values[key] = int(amount[0]) * 1024
        except ValueError:
            continue
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if total is None or available is None:
        return None, total
    return max(0, total - available), total


def _read_proc_load(path: Path) -> tuple[float | None, float | None, float | None]:
    text = _read_text(path)
    if text is None:
        return None, None, None
    values = text.split()
    if len(values) < 3:
        return None, None, None
    try:
        return float(values[0]), float(values[1]), float(values[2])
    except ValueError:
        return None, None, None


def _read_amd_sensors(device: Path) -> dict[str, float | None]:
    readings: dict[str, float | None] = {
        "edge": None,
        "junction": None,
        "memory": None,
        "power": None,
    }
    for hwmon in sorted((device / "hwmon").glob("hwmon*")):
        for input_path in sorted(hwmon.glob("temp*_input")):
            raw = _read_int(input_path)
            if raw is None:
                continue
            stem = input_path.name.removesuffix("_input")
            label = (_read_text(hwmon / f"{stem}_label") or "").lower()
            if "junction" in label:
                key = "junction"
            elif "mem" in label:
                key = "memory"
            elif "edge" in label:
                key = "edge"
            elif stem == "temp1":
                key = "edge"
            elif stem == "temp2":
                key = "junction"
            elif stem == "temp3":
                key = "memory"
            else:
                continue
            readings[key] = raw / 1000.0

        for power_path in sorted(hwmon.glob("power*_average")) + sorted(
            hwmon.glob("power*_input")
        ):
            raw = _read_int(power_path)
            if raw is None:
                continue
            readings["power"] = raw / 1_000_000.0 if raw >= 10_000 else float(raw)
            break
    return readings


def _read_amd_metrics(drm_class_path: Path) -> tuple[
    int | None,
    int | None,
    int | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    busy_values: list[int] = []
    vram_used = 0
    vram_total = 0
    have_vram = False
    sensor_values: dict[str, list[float]] = {
        "edge": [],
        "junction": [],
        "memory": [],
        "power": [],
    }
    for card in sorted(drm_class_path.glob("card[0-9]*")):
        device = card / "device"
        if (_read_text(device / "vendor") or "").lower() != "0x1002":
            continue
        busy = _read_int(device / "gpu_busy_percent")
        if busy is not None:
            busy_values.append(max(0, min(100, busy)))
        used = _read_int(device / "mem_info_vram_used")
        total = _read_int(device / "mem_info_vram_total")
        if used is not None and total is not None and total > 0:
            vram_used += max(0, used)
            vram_total += total
            have_vram = True
        for name, value in _read_amd_sensors(device).items():
            if value is not None:
                sensor_values[name].append(value)

    def maximum(name: str) -> float | None:
        values = sensor_values[name]
        return max(values) if values else None

    return (
        max(busy_values) if busy_values else None,
        vram_used if have_vram else None,
        vram_total if have_vram else None,
        maximum("edge"),
        maximum("junction"),
        maximum("memory"),
        maximum("power"),
    )


def _optional_percent(stats: object | None, name: str) -> int | None:
    """Return a valid provider percentage without inventing a replacement."""

    value = getattr(stats, name, None)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 0 <= value <= 100 else None


def _read_windows_run_telemetry(
    system_stats_reader: Callable[[], object] | None = None,
) -> RunTelemetry:
    """Adapt the lightweight GUI stats provider to durable run-log telemetry.

    Importing the provider happens only in the telemetry daemon, avoiding a
    new GUI-startup import.  Its persistent PDH reader retains its own lifetime
    and locking; this adapter neither creates another reader nor retries it.
    """

    try:
        if system_stats_reader is None:
            from jasna.gui.system_stats import read_system_stats

            system_stats_reader = read_system_stats
        stats = system_stats_reader()
    except Exception:
        # This side channel must remain fail-open.  The source marker means the
        # rendered line still identifies every unavailable field explicitly.
        stats = None
    return RunTelemetry(
        load_1=None,
        load_5=None,
        load_15=None,
        memory_used_bytes=None,
        memory_total_bytes=None,
        gpu_busy_percent=_optional_percent(stats, "gpu_util"),
        vram_used_bytes=None,
        vram_total_bytes=None,
        gpu_edge_celsius=None,
        gpu_junction_celsius=None,
        gpu_memory_celsius=None,
        gpu_power_watts=None,
        cpu_util_percent=_optional_percent(stats, "cpu_util"),
        ram_util_percent=_optional_percent(stats, "ram_util"),
        vram_util_percent=_optional_percent(stats, "vram_util"),
        telemetry_source="windows-system-stats",
    )


def read_run_telemetry(
    *,
    proc_meminfo_path: Path = _PROC_MEMINFO_PATH,
    proc_loadavg_path: Path = _PROC_LOADAVG_PATH,
    drm_class_path: Path = _DRM_CLASS_PATH,
    system_stats_reader: Callable[[], object] | None = None,
) -> RunTelemetry:
    """Read host diagnostics without importing GPU runtime libraries or tools."""
    if sys.platform == "win32":
        return _read_windows_run_telemetry(system_stats_reader)
    memory_used, memory_total = _read_proc_memory(proc_meminfo_path)
    load_1, load_5, load_15 = _read_proc_load(proc_loadavg_path)
    (
        gpu_busy,
        vram_used,
        vram_total,
        edge,
        junction,
        memory,
        power,
    ) = _read_amd_metrics(drm_class_path)
    return RunTelemetry(
        load_1=load_1,
        load_5=load_5,
        load_15=load_15,
        memory_used_bytes=memory_used,
        memory_total_bytes=memory_total,
        gpu_busy_percent=gpu_busy,
        vram_used_bytes=vram_used,
        vram_total_bytes=vram_total,
        gpu_edge_celsius=edge,
        gpu_junction_celsius=junction,
        gpu_memory_celsius=memory,
        gpu_power_watts=power,
    )


def _format_mib(value: int | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value / (1024 * 1024):.0f}MiB"


def _format_float(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.2f}"


def _format_percent(value: int | None) -> str:
    return "unavailable" if value is None else f"{value}%"


def format_run_telemetry(
    telemetry: RunTelemetry,
    context: Mapping[str, object] | None = None,
) -> str:
    """Return one deterministic, grep-friendly telemetry line for the run log."""
    parts: list[str] = []
    if context:
        for key in ("gui_pid", "worker_pid", "job"):
            value = context.get(key)
            if value is not None and str(value):
                parts.append(f"{key}={value}")
    if telemetry.telemetry_source is not None:
        parts.append(f"telemetry_source={telemetry.telemetry_source}")
    parts.extend(
        (
            "load="
            f"{_format_float(telemetry.load_1)}/"
            f"{_format_float(telemetry.load_5)}/"
            f"{_format_float(telemetry.load_15)}",
            "ram="
            f"{_format_mib(telemetry.memory_used_bytes)}/"
            f"{_format_mib(telemetry.memory_total_bytes)}",
            "gpu_busy="
            + (
                "unavailable"
                if telemetry.gpu_busy_percent is None
                else f"{telemetry.gpu_busy_percent}%"
            ),
            "vram="
            f"{_format_mib(telemetry.vram_used_bytes)}/"
            f"{_format_mib(telemetry.vram_total_bytes)}",
            f"gpu_edge={_format_float(telemetry.gpu_edge_celsius)}C",
            f"gpu_junction={_format_float(telemetry.gpu_junction_celsius)}C",
            f"gpu_memory={_format_float(telemetry.gpu_memory_celsius)}C",
            f"gpu_power={_format_float(telemetry.gpu_power_watts)}W",
        )
    )
    if telemetry.telemetry_source == "windows-system-stats":
        parts.extend(
            (
                f"cpu_util={_format_percent(telemetry.cpu_util_percent)}",
                f"ram_util={_format_percent(telemetry.ram_util_percent)}",
                f"vram_util={_format_percent(telemetry.vram_util_percent)}",
            )
        )
    return "Telemetry: " + " ".join(parts)


class RunTelemetrySampler:
    """Periodically collect platform diagnostics outside GUI/worker threads."""

    def __init__(
        self,
        on_log: Callable[[str, str], bool | None],
        *,
        is_active: Callable[[], bool],
        context_provider: Callable[[], Mapping[str, object]] | None = None,
        collect: Callable[[], RunTelemetry] = read_run_telemetry,
        interval_seconds: float = DEFAULT_TELEMETRY_INTERVAL_SECONDS,
    ) -> None:
        self._on_log = on_log
        self._is_active = is_active
        self._context_provider = context_provider
        self._collect = collect
        self._interval_seconds = max(0.01, float(interval_seconds))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="jasna-run-telemetry",
        )
        self._thread.start()

    def stop(self, timeout: float = 0.5) -> None:
        self._stop_event.set()
        thread = self._thread
        if (
            timeout > 0
            and thread is not None
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=max(0.0, timeout))

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop_event.is_set() and self._is_active():
            self._sample_once()
            if self._stop_event.wait(self._interval_seconds):
                return

    def _sample_once(self) -> None:
        try:
            telemetry = self._collect()
            context = self._context_provider() if self._context_provider else None
            self._on_log("INFO", format_run_telemetry(telemetry, context))
        except Exception:
            # A failed diagnostic read is intentionally silent to avoid turning a
            # best-effort side channel into a source of processing noise.
            return
