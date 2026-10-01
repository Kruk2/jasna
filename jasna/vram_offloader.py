from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from collections.abc import Callable

import psutil
import torch

from jasna.blend_buffer import BlendBuffer
from jasna.crop_buffer import CropBuffer
from jasna.native_worker import NATIVE_ENCODE_STALL_EXIT_CODE

_log = logging.getLogger(__name__)

VRAM_LIMIT: float | None = None
VRAM_SAFETYNET: int = 750 * 1024 * 1024
AMD_MIN_VRAM_STARTUP_BUDGET: int = 4 * 1024 * 1024 * 1024
SYSTEM_VRAM_PRESSURE_SECONDS = 2.0
SYSTEM_VRAM_RECOVERY_SECONDS = 5.0
SYSTEM_VRAM_EPISODE_COOLDOWN_SECONDS = 30.0
SYSTEM_VRAM_CRITICAL_SECONDS = 1.0
SYSTEM_VRAM_MAX_OFFLOAD_BYTES = 256 * 1024 * 1024

SYSTEM_VRAM_PRESSURE_FRACTION = 0.05
SYSTEM_VRAM_PRESSURE_MIN = 512 * 1024 * 1024
SYSTEM_VRAM_PRESSURE_MAX = 1024 * 1024 * 1024
SYSTEM_VRAM_RECOVERY_FRACTION = 0.075
SYSTEM_VRAM_RECOVERY_MIN = 768 * 1024 * 1024
SYSTEM_VRAM_RECOVERY_MAX = 1536 * 1024 * 1024
SYSTEM_VRAM_CRITICAL_FRACTION = 0.02
SYSTEM_VRAM_CRITICAL_MIN = 256 * 1024 * 1024
SYSTEM_VRAM_CRITICAL_MAX = 512 * 1024 * 1024

# Host RAM is a separate failure domain from Torch VRAM.  A long AMF/native
# session can retain allocations that psutil sees but torch cannot reclaim.  We
# stop an isolated worker before Linux's global OOM killer does so, keeping the
# original pressure reason visible to the GUI.
# The previous 82%/6% limits allowed the observed 32-GiB worker to reach
# ~23.9 GiB RSS while the machine was already at ~92% total RAM before the
# next native allocation triggered Linux's OOM killer.  Keep a two-second
# debounce, but start the episode early enough to leave room for AMF/Vulkan
# allocations that are not visible in Torch telemetry.
SYSTEM_RAM_PRESSURE_FRACTION = 0.75
SYSTEM_RAM_CRITICAL_AVAILABLE_FRACTION = 0.10
SYSTEM_RAM_PRESSURE_SECONDS = 2.0

_POLL_INTERVAL = 0.1
_MIB = 1024 * 1024
STALL_WARN_SECONDS = 30.0
_SYSTEM_PRESSURE_DEBUG_SECONDS = 30.0
_DRM_CLASS_PATH = Path("/sys/class/drm")


def default_host_memory_limit_bytes() -> int | None:
    """Return a conservative per-worker RSS ceiling for the host machine."""

    try:
        total = int(psutil.virtual_memory().total)
    except (OSError, psutil.Error, TypeError, ValueError):
        return None
    if total <= 0:
        return None
    return max(8 * 1024**3, int(total * SYSTEM_RAM_PRESSURE_FRACTION))


def read_linux_amd_system_vram(
    drm_class_path: Path = _DRM_CLASS_PATH,
) -> tuple[int, int] | None:
    """Return whole-card AMD VRAM usage from DRM sysfs.

    ``torch.cuda.mem_get_info`` on ROCm does not account for every AMF/Vulkan
    allocation.  The DRM counters do, so the restoration offloader must use
    them when native decode and encode share the card with Torch.
    """

    if sys.platform != "linux":
        return None
    for card in sorted(drm_class_path.glob("card[0-9]*")):
        device = card / "device"
        try:
            vendor = (device / "vendor").read_text(encoding="utf-8").strip().lower()
            if vendor != "0x1002":
                continue
            used = int((device / "mem_info_vram_used").read_text(encoding="utf-8"))
            total = int((device / "mem_info_vram_total").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if total > 0 and 0 <= used <= total:
            return used, total
    return None


def restoration_vram_safetynet(
    *,
    frame_width: int,
    frame_height: int,
    batch_size: int,
    ten_bit: bool,
    amd: bool,
) -> int:
    """Return the Torch-side reserve for restoration tensors.

    The original 750 MiB reserve predates the unified AMD route and covers
    neither two simultaneous AMF readers nor their Vulkan surface windows. The
    derived AMD reserve accounts for the Torch-visible packed/RGB working set.
    It deliberately does not include a fixed four-GiB whole-card target: native
    AMF/Vulkan pools are not Torch-offloadable, and treating their shortage as
    a per-poll tensor target causes CPU/GPU migration thrash. NVIDIA keeps the
    established fixed reserve.
    """

    if not amd:
        return VRAM_SAFETYNET
    width = max(1, int(frame_width))
    height = max(1, int(frame_height))
    batch = max(1, int(batch_size))
    pixels = width * height
    sample_bytes = 2 if ten_bit else 1
    rgb_batch = pixels * 3 * batch
    packed_batch = pixels * 3 * sample_bytes * batch // 2
    two_reader_batches = 2 * (rgb_batch + packed_batch)
    two_reader_surface_windows = 2 * 3 * pixels * 3 * sample_bytes // 2
    derived = VRAM_SAFETYNET + two_reader_batches + two_reader_surface_windows
    return derived


def restoration_vram_startup_budget(
    *,
    frame_width: int,
    frame_height: int,
    batch_size: int,
    ten_bit: bool,
    amd: bool,
) -> int:
    """Return the preflight native-memory budget, not a runtime reclaim line."""

    derived = restoration_vram_safetynet(
        frame_width=frame_width,
        frame_height=frame_height,
        batch_size=batch_size,
        ten_bit=ten_bit,
        amd=amd,
    )
    if not amd:
        return derived
    return max(AMD_MIN_VRAM_STARTUP_BUDGET, derived)


def system_vram_pressure_watermarks(total_bytes: int) -> tuple[int, int, int]:
    """Return pressure, recovery, and critical headroom for one whole card.

    Runtime pressure follows card capacity within bounded limits.  Four GiB is
    intentionally absent here: it is a preflight budget for choosing a route,
    not a condition that should fire every 100 ms on a normally busy GPU.
    """

    total = max(1, int(total_bytes))
    pressure = min(
        SYSTEM_VRAM_PRESSURE_MAX,
        max(SYSTEM_VRAM_PRESSURE_MIN, int(total * SYSTEM_VRAM_PRESSURE_FRACTION)),
    )
    recovery = min(
        SYSTEM_VRAM_RECOVERY_MAX,
        max(SYSTEM_VRAM_RECOVERY_MIN, int(total * SYSTEM_VRAM_RECOVERY_FRACTION)),
    )
    recovery = max(recovery, pressure + 256 * 1024 * 1024)
    critical = min(
        SYSTEM_VRAM_CRITICAL_MAX,
        max(SYSTEM_VRAM_CRITICAL_MIN, int(total * SYSTEM_VRAM_CRITICAL_FRACTION)),
    )
    critical = min(critical, max(0, pressure - 1))
    return pressure, recovery, critical


class VramStats:
    def __init__(self) -> None:
        self.min_bytes: int = 0
        self.max_bytes: int = 0
        self.sum_bytes: int = 0
        self.sample_count: int = 0
        self.offload_count: int = 0
        self.total_offloaded_bytes: int = 0
        self.system_sample_count: int = 0
        self.system_max_used_bytes: int = 0
        self.system_min_headroom_bytes: int = 0
        self.system_reclaim_count: int = 0
        self.system_pressure_episodes: int = 0
        self.host_memory_pressure_episodes: int = 0
        self.host_max_rss_bytes: int = 0
        self.host_min_available_bytes: int = 0

    def update(self, used_bytes: int) -> None:
        if self.sample_count == 0:
            self.min_bytes = used_bytes
            self.max_bytes = used_bytes
        else:
            self.min_bytes = min(self.min_bytes, used_bytes)
            self.max_bytes = max(self.max_bytes, used_bytes)
        self.sum_bytes += used_bytes
        self.sample_count += 1

    def update_system(self, used_bytes: int, total_bytes: int) -> None:
        headroom = max(0, total_bytes - used_bytes)
        if self.system_sample_count == 0:
            self.system_max_used_bytes = used_bytes
            self.system_min_headroom_bytes = headroom
        else:
            self.system_max_used_bytes = max(self.system_max_used_bytes, used_bytes)
            self.system_min_headroom_bytes = min(
                self.system_min_headroom_bytes,
                headroom,
            )
        self.system_sample_count += 1

    @property
    def avg_bytes(self) -> float:
        if self.sample_count == 0:
            return 0.0
        return self.sum_bytes / self.sample_count

    def summary(self) -> str:
        if self.sample_count == 0:
            summary = "VRAM offloader: no samples"
        else:
            summary = (
                f"VRAM — min: {self.min_bytes / _MIB:.0f} MiB, "
                f"max: {self.max_bytes / _MIB:.0f} MiB, "
                f"avg: {self.avg_bytes / _MIB:.0f} MiB | "
                f"offloads: {self.offload_count}, "
                f"total offloaded: {self.total_offloaded_bytes / _MIB:.0f} MiB"
            )
        if self.system_sample_count:
            summary += (
                f" | system peak: {self.system_max_used_bytes / _MIB:.0f} MiB, "
                f"minimum headroom: {self.system_min_headroom_bytes / _MIB:.0f} MiB, "
                f"pressure episodes: {self.system_pressure_episodes}, "
                f"critical reclaims: {self.system_reclaim_count}"
            )
        if self.host_max_rss_bytes:
            summary += (
                f" | host RSS peak: {self.host_max_rss_bytes / _MIB:.0f} MiB, "
                f"minimum available: {self.host_min_available_bytes / _MIB:.0f} MiB, "
                f"pressure episodes: {self.host_memory_pressure_episodes}"
            )
        return summary


class VramOffloader:
    def __init__(
        self,
        device: torch.device,
        blend_buffer: BlendBuffer,
        crop_buffers: dict[int, CropBuffer],
        crop_lock: threading.Lock,
        vram_limit: float | None = VRAM_LIMIT,
        safetynet: int = VRAM_SAFETYNET,
        system_vram_startup_budget: int | None = None,
        system_vram_reader: Callable[[], tuple[int, int] | None] | None = None,
        system_vram_pressure_seconds: float = SYSTEM_VRAM_PRESSURE_SECONDS,
        system_vram_recovery_seconds: float = SYSTEM_VRAM_RECOVERY_SECONDS,
        system_vram_episode_cooldown_seconds: float = SYSTEM_VRAM_EPISODE_COOLDOWN_SECONDS,
        system_vram_critical_seconds: float = SYSTEM_VRAM_CRITICAL_SECONDS,
        system_vram_max_offload_bytes: int = SYSTEM_VRAM_MAX_OFFLOAD_BYTES,
        host_memory_limit_bytes: int | None = None,
        host_memory_reader: Callable[[], tuple[int, int, int] | None] | None = None,
        host_memory_pressure_seconds: float = SYSTEM_RAM_PRESSURE_SECONDS,
        cancel_event: threading.Event | None = None,
        terminate_on_encode_stall: bool = False,
        encode_stall_timeout_seconds: float = 90.0,
        system_vram_reader_factory: Callable | None = None,
        on_system_vram_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        if system_vram_reader_factory is not None:
            if (not callable(system_vram_reader_factory) or not callable(on_system_vram_error)
                    or system_vram_reader is not None or system_vram_startup_budget is None):
                raise ValueError("owned global reader requires a factory, error callback, budget and no borrowed reader")
        self._system_vram_reader_factory = system_vram_reader_factory
        self._on_system_vram_error = on_system_vram_error
        self._owned_system_vram_reader = None
        self._system_vram_failure = None
        self._owned_reader_started = False
        self._device = device
        self._blend_buffer = blend_buffer
        self._crop_buffers = crop_buffers
        self._crop_lock = crop_lock

        if vram_limit is not None:
            gpu_total = int(vram_limit * 1024 * 1024 * 1024)
        else:
            gpu_total = torch.cuda.get_device_properties(device).total_memory
        self._threshold = max(0, gpu_total - safetynet)
        self._offload_device_type = "cuda"
        self._system_vram_startup_budget = (
            None
            if system_vram_startup_budget is None
            else max(0, int(system_vram_startup_budget))
        )
        self._system_vram_reader = (
            (system_vram_reader or read_linux_amd_system_vram)
            if self._system_vram_startup_budget is not None
            else None
        )
        self._system_pressure_seconds = max(0.0, float(system_vram_pressure_seconds))
        self._system_recovery_seconds = max(0.0, float(system_vram_recovery_seconds))
        self._system_episode_cooldown_seconds = max(
            0.0,
            float(system_vram_episode_cooldown_seconds),
        )
        self._system_critical_seconds = max(0.0, float(system_vram_critical_seconds))
        self._system_max_offload_bytes = max(0, int(system_vram_max_offload_bytes))
        self._host_memory_limit_bytes = (
            None
            if host_memory_limit_bytes is None
            else max(1, int(host_memory_limit_bytes))
        )
        self._host_memory_monitor_enabled = (
            host_memory_limit_bytes is not None or host_memory_reader is not None
        )
        self._host_memory_reader = host_memory_reader or self._read_host_memory
        self._host_memory_pressure_seconds = max(
            0.0,
            float(host_memory_pressure_seconds),
        )
        self._cancel_event = cancel_event
        self._terminate_on_encode_stall = bool(terminate_on_encode_stall)
        self._encode_stall_timeout_seconds = max(
            STALL_WARN_SECONDS,
            float(encode_stall_timeout_seconds),
        )
        self._host_pressure_since: float | None = None
        self._host_memory_pressure = False
        self._system_pressure_since: float | None = None
        self._system_pressure_active = False
        self._system_recovery_since: float | None = None
        self._system_last_episode_at: float | None = None
        self._system_critical_since: float | None = None
        self._system_critical_reclaimed = False
        self._last_system_pressure_debug_time = 0.0

        self.stats = VramStats()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="VramOffloader", daemon=True)
        self._last_encode_time: list[float | None] | None = None
        self._last_stall_warn_time: float = 0.0
        self._stall_diagnostics_dumped = False
        self._stall_exit_requested = False
        self._stall_check_paused = False
        self._pipeline_queues: dict[str, object] | None = None
        self._metadata_queue: object | None = None

        _log.info(
            "VramOffloader: Torch threshold=%d MiB (total=%d MiB, safetynet=%d MiB%s)",
            self._threshold // _MIB,
            gpu_total // _MIB,
            safetynet // _MIB,
            (
                ", whole-card watermarks=capacity-scaled"
                f", startup budget={self._system_vram_startup_budget // _MIB} MiB"
                if self._system_vram_startup_budget is not None
                else ""
            ),
        )
        if self._host_memory_monitor_enabled:
            host_limit = self._host_memory_limit_bytes
            _log.info(
                "VramOffloader: host-memory pressure monitor enabled "
                "(RSS limit=%s, available floor=%d%%, debounce=%.1fs)",
                (
                    f"{host_limit // _MIB} MiB"
                    if host_limit is not None
                    else "total-scaled"
                ),
                int(SYSTEM_RAM_CRITICAL_AVAILABLE_FRACTION * 100),
                self._host_memory_pressure_seconds,
            )

    def set_encode_heartbeat(self, shared_time: list[float | None]) -> None:
        self._last_encode_time = shared_time

    def set_pipeline_queues(
        self,
        clip_queue: object,
        secondary_queue: object,
        encode_queue: object,
        metadata_queue: object,
    ) -> None:
        self._pipeline_queues = {
            "clip_queue": clip_queue,
            "secondary_queue": secondary_queue,
            "encode_queue": encode_queue,
        }
        self._metadata_queue = metadata_queue

    def set_cancel_event(self, cancel_event: threading.Event | None) -> None:
        """Give the monitor a cancellation channel for pre-OOM shutdown."""

        self._cancel_event = cancel_event

    @property
    def host_memory_pressure(self) -> bool:
        return self._host_memory_pressure

    @staticmethod
    def _read_host_memory() -> tuple[int, int, int] | None:
        try:
            process_rss = int(psutil.Process().memory_info().rss)
            virtual = psutil.virtual_memory()
            return process_rss, int(virtual.total), int(virtual.available)
        except (OSError, psutil.Error, TypeError, ValueError):
            return None

    def start(self) -> None:
        if self._system_vram_reader_factory is None:
            self._thread.start()
            return
        if self._owned_reader_started or self._stop.is_set():
            raise RuntimeError("owned global VRAM sampler cannot be restarted")
        self._owned_reader_started = True
        try:
            self._owned_system_vram_reader = self._system_vram_reader_factory()
            if (not callable(self._owned_system_vram_reader)
                    or not callable(getattr(self._owned_system_vram_reader, "close", None))):
                raise TypeError("owned global VRAM reader must be callable and closeable")
            self._system_vram_reader = self._read_required_system_vram
            self.stats.update_system(*self._read_required_system_vram())
            self._thread.start()
        except BaseException as error:
            self._record_system_vram_failure(error)
            # No running poller owns the reader if thread startup failed.
            # Refuse to race a partially started thread, should that occur.
            if not self._thread.is_alive():
                self._close_owned_system_vram_reader()
            raise

    def stop(self) -> None:
        self._stop.set()
        if self._system_vram_reader_factory is None:
            self._thread.join(timeout=5.0)
        else:
            if self._thread.ident is not None:
                self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                raise RuntimeError("global VRAM polling thread did not retire; reader remains thread-owned")
        _log.info(self.stats.summary())
        if self._system_vram_failure is not None:
            raise RuntimeError("required whole-card VRAM monitoring failed") from self._system_vram_failure

    def _record_system_vram_failure(self, error: BaseException) -> None:
        if self._system_vram_failure is None:
            self._system_vram_failure = error
            self._stop.set()
            try:
                self._on_system_vram_error(error)
            except BaseException:
                _log.exception("Could not report whole-card VRAM sampler failure")

    def _read_required_system_vram(self) -> tuple[int, int]:
        sample = self._owned_system_vram_reader()
        if (not isinstance(sample, tuple) or len(sample) != 2
                or any(type(value) is not int for value in sample)
                or sample[1] <= 0 or not 0 <= sample[0] <= sample[1]):
            raise ValueError("required whole-card VRAM sample is unavailable or invalid")
        return sample

    def _close_owned_system_vram_reader(self) -> None:
        reader = self._owned_system_vram_reader
        if reader is None:
            return
        try:
            reader.close()
        except BaseException as error:
            self._record_system_vram_failure(error)
        else:
            self._owned_system_vram_reader = None
            self._system_vram_reader = None

    def _run(self) -> None:
        if self._system_vram_reader_factory is None:
            self._run_loop()
            return
        try:
            self._run_loop()
        except BaseException as error:
            self._record_system_vram_failure(error)
        finally:
            self._close_owned_system_vram_reader()

    def _run_loop(self) -> None:
        while not self._stop.wait(_POLL_INTERVAL):
            free, total = torch.cuda.mem_get_info(self._device)
            used = total - free
            self.stats.update(used)
            bytes_to_free = max(0, used - self._threshold)
            system_sample: tuple[int, int] | None = None
            if self._system_vram_reader is not None:
                try:
                    system_sample = self._system_vram_reader()
                except Exception:
                    if self._system_vram_reader_factory is not None:
                        raise
                    _log.debug("Could not read whole-card VRAM usage", exc_info=True)
                if system_sample is not None:
                    system_used, system_total = system_sample
                    self.stats.update_system(system_used, system_total)

            freed = 0
            if bytes_to_free > 0:
                freed = self._offload(bytes_to_free)
                if freed > 0:
                    torch.cuda.empty_cache()
                    self.stats.offload_count += 1
                    self.stats.total_offloaded_bytes += freed
                    _log.debug(
                        "[vram-offloader] offloaded %.1f MiB (used=%.0f MiB, threshold=%.0f MiB)",
                        freed / _MIB,
                        used / _MIB,
                        self._threshold / _MIB,
                    )
            self._check_system_pressure(system_sample)
            self._check_host_memory_pressure()
            self._check_encode_stall()

    def _check_host_memory_pressure(self) -> None:
        if not self._host_memory_monitor_enabled or self._host_memory_pressure:
            return
        try:
            sample = self._host_memory_reader()
        except Exception:
            _log.debug("Could not read host memory usage", exc_info=True)
            return
        if sample is None:
            return
        try:
            rss, total, available = (int(value) for value in sample)
        except (TypeError, ValueError):
            return
        if total <= 0 or rss < 0 or available < 0:
            return
        self.stats.host_max_rss_bytes = max(self.stats.host_max_rss_bytes, rss)
        if self.stats.host_min_available_bytes == 0:
            self.stats.host_min_available_bytes = available
        else:
            self.stats.host_min_available_bytes = min(
                self.stats.host_min_available_bytes,
                available,
            )
        limit = self._host_memory_limit_bytes
        if limit is None:
            limit = max(1, int(total * SYSTEM_RAM_PRESSURE_FRACTION))
        available_floor = max(
            512 * _MIB,
            int(total * SYSTEM_RAM_CRITICAL_AVAILABLE_FRACTION),
        )
        pressured = rss >= limit or available <= available_floor
        now = time.monotonic()
        if not pressured:
            self._host_pressure_since = None
            return
        if self._host_pressure_since is None:
            self._host_pressure_since = now
            return
        if now - self._host_pressure_since < self._host_memory_pressure_seconds:
            return
        self._host_memory_pressure = True
        self.stats.host_memory_pressure_episodes += 1
        _log.error(
            "[vram-offloader] host memory pressure: rss=%.0f MiB, "
            "available=%.0f MiB, limit=%.0f MiB; cancelling the worker before "
            "the OS OOM killer can terminate it",
            rss / _MIB,
            available / _MIB,
            limit / _MIB,
        )
        if self._cancel_event is not None:
            self._cancel_event.set()
        queues = self._pipeline_queues
        if queues:
            for pipeline_queue in queues.values():
                wake_all = getattr(pipeline_queue, "wake_all", None)
                if callable(wake_all):
                    wake_all()

    def _reset_system_pressure(self) -> None:
        self._system_pressure_since = None
        self._system_pressure_active = False
        self._system_recovery_since = None
        self._system_critical_since = None
        self._system_critical_reclaimed = False

    def _check_system_pressure(
        self,
        system_sample: tuple[int, int] | None,
    ) -> None:
        if system_sample is None:
            return
        used, total = system_sample
        headroom = max(0, total - used)
        pressure, recovery, critical = system_vram_pressure_watermarks(total)
        now = time.monotonic()
        if self._system_pressure_active and headroom >= recovery:
            if self._system_recovery_since is None:
                self._system_recovery_since = now
                return
            if now - self._system_recovery_since < self._system_recovery_seconds:
                return
            _log.info(
                "[vram-offloader] whole-card pressure recovered: "
                "headroom=%.0f MiB, recovery=%.0f MiB sustained for %.1fs",
                headroom / _MIB,
                recovery / _MIB,
                self._system_recovery_seconds,
            )
            self._reset_system_pressure()
            return
        if self._system_pressure_active:
            self._system_recovery_since = None
        elif headroom >= pressure:
            self._system_pressure_since = None
            return

        if not self._system_pressure_active:
            cooldown_ready = (
                self._system_last_episode_at is None
                or now - self._system_last_episode_at
                >= self._system_episode_cooldown_seconds
                or headroom < critical
            )
            if not cooldown_ready:
                self._system_pressure_since = None
                return
            if self._system_pressure_since is None:
                self._system_pressure_since = now
                return
            if now - self._system_pressure_since < self._system_pressure_seconds:
                return
            self._system_pressure_active = True
            self._system_last_episode_at = now
            self._system_recovery_since = None
            self.stats.system_pressure_episodes += 1
            target = min(
                self._system_max_offload_bytes,
                max(0, pressure - headroom),
            )
            freed = self._offload(target) if target > 0 else 0
            if freed > 0:
                # One allocator trim per pressure episode makes the moved pages
                # visible to native AMF/Vulkan without repeating the operation
                # on every telemetry poll.
                torch.cuda.empty_cache()
                self.stats.offload_count += 1
                self.stats.total_offloaded_bytes += freed
            _log.info(
                "[vram-offloader] whole-card pressure episode: "
                "headroom=%.0f MiB, pressure=%.0f MiB, recovery=%.0f MiB, "
                "limited offload=%.1f MiB",
                headroom / _MIB,
                pressure / _MIB,
                recovery / _MIB,
                freed / _MIB,
            )

        if headroom >= critical or self._system_critical_reclaimed:
            self._system_critical_since = None
            return
        if self._system_critical_since is None:
            self._system_critical_since = now
            return
        if now - self._system_critical_since < self._system_critical_seconds:
            return
        torch.cuda.empty_cache()
        self.stats.system_reclaim_count += 1
        self._system_critical_reclaimed = True
        _log.warning(
            "[vram-offloader] critical whole-card headroom: "
            "used=%.0f MiB, total=%.0f MiB, headroom=%.0f MiB, critical=%.0f MiB; "
            "performed the only emergency cache reclaim for this pressure episode",
            used / _MIB,
            total / _MIB,
            headroom / _MIB,
            critical / _MIB,
        )

    def pause_stall_check(self) -> None:
        self._stall_check_paused = True

    def _check_encode_stall(self) -> None:
        hb = self._last_encode_time
        if hb is None or self._stall_check_paused:
            return
        last_activity = hb[0]
        # Restoration can legitimately spend more than 30 seconds producing
        # the first frame of a clip.  Until write() is actually entered there
        # is no encoder activity to monitor, so reporting an encoder stall here
        # only creates a large false-positive thread dump.
        if last_activity is None:
            return
        now = time.monotonic()
        elapsed = now - last_activity
        if elapsed > STALL_WARN_SECONDS:
            # One compact warning is enough for a single stall episode.  The
            # previous 30-second full thread dumps could grow a run log by
            # megabytes while the native AMF call was permanently blocked.
            if self._last_stall_warn_time == 0.0:
                _log.warning(
                    "[vram-offloader] encode stall detected: no frame encoded for %.0fs",
                    elapsed,
                )
                self._last_stall_warn_time = now
            if (
                self._terminate_on_encode_stall
                and elapsed >= self._encode_stall_timeout_seconds
                and not self._stall_exit_requested
            ):
                self._stall_exit_requested = True
                _log.critical(
                    "AMF encode remained blocked for %.0fs; terminating the "
                    "isolated worker so the completed Smart Render fragments "
                    "can be resumed in a fresh process",
                    elapsed,
                )
                if not self._stall_diagnostics_dumped:
                    self._dump_stall_diagnostics(elapsed)
                    self._stall_diagnostics_dumped = True
                os._exit(NATIVE_ENCODE_STALL_EXIT_CODE)
        else:
            self._last_stall_warn_time = 0.0
            self._stall_diagnostics_dumped = False
            self._stall_exit_requested = False

    def _dump_stall_diagnostics(self, elapsed: float) -> None:
        lines: list[str] = [f"=== ENCODE STALL DIAGNOSTICS (stalled {elapsed:.0f}s) ==="]

        # Queue sizes and frame counts
        if self._pipeline_queues:
            for name, q in self._pipeline_queues.items():
                try:
                    lines.append(
                        f"  {name}: items={q.qsize()} frames={q.current_frames} max_frames={q._max_frames}"
                    )
                except Exception:
                    lines.append(f"  {name}: <error reading>")
        if self._metadata_queue is not None:
            try:
                lines.append(
                    f"  metadata_queue: items~={self._metadata_queue.qsize()} maxsize={self._metadata_queue.maxsize}"
                )
            except Exception:
                lines.append("  metadata_queue: <error reading>")

        # Blend buffer state
        try:
            bb = self._blend_buffer
            with bb._lock:
                pending_count = len(bb.pending_map)
                results_count = len(bb._results)
                result_track_ids = list(bb._results.keys())
                earliest_pending = min(bb.pending_map.keys()) if bb.pending_map else None
                latest_pending = max(bb.pending_map.keys()) if bb.pending_map else None
                waiting_frames = [
                    (fidx, tids)
                    for fidx, tids in sorted(bb.pending_map.items())
                    if not all(tid in bb._results for tid in tids)
                ][:5]
            lines.append(
                f"  blend_buffer: pending_frames={pending_count} results={results_count}"
                f" result_track_ids={result_track_ids}"
            )
            if earliest_pending is not None:
                lines.append(f"  blend_buffer: frame_range=[{earliest_pending}..{latest_pending}]")
            if waiting_frames:
                for fidx, tids in waiting_frames:
                    missing = [t for t in tids if t not in (bb._results if hasattr(bb, '_results') else {})]
                    lines.append(f"  blend_buffer: frame {fidx} waiting for tracks {missing}")
        except Exception as e:
            lines.append(f"  blend_buffer: <error: {e}>")

        # Crop buffers
        try:
            with self._crop_lock:
                crop_ids = list(self._crop_buffers.keys())
                crop_sizes = {k: v.frame_count for k, v in self._crop_buffers.items()}
            lines.append(f"  crop_buffers: track_ids={crop_ids} sizes={crop_sizes}")
        except Exception as e:
            lines.append(f"  crop_buffers: <error: {e}>")

        # VRAM
        try:
            free, total = torch.cuda.mem_get_info(self._device)
            alloc = torch.cuda.memory_allocated(self._device)
            reserved = torch.cuda.memory_reserved(self._device)
            lines.append(
                f"  VRAM: used={((total - free) / _MIB):.0f} MiB"
                f" allocated={alloc / _MIB:.0f} MiB"
                f" reserved={reserved / _MIB:.0f} MiB"
                f" free={free / _MIB:.0f} MiB"
            )
        except Exception as e:
            lines.append(f"  VRAM: <error: {e}>")

        # Thread stack traces
        lines.append("  --- Thread stacks ---")
        thread_names = {t.ident: t.name for t in threading.enumerate()}
        for tid, frame in sys._current_frames().items():
            name = thread_names.get(tid, f"Thread-{tid}")
            if name == "VramOffloader":
                continue
            tb = "".join(traceback.format_stack(frame))
            lines.append(f"  [{name}]\n{tb}")

        lines.append("=== END STALL DIAGNOSTICS ===")
        _log.warning("\n".join(lines))

    def _offload(self, bytes_to_free: int) -> int:
        freed = 0

        results = self._blend_buffer.offloadable_results()
        results.sort(key=lambda sr: sr.start_frame, reverse=True)

        for sr in results:
            for i, frame in enumerate(sr.restored_frames):
                if frame.device.type == self._offload_device_type:
                    nbytes = frame.nelement() * frame.element_size()
                    sr.restored_frames[i] = frame.cpu()
                    freed += nbytes
                    if freed >= bytes_to_free:
                        return freed
            for i, mask in enumerate(sr.masks):
                if mask.device.type == self._offload_device_type:
                    nbytes = mask.nelement() * mask.element_size()
                    sr.masks[i] = mask.cpu()
                    freed += nbytes

        with self._crop_lock:
            buffers = list(self._crop_buffers.values())
        buffers.sort(key=lambda cb: cb.frame_count, reverse=True)

        for cb in buffers:
            for rc in cb.crops:
                if rc.crop.device.type == self._offload_device_type:
                    nbytes = rc.crop.nelement() * rc.crop.element_size()
                    rc.crop = rc.crop.cpu()
                    freed += nbytes
                    if freed >= bytes_to_free:
                        return freed

        return freed
