from __future__ import annotations

import logging
import sys
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from queue import Queue

import torch

from jasna.blend_buffer import BlendBuffer
from jasna.crop_buffer import CropBuffer, RawCrop
from jasna.frame_queue import FrameQueue
from jasna.pipeline_items import SecondaryRestoreResult

_log = logging.getLogger(__name__)

VRAM_SAFETYNET: int = 750 * 1024 * 1024

_POLL_INTERVAL = 0.1
_MIB = 1024 * 1024
STALL_WARN_SECONDS = 30.0


class VramStats:
    def __init__(self) -> None:
        self.min_bytes: int = 0
        self.max_bytes: int = 0
        self.sum_bytes: int = 0
        self.sample_count: int = 0
        self.offload_count: int = 0
        self.total_offloaded_bytes: int = 0

    def update(self, used_bytes: int) -> None:
        if self.sample_count == 0:
            self.min_bytes = used_bytes
            self.max_bytes = used_bytes
        else:
            self.min_bytes = min(self.min_bytes, used_bytes)
            self.max_bytes = max(self.max_bytes, used_bytes)
        self.sum_bytes += used_bytes
        self.sample_count += 1

    @property
    def avg_bytes(self) -> float:
        if self.sample_count == 0:
            return 0.0
        return self.sum_bytes / self.sample_count

    def summary(self) -> str:
        if self.sample_count == 0:
            return "VRAM offloader: no samples"
        return (
            f"VRAM — min: {self.min_bytes / _MIB:.0f} MiB, "
            f"max: {self.max_bytes / _MIB:.0f} MiB, "
            f"avg: {self.avg_bytes / _MIB:.0f} MiB | "
            f"offloads: {self.offload_count}, "
            f"total offloaded: {self.total_offloaded_bytes / _MIB:.0f} MiB"
        )


@dataclass(frozen=True)
class OffloadStep:
    """One tensor on the GPU to move to the CPU; ``move`` replaces it where it is kept."""

    tensor: torch.Tensor
    nbytes: int
    move: Callable[[], None]


def _move_item_to_cpu(items: list[torch.Tensor], index: int) -> None:
    items[index] = items[index].cpu()


def _move_crop_to_cpu(raw_crop: RawCrop) -> None:
    raw_crop.crop = raw_crop.crop.cpu()


def plan_offload(
    results: list[SecondaryRestoreResult], crop_buffers: list[CropBuffer], bytes_to_free: int, device_type: str
) -> list[OffloadStep]:
    """What to move off ``device_type`` to free ``bytes_to_free``: restored frames of the latest
    clips first (each clip's masks go along with its frames), then the crops of the largest
    crop buffers."""
    steps: list[OffloadStep] = []
    planned = 0

    def add(tensor: torch.Tensor, move: Callable[[], None]) -> None:
        nonlocal planned
        nbytes = tensor.nelement() * tensor.element_size()
        steps.append(OffloadStep(tensor, nbytes, move))
        planned += nbytes

    for result in sorted(results, key=lambda result: result.start_frame, reverse=True):
        for index, frame in enumerate(result.restored_frames):
            if frame.device.type == device_type:
                add(frame, partial(_move_item_to_cpu, result.restored_frames, index))
                if planned >= bytes_to_free:
                    return steps
        for index, mask in enumerate(result.masks):
            if mask.device.type == device_type:
                add(mask, partial(_move_item_to_cpu, result.masks, index))

    for crop_buffer in sorted(crop_buffers, key=lambda crop_buffer: crop_buffer.frame_count, reverse=True):
        for raw_crop in crop_buffer.crops:
            if raw_crop.crop.device.type == device_type:
                add(raw_crop.crop, partial(_move_crop_to_cpu, raw_crop))
                if planned >= bytes_to_free:
                    return steps
    return steps


def stall_warning_due(now: float, last_encode: float, last_warning: float) -> tuple[bool, float]:
    """Whether to warn about an encode stall now, and the last warning time to keep: once per
    ``STALL_WARN_SECONDS`` while no frame was encoded for that long."""
    if now - last_encode <= STALL_WARN_SECONDS:
        return False, 0.0
    if now - last_warning >= STALL_WARN_SECONDS:
        return True, now
    return False, last_warning


class VramOffloader:
    """Moves restored frames and crops to the CPU when the GPU runs low, and logs the queues
    when encoding stalls; ``encode_heartbeat[0]`` is the last encode time (None: no stall check)."""

    def __init__(
        self,
        device: torch.device,
        blend_buffer: BlendBuffer,
        crop_buffers: dict[int, CropBuffer],
        *,
        frame_queues: dict[str, FrameQueue],
        metadata_queue: Queue,
        encode_heartbeat: list[float] | None,
    ) -> None:
        self._device = device
        self._blend_buffer = blend_buffer
        self._crop_buffers = crop_buffers

        gpu_total = torch.cuda.get_device_properties(device).total_memory
        self._threshold = max(0, gpu_total - VRAM_SAFETYNET)

        self.stats = VramStats()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="VramOffloader", daemon=True)
        self._last_encode_time = encode_heartbeat
        self._last_stall_warn_time: float = 0.0
        self._stall_check_paused = False
        self._frame_queues = frame_queues
        self._metadata_queue = metadata_queue

        _log.info(
            "VramOffloader: threshold=%d MiB (total=%d MiB, safetynet=%d MiB)",
            self._threshold // _MIB,
            gpu_total // _MIB,
            VRAM_SAFETYNET // _MIB,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)
        _log.info(self.stats.summary())

    def _run(self) -> None:
        while not self._stop.wait(_POLL_INTERVAL):
            free, total = torch.cuda.mem_get_info(self._device)
            used = total - free
            self.stats.update(used)
            if used > self._threshold:
                freed = self._offload(used - self._threshold)
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
            self._check_encode_stall()

    def pause_stall_check(self) -> None:
        self._stall_check_paused = True

    def _check_encode_stall(self) -> None:
        hb = self._last_encode_time
        if hb is None or self._stall_check_paused:
            return
        now = time.monotonic()
        warn, self._last_stall_warn_time = stall_warning_due(now, hb[0], self._last_stall_warn_time)
        if warn:
            elapsed = now - hb[0]
            _log.warning("[vram-offloader] encode stall detected: no frame encoded for %.0fs", elapsed)
            self._dump_stall_diagnostics(elapsed)

    def _dump_stall_diagnostics(self, elapsed: float) -> None:
        lines: list[str] = [f"=== ENCODE STALL DIAGNOSTICS (stalled {elapsed:.0f}s) ==="]

        for name, q in self._frame_queues.items():
            lines.append(f"  {name}: items={q.qsize()} frames={q.current_frames} max_frames={q.max_frames}")
        lines.append(f"  metadata_queue: items~={self._metadata_queue.qsize()} maxsize={self._metadata_queue.maxsize}")
        lines.extend(f"  {line}" for line in self._blend_buffer.debug_lines())
        crop_sizes = {k: v.frame_count for k, v in list(self._crop_buffers.items())}
        lines.append(f"  crop_buffers: track_ids={list(crop_sizes)} sizes={crop_sizes}")

        free, total = torch.cuda.mem_get_info(self._device)
        lines.append(
            f"  VRAM: used={((total - free) / _MIB):.0f} MiB"
            f" allocated={torch.cuda.memory_allocated(self._device) / _MIB:.0f} MiB"
            f" reserved={torch.cuda.memory_reserved(self._device) / _MIB:.0f} MiB"
            f" free={free / _MIB:.0f} MiB"
        )

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
        steps = plan_offload(
            self._blend_buffer.offloadable_results(),
            list(self._crop_buffers.values()),
            bytes_to_free,
            self._device.type,
        )
        for step in steps:
            step.move()
        return sum(step.nbytes for step in steps)
