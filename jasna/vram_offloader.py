from __future__ import annotations

import logging
import sys
import threading
import time
import traceback

import torch

from jasna.blend_buffer import BlendBuffer
from jasna.crop_buffer import CropBuffer

_log = logging.getLogger(__name__)

VRAM_LIMIT: float | None = None
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


class VramOffloader:
    def __init__(
        self,
        device: torch.device,
        blend_buffer: BlendBuffer,
        crop_buffers: dict[int, CropBuffer],
        vram_limit: float | None = VRAM_LIMIT,
        safetynet: int = VRAM_SAFETYNET,
    ) -> None:
        self._device = device
        self._blend_buffer = blend_buffer
        self._crop_buffers = crop_buffers

        if vram_limit is not None:
            gpu_total = int(vram_limit * 1024 * 1024 * 1024)
        else:
            gpu_total = torch.cuda.get_device_properties(device).total_memory
        self._threshold = max(0, gpu_total - safetynet)
        self._offload_device_type = "cuda"

        self.stats = VramStats()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="VramOffloader", daemon=True)
        self._last_encode_time: list[float] | None = None
        self._last_stall_warn_time: float = 0.0
        self._stall_check_paused = False
        self._pipeline_queues: dict[str, object] | None = None
        self._metadata_queue: object | None = None

        _log.info(
            "VramOffloader: threshold=%d MiB (total=%d MiB, safetynet=%d MiB)",
            self._threshold // _MIB,
            gpu_total // _MIB,
            safetynet // _MIB,
        )

    def set_encode_heartbeat(self, shared_time: list[float]) -> None:
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
        elapsed = now - hb[0]
        if elapsed > STALL_WARN_SECONDS:
            if now - self._last_stall_warn_time >= STALL_WARN_SECONDS:
                _log.warning(
                    "[vram-offloader] encode stall detected: no frame encoded for %.0fs",
                    elapsed,
                )
                self._dump_stall_diagnostics(elapsed)
                self._last_stall_warn_time = now
        else:
            self._last_stall_warn_time = 0.0

    def _dump_stall_diagnostics(self, elapsed: float) -> None:
        lines: list[str] = [f"=== ENCODE STALL DIAGNOSTICS (stalled {elapsed:.0f}s) ==="]

        if self._pipeline_queues:
            for name, q in self._pipeline_queues.items():
                lines.append(f"  {name}: items={q.qsize()} frames={q.current_frames} max_frames={q._max_frames}")
        if self._metadata_queue is not None:
            lines.append(
                f"  metadata_queue: items~={self._metadata_queue.qsize()} maxsize={self._metadata_queue.maxsize}"
            )
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
