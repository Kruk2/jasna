from __future__ import annotations

import logging
import threading
import time

import torch

from jasna.blend_buffer import BlendBuffer
from jasna.crop_buffer import CropBuffer

_log = logging.getLogger(__name__)

VRAM_LIMIT: float | None = None
VRAM_SAFETYNET: int = 750 * 1024 * 1024

_POLL_INTERVAL = 0.1
_MIB = 1024 * 1024


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
        crop_lock: threading.Lock,
        vram_limit: float | None = VRAM_LIMIT,
        safetynet: int = VRAM_SAFETYNET,
    ) -> None:
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

        self.stats = VramStats()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="VramOffloader", daemon=True)

        _log.info(
            "VramOffloader: threshold=%d MiB (total=%d MiB, safetynet=%d MiB)",
            self._threshold // _MIB,
            gpu_total // _MIB,
            safetynet // _MIB,
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
