from __future__ import annotations

import atexit
import collections
from dataclasses import dataclass
import logging
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np
import torch

from jasna.os_utils import get_subprocess_startup_info

logger = logging.getLogger(__name__)


def _parse_tvai_args_kv(args: str) -> dict[str, str]:
    args = (args or "").strip()
    if args == "":
        return {}
    out: dict[str, str] = {}
    for part in args.split(":"):
        part = part.strip()
        if part == "":
            continue
        if "=" not in part:
            raise ValueError(f"Invalid --tvai-args item: {part!r} (expected key=value)")
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k == "":
            raise ValueError(f"Invalid --tvai-args item: {part!r} (empty key)")
        out[k] = v
    return out


TMeta = TypeVar("TMeta")


@dataclass(frozen=True)
class _TvaiTask(Generic[TMeta]):
    frames_256: torch.Tensor
    keep_start: int
    keep_end: int
    meta: list[TMeta]


@dataclass(frozen=True)
class _FlushTask:
    """Sentinel task to flush the worker's temporal buffer."""
    timeout_s: float


@dataclass(frozen=True)
class _TvaiCompleted(Generic[TMeta]):
    meta: TMeta
    worker_idx: int
    out_buf: torch.Tensor  # (H, W, 3) uint8, CPU pinned

    def to_frame_u8(self, device: torch.device) -> torch.Tensor:
        # Pinned memory enables fast DMA transfer; permute on GPU is free
        return self.out_buf.to(device=device, non_blocking=True).permute(2, 0, 1)


@dataclass(frozen=True)
class _TvaiWorkerFatal:
    error: BaseException


class _TvaiFfmpegRestorer:
    name = "tvai"
    IN_WRITE_CHUNK_FRAMES = 4

    def __init__(
        self,
        *,
        device: torch.device,
        ffmpeg_path: str,
        tvai_args: str,
        max_clip_size: int,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError(f"TVAI secondary restorer supports only CUDA (got device={self.device})")
        self.ffmpeg_path = str(ffmpeg_path)
        self.tvai_args = str(tvai_args)
        self._tvai_kv = _parse_tvai_args_kv(self.tvai_args)

        model_name = (self._tvai_kv.get("model") or "").strip().lower()

        if ("w" in self._tvai_kv) or ("h" in self._tvai_kv):
            raise ValueError('Do not pass "w" or "h" in --tvai-args; use --tvai-scale instead')

        scale_raw = self._tvai_kv.get("scale")
        if scale_raw is None:
            raise ValueError('Missing "scale" in tvai args. Pass it via --tvai-scale (valid: 1, 2, 4)')
        scale = int(scale_raw)
        if scale not in (1, 2, 4):
            raise ValueError(f'Invalid tvai "scale": {scale} (valid: 1, 2, 4)')
        self.scale = scale

        self.out_w = 256 * scale
        self.out_h = 256 * scale

        parts: list[tuple[str, str]] = []
        if "model" in self._tvai_kv:
            parts.append(("model", self._tvai_kv["model"]))
        parts.append(("scale", str(self.scale)))
        for k, v in self._tvai_kv.items():
            if k in {"model", "scale", "w", "h"}:
                continue
            parts.append((k, v))
        self._tvai_args_effective = ":".join(f"{k}={v}" for k, v in parts)

        if max_clip_size <= 0:
            raise ValueError("--max-clip-size must be > 0")
        self.max_clip_size = max_clip_size

        logger.info(
            "TVAI init: ffmpeg_path=%r tvai_args_effective=%r model=%r scale=%d out=%dx%d device=%s max_clip_size=%d",
            self.ffmpeg_path,
            self._tvai_args_effective,
            model_name,
            self.scale,
            self.out_w,
            self.out_h,
            self.device,
            self.max_clip_size,
        )

        self._in_buf = torch.empty((self.max_clip_size, 256, 256, 3), dtype=torch.uint8, device="cpu", pin_memory=True)
        self._in_np = self._in_buf.numpy()
        self._in_frame_bytes = 256 * 256 * 3
        self._in_all_mv = memoryview(self._in_np).cast("B")
        self._fatal: BaseException | None = None
        self._stderr_buf: list[bytes] = []

        self._validate_tvai_environment()
        self._start_ffmpeg()
        atexit.register(self.close)

    def _validate_tvai_environment(self) -> None:
        data_dir = os.environ.get("TVAI_MODEL_DATA_DIR")
        model_dir = os.environ.get("TVAI_MODEL_DIR")
        logger.debug("TVAI env: TVAI_MODEL_DATA_DIR=%r TVAI_MODEL_DIR=%r", data_dir, model_dir)
        if not data_dir:
            raise RuntimeError("TVAI_MODEL_DATA_DIR env var is not set")
        if not model_dir:
            raise RuntimeError("TVAI_MODEL_DIR env var is not set")

        if not Path(data_dir).is_dir():
            raise RuntimeError(f"TVAI_MODEL_DATA_DIR does not point to an existing directory: {data_dir!r}")
        if not Path(model_dir).is_dir():
            raise RuntimeError(f"TVAI_MODEL_DIR does not point to an existing directory: {model_dir!r}")

        if not Path(self.ffmpeg_path).is_file():
            raise FileNotFoundError(f"TVAI ffmpeg not found: {self.ffmpeg_path!r}")

    def _start_ffmpeg(self) -> None:
        cmd = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "256x256",
            "-r",
            "25",
            "-i",
            "pipe:0",
            "-sws_flags",
            "spline+accurate_rnd+full_chroma_int",
            "-filter_complex",
            f"tvai_up={self._tvai_args_effective}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]

        logger.debug("TVAI starting ffmpeg: %r", cmd)
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=get_subprocess_startup_info(),
            env=os.environ.copy(),
            bufsize=0,
        )
        if self._proc.stdin is None or self._proc.stdout is None or self._proc.stderr is None:
            raise RuntimeError("Failed to start TVAI ffmpeg with pipes")
        logger.debug("TVAI ffmpeg started: pid=%s", self._proc.pid)

        self._out_lock = threading.Lock()
        self._out_cond = threading.Condition(self._out_lock)
        self._out_frames: collections.deque[np.ndarray] = collections.deque()

        def _stderr_reader() -> None:
            try:
                while True:
                    line = self._proc.stderr.readline()
                    if not line:
                        break
                    with self._out_lock:
                        self._stderr_buf.append(line)
                        if len(self._stderr_buf) > 200:
                            self._stderr_buf = self._stderr_buf[-200:]
            except BaseException as e:
                with self._out_cond:
                    if self._fatal is None:
                        self._fatal = e
                    self._out_cond.notify_all()

        def _stdout_reader() -> None:
            try:
                frames_read = 0
                frame_bytes = self.out_h * self.out_w * 3
                while True:
                    out_buf = torch.empty((self.out_h, self.out_w, 3), dtype=torch.uint8, device="cpu", pin_memory=True)
                    mv = memoryview(out_buf.numpy()).cast("B")
                    offset = 0
                    while offset < frame_bytes:
                        n = self._proc.stdout.readinto(mv[offset:])
                        if not n:
                            if offset == 0:
                                return
                            raise RuntimeError(
                                f"Unexpected EOF while reading ffmpeg stdout (got {offset} / {frame_bytes} bytes)"
                            )
                        offset += n
                    frames_read += 1
                    if (frames_read <= 3) or (frames_read % 25 == 0):
                        logger.debug("TVAI stdout: received frame=%d bytes=%d", frames_read, frame_bytes)
                    with self._out_cond:
                        self._out_frames.append(out_buf)
                        self._out_cond.notify_all()
            except BaseException as e:
                with self._out_cond:
                    if self._fatal is None:
                        self._fatal = e
                    self._out_cond.notify_all()

        self._stderr_thread = threading.Thread(target=_stderr_reader, daemon=True)
        self._stdout_thread = threading.Thread(target=_stdout_reader, daemon=True)
        self._stderr_thread.start()
        self._stdout_thread.start()

    def _check_alive(self) -> None:
        if self._fatal is not None:
            raise RuntimeError(f"TVAI ffmpeg is not healthy: {self._fatal}") from self._fatal
        rc = self._proc.poll()
        if rc is not None:
            stderr_text = b"".join(self._stderr_buf).decode(errors="replace").strip()
            raise RuntimeError(f"TVAI ffmpeg crashed (exit_code={rc}). stderr:\n{stderr_text}")

    def _check_fatal(self) -> None:
        if self._fatal is not None:
            raise RuntimeError(f"TVAI ffmpeg is not healthy: {self._fatal}") from self._fatal

    def _drain_stdout(self) -> list[np.ndarray]:
        """Drain all currently available frames from stdout deque. Must hold _out_lock."""
        out: list[np.ndarray] = []
        while self._out_frames:
            out.append(self._out_frames.popleft())
        return out

    def close(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is None:
            return
        if proc.poll() is not None:
            return
        logger.debug("TVAI closing ffmpeg: pid=%s", proc.pid)
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        finally:
            proc.terminate()

    def push_padding(self, n: int) -> list[np.ndarray]:
        """Write n black frames to stdin to flush internal temporal buffers."""
        if n <= 0:
            return []
        self._check_alive()
        black_frame = bytes(self._in_frame_bytes)
        drained: list[np.ndarray] = []
        for i in range(n):
            self._proc.stdin.write(black_frame)
            with self._out_cond:
                self._check_alive()
                drained.extend(self._drain_stdout())
        logger.debug("TVAI push_padding: wrote=%d drained=%d", n, len(drained))
        return drained

    def restore(self, frames_256: torch.Tensor, *, keep_start: int, keep_end: int) -> list[torch.Tensor]:
        t0 = time.perf_counter()
        t = frames_256.shape[0]
        if t == 0:
            return []

        if frames_256.dtype == torch.uint8:
            raise RuntimeError("TVAI secondary expects float frames in [0, 1], got uint8")

        ks = max(0, keep_start)
        ke = min(t, keep_end)
        n = ke - ks
        if n <= 0:
            return []
        if n > self.max_clip_size:
            raise RuntimeError(f"TVAI restore got n={n} which exceeds max_clip_size={self.max_clip_size}")

        logger.debug("TVAI restore(stream): t=%d keep=[%d, %d) n=%d", t, ks, ke, n)

        frames_u8 = frames_256[ks:ke].mul(255.0).round().clamp(0, 255).to(dtype=torch.uint8)
        self._in_buf[:n].copy_(frames_u8.permute(0, 2, 3, 1), non_blocking=True)
        torch.cuda.synchronize(frames_256.device)

        self._check_alive()
        drained: list[torch.Tensor] = []

        for start in range(0, n, self.IN_WRITE_CHUNK_FRAMES):
            end = min(n, start + self.IN_WRITE_CHUNK_FRAMES)
            start_b = start * self._in_frame_bytes
            end_b = end * self._in_frame_bytes
            logger.debug("TVAI stdin: writing frames [%d, %d) bytes [%d, %d)", start, end, start_b, end_b)
            self._proc.stdin.write(self._in_all_mv[start_b:end_b])

            with self._out_cond:
                self._check_alive()
                drained.extend(self._drain_stdout())

        with self._out_cond:
            self._check_alive()
            drained.extend(self._drain_stdout())

        logger.debug(
            "TVAI restore(stream) done: wrote=%d drained=%d elapsed_ms=%.1f",
            n, len(drained), (time.perf_counter() - t0) * 1000.0,
        )
        return drained

    def flush(self, *, timeout_s: float = 300.0) -> list[torch.Tensor]:
        t0 = time.perf_counter()
        proc = getattr(self, "_proc", None)
        if proc is None:
            return []

        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except BaseException:
                pass

        deadline = time.perf_counter() + timeout_s
        drained: list[torch.Tensor] = []

        while True:
            with self._out_cond:
                self._check_fatal()
                drained.extend(self._drain_stdout())

            if proc.poll() is not None:
                break

            if time.perf_counter() >= deadline:
                stderr_text = b"".join(self._stderr_buf).decode(errors="replace").strip()
                raise RuntimeError(f"TVAI ffmpeg flush timed out after {timeout_s:.1f}s. stderr:\n{stderr_text}")

            with self._out_cond:
                self._out_cond.wait(timeout=0.05)

        self._stdout_thread.join(timeout=1.0)

        with self._out_cond:
            self._check_fatal()
            drained.extend(self._drain_stdout())

        logger.debug(
            "TVAI flush done: drained=%d elapsed_ms=%.1f",
            len(drained), (time.perf_counter() - t0) * 1000.0,
        )
        return drained


class TvaiSecondaryRestorer(Generic[TMeta]):
    name = "tvai"

    # TVAI filter has an internal temporal pipeline. In practice it often needs ~25 frames
    # of future input before the last outputs of a segment become available.
    LATENCY_FRAMES = 25

    def __init__(
        self,
        *,
        device: torch.device,
        ffmpeg_path: str,
        tvai_args: str,
        max_clip_size: int,
        num_workers: int = 2,
    ) -> None:
        num_workers = int(num_workers)
        if num_workers <= 0:
            raise ValueError("num_workers must be > 0")

        self.device = torch.device(device)
        self.ffmpeg_path = str(ffmpeg_path)
        self.tvai_args = str(tvai_args)
        self.max_clip_size = int(max_clip_size)
        self.num_workers = num_workers

        self._workers: list[_TvaiFfmpegRestorer] = []
        self._worker_threads: list[threading.Thread] = []
        self._task_queues: list[queue.Queue[_TvaiTask[TMeta] | None]] = []
        self._pending_by_worker: list[collections.deque[TMeta | None]] = []
        self._completed: queue.Queue[_TvaiCompleted[TMeta] | _TvaiWorkerFatal] = queue.Queue()
        self._fatal: BaseException | None = None
        self._track_to_worker: dict[int, int] = {}
        self._worker_task_count: list[int] = [0] * num_workers
        self._closed = False
        self._slots = threading.Semaphore(num_workers)

        for i in range(num_workers):
            w = _TvaiFfmpegRestorer(
                device=self.device,
                ffmpeg_path=self.ffmpeg_path,
                tvai_args=self.tvai_args,
                max_clip_size=self.max_clip_size,
            )
            self._workers.append(w)
            self._task_queues.append(queue.Queue())
            self._pending_by_worker.append(collections.deque())

            def _run(worker_idx: int) -> None:
                try:
                    if self.device.type == "cuda":
                        torch.cuda.set_device(self.device)
                    rest = self._workers[worker_idx]
                    pending = self._pending_by_worker[worker_idx]
                    q = self._task_queues[worker_idx]
                    while True:
                        task = q.get()
                        if task is None:
                            return
                        if isinstance(task, _FlushTask):
                            real_remaining = sum(1 for m in pending if m is not None)
                            logger.debug("worker %d flush: real_pending=%d", worker_idx, real_remaining)
                            if real_remaining == 0:
                                continue

                            deadline = time.perf_counter() + task.timeout_s
                            black_frame = bytes(rest._in_frame_bytes)
                            padding_pushed = 0
                            drained_total = 0
                            last_progress_t = time.perf_counter()

                            # Push some padding upfront to advance TVAI's internal pipeline.
                            for _ in range(TvaiSecondaryRestorer.LATENCY_FRAMES):
                                rest._proc.stdin.write(black_frame)
                                pending.append(None)
                                padding_pushed += 1

                            # Drain outputs until all real frames are out.
                            while real_remaining > 0:
                                with rest._out_cond:
                                    rest._check_alive()
                                    drained = rest._drain_stdout()
                                    if not drained:
                                        now = time.perf_counter()
                                        if now >= deadline:
                                            logger.warning(
                                                "worker %d flush timed out: real_remaining=%d padding_pushed=%d",
                                                worker_idx,
                                                real_remaining,
                                                padding_pushed,
                                            )
                                            break
                                        # If no progress for a bit, push extra padding to force more output.
                                        if (now - last_progress_t) >= 0.25:
                                            for _ in range(5):
                                                rest._proc.stdin.write(black_frame)
                                                pending.append(None)
                                                padding_pushed += 1
                                            last_progress_t = now
                                        rest._out_cond.wait(timeout=0.05)
                                        continue

                                drained_total += len(drained)
                                last_progress_t = time.perf_counter()
                                for out_buf in drained:
                                    if not pending:
                                        continue
                                    meta = pending.popleft()
                                    if meta is not None:
                                        real_remaining -= 1
                                    self._completed.put(
                                        _TvaiCompleted(
                                            meta=meta,
                                            worker_idx=worker_idx,
                                            out_buf=out_buf,
                                        )
                                    )

                            # Important: do NOT clear remaining padding markers.
                            # They must stay queued so any future TVAI outputs (caused by internal latency)
                            # get consumed as padding and cannot shift alignment for subsequent real frames.
                            logger.debug(
                                "worker %d flush done: real_remaining=%d padding_pushed=%d drained=%d pending_after=%d",
                                worker_idx,
                                real_remaining,
                                padding_pushed,
                                drained_total,
                                len(pending),
                            )
                            continue
                        try:
                            pending.extend(task.meta)
                            drained = rest.restore(
                                task.frames_256, keep_start=task.keep_start, keep_end=task.keep_end
                            )
                            for out_buf in drained:
                                if not pending:
                                    break
                                meta = pending.popleft()
                                self._completed.put(
                                    _TvaiCompleted(
                                        meta=meta,
                                        worker_idx=worker_idx,
                                        out_buf=out_buf,
                                    )
                                )
                        finally:
                            self._slots.release()
                except BaseException as e:
                    self._completed.put(_TvaiWorkerFatal(error=e))

            t = threading.Thread(target=_run, args=(i,), daemon=True, name=f"tvai-worker-{i}")
            t.start()
            self._worker_threads.append(t)

        self.out_w = self._workers[0].out_w
        self.out_h = self._workers[0].out_h
        atexit.register(self.close)

    def close(self) -> None:
        self._closed = True
        for q in self._task_queues:
            try:
                q.put_nowait(None)
            except BaseException:
                pass
        for t in self._worker_threads:
            try:
                t.join(timeout=1.0)
            except BaseException:
                pass
        for w in self._workers:
            try:
                w.close()
            except BaseException:
                pass

    def _raise_if_fatal(self) -> None:
        if self._fatal is not None:
            raise RuntimeError(f"TVAI worker failed: {self._fatal}") from self._fatal

    def submit(self, frames_256: torch.Tensor, *, keep_start: int, keep_end: int, meta: list[TMeta], track_id: int = 0) -> None:
        self._raise_if_fatal()
        if self._closed:
            raise RuntimeError("TVAI pool is closed")
        if not meta:
            return
        self._slots.acquire()
        idx = self._track_to_worker.get(track_id)
        if idx is None:
            idx = min(range(self.num_workers), key=lambda i: self._worker_task_count[i])
            self._track_to_worker[track_id] = idx
        self._worker_task_count[idx] += 1
        self._task_queues[idx].put(_TvaiTask(frames_256=frames_256, keep_start=keep_start, keep_end=keep_end, meta=meta))

    def flush_track(self, track_id: int, timeout_s: float = 60.0) -> None:
        """Flush a specific track by pushing padding frames to its worker.
        
        This forces the temporal filter to output remaining real frames.
        The padding outputs will be discarded when drained.
        
        This is async - we submit the flush task and return immediately.
        The worker will process it when it reaches the task in its queue.
        """
        self._raise_if_fatal()
        if self._closed:
            logger.debug("flush_track(%d): pool is closed, skipping", track_id)
            return
        idx = self._track_to_worker.pop(track_id, None)
        if idx is None:
            logger.debug("flush_track(%d): track not mapped to any worker, skipping", track_id)
            return
        
        logger.debug("flush_track(%d): queueing async flush task to worker %d", track_id, idx)
        # Don't acquire slot - let worker process flush when it reaches it in queue.
        # This makes flush_track truly non-blocking.
        self._task_queues[idx].put(_FlushTask(timeout_s=timeout_s))

    def transfer_track(self, old_track_id: int, new_track_id: int) -> None:
        """Transfer worker mapping from old track to new track without flushing.
        
        Used when a track is split due to max_clip_size - the continuation should
        use the same worker so TVAI outputs flow naturally without padding.
        """
        idx = self._track_to_worker.pop(old_track_id, None)
        if idx is not None:
            self._track_to_worker[new_track_id] = idx

    def drain_completed(self, *, limit: int | None = None) -> list[_TvaiCompleted[TMeta]]:
        self._raise_if_fatal()
        out: list[_TvaiCompleted[TMeta]] = []
        while True:
            if limit is not None and len(out) >= limit:
                break
            try:
                item = self._completed.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, _TvaiWorkerFatal):
                self._fatal = item.error
                self._raise_if_fatal()
            if item.meta is None:
                continue
            out.append(item)
        return out

    def flush(self, *, timeout_s: float = 300.0) -> None:
        self._raise_if_fatal()
        if self._closed:
            return
        self._closed = True

        for q in self._task_queues:
            q.put(None)

        deadline = time.perf_counter() + timeout_s
        for t in self._worker_threads:
            remaining = max(0.0, deadline - time.perf_counter())
            t.join(timeout=remaining)
            if t.is_alive():
                raise RuntimeError("TVAI worker did not finish before flush deadline")

        for worker_idx, rest in enumerate(self._workers):
            remaining = max(0.0, deadline - time.perf_counter())
            drained = rest.flush(timeout_s=remaining if remaining > 0 else 0.1)
            pending = self._pending_by_worker[worker_idx]
            for out_buf in drained:
                if not pending:
                    break
                meta = pending.popleft()
                self._completed.put(
                    _TvaiCompleted(
                        meta=meta,
                        worker_idx=worker_idx,
                        out_buf=out_buf,
                    )
                )
            real_remaining = sum(1 for m in pending if m is not None)
            if real_remaining > 0:
                raise RuntimeError(f"TVAI flush did not produce all pending frames (worker={worker_idx} real_remaining={real_remaining})")
