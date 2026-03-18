from __future__ import annotations

import atexit
import logging
import os
import subprocess
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import torch

from jasna.os_utils import get_subprocess_startup_info

logger = logging.getLogger(__name__)

TVAI_MIN_FRAMES = 5
WRITE_BATCH_SIZE = 4
READ_TIMEOUT = 10.0


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


@dataclass
class _PendingClip:
    seq: int
    worker_idx: int
    real_count: int
    total_pushed: int


class _TvaiWorker:
    def __init__(self, ffmpeg_path: str, tvai_filter_args: str, scale: int) -> None:
        self.out_w = 256 * scale
        self.out_h = 256 * scale
        self._in_frame_bytes = 256 * 256 * 3
        self._out_frame_bytes = self.out_h * self.out_w * 3
        self._ffmpeg_path = ffmpeg_path
        self._tvai_filter_args = tvai_filter_args
        self._scale = scale
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._output: Queue[np.ndarray] = Queue()
        self._intentional_kill = False
        self._start()

    def _build_cmd(self) -> list[str]:
        return [
            self._ffmpeg_path,
            "-hide_banner", "-loglevel", "warning",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "256x256", "-r", "25",
            "-i", "pipe:0",
            "-sws_flags", "spline+accurate_rnd+full_chroma_int",
            "-filter_complex", f"tvai_up={self._tvai_filter_args}",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "pipe:1",
        ]

    def _start(self) -> None:
        self._intentional_kill = False
        self._output = Queue()
        self._proc = subprocess.Popen(
            self._build_cmd(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.environ.get("TVAI_MODEL_DATA_DIR"),
            startupinfo=get_subprocess_startup_info(),
            env=os.environ.copy(),
        )
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()
        logger.info("TVAI worker started pid=%s", self._proc.pid)

    def _reader_loop(self) -> None:
        proc = self._proc
        while True:
            data = proc.stdout.read(self._out_frame_bytes)
            if not data or len(data) < self._out_frame_bytes:
                break
            frame = np.frombuffer(
                data, dtype=np.uint8, count=self._out_frame_bytes,
            ).reshape(self.out_h, self.out_w, 3).copy()
            self._output.put(frame)
        rc = proc.wait()
        if rc != 0 and not self._intentional_kill:
            stderr = proc.stderr.read() if proc.stderr else b""
            stderr_text = stderr.decode(errors="replace").strip() if stderr else ""
            logger.error(
                "TVAI worker pid=%s died (code %s). stderr: %s",
                proc.pid, rc, stderr_text[-1000:],
            )

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def push_frames(self, frames_hwc_u8: np.ndarray) -> None:
        n = frames_hwc_u8.shape[0]
        logger.info("TVAI worker pid=%s: pushing %d frames", self._proc.pid, n)
        data = memoryview(frames_hwc_u8).cast("B")
        batch_bytes = WRITE_BATCH_SIZE * self._in_frame_bytes
        for offset in range(0, len(data), batch_bytes):
            self._proc.stdin.write(bytes(data[offset:offset + batch_bytes]))
            self._proc.stdin.flush()

    def read_frames(self, n: int, timeout: float = READ_TIMEOUT) -> list[np.ndarray]:
        results: list[np.ndarray] = []
        for _ in range(n):
            try:
                results.append(self._output.get(timeout=timeout))
            except Empty:
                break
        logger.info("TVAI worker pid=%s: read %d/%d frames", self._proc.pid, len(results), n)
        return results

    def read_available(self) -> list[np.ndarray]:
        results: list[np.ndarray] = []
        while True:
            try:
                results.append(self._output.get_nowait())
            except Empty:
                break
        return results

    def flush(self) -> list[np.ndarray]:
        logger.info("TVAI worker pid=%s: flushing", self._proc.pid)
        self._proc.stdin.close()
        if self._reader is not None:
            self._reader.join(timeout=10)
        remaining: list[np.ndarray] = []
        while not self._output.empty():
            remaining.append(self._output.get_nowait())
        logger.info("TVAI worker pid=%s: flushed %d remaining frames", self._proc.pid, len(remaining))
        return remaining

    def restart(self) -> None:
        logger.info("TVAI worker pid=%s: restarting", self._proc.pid)
        self.kill()
        if self._reader is not None:
            self._reader.join(timeout=5)
        self._start()

    def kill(self) -> None:
        self._intentional_kill = True
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait()


class TvaiSecondaryRestorer:
    name = "tvai"

    def __init__(self, *, ffmpeg_path: str, tvai_args: str, scale: int, num_workers: int) -> None:
        self.ffmpeg_path = str(ffmpeg_path)
        self.num_workers = int(num_workers)

        kv = _parse_tvai_args_kv(tvai_args)
        if scale not in (1, 2, 4):
            raise ValueError(f'Invalid tvai scale: {scale} (valid: 1, 2, 4)')
        self.scale = scale
        self.out_w = 256 * scale
        self.out_h = 256 * scale

        parts: list[tuple[str, str]] = []
        if "model" in kv:
            parts.append(("model", kv["model"]))
        parts.append(("scale", str(self.scale)))
        for k, v in kv.items():
            if k in {"model", "scale", "w", "h"}:
                continue
            parts.append((k, v))
        self._tvai_filter_args = ":".join(f"{k}={v}" for k, v in parts)

        self._validate_environment()

        self._workers: list[_TvaiWorker] = [
            _TvaiWorker(self.ffmpeg_path, self._tvai_filter_args, self.scale)
            for _ in range(self.num_workers)
        ]
        self._worker_locks = [threading.Lock() for _ in range(self.num_workers)]
        self._worker_pending_frames = [0] * self.num_workers
        self._assign_lock = threading.Lock()

        self._next_seq = 0
        self._pending_clips: deque[_PendingClip] = deque()
        self._worker_output_buf: list[list[np.ndarray]] = [[] for _ in range(self.num_workers)]
        self._worker_last_frame: list[np.ndarray | None] = [None] * self.num_workers

        logger.info(
            "TvaiSecondaryRestorer: ffmpeg=%r filter_args=%r scale=%d num_workers=%d",
            self.ffmpeg_path, self._tvai_filter_args, self.scale, self.num_workers,
        )

        atexit.register(self.close)

    def _validate_environment(self) -> None:
        data_dir = os.environ.get("TVAI_MODEL_DATA_DIR")
        model_dir = os.environ.get("TVAI_MODEL_DIR")
        if not data_dir:
            raise RuntimeError("TVAI_MODEL_DATA_DIR env var is not set")
        if not model_dir:
            raise RuntimeError("TVAI_MODEL_DIR env var is not set")
        if not Path(data_dir).is_dir():
            raise RuntimeError(f"TVAI_MODEL_DATA_DIR not a directory: {data_dir!r}")
        if not Path(model_dir).is_dir():
            raise RuntimeError(f"TVAI_MODEL_DIR not a directory: {model_dir!r}")
        if not Path(self.ffmpeg_path).is_file():
            raise FileNotFoundError(f"TVAI ffmpeg not found: {self.ffmpeg_path!r}")

    def _get_worker_index(self, frame_count: int = 0) -> int:
        with self._assign_lock:
            idx = min(range(self.num_workers), key=lambda i: self._worker_pending_frames[i])
            self._worker_pending_frames[idx] += frame_count
        return idx

    def _to_numpy_hwc(self, frames: torch.Tensor) -> np.ndarray:
        frames_u8 = frames.mul(255.0).round().clamp(0, 255).to(dtype=torch.uint8)
        return frames_u8.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    def _pad_to_minimum(self, frames_hwc: np.ndarray) -> tuple[np.ndarray, int]:
        n = frames_hwc.shape[0]
        if n >= TVAI_MIN_FRAMES:
            return frames_hwc, 0
        pad_count = TVAI_MIN_FRAMES - n
        padding = np.repeat(frames_hwc[-1:], pad_count, axis=0)
        return np.concatenate([frames_hwc, padding], axis=0), pad_count

    def _run_worker(self, frames_hwc: np.ndarray) -> list[np.ndarray]:
        n = frames_hwc.shape[0]
        idx = self._get_worker_index(n)
        worker = self._workers[idx]
        lock = self._worker_locks[idx]

        with lock:
            if not worker.alive:
                worker.restart()
            worker.push_frames(frames_hwc)
            results = worker.read_frames(n)

            if len(results) < n:
                logger.info(
                    "TVAI worker %d: got %d/%d, flushing to get remaining",
                    idx, len(results), n,
                )
                remaining = worker.flush()
                results.extend(remaining[:n - len(results)])
                worker.restart()

            if len(results) < n:
                raise RuntimeError(f"TVAI worker {idx}: expected {n} output frames, got {len(results)}")

            with self._assign_lock:
                self._worker_pending_frames[idx] -= n
            return results

    def _to_tensors(self, frames_np: list[np.ndarray]) -> list[torch.Tensor]:
        return [torch.from_numpy(f).permute(2, 0, 1) for f in frames_np]

    _restore_call_count = 0

    def restore_numpy(self, frames_hwc: np.ndarray) -> list[np.ndarray]:
        n = frames_hwc.shape[0]
        frames_hwc, pad_count = self._pad_to_minimum(frames_hwc)
        out_np = self._run_worker(frames_hwc)
        if pad_count > 0:
            out_np = out_np[:n]
        return out_np

    def restore(self, frames_256: torch.Tensor, *, keep_start: int, keep_end: int) -> list[torch.Tensor]:
        self._restore_call_count += 1
        t = frames_256.shape[0]
        ks = max(0, keep_start)
        ke = min(t, keep_end)
        if ks >= ke:
            return []

        kept = frames_256[ks:ke]
        n = kept.shape[0]
        logger.debug("TVAI restore #%d: %d total frames, keep [%d:%d] = %d frames", self._restore_call_count, t, ks, ke, n)

        frames_hwc = self._to_numpy_hwc(kept)
        del kept, frames_256
        out_np = self.restore_numpy(frames_hwc)
        return self._to_tensors(out_np)

    def push_clip(self, frames_256: torch.Tensor, keep_start: int, keep_end: int) -> int:
        seq = self._next_seq
        self._next_seq += 1

        t = frames_256.shape[0]
        ks = max(0, keep_start)
        ke = min(t, keep_end)
        if ks >= ke:
            self._pending_clips.append(_PendingClip(seq=seq, worker_idx=-1, real_count=0, total_pushed=0))
            return seq

        kept = frames_256[ks:ke]
        frames_hwc = self._to_numpy_hwc(kept)
        n = frames_hwc.shape[0]

        idx = self._get_worker_index(n)
        worker = self._workers[idx]
        lock = self._worker_locks[idx]
        with lock:
            if not worker.alive:
                worker.restart()
            worker.push_frames(frames_hwc)
        self._worker_last_frame[idx] = frames_hwc[-1]

        self._pending_clips.append(_PendingClip(seq=seq, worker_idx=idx, real_count=n, total_pushed=n))
        logger.info("TVAI push_clip seq=%d: %d frames -> worker %d", seq, n, idx)
        return seq

    def _drain_workers(self) -> None:
        for idx, worker in enumerate(self._workers):
            frames = worker.read_available()
            if frames:
                self._worker_output_buf[idx].extend(frames)

    def pop_completed(self) -> list[tuple[int, list[torch.Tensor]]]:
        self._drain_workers()
        completed: list[tuple[int, list[torch.Tensor]]] = []
        while self._pending_clips:
            clip = self._pending_clips[0]
            if clip.real_count == 0:
                self._pending_clips.popleft()
                completed.append((clip.seq, []))
                continue
            buf = self._worker_output_buf[clip.worker_idx]
            if len(buf) >= clip.total_pushed:
                real_frames = buf[:clip.real_count]
                del buf[:clip.total_pushed]
                self._pending_clips.popleft()
                with self._assign_lock:
                    self._worker_pending_frames[clip.worker_idx] -= clip.total_pushed
                completed.append((clip.seq, self._to_tensors(real_frames)))
            else:
                break
        return completed

    def flush_all(self) -> None:
        with self._assign_lock:
            busy = [i for i in range(len(self._workers)) if self._worker_pending_frames[i] > 0]
        if not busy:
            return
        logger.info("TVAI flush_all: flushing %d/%d workers", len(busy), len(self._workers))
        threads = [threading.Thread(target=self._flush_worker, args=(idx,), daemon=True) for idx in busy]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def _flush_worker(self, idx: int) -> None:
        worker = self._workers[idx]
        with self._worker_locks[idx]:
            if worker.alive:
                pending = self._worker_pending_frames[idx]
                pad_count = 0
                if pending < TVAI_MIN_FRAMES and self._worker_last_frame[idx] is not None:
                    pad_count = TVAI_MIN_FRAMES - pending
                    padding = np.repeat(self._worker_last_frame[idx][np.newaxis], pad_count, axis=0)
                    worker.push_frames(padding)
                remaining = worker.flush()
                self._worker_output_buf[idx].extend(remaining)
                if pad_count > 0:
                    self._worker_output_buf[idx] = self._worker_output_buf[idx][:pending]
                worker.restart()

    def close(self) -> None:
        logger.info("TvaiSecondaryRestorer: closing %d workers", len(self._workers))
        for worker in self._workers:
            worker.kill()
        self._pending_clips.clear()
        self._worker_output_buf = [[] for _ in range(self.num_workers)]
        self._worker_pending_frames = [0] * self.num_workers
        self._worker_last_frame = [None] * self.num_workers
