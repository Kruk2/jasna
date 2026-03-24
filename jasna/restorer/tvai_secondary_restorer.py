from __future__ import annotations

import logging
import os
import subprocess
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue, Empty

import numpy as np

from jasna.frame_queue import FrameQueue
import torch

logger = logging.getLogger(__name__)

TVAI_PIPELINE_DELAY = 20


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
class _ClipSegment:
    seq: int
    expected: int
    collected: list[np.ndarray] = field(default_factory=list)


@dataclass
class _FillerSegment:
    remaining: int


_Segment = _ClipSegment | _FillerSegment


class _TvaiWorker:
    def __init__(self, cmd: list[str], out_frame_bytes: int, out_size: int, max_write_frames: int) -> None:
        self._cmd = cmd
        self._out_frame_bytes = out_frame_bytes
        self._out_size = out_size
        self._max_write_frames = max_write_frames
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._writer: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._frame_queue: Queue[np.ndarray | None] = Queue()
        self._write_queue = FrameQueue(max_write_frames)
        self._error: BaseException | None = None

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        self._error = None
        self._frame_queue = Queue()
        self._write_queue = FrameQueue(self._max_write_frames)
        self._proc = subprocess.Popen(
            self._cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()
        self._writer = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer.start()
        self._stderr_reader = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stderr_reader.start()

    def _reader_loop(self) -> None:
        assert self._proc is not None
        stdout = self._proc.stdout
        assert stdout is not None
        try:
            while True:
                data = stdout.read(self._out_frame_bytes)
                if len(data) < self._out_frame_bytes:
                    break
                frame = np.frombuffer(data, dtype=np.uint8).reshape(
                    self._out_size, self._out_size, 3
                ).copy()
                self._frame_queue.put(frame)
        except Exception as e:
            self._error = e
        finally:
            self._frame_queue.put(None)

    def _stderr_loop(self) -> None:
        assert self._proc is not None
        stderr = self._proc.stderr
        if stderr is None:
            return
        try:
            for line in stderr:
                msg = line.decode("utf-8", errors="replace").rstrip()
                if msg:
                    logger.debug("TVAI ffmpeg stderr: %s", msg)
        except Exception:
            pass

    def _writer_loop(self) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        stdin = self._proc.stdin
        try:
            while True:
                data = self._write_queue.get()
                if data is None:
                    self._write_queue.task_done()
                    break
                stdin.write(memoryview(data))
                stdin.flush()
                self._write_queue.task_done()
        except Exception as e:
            self._error = e

    def check_error(self) -> None:
        if self._error is not None:
            e = self._error
            self._error = None
            raise RuntimeError("TVAI worker thread crashed") from e

    def push_frames(self, frames_hwc: np.ndarray) -> None:
        self.check_error()
        data = np.ascontiguousarray(frames_hwc)
        self._write_queue.put(data, frame_count=data.shape[0])

    def drain_writes(self) -> None:
        self.check_error()
        self._write_queue.join()

    def drain_available(self) -> list[np.ndarray]:
        self.check_error()
        frames: list[np.ndarray] = []
        while True:
            try:
                f = self._frame_queue.get_nowait()
                if f is None:
                    break
                frames.append(f)
            except Empty:
                break
        return frames

    def close_stdin_and_drain(self, timeout: float = 30.0) -> list[np.ndarray]:
        self.check_error()
        self.drain_writes()
        self._write_queue.put(None)
        if self._writer is not None:
            self._writer.join(timeout=timeout)
            self._writer = None
        if self._proc is not None and self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
        if self._reader is not None:
            self._reader.join(timeout=timeout)
        if self._stderr_reader is not None:
            self._stderr_reader.join(timeout=timeout)
            self._stderr_reader = None

        frames: list[np.ndarray] = []
        while True:
            try:
                f = self._frame_queue.get_nowait()
                if f is None:
                    break
                frames.append(f)
            except Empty:
                break
        return frames

    def kill(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
            except OSError:
                pass
            self._proc.wait(timeout=5)
            self._proc = None
        if self._writer is not None:
            self._write_queue.put(None)
            self._writer.join(timeout=5)
            self._writer = None
        if self._reader is not None:
            self._reader.join(timeout=5)
            self._reader = None
        if self._stderr_reader is not None:
            self._stderr_reader.join(timeout=5)
            self._stderr_reader = None

    def restart(self) -> None:
        self.kill()
        self.start()


class TvaiSecondaryRestorer:
    name = "tvai"
    prefers_cpu_input = True
    _INPUT_SIZE = 256

    def __init__(self, *, ffmpeg_path: str, tvai_args: str, scale: int, num_workers: int, max_clip_size: int = 180) -> None:
        self.ffmpeg_path = str(ffmpeg_path)
        self.tvai_args = str(tvai_args)
        self.scale = int(scale)
        self.num_workers = int(num_workers)
        if self.scale not in (1, 2, 4):
            raise ValueError(f"Invalid tvai scale: {self.scale} (valid: 1, 2, 4)")
        kv = _parse_tvai_args_kv(self.tvai_args)
        parts: list[tuple[str, str]] = []
        if "model" in kv:
            parts.append(("model", kv["model"]))
        parts.append(("scale", str(self.scale)))
        for key, value in kv.items():
            if key in {"model", "scale", "w", "h"}:
                continue
            parts.append((key, value))
        self.tvai_filter_args = ":".join(f"{key}={value}" for key, value in parts)
        self._out_size = self._INPUT_SIZE * self.scale
        self._in_frame_bytes = self._INPUT_SIZE * self._INPUT_SIZE * 3
        self._out_frame_bytes = self._out_size * self._out_size * 3
        self._validated = False
        self._started = False
        self._max_clip_size = max_clip_size
        self._next_seq = 0
        self._workers: list[_TvaiWorker] = []
        self._worker_segments: list[deque[_Segment]] = []
        self._completed: dict[int, list[np.ndarray]] = {}
        self._seq_lock = threading.Lock()
        self._worker_locks: list[threading.Lock] = []

    @property
    def preferred_queue_size(self) -> int:
        return 2

    def _validate_environment(self) -> None:
        data_dir = os.environ.get("TVAI_MODEL_DATA_DIR")
        if not data_dir:
            raise RuntimeError("TVAI_MODEL_DATA_DIR environment variable is not set")
        if not Path(data_dir).is_dir():
            raise RuntimeError(f"TVAI_MODEL_DATA_DIR is not a directory: {data_dir}")

        model_dir = os.environ.get("TVAI_MODEL_DIR")
        if not model_dir:
            raise RuntimeError("TVAI_MODEL_DIR environment variable is not set")
        if not Path(model_dir).is_dir():
            raise RuntimeError(f"TVAI_MODEL_DIR is not a directory: {model_dir}")

        if not Path(self.ffmpeg_path).is_file():
            raise FileNotFoundError(f"TVAI ffmpeg not found: {self.ffmpeg_path}")

    def _ensure_started(self) -> None:
        if self._started:
            return
        if not self._validated:
            self._validate_environment()
            self._validated = True
        cmd = self.build_ffmpeg_cmd()
        max_write_frames = self._max_clip_size * 2
        for _ in range(self.num_workers):
            w = _TvaiWorker(cmd, self._out_frame_bytes, self._out_size, max_write_frames)
            w.start()
            self._workers.append(w)
            self._worker_segments.append(deque())
            self._worker_locks.append(threading.Lock())
        self._started = True

    def build_ffmpeg_cmd(self) -> list[str]:
        size = f"{self._INPUT_SIZE}x{self._INPUT_SIZE}"
        return [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            size,
            "-r",
            "25",
            "-i",
            "pipe:0",
            "-sws_flags",
            "spline+accurate_rnd+full_chroma_int",
            "-filter_complex",
            f"tvai_up={self.tvai_filter_args}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]

    @staticmethod
    def _to_numpy_hwc(frames_nchw: np.ndarray) -> np.ndarray:
        x = frames_nchw * np.float32(255.0)
        np.nan_to_num(x, nan=0.0, copy=False)
        np.clip(x, 0, 255, out=x)
        return np.ascontiguousarray(x.transpose(0, 2, 3, 1), dtype=np.uint8)

    @staticmethod
    def _to_tensors(frames_np: list[np.ndarray]) -> torch.Tensor:
        if not frames_np:
            return torch.empty(0)
        batch = np.stack(frames_np)
        batch = np.ascontiguousarray(batch.transpose(0, 3, 1, 2))
        return torch.from_numpy(batch)

    def _pending_frames(self, wi: int) -> int:
        total = 0
        for seg in list(self._worker_segments[wi]):
            if isinstance(seg, _ClipSegment):
                total += seg.expected - len(seg.collected)
            elif isinstance(seg, _FillerSegment):
                total += seg.remaining
        return total

    def _least_pending_worker(self) -> int:
        return min(range(self.num_workers), key=self._pending_frames)

    def push_clip(
        self,
        frames_256: torch.Tensor,
        *,
        keep_start: int,
        keep_end: int,
    ) -> int:
        t = int(frames_256.shape[0])
        ks = max(0, int(keep_start))
        ke = min(t, int(keep_end))
        if ks >= ke:
            seq = self._next_seq
            self._next_seq += 1
            self._completed[seq] = []
            return seq

        self._ensure_started()

        kept_np = frames_256[ks:ke].cpu().numpy()
        frames_hwc = self._to_numpy_hwc(kept_np)
        n = len(frames_hwc)

        with self._seq_lock:
            seq = self._next_seq
            self._next_seq += 1
            wi = self._least_pending_worker()

        with self._worker_locks[wi]:
            self._worker_segments[wi].append(_ClipSegment(seq=seq, expected=n))
            self._workers[wi].push_frames(frames_hwc)
        logger.debug("TVAI push seq=%d frames=%d -> worker %d", seq, n, wi)
        return seq

    def _process_drained_frames(self, wi: int, frames: list[np.ndarray]) -> None:
        segments = self._worker_segments[wi]
        for frame in frames:
            if not segments:
                logger.warning("TVAI worker %d: unexpected output frame (no pending segments)", wi)
                continue
            seg = segments[0]
            if isinstance(seg, _FillerSegment):
                seg.remaining -= 1
                if seg.remaining <= 0:
                    segments.popleft()
                continue
            seg.collected.append(frame)
            if len(seg.collected) >= seg.expected:
                self._completed[seg.seq] = seg.collected
                segments.popleft()

    def pop_completed(self) -> list[tuple[int, list[np.ndarray]]]:
        for wi in range(len(self._workers)):
            frames = self._workers[wi].drain_available()
            if frames:
                with self._worker_locks[wi]:
                    self._process_drained_frames(wi, frames)
        result: list[tuple[int, list[np.ndarray]]] = []
        for seq in sorted(self._completed.keys()):
            result.append((seq, self._completed.pop(seq)))
        return result

    @property
    def has_pending(self) -> bool:
        return any(
            isinstance(s, _ClipSegment)
            for segs in self._worker_segments
            for s in list(segs)
        )

    def flush_pending(self, target_seqs: set[int] | None = None) -> bool:
        if not self._started:
            return False
        filler = np.zeros(
            (TVAI_PIPELINE_DELAY, self._INPUT_SIZE, self._INPUT_SIZE, 3),
            dtype=np.uint8,
        )
        flushed = False
        for wi in range(len(self._workers)):
            if not self._worker_locks[wi].acquire(blocking=False):
                continue
            try:
                segs = self._worker_segments[wi]
                if target_seqs is None:
                    has_target = any(isinstance(s, _ClipSegment) for s in segs)
                else:
                    has_target = any(isinstance(s, _ClipSegment) and s.seq in target_seqs for s in segs)
                if not has_target:
                    continue
                if segs and isinstance(segs[-1], _FillerSegment):
                    continue
                self._workers[wi].push_frames(filler)
                segs.append(_FillerSegment(remaining=TVAI_PIPELINE_DELAY))
                flushed = True
                logger.debug("TVAI flush_pending: pushed %d filler frames to worker %d (target_seqs=%s)", TVAI_PIPELINE_DELAY, wi, target_seqs)
            finally:
                self._worker_locks[wi].release()
        return flushed

    def flush_all(self) -> None:
        if not self._started:
            return
        for wi in range(len(self._workers)):
            with self._worker_locks[wi]:
                remaining = self._workers[wi].close_stdin_and_drain()
            segments = self._worker_segments[wi]
            for frame in remaining:
                if not segments:
                    break
                seg = segments[0]
                if isinstance(seg, _FillerSegment):
                    seg.remaining -= 1
                    if seg.remaining <= 0:
                        segments.popleft()
                    continue
                seg.collected.append(frame)
                if len(seg.collected) >= seg.expected:
                    self._completed[seg.seq] = seg.collected
                    segments.popleft()
            for seg in list(segments):
                if isinstance(seg, _ClipSegment) and seg.collected:
                    logger.warning(
                        "TVAI flush_all: seq=%d incomplete (%d/%d frames)",
                        seg.seq, len(seg.collected), seg.expected,
                    )
                    self._completed[seg.seq] = seg.collected
            segments.clear()
            self._workers[wi].restart()
        logger.debug("TVAI flush_all: all workers restarted")

    def restore(self, frames_256: torch.Tensor, *, keep_start: int, keep_end: int) -> list[torch.Tensor]:
        device = frames_256.device
        seq = self.push_clip(frames_256, keep_start=keep_start, keep_end=keep_end)
        self.flush_all()
        completed = self.pop_completed()
        result_np: list[np.ndarray] = []
        for s, frames in completed:
            if s == seq:
                result_np = frames
                break
        batch = self._to_tensors(result_np)
        if batch.numel() == 0:
            return []
        if device.type != "cpu":
            batch = batch.to(device, non_blocking=True)
        return list(batch.unbind(0))

    def close(self) -> None:
        for w in self._workers:
            w.kill()
        self._workers.clear()
        self._worker_segments.clear()
        self._worker_locks.clear()
        self._completed.clear()
        self._started = False
