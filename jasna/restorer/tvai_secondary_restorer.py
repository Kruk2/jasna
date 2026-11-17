from __future__ import annotations

import logging
import subprocess
import threading
from collections import deque
from dataclasses import dataclass, field
from queue import Queue, Empty

import numpy as np
import torch

from jasna.frame_queue import FrameQueue
from jasna.os_utils import subprocess_no_window_kwargs, tvai_setup_problems

logger = logging.getLogger(__name__)

TVAI_PIPELINE_DELAY = 20
TVAI_MIN_STREAM_FRAMES_TO_EMIT = 8
# Frames queued for one worker's ffmpeg stdin before push_clip waits: two full-size clips.
_MAX_QUEUED_WRITE_FRAMES = 360
_TVAI_DENOISE_ARGS = (
    "model=nyx-3:preblur=-1:noise=-0.10:details=-1:halo=-1:blur=-1:"
    "compression=-1:estimate=8:device=-2:vram=1:instances=0"
)


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


def _build_tvai_filter_args(args: str, *, scale: int) -> str:
    kv = _parse_tvai_args_kv(args)
    parts: list[tuple[str, str]] = []
    if "model" in kv:
        parts.append(("model", kv["model"]))
    parts.append(("scale", str(scale)))
    for key, value in kv.items():
        if key in {"model", "scale", "w", "h"}:
            continue
        parts.append((key, value))
    return ":".join(f"{key}={value}" for key, value in parts)


def frames_to_tensor(frames_hwc: list[np.ndarray]) -> torch.Tensor:
    """HWC uint8 frames as one ``(T, C, H, W)`` uint8 CPU tensor (empty for no frames)."""
    if not frames_hwc:
        return torch.empty(0)
    return torch.from_numpy(np.ascontiguousarray(np.stack(frames_hwc).transpose(0, 3, 1, 2)))


@dataclass
class _ClipSegment:
    seq: int
    expected: int
    collected: list[np.ndarray] = field(default_factory=list)


@dataclass
class _FillerSegment:
    remaining: int


class WorkerLedger:
    """What one worker's ffmpeg was sent, in order (clip frames and filler frames), so its output
    frames, which come back in the same order, can be matched to their clips."""

    def __init__(self) -> None:
        self._segments: deque[_ClipSegment | _FillerSegment] = deque()

    @property
    def pending_frames(self) -> int:
        return sum(
            segment.expected - len(segment.collected) if isinstance(segment, _ClipSegment) else segment.remaining
            for segment in list(self._segments)
        )

    def holds_clip(self, seqs: set[int] | None = None) -> bool:
        """Whether a clip (one of ``seqs``, when given) still waits for output frames."""
        return any(
            isinstance(segment, _ClipSegment) and (seqs is None or segment.seq in seqs)
            for segment in list(self._segments)
        )

    def add_clip(self, seq: int, frame_count: int) -> None:
        self._segments.append(_ClipSegment(seq=seq, expected=frame_count))

    def add_filler(self, frame_count: int) -> None:
        if self._segments and isinstance(self._segments[-1], _FillerSegment):
            self._segments[-1].remaining += frame_count
        else:
            self._segments.append(_FillerSegment(remaining=frame_count))

    def consume(self, frames: list[np.ndarray]) -> list[tuple[int, list[np.ndarray]]]:
        """Match output ``frames`` to the segments; returns the clips they complete, in order."""
        completed: list[tuple[int, list[np.ndarray]]] = []
        for frame in frames:
            if not self._segments:
                logger.warning("TVAI worker returned a frame nothing was waiting for")
                continue
            segment = self._segments[0]
            if isinstance(segment, _FillerSegment):
                segment.remaining -= 1
                if segment.remaining <= 0:
                    self._segments.popleft()
                continue
            segment.collected.append(frame)
            if len(segment.collected) >= segment.expected:
                completed.append((segment.seq, segment.collected))
                self._segments.popleft()
        return completed

    def unfinished_clips(self) -> list[int]:
        return [segment.seq for segment in self._segments if isinstance(segment, _ClipSegment)]

    def clear(self) -> None:
        self._segments.clear()


def least_busy_worker(ledgers: list[WorkerLedger]) -> int:
    return min(range(len(ledgers)), key=lambda index: ledgers[index].pending_frames)


class _TvaiWorker:
    def __init__(self, cmd: list[str], out_frame_bytes: int, out_size: int) -> None:
        self._cmd = cmd
        self._out_frame_bytes = out_frame_bytes
        self._out_size = out_size
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._writer: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._frame_queue: Queue[np.ndarray | None] = Queue()
        self._write_queue = FrameQueue(_MAX_QUEUED_WRITE_FRAMES)
        self._error: BaseException | None = None
        self.frames_pushed = 0

    def start(self) -> None:
        self._error = None
        self._frame_queue = Queue()
        self._write_queue = FrameQueue(_MAX_QUEUED_WRITE_FRAMES)
        self.frames_pushed = 0
        self._proc = subprocess.Popen(
            self._cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **subprocess_no_window_kwargs(),
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
            logger.debug("TVAI stderr reader loop stopped", exc_info=True)

    def _writer_loop(self) -> None:
        # After a failed write (ffmpeg gone) the loop keeps taking and dropping queued frames, so a
        # push waiting for room or a drain waiting for the queue returns and sees the error.
        assert self._proc is not None and self._proc.stdin is not None
        stdin = self._proc.stdin
        while True:
            data = self._write_queue.get()
            try:
                if data is None:
                    return
                if self._error is None:
                    stdin.write(memoryview(data))
                    stdin.flush()
            except Exception as e:
                self._error = e
            finally:
                self._write_queue.task_done()

    def check_error(self) -> None:
        if self._error is not None:
            e = self._error
            self._error = None
            raise RuntimeError("TVAI worker thread crashed") from e

    def push_frames(self, frames_hwc: np.ndarray) -> None:
        self.check_error()
        data = np.ascontiguousarray(frames_hwc)
        self.frames_pushed += data.shape[0]
        self._write_queue.put(data, frame_count=data.shape[0])
        self.check_error()

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
        self._write_queue.join()
        self._write_queue.put(None)
        if self._writer is not None:
            self._writer.join(timeout=timeout)
            self._writer = None
        if self._proc is not None and self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except OSError:
                logger.debug("TVAI ffmpeg stdin already closed", exc_info=True)
        if self._reader is not None:
            self._reader.join(timeout=timeout)
        if self._stderr_reader is not None:
            self._stderr_reader.join(timeout=timeout)
            self._stderr_reader = None
        return self.drain_available()

    def kill(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
            except OSError:
                logger.debug("TVAI ffmpeg already exited", exc_info=True)
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

    def __init__(self, *, ffmpeg_path: str, tvai_args: str, scale: int, num_workers: int, tvai_denoise: bool) -> None:
        self.ffmpeg_path = str(ffmpeg_path)
        self.tvai_args = str(tvai_args)
        self.scale = int(scale)
        self.num_workers = int(num_workers)
        if self.scale not in (1, 2, 4):
            raise ValueError(f"Invalid tvai scale: {self.scale} (valid: 1, 2, 4)")
        self.tvai_filter_args = _build_tvai_filter_args(self.tvai_args, scale=self.scale)
        self.tvai_denoise_filter_args = (
            _build_tvai_filter_args(_TVAI_DENOISE_ARGS, scale=1)
            if tvai_denoise
            else None
        )
        filter_passes = 3 if self.tvai_denoise_filter_args else 1
        self._pipeline_delay = TVAI_PIPELINE_DELAY * filter_passes
        self._minimum_stream_frames = TVAI_MIN_STREAM_FRAMES_TO_EMIT * filter_passes
        self._out_size = self._INPUT_SIZE * self.scale
        self._out_frame_bytes = self._out_size * self._out_size * 3
        self._validated = False
        self._started = False
        self._next_seq = 0
        self._workers: list[_TvaiWorker] = []
        self._ledgers: list[WorkerLedger] = []
        self._completed: dict[int, list[np.ndarray]] = {}
        self._seq_lock = threading.Lock()
        self._worker_locks: list[threading.Lock] = []

    def _validate_environment(self) -> None:
        problems = tvai_setup_problems(self.ffmpeg_path)
        if problems:
            error = FileNotFoundError if problems[0].key == "ffmpeg_not_found" else RuntimeError
            raise error(problems[0].message)

    def _ensure_started(self) -> None:
        if self._started:
            return
        if not self._validated:
            self._validate_environment()
            self._validated = True
        cmd = self.build_ffmpeg_cmd()
        for _ in range(self.num_workers):
            w = _TvaiWorker(cmd, self._out_frame_bytes, self._out_size)
            w.start()
            self._workers.append(w)
            self._ledgers.append(WorkerLedger())
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
            self._build_filter_complex(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]

    def _build_filter_complex(self) -> str:
        enhance = f"tvai_up={self.tvai_filter_args}"
        if self.tvai_denoise_filter_args is None:
            return enhance
        denoise = f"tvai_up={self.tvai_denoise_filter_args}"
        return (
            "split=3[i0][i1][i2];"
            f"[i0]{denoise}[d1p];"
            "[i1][d1p]blend=all_mode=grainextract,split[nm1][nm1_copy];"
            f"[nm1]{denoise}[dnm];"
            "[dnm][nm1_copy]blend=all_mode=grainextract[temp];"
            f"[i2][temp]blend=all_mode=grainmerge,{enhance}"
        )

    @staticmethod
    def _to_numpy_hwc(frames_nchw: np.ndarray) -> np.ndarray:
        x = frames_nchw * np.float32(255.0)
        np.nan_to_num(x, nan=0.0, copy=False)
        np.clip(x, 0, 255, out=x)
        return np.ascontiguousarray(x.transpose(0, 2, 3, 1), dtype=np.uint8)

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
            wi = least_busy_worker(self._ledgers)

        with self._worker_locks[wi]:
            self._ledgers[wi].add_clip(seq, n)
            self._workers[wi].push_frames(frames_hwc)
        logger.debug("TVAI push seq=%d frames=%d -> worker %d", seq, n, wi)
        return seq

    def pop_completed(self) -> list[tuple[int, torch.Tensor]]:
        """Finished clips as ``(seq, (T, C, H, W) uint8 CPU tensor)``, oldest first."""
        for wi, worker in enumerate(self._workers):
            frames = worker.drain_available()
            if frames:
                with self._worker_locks[wi]:
                    self._completed.update(self._ledgers[wi].consume(frames))
        return [(seq, frames_to_tensor(self._completed.pop(seq))) for seq in sorted(self._completed)]

    @property
    def has_pending(self) -> bool:
        return any(ledger.holds_clip() for ledger in self._ledgers)

    def flush_pending(self, target_seqs: set[int] | None = None) -> bool:
        if not self._started:
            return False
        filler = np.zeros(
            (self._pipeline_delay, self._INPUT_SIZE, self._INPUT_SIZE, 3),
            dtype=np.uint8,
        )
        flushed = False
        for wi in range(len(self._workers)):
            if not self._worker_locks[wi].acquire(blocking=False):
                continue
            try:
                if not self._ledgers[wi].holds_clip(target_seqs):
                    continue
                self._ledgers[wi].add_filler(self._pipeline_delay)
                self._workers[wi].push_frames(filler)
                flushed = True
                logger.debug("TVAI flush_pending: pushed %d filler frames to worker %d (target_seqs=%s)", self._pipeline_delay, wi, target_seqs)
            finally:
                self._worker_locks[wi].release()
        return flushed

    def _pad_stream_too_short_for_tvai_to_emit(self, wi: int) -> None:
        worker = self._workers[wi]
        deficit = self._minimum_stream_frames - worker.frames_pushed
        if deficit <= 0 or not self._ledgers[wi].holds_clip():
            return
        self._ledgers[wi].add_filler(deficit)
        worker.push_frames(np.zeros((deficit, self._INPUT_SIZE, self._INPUT_SIZE, 3), dtype=np.uint8))
        logger.debug("TVAI flush_all: padded worker %d with %d filler frames (stream too short)", wi, deficit)

    def flush_all(self) -> None:
        """Close every worker's stdin, collect all of its output and restart it. Raises when a
        worker's output ends before every clip sent to it is complete."""
        if not self._started:
            return
        for wi, (worker, ledger) in enumerate(zip(self._workers, self._ledgers)):
            with self._worker_locks[wi]:
                self._pad_stream_too_short_for_tvai_to_emit(wi)
                remaining = worker.close_stdin_and_drain()
            self._completed.update(ledger.consume(remaining))
            unfinished = ledger.unfinished_clips()
            ledger.clear()
            worker.restart()
            if unfinished:
                raise RuntimeError(f"TVAI ffmpeg stopped before returning every frame of clips {unfinished}")
        logger.debug("TVAI flush_all: all workers restarted")

    def restore(self, frames_256: torch.Tensor, *, keep_start: int, keep_end: int) -> list[torch.Tensor]:
        seq = self.push_clip(frames_256, keep_start=keep_start, keep_end=keep_end)
        self.flush_all()
        batch = dict(self.pop_completed()).get(seq, torch.empty(0))
        if batch.numel() == 0:
            return []
        if frames_256.device.type != "cpu":
            batch = batch.to(frames_256.device, non_blocking=True)
        return list(batch.unbind(0))

    def close(self) -> None:
        for w in self._workers:
            w.kill()
        self._workers.clear()
        self._ledgers.clear()
        self._worker_locks.clear()
        self._completed.clear()
        self._started = False
