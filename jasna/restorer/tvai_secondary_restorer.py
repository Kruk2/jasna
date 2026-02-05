from __future__ import annotations

import atexit
import collections
import logging
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

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


class TvaiSecondaryRestorer:
    name = "tvai"
    DEFAULT_OUT_SIZE = 1024
    OUT_BUFFER_POOL_SIZE = 32
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

        model_name = str(self._tvai_kv.get("model") or "").strip().lower()

        if ("w" in self._tvai_kv) or ("h" in self._tvai_kv):
            raise ValueError('Do not pass "w" or "h" in --tvai-args; use --tvai-scale instead')

        scale_raw = self._tvai_kv.get("scale")
        if scale_raw is None:
            raise ValueError('Missing "scale" in tvai args. Pass it via --tvai-scale (valid: 1, 2, 4)')
        scale = int(scale_raw)
        if scale not in (1, 2, 4):
            raise ValueError(f'Invalid tvai "scale": {scale} (valid: 1, 2, 4)')
        self.scale = scale

        self.out_w = int(256 * scale)
        self.out_h = int(256 * scale)

        parts: list[tuple[str, str]] = []
        if "model" in self._tvai_kv:
            parts.append(("model", str(self._tvai_kv["model"])))
        parts.append(("scale", str(int(self.scale))))
        for k, v in self._tvai_kv.items():
            if k in {"model", "scale", "w", "h"}:
                continue
            parts.append((str(k), str(v)))
        self._tvai_args_effective = ":".join(f"{k}={v}" for k, v in parts)

        logger.debug(
            "TVAI init: ffmpeg_path=%r tvai_args=%r tvai_args_effective=%r parsed_args=%r model=%r scale=%d out=%dx%d",
            self.ffmpeg_path,
            self.tvai_args,
            self._tvai_args_effective,
            self._tvai_kv,
            model_name,
            int(self.scale),
            int(self.out_w),
            int(self.out_h),
        )

        max_clip_size = int(max_clip_size)
        if max_clip_size <= 0:
            raise ValueError("--max-clip-size must be > 0")
        self.max_clip_size = max_clip_size
        logger.debug(
            "TVAI init: device=%s out_w=%d out_h=%d max_clip_size=%d",
            self.device,
            int(self.out_w),
            int(self.out_h),
            int(self.max_clip_size),
        )

        self._in_buf = torch.empty(
            (self.max_clip_size, 256, 256, 3),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        self._in_np = self._in_buf.numpy()
        self._in_frame_bytes = int(256 * 256 * 3)
        self._in_all_mv = memoryview(self._in_np).cast("B")
        self._fatal: BaseException | None = None
        self._stderr_buf: list[bytes] = []

        self._validate_tvai_environment()
        self._start_ffmpeg()
        self._init_output_buffers()
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
        out_w = int(self.out_w)
        out_h = int(self.out_h)

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
        self._out_frames: collections.deque[tuple[torch.Tensor, memoryview, np.ndarray]] = collections.deque()
        self._out_pool: queue.SimpleQueue[tuple[torch.Tensor, memoryview, np.ndarray]] = queue.SimpleQueue()

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
                while True:
                    buf, mv, buf_np = self._out_pool.get()
                    view = mv
                    offset = 0
                    while offset < len(view):
                        n = self._proc.stdout.readinto(view[offset:])
                        if not n:
                            if offset == 0:
                                self._out_pool.put((buf, mv, buf_np))
                                return
                            raise RuntimeError(
                                f"Unexpected EOF while reading ffmpeg stdout (got {offset} / {len(view)} bytes)"
                            )
                        offset += n
                    frames_read += 1
                    if (frames_read <= 3) or (frames_read % 25 == 0):
                        logger.debug("TVAI stdout: received frame=%d bytes=%d", frames_read, int(len(view)))
                    with self._out_cond:
                        self._out_frames.append((buf, mv, buf_np))
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

    def _init_output_buffers(self) -> None:
        out_w = int(self.out_w)
        out_h = int(self.out_h)
        self._out_batch_hwc = torch.empty(
            (self.max_clip_size, out_h, out_w, 3),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        self._out_batch_np = self._out_batch_hwc.numpy()
        for _ in range(int(self.OUT_BUFFER_POOL_SIZE)):
            buf = torch.empty((out_h, out_w, 3), dtype=torch.uint8, device="cpu", pin_memory=True)
            buf_np = buf.numpy()
            mv = memoryview(buf_np).cast("B")
            self._out_pool.put((buf, mv, buf_np))
        logger.debug("TVAI output buffers: out_w=%d out_h=%d pool_size=%d", out_w, out_h, int(self.OUT_BUFFER_POOL_SIZE))

    def close(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is None:
            return
        if proc.poll() is not None:
            return
        logger.debug("TVAI closing ffmpeg: pid=%s", getattr(proc, "pid", None))
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        finally:
            proc.terminate()

    def restore(self, frames_256: torch.Tensor, *, keep_start: int, keep_end: int) -> torch.Tensor:
        t0 = time.perf_counter()
        t = int(frames_256.shape[0])
        if t == 0:
            return torch.empty((0, 3, int(self.out_h), int(self.out_w)), dtype=torch.uint8, device=self.device)

        if frames_256.dtype == torch.uint8:
            raise RuntimeError("TVAI secondary expects float frames in [0, 1], got uint8")

        out_w = int(self.out_w)
        out_h = int(self.out_h)

        restore_start = int(keep_start)
        restore_end = int(keep_end)

        if restore_start < 0:
            restore_start = 0
        if restore_end > t:
            restore_end = t

        n = int(restore_end - restore_start)
        if n <= 0:
            return torch.empty((0, 3, int(self.out_h), int(self.out_w)), dtype=torch.uint8, device=self.device)

        slice_end = int(restore_end)
        logger.debug(
            "TVAI restore(stream): t=%d keep_start=%d keep_end=%d restore_start=%d restore_end=%d n=%d",
            t,
            int(keep_start),
            int(keep_end),
            restore_start,
            restore_end,
            n,
        )
        frames_u8 = frames_256[restore_start:slice_end].mul(255.0).round().clamp(0, 255).to(dtype=torch.uint8)

        def _ensure_alive() -> None:
            if self._fatal is not None:
                raise RuntimeError(f"TVAI ffmpeg is not healthy: {self._fatal}") from self._fatal
            rc = self._proc.poll()
            if rc is not None:
                stderr_text = b"".join(self._stderr_buf).decode(errors="replace").strip()
                raise RuntimeError(f"TVAI ffmpeg crashed (exit_code={rc}). stderr:\n{stderr_text}")

        if n > int(self.max_clip_size):
            raise RuntimeError(f"TVAI restore got n={n} which exceeds max_clip_size={self.max_clip_size}")

        frames_cpu_hwc = self._in_buf[:n, :, :, :]
        frames_cpu_hwc[:n].copy_(
            frames_u8[:n].permute(0, 2, 3, 1),
            non_blocking=True,
        )

        torch.cuda.synchronize(frames_256.device)

        def _ensure_alive_locked() -> None:
            _ensure_alive()

        def _drain_one_batch_locked() -> int:
            k = min(int(len(self._out_frames)), int(self.max_clip_size))
            for i in range(k):
                buf, mv, buf_np = self._out_frames.popleft()
                np.copyto(self._out_batch_np[i], buf_np)
                self._out_pool.put((buf, mv, buf_np))
            return k

        def _copy_batch_to_gpu(k: int) -> torch.Tensor:
            if k <= 0:
                return torch.empty((0, 3, out_h, out_w), dtype=torch.uint8, device=self.device)
            chunk_gpu_hwc = torch.empty((k, out_h, out_w, 3), dtype=torch.uint8, device=self.device)
            chunk_gpu_hwc.copy_(self._out_batch_hwc[:k], non_blocking=False)
            return chunk_gpu_hwc.permute(0, 3, 1, 2)

        _ensure_alive()
        out_chunks: list[torch.Tensor] = []
        chunk_frames = int(self.IN_WRITE_CHUNK_FRAMES)
        for start in range(0, int(n), chunk_frames):
            end = min(int(n), start + chunk_frames)
            logger.debug(
                "TVAI stdin: writing frames [%d, %d) bytes [%d, %d)",
                int(start),
                int(end),
                int(start) * int(self._in_frame_bytes),
                int(end) * int(self._in_frame_bytes),
            )
            start_b = int(start) * int(self._in_frame_bytes)
            end_b = int(end) * int(self._in_frame_bytes)
            self._proc.stdin.write(self._in_all_mv[start_b:end_b])

            while True:
                with self._out_cond:
                    _ensure_alive_locked()
                    if not self._out_frames:
                        break
                    k = _drain_one_batch_locked()
                if k > 0:
                    out_chunks.append(_copy_batch_to_gpu(k))

        while True:
            with self._out_cond:
                _ensure_alive_locked()
                if not self._out_frames:
                    break
                k = _drain_one_batch_locked()
            if k > 0:
                out_chunks.append(_copy_batch_to_gpu(k))

        out = out_chunks[0] if len(out_chunks) == 1 else torch.cat(out_chunks, dim=0) if out_chunks else torch.empty((0, 3, out_h, out_w), dtype=torch.uint8, device=self.device)
        logger.debug(
            "TVAI restore(stream) done: wrote=%d drained=%d elapsed_ms=%.1f",
            int(n),
            int(out.shape[0]),
            (time.perf_counter() - t0) * 1000.0,
        )
        return out

    def flush(self, *, timeout_s: float = 30.0) -> torch.Tensor:
        t0 = time.perf_counter()
        proc = getattr(self, "_proc", None)
        if proc is None:
            return torch.empty((0, 3, int(self.out_h), int(self.out_w)), dtype=torch.uint8, device=self.device)

        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except BaseException:
                pass

        out_w = int(self.out_w)
        out_h = int(self.out_h)

        def _drain_one_batch_locked() -> int:
            k = min(int(len(self._out_frames)), int(self.max_clip_size))
            for i in range(k):
                buf, mv, buf_np = self._out_frames.popleft()
                np.copyto(self._out_batch_np[i], buf_np)
                self._out_pool.put((buf, mv, buf_np))
            return k

        def _copy_batch_to_gpu(k: int) -> torch.Tensor:
            if k <= 0:
                return torch.empty((0, 3, out_h, out_w), dtype=torch.uint8, device=self.device)
            chunk_gpu_hwc = torch.empty((k, out_h, out_w, 3), dtype=torch.uint8, device=self.device)
            chunk_gpu_hwc.copy_(self._out_batch_hwc[:k], non_blocking=False)
            return chunk_gpu_hwc.permute(0, 3, 1, 2)

        deadline = time.perf_counter() + float(timeout_s)
        out_chunks: list[torch.Tensor] = []

        while True:
            while True:
                with self._out_cond:
                    if self._fatal is not None:
                        raise RuntimeError(f"TVAI ffmpeg is not healthy: {self._fatal}") from self._fatal
                    if not self._out_frames:
                        break
                    k = _drain_one_batch_locked()
                if k > 0:
                    out_chunks.append(_copy_batch_to_gpu(k))

            rc = proc.poll()
            if rc is not None:
                break

            if time.perf_counter() >= deadline:
                proc.terminate()
                break

            with self._out_cond:
                self._out_cond.wait(timeout=0.05)

        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=5.0)

        if hasattr(self, "_stdout_thread"):
            self._stdout_thread.join(timeout=1.0)

        while True:
            with self._out_cond:
                if self._fatal is not None:
                    raise RuntimeError(f"TVAI ffmpeg is not healthy: {self._fatal}") from self._fatal
                if not self._out_frames:
                    break
                k = _drain_one_batch_locked()
            if k > 0:
                out_chunks.append(_copy_batch_to_gpu(k))

        out = (
            out_chunks[0]
            if len(out_chunks) == 1
            else torch.cat(out_chunks, dim=0)
            if out_chunks
            else torch.empty((0, 3, out_h, out_w), dtype=torch.uint8, device=self.device)
        )
        logger.debug(
            "TVAI flush done: drained=%d elapsed_ms=%.1f",
            int(out.shape[0]),
            (time.perf_counter() - t0) * 1000.0,
        )
        return out

