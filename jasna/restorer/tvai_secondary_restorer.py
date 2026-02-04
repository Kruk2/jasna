from __future__ import annotations

import atexit
import collections
import os
import queue
import subprocess
import threading
from pathlib import Path

import numpy as np
import torch

from jasna.os_utils import get_subprocess_startup_info


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
    MIN_TVAI_FRAMES = 4
    TAIL_PAD_FRAMES = 8
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
        if ("w" in self._tvai_kv) or ("h" in self._tvai_kv):
            if ("w" not in self._tvai_kv) or ("h" not in self._tvai_kv):
                raise ValueError('If you pass "w" or "h" in --tvai-args, you must pass both')
            self.out_w = int(self._tvai_kv["w"])
            self.out_h = int(self._tvai_kv["h"])
            if self.out_w <= 0 or self.out_h <= 0:
                raise ValueError('--tvai-args "w" and "h" must be > 0')
        else:
            self.out_w = int(self.DEFAULT_OUT_SIZE)
            self.out_h = int(self.DEFAULT_OUT_SIZE)

        max_clip_size = int(max_clip_size)
        if max_clip_size <= 0:
            raise ValueError("--max-clip-size must be > 0")
        self.max_clip_size = max_clip_size

        self._in_buf = torch.empty(
            (self.max_clip_size + int(self.TAIL_PAD_FRAMES), 256, 256, 3),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        self._in_np = self._in_buf.numpy()
        self._in_frame_bytes = int(256 * 256 * 3)
        self._in_all_mv = memoryview(self._in_np).cast("B")
        self._fatal: BaseException | None = None
        self._stderr_buf: list[bytes] = []
        self._pending_discard_outputs = 0

        self._validate_tvai_environment()
        self._start_ffmpeg()
        self._init_output_buffers()
        atexit.register(self.close)

    def _validate_tvai_environment(self) -> None:
        data_dir = os.environ.get("TVAI_MODEL_DATA_DIR")
        model_dir = os.environ.get("TVAI_MODEL_DIR")
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
            f"tvai_up={self.tvai_args},scale={out_w}:{out_h}:flags=spline",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]

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
                while True:
                    buf, mv, buf_np = self._out_pool.get()
                    view = mv
                    offset = 0
                    while offset < len(view):
                        n = self._proc.stdout.readinto(view[offset:])
                        if not n:
                            raise RuntimeError(f"Unexpected EOF while reading ffmpeg stdout (got {offset} / {len(view)} bytes)")
                        offset += n
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

    def close(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is None:
            return
        if proc.poll() is not None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        finally:
            proc.terminate()

    def restore(self, frames_256: torch.Tensor, *, keep_start: int, keep_end: int) -> torch.Tensor:
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
        padded_n = max(int(n + self.TAIL_PAD_FRAMES), int(self.MIN_TVAI_FRAMES))
        tail_ctx = min(int(self.TAIL_PAD_FRAMES), max(0, t - restore_end))

        slice_end = int(restore_end + tail_ctx)
        frames_u8 = frames_256[restore_start:slice_end].mul(255.0).round().clamp(0, 255).to(dtype=torch.uint8)

        def _ensure_alive() -> None:
            if self._fatal is not None:
                raise RuntimeError(f"TVAI ffmpeg is not healthy: {self._fatal}") from self._fatal
            rc = self._proc.poll()
            if rc is not None:
                stderr_text = b"".join(self._stderr_buf).decode(errors="replace").strip()
                raise RuntimeError(f"TVAI ffmpeg crashed (exit_code={rc}). stderr:\n{stderr_text}")

        frames_cpu_hwc = self._in_buf[:padded_n, :, :, :]
        frames_cpu_hwc[:n].copy_(
            frames_u8[:n].permute(0, 2, 3, 1),
            non_blocking=True,
        )
        if tail_ctx > 0:
            frames_cpu_hwc[n : n + tail_ctx].copy_(
                frames_u8[n : n + tail_ctx].permute(0, 2, 3, 1),
                non_blocking=True,
            )

        filled = n + tail_ctx
        torch.cuda.synchronize(frames_256.device)
        if padded_n > filled:
            self._in_np[filled:padded_n] = self._in_np[filled - 1]

        need_discard = int(self._pending_discard_outputs)
        need_collect = int(n)
        tail_outputs = int(padded_n - n)
        out_write_idx = 0

        def _ensure_alive_locked() -> None:
            _ensure_alive()

        def _drain_outputs_locked() -> None:
            nonlocal need_discard, need_collect, out_write_idx
            while self._out_frames and ((need_discard > 0) or (need_collect > 0)):
                buf, mv, buf_np = self._out_frames.popleft()
                if need_discard > 0:
                    need_discard -= 1
                    self._out_pool.put((buf, mv, buf_np))
                    continue
                np.copyto(self._out_batch_np[out_write_idx], buf_np)
                out_write_idx += 1
                need_collect -= 1
                self._out_pool.put((buf, mv, buf_np))

        _ensure_alive()
        chunk_frames = int(self.IN_WRITE_CHUNK_FRAMES)
        for start in range(0, int(padded_n), chunk_frames):
            end = min(int(padded_n), start + chunk_frames)
            start_b = int(start) * int(self._in_frame_bytes)
            end_b = int(end) * int(self._in_frame_bytes)
            self._proc.stdin.write(self._in_all_mv[start_b:end_b])
            with self._out_cond:
                _ensure_alive_locked()
                _drain_outputs_locked()

        with self._out_cond:
            while (need_discard > 0) or (need_collect > 0):
                _ensure_alive_locked()
                _drain_outputs_locked()

                if (need_discard > 0) or (need_collect > 0):
                    self._out_cond.wait(timeout=0.1)

        self._pending_discard_outputs = tail_outputs

        out_full_hwc = torch.empty((t, out_h, out_w, 3), dtype=torch.uint8, device=self.device)
        out_full_hwc[restore_start:restore_end].copy_(self._out_batch_hwc[:n], non_blocking=True)
        return out_full_hwc.permute(0, 3, 1, 2)

