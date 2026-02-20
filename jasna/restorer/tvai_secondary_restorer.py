from __future__ import annotations

import atexit
import logging
import os
import subprocess
import threading
from pathlib import Path

import numpy as np
import torch

from jasna.os_utils import get_subprocess_startup_info

logger = logging.getLogger(__name__)

TVAI_MIN_FRAMES = 5


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


class _TvaiWorker:
    def __init__(self, ffmpeg_path: str, tvai_filter_args: str, scale: int) -> None:
        self.out_w = 256 * scale
        self.out_h = 256 * scale
        self._in_frame_bytes = 256 * 256 * 3
        self._out_frame_bytes = self.out_h * self.out_w * 3

        cmd = [
            ffmpeg_path,
            "-hide_banner", "-loglevel", "warning",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "256x256", "-r", "25",
            "-i", "pipe:0",
            "-sws_flags", "spline+accurate_rnd+full_chroma_int",
            "-filter_complex", f"tvai_up={tvai_filter_args}",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "pipe:1",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=get_subprocess_startup_info(),
            env=os.environ.copy(),
        )

    def warm_up(self) -> None:
        self._proc.stdin.write(bytes(self._in_frame_bytes))

    def restore_frames(self, frames_hwc_u8: np.ndarray) -> list[np.ndarray]:
        n = frames_hwc_u8.shape[0]
        pid = self._proc.pid
        logger.debug("TVAI worker pid=%s: sending %d frames (%d bytes input)", pid, n, n * self._in_frame_bytes)
        input_data = bytes(memoryview(frames_hwc_u8).cast("B"))

        stdout_data, stderr_data = self._proc.communicate(input=input_data)
        stderr_text = stderr_data.decode(errors="replace").strip() if stderr_data else ""
        if stderr_text:
            logger.debug("TVAI worker pid=%s stderr: %s", pid, stderr_text[-500:])

        if self._proc.returncode != 0:
            raise RuntimeError(f"TVAI ffmpeg pid={pid} crashed (code {self._proc.returncode}) after receiving {n} frames. stderr: {stderr_text}")

        actual_frames = len(stdout_data) // self._out_frame_bytes
        logger.debug("TVAI worker pid=%s: got %d output frames (%d bytes), expected %d (1 warmup + %d real)", pid, actual_frames, len(stdout_data), 1 + n, n)
        if actual_frames < 1 + n:
            raise RuntimeError(f"TVAI pid={pid}: expected {1 + n} output frames, got {actual_frames}. stderr: {stderr_text}")

        results: list[np.ndarray] = []
        for i in range(1, 1 + n):
            offset = i * self._out_frame_bytes
            frame = np.frombuffer(stdout_data, dtype=np.uint8, count=self._out_frame_bytes, offset=offset).reshape(self.out_h, self.out_w, 3).copy()
            results.append(frame)
        return results

    def kill(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()


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

        self._pool: list[_TvaiWorker] = []
        self._pool_lock = threading.Lock()
        self._closed = False

        logger.info(
            "TvaiSecondaryRestorer: ffmpeg=%r filter_args=%r scale=%d num_workers=%d",
            self.ffmpeg_path, self._tvai_filter_args, self.scale, self.num_workers,
        )

        for _ in range(self.num_workers):
            self._pool.append(self._spawn_warm_worker())

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

    def _spawn_warm_worker(self) -> _TvaiWorker:
        worker = _TvaiWorker(self.ffmpeg_path, self._tvai_filter_args, self.scale)
        worker.warm_up()
        return worker

    def _take_worker(self) -> _TvaiWorker:
        with self._pool_lock:
            if self._pool:
                return self._pool.pop()
        return self._spawn_warm_worker()

    def _replenish_async(self) -> None:
        def _do() -> None:
            if self._closed:
                return
            worker = self._spawn_warm_worker()
            with self._pool_lock:
                if self._closed:
                    worker.kill()
                else:
                    self._pool.append(worker)
        threading.Thread(target=_do, daemon=True).start()

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
        worker = self._take_worker()
        try:
            return worker.restore_frames(frames_hwc)
        finally:
            self._replenish_async()

    def _to_tensors(self, frames_np: list[np.ndarray], device: torch.device) -> list[torch.Tensor]:
        return [torch.from_numpy(f).to(device).permute(2, 0, 1) for f in frames_np]

    _restore_call_count = 0

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
        frames_hwc, pad_count = self._pad_to_minimum(frames_hwc)

        out_np = self._run_worker(frames_hwc)
        if pad_count > 0:
            out_np = out_np[:n]

        return self._to_tensors(out_np, frames_256.device)

    def close(self) -> None:
        self._closed = True
        with self._pool_lock:
            workers = list(self._pool)
            self._pool.clear()
        for w in workers:
            w.kill()
