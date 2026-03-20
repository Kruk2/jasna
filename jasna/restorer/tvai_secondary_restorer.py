from __future__ import annotations

import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

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


class TvaiSecondaryRestorer:
    name = "tvai"
    num_workers = 1
    _INPUT_SIZE = 256

    def __init__(self, *, ffmpeg_path: str, tvai_args: str, scale: int, num_workers: int) -> None:
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
        self._pool: ThreadPoolExecutor | None = None
        self._validated = False

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

    def _ensure_pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            if not self._validated:
                self._validate_environment()
                self._validated = True
            self._pool = ThreadPoolExecutor(max_workers=self.num_workers)
        return self._pool

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
    def _to_numpy_hwc(frames_256: np.ndarray) -> np.ndarray:
        x = np.clip(frames_256, 0.0, 1.0)
        x = np.round(x * 255.0).clip(0, 255).astype(np.uint8)
        return np.ascontiguousarray(x.transpose(0, 2, 3, 1))

    @staticmethod
    def _to_tensors(frames_np: list[np.ndarray]) -> list[torch.Tensor]:
        return [torch.from_numpy(np.ascontiguousarray(f.transpose(2, 0, 1))) for f in frames_np]

    def _communicate(self, frames_np: np.ndarray) -> list[np.ndarray]:
        n = len(frames_np)
        if n == 0:
            return []

        padded = n < TVAI_MIN_FRAMES
        if padded:
            pad_count = TVAI_MIN_FRAMES - n
            padding = np.repeat(frames_np[-1:], pad_count, axis=0)
            frames_np = np.concatenate([frames_np, padding], axis=0)

        stdin_bytes = frames_np.tobytes()
        cmd = self.build_ffmpeg_cmd()
        logger.debug("TVAI communicate: %d frames (%d padded) cmd=%s", n, len(frames_np), cmd[0])

        proc = subprocess.run(
            cmd,
            input=stdin_bytes,
            capture_output=True,
        )

        if proc.returncode != 0:
            stderr_text = proc.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"TVAI ffmpeg failed (exit {proc.returncode}): {stderr_text}")

        stdout_bytes = proc.stdout
        expected_total = len(frames_np) * self._out_frame_bytes
        if len(stdout_bytes) < expected_total:
            raise RuntimeError(
                f"TVAI ffmpeg output too short: got {len(stdout_bytes)} bytes, "
                f"expected {expected_total} ({len(frames_np)} frames x {self._out_frame_bytes})"
            )

        out_frames = []
        for i in range(len(frames_np)):
            start = i * self._out_frame_bytes
            end = start + self._out_frame_bytes
            frame = np.frombuffer(stdout_bytes[start:end], dtype=np.uint8).reshape(
                self._out_size, self._out_size, 3
            ).copy()
            out_frames.append(frame)

        if padded:
            out_frames = out_frames[:n]

        return out_frames

    def restore(self, frames_256: torch.Tensor, *, keep_start: int, keep_end: int) -> list[torch.Tensor]:
        t = int(frames_256.shape[0])
        ks = max(0, int(keep_start))
        ke = min(t, int(keep_end))
        if ks >= ke:
            return []

        device = frames_256.device
        kept_np = frames_256[ks:ke].cpu().numpy()
        frames_np = self._to_numpy_hwc(kept_np)

        pool = self._ensure_pool()
        future = pool.submit(self._communicate, frames_np)
        result_np = future.result()

        tensors = self._to_tensors(result_np)
        if device.type != "cpu" and tensors:
            return torch.stack(tensors).to(device, non_blocking=True).unbind(0)
        return tensors

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
