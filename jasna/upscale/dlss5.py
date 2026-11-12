"""Whole-frame upscaling with NVIDIA DLSS 5 (draft, Windows only).

NVIDIA has no public DLSS 5 SDK, so Jasna drives the video worker of the community DLSS 5
Visual Enhancer: its ``nvngx.dll`` (an executable despite the name) hosts ReShade and the RenoDX
DLSS 5 add-on, runs DLSS Super Resolution and then DLSS 5 neural rendering on Direct3D 12. The
user copies that worker with its runtime files into ``model_weights/dlss5``. Frames travel over
the worker's stdin/stdout as RGBA8 with optical-flow motion vectors, since a video has none."""

from __future__ import annotations

import struct
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from jasna.engine_paths import model_weights_dir
from jasna.upscale import Size, finish_frame

DLSS5_RUNTIME_FILES = ("nvngx.dll", "dxgi.dll", "renodx-dlss5.addon64", "nvngx_dlss.dll", "nvngx_dlssnr.dll")
DLSS5_PRESETS = {"default": 0, "j": 10, "k": 11, "l": 12, "m": 13}
DLSS5_STYLES = {"default": 0, "natural": 1, "cinematic": 2}
# DLSS Super Resolution modes, smallest scale first: (scale, the worker's mode number).
_MODES = ((1.5, 2), (1.724, 1), (2.0, 0), (3.0, 3))

_VIDEO_HEADER = struct.Struct("<14I4f")
_SETUP = struct.Struct("<12I")
_FRAME_HEADER = struct.Struct("<4Iq")
_RESULT_HEADER = struct.Struct("<5Iq")
_VIDEO_MAGIC = 0x34563544
_SETUP_MAGIC = 0x34505553
_FRAME_MAGIC = 0x314D5246
_RESULT_MAGIC = 0x3154554F
_NGX_SUCCESS = 1
# The worker wants the frame count up front; Jasna sends the largest and closes the stream after
# the last frame, which the worker reports as a truncated stream.
_UNKNOWN_FRAME_COUNT = 2**31 - 1
_TRUNCATED_STREAM_EXIT = 7
_FLOW_WIDTH = 640
_SCENE_CUT_DIFFERENCE = 0.24
_LOG_LINES = 200


def dlss5_runtime_dir() -> Path:
    return model_weights_dir() / "dlss5"


def dlss5_unavailable_reason(device: torch.device, runtime_dir: Path) -> str | None:
    from jasna.accelerator import is_nvidia_device

    if sys.platform != "win32":
        return "DLSS 5 upscaling runs on Windows only"
    if not is_nvidia_device(device):
        return "DLSS 5 upscaling needs an NVIDIA RTX GPU"
    missing = [name for name in DLSS5_RUNTIME_FILES if not (runtime_dir / name).is_file()]
    if missing:
        return f"DLSS 5 upscaling needs these runtime files in {runtime_dir}: {', '.join(missing)}"
    return None


def _even(value: float) -> int:
    return max(2, round(value / 2) * 2)


def dlss5_mode(source_size: Size, output_size: Size) -> tuple[Size, int]:
    """The size DLSS renders at and its mode: the smallest scale that reaches the output height,
    else the largest. The final resize fits it to the exact output size."""
    width, height = source_size
    scale, mode = next(((s, m) for s, m in _MODES if height * s >= output_size[1]), _MODES[-1])
    return (_even(width * scale), _even(height * scale)), mode


def video_header(
    source_size: Size, dlss_size: Size, mode: int, *, preset: str, style: str, intensity: float
) -> bytes:
    return _VIDEO_HEADER.pack(
        _VIDEO_MAGIC, *source_size, *dlss_size, 0, _UNKNOWN_FRAME_COUNT, mode, DLSS5_PRESETS[preset],
        0, 0, DLSS5_STYLES[style], 0, 0, intensity, 1.0, 1.0, -1.0,
    )


def render_size_from_setup(data: bytes, dlss_size: Size, preset: str) -> Size:
    """The input size the worker accepted, after checking it set up DLSS as asked."""
    magic, ok, result, render_width, render_height, out_width, out_height, *_, applied_preset = _SETUP.unpack(data)
    if magic != _SETUP_MAGIC:
        raise RuntimeError("The DLSS 5 worker does not speak the version-4 video protocol")
    if not ok:
        raise RuntimeError(f"DLSS 5 could not start for {dlss_size[0]}x{dlss_size[1]} (NGX 0x{result:08X})")
    if (out_width, out_height) != dlss_size or applied_preset != DLSS5_PRESETS[preset]:
        raise RuntimeError("The DLSS 5 worker set up a different output size or model than asked")
    return render_width, render_height


class MotionGuide:
    """Backward optical flow between consecutive small grayscale frames, in small-frame pixels
    (where each pixel was in the previous frame), and a reset at the first frame and scene cuts."""

    def __init__(self) -> None:
        self._previous: np.ndarray | None = None
        self._flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self._flow.setUseSpatialPropagation(True)
        self._flow.setFinestScale(1)

    def next(self, gray: np.ndarray) -> np.ndarray | None:
        """The flow to ``gray`` from the previous frame, or ``None`` to reset the history."""
        previous, self._previous = self._previous, gray
        if previous is None:
            return None
        if np.mean(cv2.absdiff(gray, previous)) / 255.0 > _SCENE_CUT_DIFFERENCE:
            return None
        return self._flow.calc(gray, previous, None)


class Dlss5FrameUpscaler:
    def __init__(
        self,
        device: torch.device,
        *,
        source_size: Size,
        output_size: Size,
        preset: str,
        style: str,
        intensity: float,
        strength: float,
        runtime_dir: Path,
    ) -> None:
        self._device = device
        self._output_size = output_size
        self._strength = strength
        self._dlss_size, mode = dlss5_mode(source_size, output_size)
        self._log: deque[str] = deque(maxlen=_LOG_LINES)
        self._worker = subprocess.Popen(
            [str(runtime_dir / "nvngx.dll"), "--video"], cwd=runtime_dir,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self._log_thread = threading.Thread(target=self._drain_log, name="dlss5-log", daemon=True)
        self._log_thread.start()
        self._send(video_header(source_size, self._dlss_size, mode, preset=preset, style=style, intensity=intensity))
        self._render_size = render_size_from_setup(self._read(_SETUP.size), self._dlss_size, preset)
        render_width, render_height = self._render_size
        scale = min(1.0, _FLOW_WIDTH / render_width)
        self._flow_size = (_even(render_width * scale), _even(render_height * scale))
        self._flow_to_render = torch.tensor([render_width / self._flow_size[0], render_height / self._flow_size[1]])
        self._guide = MotionGuide()
        self._zero_motion = np.zeros((render_height, render_width, 2), np.float16)
        self._output = np.empty((self._dlss_size[1], self._dlss_size[0], 4), np.uint8)
        self._index = 0

    def _drain_log(self) -> None:
        for line in self._worker.stderr:
            self._log.append(line.decode(errors="replace").rstrip())

    def _failure(self, message: str) -> RuntimeError:
        return RuntimeError(f"{message}\n" + "\n".join(list(self._log)[-40:]))

    def _send(self, *chunks) -> None:
        for chunk in chunks:
            self._worker.stdin.write(chunk)
        self._worker.stdin.flush()

    def _read(self, size: int) -> bytes:
        data = self._worker.stdout.read(size)
        if len(data) != size:
            raise self._failure(f"The DLSS 5 worker stopped (exit {self._worker.wait()})")
        return data

    def _read_into(self, target: np.ndarray) -> None:
        view = memoryview(target.reshape(-1))
        while view:
            count = self._worker.stdout.readinto(view)
            if not count:
                raise self._failure(f"The DLSS 5 worker stopped (exit {self._worker.wait()})")
            view = view[count:]

    def _motion(self, image: torch.Tensor) -> tuple[np.ndarray, bool]:
        """The frame's motion vectors at the render size, and whether DLSS drops its history."""
        width, height = self._flow_size
        small = F.interpolate(image[None].float(), size=(height, width), mode="area")[0]
        gray = (small[0] * 0.299 + small[1] * 0.587 + small[2] * 0.114).round_().to(torch.uint8)
        flow = self._guide.next(gray.cpu().numpy())
        if flow is None:
            return self._zero_motion, True
        render_width, render_height = self._render_size
        flow = torch.from_numpy(flow).to(self._device).permute(2, 0, 1)[None]
        flow = F.interpolate(flow, size=(render_height, render_width), mode="bilinear")[0].permute(1, 2, 0)
        return flow.mul(self._flow_to_render.to(self._device)).half().contiguous().cpu().numpy(), False

    @torch.no_grad()
    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        render_width, render_height = self._render_size
        image = frame
        if tuple(frame.shape[-2:]) != (render_height, render_width):
            image = F.interpolate(frame[None].float(), size=(render_height, render_width), mode="bicubic", antialias=True)
            image = image[0].clamp_(0, 255).round_().to(torch.uint8)
        motion, reset = self._motion(image)
        rgba = torch.full((render_height, render_width, 4), 255, dtype=torch.uint8, device=image.device)
        rgba[..., :3] = image.permute(1, 2, 0)
        index, self._index = self._index, self._index + 1
        self._send(_FRAME_HEADER.pack(_FRAME_MAGIC, index, int(reset), 0, index), rgba.cpu().numpy().data, motion.data)
        magic, out_index, ok, byte_count, ngx_result, _ = _RESULT_HEADER.unpack(self._read(_RESULT_HEADER.size))
        if magic != _RESULT_MAGIC or not ok or out_index != index or byte_count != self._output.nbytes:
            raise self._failure(f"The DLSS 5 worker sent an invalid result for frame {index}")
        if ngx_result != _NGX_SUCCESS:
            raise self._failure(f"DLSS 5 failed on frame {index} (NGX 0x{ngx_result:08X})")
        self._read_into(self._output)
        upscaled = torch.from_numpy(self._output).to(self._device)[..., :3].permute(2, 0, 1).contiguous()[None]
        return finish_frame(upscaled.float().div_(255.0), frame, self._output_size, self._strength)

    def close(self) -> None:
        self._worker.stdin.close()
        exit_code = self._worker.wait(timeout=60)
        self._log_thread.join(timeout=2)
        if exit_code not in (0, _TRUNCATED_STREAM_EXIT):
            raise self._failure(f"The DLSS 5 worker failed (exit {exit_code})")
