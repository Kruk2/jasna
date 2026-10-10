"""AMD GPU super-resolution through FFmpeg's AMF filters, as a secondary backend.

Engines:

``amf-sr``
    AMD's own video super resolution, executed by the AMF filter ``sr_amf``.
    AMF is AMD's media framework; on Windows it is initialised over **D3D11**
    (``AMF initialisation succeeded via D3D11``), not Vulkan.  ``algorithm=sr1-0``
    / ``sr1-1`` are the "Video SR 1.0 / 1.1" models behind AMD Software's video
    upscaling; ``bicubic``/``bilinear``/``point`` are the non-ML fallbacks.

The two network engines (``--amd-upscale-engine real-esr`` for SRVGGNetCompact and
``realesrgan`` for RRDBNet) live in ``jasna.restorer.realesrgan_secondary_restorer``
and run their network on the ROCm device; all three backends share the same
``SecondaryRestorer`` contract.

Execution model
---------------
One FFmpeg process **per clip**, with stdin closed as soon as the frames are
written.  The AMF filters buffer the tail of the stream inside the AMF component
and only release it once the input ends: FFmpeg's own source notes that the
AMF filters ran off a ``filter_frame`` callback "which has no way to tell a
component that no more input is coming".  A long-lived process therefore stalls
one frame short of every clip, which is exactly what a continuous pipe does.
Closing stdin per clip (EOF) flushes the tail deterministically, and
``subprocess`` gives us a hard timeout for free.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from jasna.os_utils import find_executable, subprocess_no_window_kwargs

logger = logging.getLogger(__name__)

AMD_UPSCALE_INPUT_SIZE = 256
# AMF accepts an arbitrary output size and keeps running its SR engine at the
# higher factors, so 6x/8x are real super resolution, not a silent fallback to
# plain scaling.  The cost is compute: in-process restorer throughput measured on
# an RX 7900 XT (30-frame clips, 256x256 crops) is ~84 fps at 2x, ~55 at 4x,
# ~31 at 6x and ~18 at 8x.  The blend step downsamples the restored crop back to
# the mosaic size, so a higher factor behaves as supersampling: cleaner edges and
# less aliasing, at a proportional time cost.
AMD_UPSCALE_SCALE_CHOICES = (2, 4, 6, 8)
AMD_UPSCALE_ENGINE_CHOICES = ("amf-sr",)
AMD_UPSCALE_ALGORITHM_CHOICES = ("sr1-0", "sr1-1", "bicubic", "bilinear", "point")
AMD_UPSCALE_DEFAULT_ENGINE = "amf-sr"
AMD_UPSCALE_DEFAULT_ALGORITHM = "sr1-0"
AMD_UPSCALE_DEFAULT_DEVICE_SPEC = "amd"
AMD_UPSCALE_SHARPNESS_MIN = -1.0
AMD_UPSCALE_SHARPNESS_MAX = 2.0
AMD_UPSCALE_DEFAULT_TIMEOUT_S = 120.0

_PIX_FMT = "rgb24"
_STDERR_TAIL_CHARS = 400


class AmdUpscaleTimeout(RuntimeError):
    """FFmpeg did not finish the clip within the configured timeout."""


def build_filter_chain(
    *,
    engine: str,
    output_size: int,
    algorithm: str = AMD_UPSCALE_DEFAULT_ALGORITHM,
    sharpness: float = -1.0,
) -> str:
    """FFmpeg ``-vf`` chain for the requested engine.

    AMF runs on AMF hardware frames, so the chain uploads software RGB frames to
    NV12 on the AMF device, upscales there, and downloads them again.
    """
    if engine == "amf-sr":
        return (
            "format=nv12,"
            "hwupload=derive_device=amf,"
            f"sr_amf=w={output_size}:h={output_size}:"
            f"algorithm={algorithm}:sharpness={sharpness:g},"
            "hwdownload,format=nv12,format=rgb24"
        )
    raise ValueError(
        f"Unsupported AMD upscale engine: {engine!r} (valid: {', '.join(AMD_UPSCALE_ENGINE_CHOICES)})"
    )


def _resolve_ffmpeg(ffmpeg_path: str | None) -> str:
    """Absolute, existence-checked FFmpeg path (Windows-safe).

    Order: explicit override -> bundled ``tools/ffmpeg.exe`` (frozen builds) ->
    ``PATH``.  The result is passed to ``subprocess`` as a list element, so
    spaces and backslashes never need quoting or escaping.
    """
    if ffmpeg_path:
        candidate = Path(ffmpeg_path).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"AMD upscale ffmpeg not found: {candidate.resolve(strict=False)}"
            )
        return str(candidate.resolve())

    found = find_executable("ffmpeg")
    if not found:
        raise FileNotFoundError(
            "'ffmpeg' was not found on PATH and no bundled copy is available. "
            "AMD upscaling needs FFmpeg with the AMF filters (sr_amf/vpp_amf)."
        )
    return str(Path(found).resolve())


class AmdUpscaleSecondaryRestorer:
    """AMD super resolution via FFmpeg, one process per clip."""

    name = "amd-upscale"
    prefers_cpu_input = True
    num_workers = 1

    def __init__(
        self,
        *,
        device: torch.device | None = None,
        scale: int = 4,
        engine: str = AMD_UPSCALE_DEFAULT_ENGINE,
        algorithm: str = AMD_UPSCALE_DEFAULT_ALGORITHM,
        sharpness: float = -1.0,
        device_spec: str = AMD_UPSCALE_DEFAULT_DEVICE_SPEC,
        input_size: int = AMD_UPSCALE_INPUT_SIZE,
        ffmpeg_path: str | None = None,
        timeout_s: float = AMD_UPSCALE_DEFAULT_TIMEOUT_S,
    ) -> None:
        scale = int(scale)
        if scale not in AMD_UPSCALE_SCALE_CHOICES:
            raise ValueError(
                f"Invalid AMD upscale factor: {scale} "
                f"(valid: {', '.join(map(str, AMD_UPSCALE_SCALE_CHOICES))})"
            )
        engine = str(engine).lower()
        if engine not in AMD_UPSCALE_ENGINE_CHOICES:
            raise ValueError(
                f"Invalid AMD upscale engine: {engine!r} (valid: {', '.join(AMD_UPSCALE_ENGINE_CHOICES)})"
            )
        algorithm = str(algorithm).lower()
        if engine == "amf-sr" and algorithm not in AMD_UPSCALE_ALGORITHM_CHOICES:
            raise ValueError(
                f"Invalid AMD upscale algorithm: {algorithm!r} "
                f"(valid: {', '.join(AMD_UPSCALE_ALGORITHM_CHOICES)})"
            )
        sharpness = float(sharpness)
        if not AMD_UPSCALE_SHARPNESS_MIN <= sharpness <= AMD_UPSCALE_SHARPNESS_MAX:
            raise ValueError(
                f"AMD upscale sharpness must be in "
                f"[{AMD_UPSCALE_SHARPNESS_MIN:g}, {AMD_UPSCALE_SHARPNESS_MAX:g}], got {sharpness:g}"
            )
        if input_size < 1:
            raise ValueError("input_size must be positive")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")

        self.device = torch.device(device) if device is not None else None
        self.input_size = int(input_size)
        self.scale = scale
        self.output_size = int(input_size) * scale
        self.engine = engine
        self.algorithm = algorithm
        self.sharpness = sharpness
        self.device_spec = str(device_spec)
        self.timeout_s = float(timeout_s)
        self.ffmpeg_path = _resolve_ffmpeg(ffmpeg_path)

        self._in_frame_bytes = self.input_size * self.input_size * 3
        self._out_frame_bytes = self.output_size * self.output_size * 3
        self._clips = 0
        self._frames = 0
        self._seconds = 0.0
        self._last_stderr = ""

        logger.info(
            "AmdUpscaleSecondaryRestorer: engine=%s scale=%dx %s sharpness=%g "
            "(%dx%d -> %dx%d) ffmpeg=%s timeout=%gs",
            self.engine,
            self.scale,
            self.algorithm,
            self.sharpness,
            self.input_size,
            self.input_size,
            self.output_size,
            self.output_size,
            self.ffmpeg_path,
            self.timeout_s,
        )

    # -- command ------------------------------------------------------------

    def build_ffmpeg_cmd(self) -> list[str]:
        cmd: list[str] = [self.ffmpeg_path, "-hide_banner", "-loglevel", "error"]
        if self.engine == "amf-sr":
            cmd += ["-init_hw_device", f"amf={self.device_spec}"]
        cmd += [
            "-f", "rawvideo",
            "-pix_fmt", _PIX_FMT,
            "-s", f"{self.input_size}x{self.input_size}",
            "-r", "25",
            "-i", "pipe:0",
            "-vf", build_filter_chain(
                engine=self.engine,
                output_size=self.output_size,
                algorithm=self.algorithm,
                sharpness=self.sharpness,
            ),
            "-f", "rawvideo",
            "-pix_fmt", _PIX_FMT,
            "pipe:1",
        ]
        return cmd

    # -- frames -------------------------------------------------------------

    def _frames_to_hwc_bytes(self, frames: torch.Tensor) -> bytes:
        """(T,3,H,W) float [0,1] or uint8 -> contiguous HWC uint8 bytes."""
        x = frames
        if x.is_cuda:
            # arithmetic on the GPU, then one device->host copy of the small uint8 data
            x = x.to(dtype=torch.float32).clamp_(0.0, 1.0).mul_(255.0).round_().clamp_(0.0, 255.0)
            x = x.to(dtype=torch.uint8).permute(0, 2, 3, 1).contiguous().cpu()
        else:
            x = x.to(dtype=torch.float32).clamp(0.0, 1.0).mul(255.0).round().clamp(0.0, 255.0)
            x = x.to(dtype=torch.uint8).permute(0, 2, 3, 1).contiguous()
        return x.numpy().tobytes()

    def _hwc_to_frame_tensors(self, buf: bytes, count: int) -> list[torch.Tensor]:
        # bytearray keeps the array writable, which torch.from_numpy requires
        arr = np.frombuffer(bytearray(buf), dtype=np.uint8)
        frames = torch.from_numpy(arr).reshape(count, self.output_size, self.output_size, 3)
        return list(frames.permute(0, 3, 1, 2).contiguous().unbind(0))

    def _run_ffmpeg(self, payload: bytes, count: int) -> bytes:
        expected = count * self._out_frame_bytes
        cmd = self.build_ffmpeg_cmd()
        try:
            completed = subprocess.run(
                cmd,
                input=payload,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_s,
                **subprocess_no_window_kwargs(),
            )
        except subprocess.TimeoutExpired as exc:
            raise AmdUpscaleTimeout(
                f"AMD upscale ({self.engine}): FFmpeg did not finish {count} frames "
                f"within {self.timeout_s:g}s"
            ) from exc

        stderr = completed.stderr.decode("utf-8", errors="replace").strip() if completed.stderr else ""
        self._last_stderr = stderr[-_STDERR_TAIL_CHARS:]
        if completed.returncode != 0:
            raise RuntimeError(
                f"AMD upscale ({self.engine}): FFmpeg exited with {completed.returncode}: "
                f"{self._last_stderr or '(no stderr)'}"
            )
        produced = len(completed.stdout)
        if produced < expected:
            raise RuntimeError(
                f"AMD upscale ({self.engine}): FFmpeg returned {produced // self._out_frame_bytes} "
                f"of {count} frames (expected {expected} bytes, got {produced}). "
                f"{self._last_stderr or '(no stderr)'}"
            )
        return completed.stdout[:expected]

    def restore(
        self,
        frames: torch.Tensor,
        *,
        keep_start: int,
        keep_end: int,
    ) -> list[torch.Tensor]:
        t = int(frames.shape[0])
        if t == 0:
            return []
        if frames.ndim != 4 or tuple(frames.shape[1:]) != (3, self.input_size, self.input_size):
            raise ValueError(
                f"expected frames shaped (T, 3, {self.input_size}, {self.input_size}), "
                f"got {tuple(frames.shape)}"
            )

        ks = max(0, int(keep_start))
        ke = min(t, int(keep_end))
        if ks >= ke:
            return []
        batch = frames[ks:ke]
        count = int(batch.shape[0])

        payload = self._frames_to_hwc_bytes(batch)
        start = time.monotonic()
        raw = self._run_ffmpeg(payload, count)
        elapsed = time.monotonic() - start

        self._clips += 1
        self._frames += count
        self._seconds += elapsed
        if self._clips % 20 == 0:
            logger.info(
                "AMD upscale (%s): %d clips / %d frames, %.1f fps average",
                self.engine,
                self._clips,
                self._frames,
                self._frames / max(self._seconds, 1e-6),
            )
        return self._hwc_to_frame_tensors(raw, count)

    def close(self) -> None:
        """Nothing is persistent: each clip owns its FFmpeg process."""
        if self._frames:
            logger.info(
                "AMD upscale (%s) finished: %d clips / %d frames in %.1fs (%.1f fps)",
                self.engine,
                self._clips,
                self._frames,
                self._seconds,
                self._frames / max(self._seconds, 1e-6),
            )
