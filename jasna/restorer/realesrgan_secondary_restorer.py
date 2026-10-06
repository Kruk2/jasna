"""Real-ESRGAN super resolution: an RRDBNet running in-process on the AMD GPU.

This is the AMD counterpart of the RTX Super Res restorer and follows the same
strategy: no subprocess, no FFmpeg filter graph and no Vulkan/D3D — the crops are
upscaled by a PyTorch network on the ROCm device, so ``prefers_cpu_input`` stays
``False`` and the tensors never leave VRAM.

Model: any Real-ESRGAN RRDBNet checkpoint (``RealESRGAN_x4plus.pth`` and friends).
The architecture is derived from the checkpoint itself (see ``rrdbnet.py``), so a
4x, a 2x/pixel-unshuffle or a 6-block anime checkpoint all work.  When the network
is more powerful than the requested factor (4x net, ``scale=2``) the result is
area-downsampled to the requested size, which is what the 256 -> 256*scale
contract expects.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from jasna.engine_paths import model_weights_dir
from jasna.restorer.rrdbnet import RrdbNetSpec, load_rrdbnet

logger = logging.getLogger(__name__)

REALESRGAN_INPUT_SIZE = 256
REALESRGAN_SCALE_CHOICES = (2, 4)
# Measured on an RX 7900 XT (ROCm 7.1, fp16, 256x256 -> 1024x1024):
#   batch 1 -> 15.7 fps, batch 2 -> 16.5 fps, batch 4 -> 0.24 fps (!), batch 8 -> 13.6 fps.
# Batch 4 makes MIOpen pick a pathological kernel for this shape, so keep the
# default at 2 and never batch this network in fours.
REALESRGAN_DEFAULT_BATCH = 2

# Checked in this order when ``--amd-upscale-model-path`` is not given.
REALESRGAN_WEIGHT_CANDIDATES = (
    "realesrgan_x4plus.pth",
    "RealESRGAN_x4plus.pth",
    "realesrgan-x4plus.pth",
    "realesrgan_x4plus_anime_6B.pth",
    "RealESRGAN_x4plus_anime_6B.pth",
    "realesrgan_x2plus.pth",
    "RealESRGAN_x2plus.pth",
)

REALESRGAN_WEIGHT_HELP = (
    "Put a Real-ESRGAN RRDBNet checkpoint in model_weights/ (for example "
    "realesrgan_x4plus.pth, https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth) "
    "or pass an existing file with --amd-upscale-model-path."
)


def find_default_weights() -> Path | None:
    directory = model_weights_dir()
    for name in REALESRGAN_WEIGHT_CANDIDATES:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def resolve_weights_path(model_path: str | Path | None) -> Path:
    """Absolute, existence-checked checkpoint path (Windows-safe: no shell quoting)."""
    if model_path:
        candidate = Path(model_path).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"AMD super-res model not found: {candidate.resolve(strict=False)}. "
                f"{REALESRGAN_WEIGHT_HELP}"
            )
        return candidate.resolve()
    found = find_default_weights()
    if found is None:
        raise FileNotFoundError(
            f"No AMD super-res model found in {model_weights_dir().resolve(strict=False)}. "
            f"{REALESRGAN_WEIGHT_HELP}"
        )
    return found.resolve()


class RealEsrganSecondaryRestorer:
    """Real-ESRGAN upscaling of restored crops, on the ROCm device."""

    name = "amd-upscale"
    num_workers = 1
    prefers_cpu_input = False

    def __init__(
        self,
        *,
        device: torch.device,
        scale: int = 4,
        model_path: str | Path | None = None,
        fp16: bool = True,
        batch_size: int = REALESRGAN_DEFAULT_BATCH,
        input_size: int = REALESRGAN_INPUT_SIZE,
    ) -> None:
        scale = int(scale)
        if scale not in REALESRGAN_SCALE_CHOICES:
            raise ValueError(
                f"Invalid AMD super-res factor: {scale} "
                f"(valid: {', '.join(map(str, REALESRGAN_SCALE_CHOICES))})"
            )
        if input_size < 1:
            raise ValueError("input_size must be positive")
        if int(batch_size) < 1:
            raise ValueError("batch_size must be > 0")

        self.device = torch.device(device)
        self.input_size = int(input_size)
        self.scale = scale
        self.output_size = int(input_size) * scale
        self.fp16 = bool(fp16)
        self.batch_size = int(batch_size)
        self.model_path = resolve_weights_path(model_path)

        model, spec = load_rrdbnet(self.model_path, device=self.device, fp16=self.fp16)
        self.model = model
        self.spec: RrdbNetSpec = spec

        self._clips = 0
        self._frames = 0
        self._seconds = 0.0
        logger.info(
            "RealEsrganSecondaryRestorer: %s (RRDBNet %d blocks, %d feat, native %dx) "
            "on %s fp16=%s (%dx%d -> %dx%d, batch %d)",
            self.model_path.name,
            spec.num_block,
            spec.num_feat,
            spec.native_scale,
            self.device,
            self.fp16,
            self.input_size,
            self.input_size,
            self.output_size,
            self.output_size,
            self.batch_size,
        )

    def _upscale_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        dtype = torch.float16 if self.fp16 else torch.float32
        x = chunk.to(device=self.device, dtype=dtype, non_blocking=True)
        with torch.no_grad():
            out = self.model(x)
        out = out.float().clamp_(0.0, 1.0)
        if out.shape[-1] != self.output_size or out.shape[-2] != self.output_size:
            out = F.interpolate(
                out,
                size=(self.output_size, self.output_size),
                mode="area" if out.shape[-1] > self.output_size else "bilinear",
                align_corners=False if out.shape[-1] <= self.output_size else None,
            )
        return out

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

        start = time.monotonic()
        out_frames: list[torch.Tensor] = []
        for offset in range(0, count, self.batch_size):
            chunk = batch[offset:offset + self.batch_size]
            upscaled = self._upscale_chunk(chunk)
            u8 = (
                upscaled.mul_(255.0)
                .round_()
                .clamp_(0.0, 255.0)
                .to(dtype=torch.uint8)
                .contiguous()
            )
            out_frames.extend(u8.unbind(0))

        elapsed = time.monotonic() - start
        self._clips += 1
        self._frames += count
        self._seconds += elapsed
        if self._clips % 20 == 0:
            logger.info(
                "Real-ESRGAN: %d clips / %d frames, %.1f fps average",
                self._clips,
                self._frames,
                self._frames / max(self._seconds, 1e-6),
            )
        return out_frames

    def close(self) -> None:
        """Drop the network and release its VRAM."""
        if self.model is None:
            return
        if self._frames:
            logger.info(
                "Real-ESRGAN finished: %d clips / %d frames in %.1fs (%.1f fps)",
                self._clips,
                self._frames,
                self._seconds,
                self._frames / max(self._seconds, 1e-6),
            )
        self.model = None
        if self.device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:  # pragma: no cover - ROCm runtime quirk
                logger.debug("empty_cache failed during close", exc_info=True)
