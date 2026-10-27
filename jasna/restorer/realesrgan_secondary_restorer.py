"""Real-ESR family super resolution: a PyTorch network running in-process on the GPU.

This is the AMD counterpart of the RTX Super Res restorer and follows the same
strategy: no subprocess, no FFmpeg filter graph and no Vulkan/D3D - the crops are
upscaled by a PyTorch network on the device, so ``prefers_cpu_input`` stays
``False`` and the tensors never leave VRAM.

All three shipped checkpoints are ``SRVGGNetCompact`` (see ``srvggnet.py``): the
body is a plain VGG-style stack on the low-resolution feature map, no convolution
runs at the output size, which makes the family several times cheaper than an
RRDBNet of comparable width.

* ``realesr-general-x4v3`` / ``-wdn-x4v3`` -- 32 convs, PReLU, 4x.
* ``4xLSDIRCompactC3`` -- 16 convs, trained on LSDIR live-action photos; more even
  across crops than the 32-conv net.
* ``2xParimgCompact`` -- 16 convs, 2x-native, the fastest of the three.

A 4x network asked for ``scale=2`` is area-downsampled, which is what the
256 -> 256*scale contract expects.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from jasna.engine_paths import model_weights_dir
from jasna.restorer.srvggnet import load_srvggnet, read_srvggnet_spec

logger = logging.getLogger(__name__)

REALESRGAN_INPUT_SIZE = 256
REALESRGAN_SCALE_CHOICES = (2, 4)
REALESRGAN_DEFAULT_BATCH = 2

ARCH_SRVGGNET = "srvggnet-compact"

REALESRGAN_MODEL_CHOICES = (
    "x4v3",
    "lsdir-c3",
    "parimg-2x",
)
REALESRGAN_MODEL_FILES: dict[str, tuple[str, ...]] = {
    "x4v3": ("realesr-general-x4v3.pth", "RealESRGAN_x4v3.pth"),
    "lsdir-c3": ("4xLSDIRCompactC3.pth", "4xLSDIRCompactC3_fp16.pth"),
    "parimg-2x": ("2xParimgCompact.pth", "2xParimgCompact_fp16.pth"),
}

REALESRGAN_WEIGHT_HELP = (
    "Put one of the AMD super-res checkpoints in model_weights/ "
    "(realesr-general-x4v3.pth, 4xLSDIRCompactC3.pth or 2xParimgCompact.pth) "
    "or pass an existing file with --amd-upscale-model-path."
)

_WEIGHT_WRAPPERS = ("params_ema", "params", "state_dict")


@dataclass(frozen=True)
class UpscaleNetSpec:
    """Architecture-agnostic view of a loaded super-resolution checkpoint."""

    arch: str
    num_feat: int
    native_scale: int
    num_conv: int = 0
    act_type: str = ""

    def describe(self) -> str:
        return f"SRVGGNetCompact {self.num_conv} convs, {self.num_feat} feat, {self.act_type}"


def _load_state_dict(path: str | Path) -> dict:
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        for wrapper in _WEIGHT_WRAPPERS:
            if wrapper in state and isinstance(state[wrapper], dict):
                return state[wrapper]
    if not isinstance(state, dict):
        raise ValueError(
            f"unsupported checkpoint: expected a state dict, got {type(state).__name__}"
        )
    return state


def detect_upscale_arch(state: dict) -> str:
    """Which network family a checkpoint belongs to, decided on its key names only."""
    first = state.get("body.0.weight")
    if first is not None and getattr(first, "ndim", 0) == 4:
        return ARCH_SRVGGNET
    raise ValueError(
        "unsupported checkpoint: not an SRVGGNetCompact state dict (no body.0.weight); "
        "this build ships x4v3, lsdir-c3 and parimg-2x only"
    )


def read_upscale_spec(path: str | Path) -> UpscaleNetSpec:
    """The architecture of a checkpoint without building the network."""
    spec = read_srvggnet_spec(path)
    return UpscaleNetSpec(
        arch=ARCH_SRVGGNET,
        num_feat=spec.num_feat,
        native_scale=spec.native_scale,
        num_conv=spec.num_conv,
        act_type=spec.act_type,
    )


def load_upscale_model(
    path: str | Path, *, device: torch.device, fp16: bool
) -> tuple[nn.Module, UpscaleNetSpec]:
    """Build the network from its checkpoint and move it to ``device``."""
    detect_upscale_arch(_load_state_dict(path))
    model, spec = load_srvggnet(path, device=device, fp16=fp16)
    return model, UpscaleNetSpec(
        arch=ARCH_SRVGGNET,
        num_feat=spec.num_feat,
        native_scale=spec.native_scale,
        num_conv=spec.num_conv,
        act_type=spec.act_type,
    )


def resolve_weights_path(model_path: str | Path | None, model: str = "parimg-2x") -> Path:
    """Absolute, existence-checked checkpoint path (Windows-safe: no shell quoting).

    ``model_path`` wins when given, otherwise ``model`` picks one of the three
    shipped presets. An explicit ``--amd-upscale-model-path`` always wins.
    """
    if model_path:
        candidate = Path(model_path).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"AMD super-res model not found: {candidate.resolve(strict=False)}. "
                f"{REALESRGAN_WEIGHT_HELP}"
            )
        return candidate.resolve()

    preset = str(model or "parimg-2x").strip().lower()
    if preset not in REALESRGAN_MODEL_CHOICES:
        raise ValueError(
            f"Invalid AMD super-res model preset: {model!r} "
            f"(valid: {', '.join(REALESRGAN_MODEL_CHOICES)})"
        )
    directory = model_weights_dir()
    for name in REALESRGAN_MODEL_FILES[preset]:
        candidate = directory / name
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"The AMD super-res preset '{preset}' needs "
        f"{REALESRGAN_MODEL_FILES[preset][0]} in {directory.resolve(strict=False)}, "
        f"or pass an existing checkpoint with --amd-upscale-model-path."
    )


class RealEsrganSecondaryRestorer:
    """SRVGGNetCompact upscaling of restored crops, on the GPU."""

    name = "amd-upscale"
    num_workers = 1
    prefers_cpu_input = False

    def __init__(
        self,
        *,
        device: torch.device,
        scale: int = 2,
        model_path: str | Path | None = None,
        model: str = "parimg-2x",
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
        self.model_path = resolve_weights_path(model_path, model)

        self.model, self.spec = load_upscale_model(
            self.model_path, device=self.device, fp16=self.fp16
        )

        self._clips = 0
        self._frames = 0
        self._seconds = 0.0
        logger.info(
            "RealEsrganSecondaryRestorer: %s (%s, native %dx) on %s fp16=%s "
            "(%dx%d -> %dx%d, batch %d)",
            self.model_path.name,
            self.spec.describe(),
            self.spec.native_scale,
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
