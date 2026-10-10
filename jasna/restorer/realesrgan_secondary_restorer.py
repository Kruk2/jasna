"""Real-ESRGAN super resolution: a plain PyTorch network running in-process on the AMD GPU.

This is the AMD counterpart of the RTX Super Res restorer and follows the same
strategy: no subprocess, no FFmpeg filter graph and no Vulkan/D3D — the crops are
upscaled by a PyTorch network on the ROCm device, so ``prefers_cpu_input`` stays
``False`` and the tensors never leave VRAM.

Two network families are understood, and the family is derived from the checkpoint
itself (see ``rrdbnet.py`` / ``srvggnet.py``):

* ``SRVGGNetCompact`` — ``realesr-general-x4v3`` / ``-wdn-x4v3`` (32 convs, PReLU),
  ``realesr-animevideov3`` (16 convs), ``4xLSDIRCompactC3`` / ``4xLSDIRCompactv2``
  (16 convs, live-action LSDIR weights) and ``2xHFA2kCompact`` (16 convs, 2x-native).
  No convolution runs at the high-resolution size, which makes them
  several times cheaper than an RRDBNet.  ``realesr-general-x4v3`` is the default: on
  an RX 7900 XT at 256x256 -> 1024x1024 (fp16, batch 2) it measures ~238 fps against
  ~52 fps for the 6-block anime net, uses 38 MB of VRAM instead of 604 MB, and scores
  higher PSNR/SSIM on live-action footage.  ``4xLSDIRCompactC3`` is the smaller
  LSDIR-trained member of the same family (~0.6 M params).
* ``RRDBNet`` — ``RealESRGAN_x4plus`` (23 blocks), ``RealESRGAN_x4plus_anime_6B``
  (6 blocks), pixel-unshuffle 2x variants, and KAIR checkpoints such as ``BSRNet``
  (23 blocks, BSRGAN, the highest-fidelity option measured on live-action crops).

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
from jasna.restorer.rrdbnet import load_rrdbnet, read_checkpoint_spec as read_rrdbnet_spec
from jasna.restorer.srvggnet import load_srvggnet, read_srvggnet_spec

logger = logging.getLogger(__name__)

REALESRGAN_INPUT_SIZE = 256
REALESRGAN_SCALE_CHOICES = (2, 4)
# Measured on an RX 7900 XT (ROCm 7.1, fp16, 256x256 -> 1024x1024):
#   batch 1 -> 15.7 fps, batch 2 -> 16.5 fps, batch 4 -> 0.24 fps (!), batch 8 -> 13.6 fps.
# Batch 4 makes MIOpen pick a pathological kernel for the 23-block RRDBNet at this
# shape, so keep the default at 2 and never batch that network in fours.  The
# SRVGGNetCompact default is faster at batch 4 (measured ~254 fps vs ~222 fps for
# realesr-general-x4v3), but the constant is shared, so it stays at 2 until an
# end-to-end A/B justifies a per-architecture split.
REALESRGAN_DEFAULT_BATCH = 2

ARCH_RRDBNET = "rrdbnet"
ARCH_SRVGGNET = "srvggnet-compact"

# Checked in this order when ``--amd-upscale-model-path`` is not given. The
# realesr-general checkpoints come first: measured on an RX 7900 XT
# (256x256 -> 1024x1024, fp16) realesr-general-x4v3 runs at ~238 fps against ~52 fps
# for the 6-block anime net and ~17 fps for the 23-block x4plus, at equal or better
# PSNR/SSIM on live-action footage, with 38 MB of VRAM instead of 604 MB.
REALESRGAN_WEIGHT_CANDIDATES = (
    "realesr-general-x4v3.pth",
    "RealESRGAN_x4v3.pth",
    "realesr-general-wdn-x4v3.pth",
    "RealESRGAN_x4v3_wdn.pth",
    "realesrgan_x4plus_anime_6B.pth",
    "RealESRGAN_x4plus_anime_6B.pth",
    "realesrgan_x4plus.pth",
    "RealESRGAN_x4plus.pth",
    "realesrgan-x4plus.pth",
    "realesrgan_x2plus.pth",
    "RealESRGAN_x2plus.pth",
    "4xLSDIRCompactC3.pth",
    "4xLSDIRCompactv2.pth",
    "2xHFA2kCompact.pth",
    "BSRNet.pth",
)

# Named presets for ``--amd-upscale-model`` / the GUI's model picker.  Two families
# share this ROCm engine, and the panel's engine row splits them the same way
# (see ``jasna.session_config.AMD_UPSCALE_ENGINE_MODELS``):
#   real-esr   - SRVGGNetCompact: x4v3 / wdn-x4v3 / lsdir-c3 / lsdir-v2 / hfa2k-2x
#                (~4.5x cheaper)
#   realesrgan - RRDBNet: x4plus / anime-6b / bsrnet (quality ceiling)
# The default is x4v3: fastest, highest SSIM, and the most even across mosaic crops
# (per-crop colour drift and detail ratio both stay below the 6-block anime net),
# which is what a restore-and-blend pipeline cares about most.
REALESRGAN_MODEL_CHOICES = (
    "auto",
    "x4v3",
    "wdn-x4v3",
    "lsdir-c3",
    "lsdir-v2",
    "hfa2k-2x",
    "x4plus",
    "anime-6b",
    "bsrnet",
)
REALESRGAN_MODEL_FILES: dict[str, tuple[str, ...]] = {
    "x4v3": ("realesr-general-x4v3.pth", "RealESRGAN_x4v3.pth"),
    "wdn-x4v3": ("realesr-general-wdn-x4v3.pth", "RealESRGAN_x4v3_wdn.pth"),
    "lsdir-c3": ("4xLSDIRCompactC3.pth", "4xLSDIRCompactC3_fp16.pth"),
    "lsdir-v2": ("4xLSDIRCompactv2.pth", "4xLSDIRCompactv2_fp16.pth"),
    "hfa2k-2x": ("2xHFA2kCompact.pth", "2xHFA2kCompact_fp16.pth"),
    "x4plus": ("realesrgan_x4plus.pth", "RealESRGAN_x4plus.pth", "realesrgan-x4plus.pth"),
    "anime-6b": (
        "realesrgan_x4plus_anime_6B.pth",
        "RealESRGAN_x4plus_anime_6B.pth",
        "realesrgan-x4plus-anime-6b.pth",
    ),
    "bsrnet": ("BSRNet.pth", "bsrnet.pth", "BSRGAN.pth"),
}
REALESRGAN_MODEL_URLS = {
    "x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    "wdn-x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
    "lsdir-c3": "https://github.com/Phhofm/models/releases/tag/4xLSDIRCompactC3",
    "lsdir-v2": "https://github.com/Phhofm/models/releases/tag/4xLSDIRCompact2",
    "hfa2k-2x": "https://github.com/Phhofm/models/releases/tag/2xHFA2kCompact",
    "x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "anime-6b": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "bsrnet": "https://github.com/cszn/KAIR/releases/download/v1.0/BSRNet.pth",
}

REALESRGAN_WEIGHT_HELP = (
    "Put a Real-ESRGAN checkpoint in model_weights/ (for example "
    "realesr-general-x4v3.pth, https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth) "
    "or pass an existing file with --amd-upscale-model-path."
)

_WEIGHT_WRAPPERS = ("params_ema", "params", "state_dict")


@dataclass(frozen=True)
class UpscaleNetSpec:
    """Architecture-agnostic view of a loaded super-resolution checkpoint."""

    arch: str
    num_feat: int
    native_scale: int
    num_block: int = 0
    num_conv: int = 0
    act_type: str = ""

    def describe(self) -> str:
        if self.arch == ARCH_RRDBNET:
            return f"RRDBNet {self.num_block} blocks, {self.num_feat} feat"
        if self.arch == ARCH_SRVGGNET:
            return f"SRVGGNetCompact {self.num_conv} convs, {self.num_feat} feat, {self.act_type}"
        return f"{self.arch} {self.num_feat} feat"


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
    keys = set(state)
    if any(key.startswith("conv_first") or ".rdb1." in key for key in keys):
        return ARCH_RRDBNET
    first = state.get("body.0.weight")
    if first is not None and getattr(first, "ndim", 0) == 4:
        return ARCH_SRVGGNET
    raise ValueError(
        "unsupported checkpoint: neither an RRDBNet (conv_first / body.*.rdb1) nor an "
        "SRVGGNetCompact (body.0.weight) state dict"
    )


def read_upscale_spec(path: str | Path) -> UpscaleNetSpec:
    """The architecture of a checkpoint without building the network."""
    arch = detect_upscale_arch(_load_state_dict(path))
    if arch == ARCH_RRDBNET:
        spec = read_rrdbnet_spec(path)
        return UpscaleNetSpec(
            arch=arch,
            num_feat=spec.num_feat,
            native_scale=spec.native_scale,
            num_block=spec.num_block,
        )
    spec = read_srvggnet_spec(path)
    return UpscaleNetSpec(
        arch=arch,
        num_feat=spec.num_feat,
        native_scale=spec.native_scale,
        num_conv=spec.num_conv,
        act_type=spec.act_type,
    )


def load_upscale_model(
    path: str | Path, *, device: torch.device, fp16: bool
) -> tuple[nn.Module, UpscaleNetSpec]:
    """Build either network family from its checkpoint and move it to ``device``.

    RRDBNet and SRVGGNetCompact share the same 256 -> 256*scale contract, so the
    restorer treats them interchangeably.
    """
    arch = detect_upscale_arch(_load_state_dict(path))
    if arch == ARCH_RRDBNET:
        model, spec = load_rrdbnet(path, device=device, fp16=fp16)
        return model, UpscaleNetSpec(
            arch=arch,
            num_feat=spec.num_feat,
            native_scale=spec.native_scale,
            num_block=spec.num_block,
        )
    model, spec = load_srvggnet(path, device=device, fp16=fp16)
    return model, UpscaleNetSpec(
        arch=arch,
        num_feat=spec.num_feat,
        native_scale=spec.native_scale,
        num_conv=spec.num_conv,
        act_type=spec.act_type,
    )


def find_default_weights() -> Path | None:
    directory = model_weights_dir()
    for name in REALESRGAN_WEIGHT_CANDIDATES:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def resolve_weights_path(model_path: str | Path | None, model: str = "auto") -> Path:
    """Absolute, existence-checked checkpoint path (Windows-safe: no shell quoting).

    ``model_path`` wins when given. ``model`` otherwise picks one of the named
    presets (``x4v3``, ``wdn-x4v3``, ``lsdir-c3``, ``x4plus``, ``anime-6b``,
    ``bsrnet``); ``auto`` keeps the candidate search, which also prefers
    ``realesr-general-x4v3``.  The preset decides the checkpoint, not the engine
    class, so an explicit ``--amd-upscale-model-path`` always wins.
    """
    if model_path:
        candidate = Path(model_path).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"AMD super-res model not found: {candidate.resolve(strict=False)}. "
                f"{REALESRGAN_WEIGHT_HELP}"
            )
        return candidate.resolve()

    preset = str(model or "auto").strip().lower()
    if preset not in REALESRGAN_MODEL_CHOICES:
        raise ValueError(
            f"Invalid AMD super-res model preset: {model!r} "
            f"(valid: {', '.join(REALESRGAN_MODEL_CHOICES)})"
        )
    if preset != "auto":
        directory = model_weights_dir()
        for name in REALESRGAN_MODEL_FILES[preset]:
            candidate = directory / name
            if candidate.is_file():
                return candidate.resolve()
        wanted = REALESRGAN_MODEL_FILES[preset][0]
        url = REALESRGAN_MODEL_URLS[preset]
        raise FileNotFoundError(
            f"The AMD super-res preset '{preset}' needs {wanted} in "
            f"{directory.resolve(strict=False)} ({url}), "
            f"or pass an existing checkpoint with --amd-upscale-model-path."
        )

    found = find_default_weights()
    if found is None:
        raise FileNotFoundError(
            f"No AMD super-res model found in {model_weights_dir().resolve(strict=False)}. "
            f"{REALESRGAN_WEIGHT_HELP}"
        )
    return found.resolve()


class RealEsrganSecondaryRestorer:
    """Real-ESRGAN family upscaling of restored crops, on the ROCm device."""

    name = "amd-upscale"
    num_workers = 1
    prefers_cpu_input = False

    def __init__(
        self,
        *,
        device: torch.device,
        scale: int = 4,
        model_path: str | Path | None = None,
        model: str = "auto",
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
