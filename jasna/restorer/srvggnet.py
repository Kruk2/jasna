"""SRVGGNetCompact (the realesr-general / realesr-animevideov3 family) in plain PyTorch.

Self-contained so an AMD host does not need ``basicsr``/``realesrgan`` (both pull in
CUDA-only build steps).  The architecture matches BasicSR's ``SRVGGNetCompact``
exactly, which is what the public checkpoints were trained with:

* ``realesr-general-x4v3`` / ``realesr-general-wdn-x4v3``  32 convs, 64 feat, PReLU, 4x
* ``realesr-animevideov3``                                 16 convs, 64 feat, PReLU, 4x
* ``4xLSDIRCompactC3`` (Phhofm)                            16 convs, 64 feat, PReLU, 4x
  - the LSDIR live-action compact net; ~0.62 M params, ~2.4 MB on disk.

No convolution ever runs at the high-resolution size: the body is a plain VGG-style
stack on the low-resolution feature map, the last convolution expands to
``num_out_ch * upscale**2`` channels and ``PixelShuffle`` does the upsampling.  That
is why the family is several times cheaper than an RRDBNet of comparable width --
measured on an RX 7900 XT (256x256 -> 1024x1024, fp16) ``realesr-general-x4v3`` runs
at ~238 fps against ~52 fps for ``RealESRGAN_x4plus_anime_6B``.  The network learns
the residual over a nearest-neighbour upsample of the input, which is added back
after the shuffle.

The loader reads ``body.0.weight`` for the feature width, counts the 4-D convolution
weights for the depth and reads the last one for the upscale factor, so any
checkpoint of the family (or a fine-tune of one) loads without configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_ACT_TYPES = ("prelu", "relu", "leakyrelu")


@dataclass(frozen=True)
class SrvggNetSpec:
    num_in_ch: int
    num_out_ch: int
    num_feat: int
    num_conv: int
    upscale: int
    act_type: str

    @property
    def native_scale(self) -> int:
        """Upscale factor the network was trained at (4 for the general checkpoints)."""
        return self.upscale


def _activation(act_type: str, num_feat: int) -> nn.Module:
    if act_type == "prelu":
        return nn.PReLU(num_parameters=num_feat)
    if act_type == "relu":
        return nn.ReLU(inplace=True)
    if act_type == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    raise ValueError(f"unsupported activation {act_type!r} (valid: {', '.join(_ACT_TYPES)})")


class SRVGGNetCompact(nn.Module):
    def __init__(
        self,
        *,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 32,
        upscale: int = 4,
        act_type: str = "prelu",
    ) -> None:
        super().__init__()
        if num_conv < 1:
            raise ValueError("num_conv must be >= 1")
        if upscale < 1:
            raise ValueError("upscale must be >= 1")
        if act_type not in _ACT_TYPES:
            raise ValueError(f"unsupported activation {act_type!r} (valid: {', '.join(_ACT_TYPES)})")

        self.num_in_ch = int(num_in_ch)
        self.num_out_ch = int(num_out_ch)
        self.num_feat = int(num_feat)
        self.num_conv = int(num_conv)
        self.upscale = int(upscale)
        self.act_type = act_type

        body = nn.ModuleList()
        body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        body.append(_activation(act_type, num_feat))
        for _ in range(num_conv):
            body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            body.append(_activation(act_type, num_feat))
        body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.body = body
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        # the network predicts the residual over a nearest-neighbour upsample of the
        # input, so the base image is added back after the shuffle.
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


def _unwrap_state(state: object) -> dict:
    if isinstance(state, dict):
        for wrapper in ("params_ema", "params", "state_dict"):
            if wrapper in state and isinstance(state[wrapper], dict):
                return state[wrapper]
    if not isinstance(state, dict):
        raise ValueError(f"unsupported checkpoint: expected a state dict, got {type(state).__name__}")
    return state


def read_srvggnet_spec(path: str | Path) -> SrvggNetSpec:
    """Derive the architecture from an SRVGGNetCompact checkpoint without building it."""
    state = _unwrap_state(torch.load(str(path), map_location="cpu", weights_only=True))
    first = state.get("body.0.weight")
    if first is None or first.ndim != 4:
        raise ValueError("unsupported checkpoint: no body.0.weight (not an SRVGGNetCompact network)")
    num_feat = int(first.shape[0])
    num_in_ch = int(first.shape[1])

    conv_indices = sorted(
        int(key.split(".")[1])
        for key, value in state.items()
        if key.startswith("body.") and key.endswith(".weight") and value.ndim == 4
    )
    if not conv_indices or conv_indices[0] != 0:
        raise ValueError("unsupported checkpoint: expected a body.0 convolution")
    last = conv_indices[-1]
    if (last - 2) % 2 != 0:
        raise ValueError(f"unsupported checkpoint: unexpected last convolution index body.{last}")
    num_conv = (last - 2) // 2

    # PReLU keeps a learnable slope per channel; ReLU/LeakyReLU have no parameters, so
    # the presence of 1-D weights at the odd body positions identifies a PReLU network.
    has_prelu = any(
        key.startswith("body.") and key.endswith(".weight") and value.ndim == 1
        for key, value in state.items()
    )
    act_type = "prelu" if has_prelu else "relu"

    # RGB in, RGB out: the final convolution emits num_out_ch * upscale**2 channels.
    num_out_ch = num_in_ch
    last_out = int(state[f"body.{last}.weight"].shape[0])
    if last_out % num_out_ch != 0:
        raise ValueError(
            f"unsupported checkpoint: last convolution emits {last_out} channels, "
            f"not a multiple of {num_out_ch}"
        )
    area = last_out // num_out_ch
    upscale = int(round(area**0.5))
    if upscale * upscale != area:
        raise ValueError(f"unsupported checkpoint: {area} output channels per image are not a square")

    return SrvggNetSpec(
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        num_feat=num_feat,
        num_conv=num_conv,
        upscale=upscale,
        act_type=act_type,
    )


def build_srvggnet(spec: SrvggNetSpec) -> SRVGGNetCompact:
    return SRVGGNetCompact(
        num_in_ch=spec.num_in_ch,
        num_out_ch=spec.num_out_ch,
        num_feat=spec.num_feat,
        num_conv=spec.num_conv,
        upscale=spec.upscale,
        act_type=spec.act_type,
    )


def load_srvggnet(
    path: str | Path, *, device: torch.device, fp16: bool
) -> tuple[SRVGGNetCompact, SrvggNetSpec]:
    """Build the network from its checkpoint and move it to ``device``."""
    state = _unwrap_state(torch.load(str(path), map_location="cpu", weights_only=True))
    spec = read_srvggnet_spec(path)
    model = build_srvggnet(spec)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            "checkpoint does not match SRVGGNetCompact "
            f"({spec.num_conv} convs, {spec.num_feat} feat, {spec.act_type}, {spec.upscale}x): "
            f"missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}"
        )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model = model.to(device=device, dtype=torch.float16 if fp16 else torch.float32)
    return model, spec
