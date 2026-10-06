"""RRDBNet (Real-ESRGAN) in plain PyTorch.

Self-contained so an AMD host does not need ``basicsr``/``realesrgan`` (both pull
in CUDA-only build steps).  The architecture matches BasicSR's ``RRDBNet`` exactly,
which is what the public Real-ESRGAN checkpoints were trained with:

* ``RealESRGAN_x4plus``      23 blocks, 64 features, 4x
* ``RealESRGAN_x4plus_anime_6B``  6 blocks, 64 features, 4x
* ``RealESRGAN_x2plus``      pixel-unshuffle 2 + 4x network = 2x

The loader reads the shape of ``conv_first.weight`` to derive ``num_feat`` and the
pixel-unshuffle factor, and counts ``body.*`` keys for ``num_block``, so any of the
above checkpoints (or a fine-tune of one) loads without configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RrdbNetSpec:
    num_feat: int
    num_block: int
    num_grow_ch: int
    pixel_unshuffle: int
    native_scale: int


def _pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    b, c, h, w = x.shape
    out_c = c * scale * scale
    h, w = h // scale, w // scale
    x = x.view(b, c, h, scale, w, scale)
    return x.permute(0, 1, 3, 5, 2, 4).reshape(b, out_c, h, w)


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RrdbBlock(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(
        self,
        *,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
        pixel_unshuffle: int = 1,
    ) -> None:
        super().__init__()
        self.pixel_unshuffle = int(pixel_unshuffle)
        in_ch = num_in_ch * self.pixel_unshuffle * self.pixel_unshuffle
        self.conv_first = nn.Conv2d(in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RrdbBlock(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pixel_unshuffle > 1:
            x = _pixel_unshuffle(x, self.pixel_unshuffle)
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def read_checkpoint_spec(path: str | Path) -> RrdbNetSpec:
    """Derive the architecture from a Real-ESRGAN checkpoint without loading weights."""
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        for wrapper in ("params_ema", "params", "state_dict"):
            if wrapper in state and isinstance(state[wrapper], dict):
                state = state[wrapper]
                break
    conv_first = state["conv_first.weight"]
    num_feat = int(conv_first.shape[0])
    in_ch = int(conv_first.shape[1])
    if in_ch % 3 != 0:
        raise ValueError(f"unsupported checkpoint: conv_first expects {in_ch} input channels")
    unshuffle_area = in_ch // 3
    pixel_unshuffle = 1
    while pixel_unshuffle * pixel_unshuffle < unshuffle_area:
        pixel_unshuffle += 1
    if pixel_unshuffle * pixel_unshuffle != unshuffle_area:
        raise ValueError(f"unsupported checkpoint: conv_first expects {in_ch} input channels")
    blocks = {int(k.split(".")[1]) for k in state if k.startswith("body.") and k.split(".")[1].isdigit()}
    num_block = max(blocks) + 1 if blocks else 0
    grow = int(state["body.0.rdb1.conv1.weight"].shape[0])
    return RrdbNetSpec(
        num_feat=num_feat,
        num_block=num_block,
        num_grow_ch=grow,
        pixel_unshuffle=pixel_unshuffle,
        native_scale=4 // pixel_unshuffle,
    )


def build_rrdbnet(spec: RrdbNetSpec) -> RRDBNet:
    return RRDBNet(
        num_feat=spec.num_feat,
        num_block=spec.num_block,
        num_grow_ch=spec.num_grow_ch,
        pixel_unshuffle=spec.pixel_unshuffle,
    )


def load_rrdbnet(path: str | Path, *, device: torch.device, fp16: bool) -> tuple[RRDBNet, RrdbNetSpec]:
    """Build the network from its checkpoint and move it to ``device``."""
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        for wrapper in ("params_ema", "params", "state_dict"):
            if wrapper in state and isinstance(state[wrapper], dict):
                state = state[wrapper]
                break
    spec = read_checkpoint_spec(path)
    model = build_rrdbnet(spec)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            "checkpoint does not match RRDBNet "
            f"({spec.num_block} blocks, {spec.num_feat} feat, pixel_unshuffle={spec.pixel_unshuffle}): "
            f"missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}"
        )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model = model.to(device=device, dtype=torch.float16 if fp16 else torch.float32)
    return model, spec
