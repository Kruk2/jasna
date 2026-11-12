"""Fold a SPAN training checkpoint into the inference weights ``jasna.models.span`` loads.

Every ``Conv3XC`` (1x1 -> 3x3 -> 1x1 convolutions plus a 1x1 skip) becomes the one 3x3
convolution it equals at inference, the same folding SPAN's own ``update_params`` does.

    python scripts/fuse_span_weights.py 2x-NomosUni_span_multijpg_ldl.pth model_weights/span-nomosuni-2x.safetensors
"""

from __future__ import annotations

import argparse

import torch
from torch.nn import functional as F

CONV3XC_LAYERS = ("conv_1", "conv_2", *(f"block_{b}.c{c}_r" for b in range(1, 7) for c in (1, 2, 3)))
PLAIN_LAYERS = {"conv_cat": "conv_cat", "upsampler.0": "upsampler"}


def _fold(state: dict[str, torch.Tensor], name: str) -> tuple[torch.Tensor, torch.Tensor]:
    w1, b1 = state[f"{name}.conv.0.weight"], state[f"{name}.conv.0.bias"]
    w2, b2 = state[f"{name}.conv.1.weight"], state[f"{name}.conv.1.bias"]
    w3, b3 = state[f"{name}.conv.2.weight"], state[f"{name}.conv.2.bias"]
    weight = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2).flip(2, 3).permute(1, 0, 2, 3)
    bias = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2
    weight = F.conv2d(weight.flip(2, 3).permute(1, 0, 2, 3), w3).flip(2, 3).permute(1, 0, 2, 3)
    bias = (w3 * bias.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3
    weight = weight + F.pad(state[f"{name}.sk.weight"], [1, 1, 1, 1])
    return weight, bias + state[f"{name}.sk.bias"]


def fused_weights(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {}
    for name in CONV3XC_LAYERS:
        weight, bias = _fold(state, name)
        target = name.removesuffix("_r")
        weights[f"{target}.weight"], weights[f"{target}.bias"] = weight, bias
    for source, target in PLAIN_LAYERS.items():
        weights[f"{target}.weight"] = state[f"{source}.weight"]
        weights[f"{target}.bias"] = state[f"{source}.bias"]
    return {name: tensor.float().contiguous() for name, tensor in weights.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()

    from safetensors.torch import save_file

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = state.get("params_ema", state.get("params", state))
    if "no_norm" in state:
        raise SystemExit("SPAN checkpoints without input normalization are not supported")
    with torch.no_grad():
        save_file(fused_weights(state), args.output)


if __name__ == "__main__":
    main()
