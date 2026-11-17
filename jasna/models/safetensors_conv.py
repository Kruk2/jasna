"""Building inference networks straight from a safetensors state dict."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


def conv_from_weights(weights: Mapping[str, torch.Tensor], name: str) -> nn.Conv2d:
    """The same-padded ``Conv2d`` holding ``weights[name + ".weight"]`` and its bias."""
    weight = weights[f"{name}.weight"]
    out_channels, in_channels, kernel, _ = weight.shape
    conv = nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2, device="meta")
    conv.weight = nn.Parameter(weight, requires_grad=False)
    conv.bias = nn.Parameter(weights[f"{name}.bias"], requires_grad=False)
    return conv
