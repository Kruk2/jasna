# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import re
import warnings

import torch

from jasna.models.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGanNet

warnings.filterwarnings(
    "ignore",
    message="The given buffer is not writable, and PyTorch does not support non-writable tensors",
    module="torch",
)

_INFERENCE_GENERATOR_PREFIXES = ("generator_ema.", "generator.")


def inference_generator_state(checkpoint: dict) -> dict[str, torch.Tensor]:
    """The generator weights a restoration checkpoint is run with: its EMA generator's, or the
    trained generator's when it has no EMA copy. Accepts a nested ``state_dict`` and
    ``module.``-prefixed (DataParallel) keys."""
    state = checkpoint.get("state_dict", checkpoint)
    state = {re.sub(r"^module\.", "", key): value for key, value in state.items()}
    for prefix in _INFERENCE_GENERATOR_PREFIXES:
        weights = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
        if weights:
            return weights
    raise ValueError("The checkpoint holds no BasicVSR++ generator weights")


def load_model(checkpoint_path: str, device: torch.device, fp16: bool) -> BasicVSRPlusPlusGanNet:
    """The BasicVSR++ restoration network of a Lada GAN checkpoint, in eval mode on ``device``."""
    generator = BasicVSRPlusPlusGanNet(mid_channels=64, num_blocks=15, spynet_pretrained=None)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    generator.load_state_dict(inference_generator_state(checkpoint))
    generator = generator.to(device).eval()
    return generator.half() if fp16 else generator
