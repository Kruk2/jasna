"""The DiffVAE's AMD paths: the encoder's comfy-kitchen conv3d head against MIOpen. Needs an AMD
GPU with Triton (Linux ROCm)."""

import importlib.util

import pytest
import torch

from jasna.models.ltx_vae import video_encoder

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is None
    or importlib.util.find_spec("triton") is None,
    reason="needs an AMD GPU and Triton",
)


def test_kitchen_resnet_block_matches_miopen(monkeypatch):
    torch.manual_seed(0)
    block = video_encoder.ResnetBlock3D(128).cuda().half().eval().to(memory_format=torch.channels_last_3d)
    x = torch.randn(1, 128, 9, 32, 32, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        reference = block.float()(x.float())
        block.half()
        monkeypatch.setattr(video_encoder, "_kitchen_conv3d", True)
        video_encoder.enable_kitchen_conv3d()
        kitchen = block(x)

    assert kitchen.is_contiguous(memory_format=torch.channels_last_3d)
    torch.testing.assert_close(kitchen.float(), reference, atol=2e-2, rtol=2e-2)
