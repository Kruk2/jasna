"""The DiffVAE's AMD paths: the ahead-of-time NA kernel against the JIT one, and the encoder's
comfy-kitchen conv3d head against MIOpen. Needs an AMD GPU with Triton (Linux ROCm)."""

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


@pytest.mark.parametrize("kernel_size, shape, query_w", [
    ((11, 11, 11), (1, 24, 20, 30, 4, 64), (5, 20)),
    ((11, 11, 11), (1, 12, 13, 14, 2, 64), (0, 14)),
    ((3, 7, 7), (1, 9, 16, 16, 8, 64), (0, 16)),
    ((3, 5, 5), (2, 5, 9, 12, 3, 64), (2, 7)),
])
def test_ahead_of_time_na3d_matches_the_jit_kernel(kernel_size, shape, query_w):
    from jasna.models.ltx_vae import na_hip, na_triton

    generator = torch.Generator(device="cuda").manual_seed(0)
    k, v, q_full = (torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(3))
    w_start, w_len = query_w
    q = q_full[:, :, :, w_start : w_start + w_len].contiguous() * 0.125

    aot = na_hip.na3d(q, k, v, list(kernel_size), w_start)
    jit = na_triton.na3d(q, k, v, list(kernel_size), w_start)

    torch.testing.assert_close(aot, jit, atol=1e-2, rtol=1e-2)


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
