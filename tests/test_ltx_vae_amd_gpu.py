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
    ((9, 9, 9), (1, 24, 20, 30, 4, 64), (5, 20)),
    ((9, 9, 9), (1, 12, 13, 14, 2, 64), (0, 14)),
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


def test_kitchen_resnet_block_matches_miopen():
    torch.manual_seed(0)
    block = video_encoder.ResnetBlock3D(128).cuda().half().eval().to(memory_format=torch.channels_last_3d)
    x = torch.randn(1, 128, 9, 32, 32, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        reference = block.float()(x.float())
        block.half()
        import comfy_kitchen  # noqa: F401

        for conv in (block.conv1, block.conv2):
            conv.kitchen = True
        kitchen = block(x)

    assert kitchen.is_contiguous(memory_format=torch.channels_last_3d)
    torch.testing.assert_close(kitchen.float(), reference, atol=2e-2, rtol=2e-2)


SHIPPED_ENCODER_BLOCKS = [
    ["res_x", {"num_layers": 2}], ["compress_space_res", {"multiplier": 2}], ["res_x", {"num_layers": 2}],
    ["compress_time_res", {"multiplier": 2}], ["res_x", {"num_layers": 2}], ["compress_all_res", {"multiplier": 2}],
    ["res_x", {"num_layers": 2}], ["compress_all_res", {"multiplier": 1}], ["res_x", {"num_layers": 2}],
]


@pytest.mark.parametrize("size", [(17, 64, 96), (9, 96, 64)])
def test_the_prebuilt_encoder_is_as_close_to_fp32_as_the_eager_one(size):
    import copy

    from jasna.models.ltx_vae import encoder_hip

    torch.manual_seed(0)
    exact = video_encoder.VideoEncoder(3, 128, SHIPPED_ENCODER_BLOCKS, 4).cuda().eval()
    encoder = copy.deepcopy(exact).bfloat16()
    encoder.to_fp16_head(video_encoder.AMD_KITCHEN_DOWN_BLOCKS)
    encoder.run_fp16_convs_on_kitchen()
    encoder.to(memory_format=torch.channels_last_3d)
    pixels = torch.rand(1, 3, *size, device="cuda") * 2 - 1
    with torch.inference_mode():
        reference = video_encoder.encode_cropped(exact, pixels)
        eager = video_encoder.encode_cropped(encoder, pixels.half()).float()
        prebuilt = encoder_hip.encode_prebuilt(encoder, pixels.half()).float()

    def distance(latent):
        return ((latent - reference).norm() / reference.norm()).item()

    assert prebuilt.shape == reference.shape
    assert distance(prebuilt) < 1.1 * distance(eager)
