"""The DiffVAE decoder's CUDA paths (Triton ``na3d``, compiled stage-5 slab, MLP and context inject)
against the eager torch reference. Needs an NVIDIA GPU and Triton."""

import importlib.util

import pytest
import torch

from jasna.models.ltx_vae import na_triton, video_encoder
from jasna.models.ltx_vae import transformer as T
from jasna.models.ltx_vae.eager_na import na3d
from jasna.models.ltx_vae.loader import encoder_kwargs
from jasna.tensor_utils import compile_static

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or importlib.util.find_spec("triton") is None,
    reason="needs an NVIDIA GPU and Triton",
)


def _qkv(shape: tuple[int, ...]) -> list[torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(0)
    return [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.5 for _ in range(3)
    ]


@pytest.mark.parametrize("kernel", [na_triton.na3d, T._fp16_eager_na3d], ids=["triton", "fp16 eager"])
@pytest.mark.parametrize("kernel_size", [(3, 5, 5), (11, 11, 11)])
@pytest.mark.parametrize("query_w", [(0, 14), (5, 4)])
def test_cuda_na3d_matches_the_eager_windows(kernel, kernel_size, query_w, monkeypatch):
    monkeypatch.setattr(T, "_fused_na3d", kernel)
    attn = T.NeighborhoodAttention3D(128, kernel_size, head_dim=64)
    q, k, v = _qkv((1, 12, 13, 14, 2, 64))
    w_start, w_len = query_w

    fused = T.neighborhood_attention(attn, q[:, :, :, w_start : w_start + w_len].contiguous(), k, v, w_start)
    reference = na3d(q.float(), k.float(), v.float(), kernel_size, 0)[:, :, :, w_start : w_start + w_len]

    torch.testing.assert_close(fused.float(), reference, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("capability, kernel", [((7, 5), T._fp16_eager_na3d), ((8, 6), na_triton.na3d)])
def test_gpus_without_triton_tensor_core_dots_use_the_fp16_eager_na3d(capability, kernel, monkeypatch):
    monkeypatch.setattr(T, "_fused_na3d", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    T.enable_fused_kernels(torch.device("cuda"))
    assert T._fused_na3d is kernel


@pytest.mark.parametrize("kernel, breaks", [(na_triton.na3d, 0), (T._fp16_eager_na3d, 1)], ids=["triton", "fp16 eager"])
def test_na3d_compiles_into_the_slab_with_at_most_one_graph_break(kernel, breaks, monkeypatch):
    monkeypatch.setattr(T, "_fused_na3d", kernel)
    torch.manual_seed(0)
    attn = T.NeighborhoodAttention3D(128, (3, 5, 5), head_dim=64).cuda().bfloat16().eval()
    norm = torch.nn.RMSNorm(128).cuda().bfloat16()
    x = torch.randn(1, 4, 8, 16, 128, device="cuda", dtype=torch.bfloat16)
    scale, shift = torch.randn(2, 1, 1, 1, 1, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    left_halo = torch.randn(1, 4, 8, 2, 128, device="cuda", dtype=torch.bfloat16)
    columns = torch.arange(2, 16, device="cuda")
    w_pos = torch.arange(16, device="cuda", dtype=torch.float32)

    torch._dynamo.reset()
    with torch.inference_mode():
        slab = (attn, norm, left_halo, x, columns, w_pos, scale, shift, 12)
        explanation = torch._dynamo.explain(T._attn_on_w_slab)(*slab)

    assert explanation.graph_count == breaks + 1
    assert explanation.graph_break_count == breaks


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_compiled_diffusion_block_matches_eager(dtype, monkeypatch):
    monkeypatch.setattr(T, "_fused_na3d", None)
    monkeypatch.setattr(T, "_slab_attention", T._attn_on_w_slab)
    monkeypatch.setattr(T, "_mlp_chunk", T.modulated_swiglu)
    monkeypatch.setattr(T, "_add_context", T.add_folded_context)
    torch.manual_seed(0)
    block = T.ChunkedDiffusionNABlock(64, (3, 5, 5), context_channels=16, head_dim=32).cuda().to(dtype).eval()
    upsample = T.LinearPixelShuffleUpsample(32, (2, 2, 2), 2).cuda().to(dtype).eval()
    x = torch.randn(1, 7, 16, 16, 64, device="cuda", dtype=dtype)
    stage4 = torch.randn(1, 4, 8, 8, 32, device="cuda", dtype=dtype)
    modulation = tuple(torch.randn(1, 1, 1, 1, 64, device="cuda", dtype=dtype) * 0.1 for _ in range(7))

    def run() -> torch.Tensor:
        with torch.inference_mode():
            return block(x.clone(), stage4, modulation, upsample, drop_leading_frame=True)

    eager = run()
    T.enable_fused_kernels(torch.device("cuda"))
    T.compile_diffusion_blocks(torch.device("cuda"), dtype)
    compiled = run()

    torch.testing.assert_close(compiled.float(), eager.float(), atol=5e-2, rtol=5e-2)


def test_compiled_channels_last_encoder_matches_eager(monkeypatch):
    config = {"encoder": {"in_channels": 3, "out_channels": 8, "patch_size": 4, "dims": 3, "norm_layer": "pixel_norm",
                          "latent_log_var": "constant", "spatial_padding_mode": "zeros",
                          "blocks": [["res_x", {"num_layers": 1}], ["compress_space_res", {"multiplier": 2}],
                                     ["compress_time_res", {"multiplier": 2}], ["compress_all_res", {"multiplier": 2}],
                                     ["res_x", {"num_layers": 1}]]}}
    torch.manual_seed(0)
    encoder = video_encoder.VideoEncoder(**encoder_kwargs(config)).cuda().bfloat16().eval()
    pixels = torch.rand(1, 3, 17, 64, 96, device="cuda", dtype=torch.bfloat16) * 2 - 1
    monkeypatch.setattr(video_encoder, "_encode_cropped", video_encoder.encode_cropped)
    with torch.inference_mode():
        eager = encoder(pixels)
        encoder.to(memory_format=torch.channels_last_3d)
        monkeypatch.setattr(video_encoder, "_encode_cropped",
                            compile_static(video_encoder.encode_cropped, mode="default", device=torch.device("cuda"),
                                           dtype=torch.bfloat16))
        compiled = encoder(pixels)

    torch.testing.assert_close(compiled.float(), eager.float(), atol=5e-2, rtol=5e-2)


def test_compiled_fp16_encoder_with_float32_tail_survives_large_activations(monkeypatch):
    config = {"encoder": {"in_channels": 3, "out_channels": 8, "patch_size": 4, "dims": 3, "norm_layer": "pixel_norm",
                          "latent_log_var": "constant", "spatial_padding_mode": "zeros",
                          "blocks": [["res_x", {"num_layers": 1}], ["compress_space_res", {"multiplier": 2}],
                                     ["compress_time_res", {"multiplier": 2}], ["compress_all_res", {"multiplier": 2}],
                                     ["res_x", {"num_layers": 1}]]}}
    torch.manual_seed(0)
    encoder = video_encoder.VideoEncoder(**encoder_kwargs(config)).cuda().eval()
    with torch.no_grad():
        encoder.conv_in.conv.weight.mul_(300)
    pixels = torch.rand(1, 3, 17, 64, 96, device="cuda") * 2 - 1
    monkeypatch.setattr(video_encoder, "_encode_cropped", video_encoder.encode_cropped)
    with torch.inference_mode():
        reference = encoder(pixels)
        encoder.to_fp16_with_float32_tail()
        monkeypatch.setattr(video_encoder, "_encode_cropped",
                            compile_static(video_encoder.encode_cropped, mode="default", device=torch.device("cuda"),
                                           dtype=torch.float16))
        compiled = encoder(pixels.half())

    assert compiled.dtype == torch.bfloat16
    torch.testing.assert_close(compiled.float(), reference.float(), atol=5e-2, rtol=5e-2)
