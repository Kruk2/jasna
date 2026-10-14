"""The DiffVAE decoder's CUDA paths (Triton ``na3d``, compiled stage-5 slab and MLP) against the
eager torch reference. Needs an NVIDIA GPU and Triton."""

import importlib.util

import pytest
import torch

from jasna.models.ltx_vae import na_triton
from jasna.models.ltx_vae import transformer as T
from jasna.models.ltx_vae.eager_na import na3d

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
    T.enable_fused_attention(torch.device("cuda"))
    assert T._fused_na3d is kernel


@pytest.mark.parametrize("kernel, breaks", [(na_triton.na3d, 0), (T._fp16_eager_na3d, 1)], ids=["triton", "fp16 eager"])
def test_na3d_compiles_into_the_slab_with_at_most_one_graph_break(kernel, breaks, monkeypatch):
    monkeypatch.setattr(T, "_fused_na3d", kernel)
    torch.manual_seed(0)
    attn = T.NeighborhoodAttention3D(128, (3, 5, 5), head_dim=64).cuda().bfloat16().eval()
    norm = torch.nn.RMSNorm(128).cuda().bfloat16()
    x = torch.randn(1, 4, 8, 16, 128, device="cuda", dtype=torch.bfloat16)
    scale, shift = torch.randn(2, 1, 1, 1, 1, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    w_pos = torch.arange(16, device="cuda", dtype=torch.float32)

    torch._dynamo.reset()
    with torch.inference_mode():
        explanation = torch._dynamo.explain(T._attn_on_w_slab)(attn, norm, x, w_pos, scale, shift, 12)

    assert explanation.graph_count == breaks + 1
    assert explanation.graph_break_count == breaks


def test_compiled_diffusion_block_matches_eager(monkeypatch):
    monkeypatch.setattr(T, "_fused_na3d", None)
    monkeypatch.setattr(T, "_slab_attention", T._attn_on_w_slab)
    monkeypatch.setattr(T, "_mlp_chunk", T.modulated_swiglu)
    torch.manual_seed(0)
    block = T.ChunkedDiffusionNABlock(64, (3, 5, 5), context_channels=16, head_dim=32).cuda().bfloat16().eval()
    upsample = T.LinearPixelShuffleUpsample(32, (2, 2, 2), 2).cuda().bfloat16().eval()
    x = torch.randn(1, 7, 16, 16, 64, device="cuda", dtype=torch.bfloat16)
    stage4 = torch.randn(1, 4, 8, 8, 32, device="cuda", dtype=torch.bfloat16)
    modulation = tuple(torch.randn(1, 1, 1, 1, 64, device="cuda", dtype=torch.bfloat16) * 0.1 for _ in range(7))

    def run() -> torch.Tensor:
        with torch.inference_mode():
            return block(x.clone(), stage4, modulation, upsample, drop_leading_frame=True)

    eager = run()
    T.enable_fused_attention(torch.device("cuda"))
    T.compile_diffusion_blocks(torch.device("cuda"))
    compiled = run()

    torch.testing.assert_close(compiled.float(), eager.float(), atol=5e-2, rtol=5e-2)
