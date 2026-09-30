"""The DiffVAE decoder's CUDA paths (comfy-kitchen ``na3d``, compiled stage-5 slab and MLP)
against the eager torch reference. Needs an NVIDIA GPU, comfy-kitchen and Triton."""

import importlib.util

import pytest
import torch

from jasna.models.ltx_vae import transformer as T
from jasna.models.ltx_vae.eager_na import na3d

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or importlib.util.find_spec("comfy_kitchen") is None
    or importlib.util.find_spec("triton") is None,
    reason="needs an NVIDIA GPU, comfy-kitchen and Triton",
)


def _qkv(shape: tuple[int, ...]) -> list[torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(0)
    return [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.5 for _ in range(3)
    ]


@pytest.mark.parametrize("kernel_size", [(3, 5, 5), (11, 11, 11)])
def test_kitchen_na3d_matches_the_eager_windows(kernel_size, monkeypatch):
    monkeypatch.setattr(T, "_fused_na3d", None)
    assert T.enable_fused_attention() is True
    attn = T.NeighborhoodAttention3D(128, kernel_size, head_dim=64)
    q, k, v = _qkv((1, 12, 13, 14, 2, 64))

    fused = T.neighborhood_attention(attn, q, k, v)
    reference = na3d(q.float(), k.float(), v.float(), kernel_size=kernel_size)

    torch.testing.assert_close(fused.float(), reference, atol=2e-2, rtol=2e-2)


def test_compiled_diffusion_block_matches_eager(monkeypatch):
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
    assert T.compile_diffusion_blocks() is True
    compiled = run()

    torch.testing.assert_close(compiled.float(), eager.float(), atol=5e-2, rtol=5e-2)
