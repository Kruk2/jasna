"""The decoder's fused residual SwiGLU kernel (``swiglu_triton`` and its ahead-of-time build) against the
torch chunk it replaces, and its AMD-only routing. Needs a GPU and Triton; the kernel tests an AMD GPU
(its tiles are sized for RDNA)."""

import importlib.util

import pytest
import torch

from jasna.accelerator import is_amd_device
from jasna.models.ltx_vae import swiglu_hip
from jasna.models.ltx_vae import transformer as T

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None, reason="needs a GPU and Triton"
)


amd_only = pytest.mark.skipif(not torch.cuda.is_available() or not is_amd_device("cuda"), reason="needs an AMD GPU")


def _operands(rows: int):
    generator = torch.Generator(device="cuda").manual_seed(0)

    def randn(*shape, scale=1.0):
        return (torch.randn(shape, device="cuda", generator=generator) * scale).to(torch.bfloat16)

    dim, hidden = swiglu_hip.SHAPES[0]
    x = randn(rows, dim, scale=2.0)
    return x, randn(dim), randn(dim, scale=0.2), randn(dim, scale=0.2), randn(hidden, dim, scale=dim**-0.5), \
        randn(hidden, dim, scale=dim**-0.5), randn(dim, hidden, scale=hidden**-0.5)


def _reference(x, norm, scale, shift, w_gate, w_up, w_down):
    return x + T.modulated_swiglu(x, norm, scale, shift, torch.cat([w_gate, w_up]), w_down)


@amd_only
@pytest.mark.parametrize("rows", [1, 300, 16_384])
def test_fused_residual_swiglu_matches_the_torch_chunk(rows):
    from jasna.models.ltx_vae import swiglu_triton

    x, *weights = _operands(rows)
    expected = _reference(x, *weights)

    swiglu_triton.residual_swiglu_(x, *weights)

    torch.testing.assert_close(x, expected, atol=6e-2, rtol=2e-2)


@amd_only
def test_the_ahead_of_time_kernel_matches_the_jit_one():
    from jasna.models.ltx_vae import swiglu_triton

    x, *weights = _operands(5000)
    jit = x.clone()
    swiglu_triton.residual_swiglu_(jit, *weights)

    swiglu_hip.residual_swiglu_(x, *weights)

    torch.testing.assert_close(x, jit, atol=0, rtol=0)


def test_amd_routes_the_diffusion_mlp_through_the_fused_kernel_and_nvidia_keeps_the_torch_chunk(monkeypatch):
    from jasna.models.ltx_vae import swiglu_triton

    monkeypatch.setattr(T, "_fused_residual_swiglu", None)
    T.enable_fused_kernels(torch.device("cuda"))

    expected = swiglu_triton.residual_swiglu_ if is_amd_device("cuda") else None
    assert T._fused_residual_swiglu is expected


@amd_only
def test_the_fused_diffusion_mlp_covers_every_token_chunk(monkeypatch):
    from jasna.models.ltx_vae import swiglu_triton

    dim = swiglu_hip.SHAPES[0][0]
    torch.manual_seed(0)
    mlp = T.SwiGLU(dim, T._swiglu_hidden_dim(dim)).cuda().bfloat16()
    norm = torch.nn.RMSNorm(dim).cuda().bfloat16()
    x = torch.randn(1, 3, 5, 7, dim, device="cuda", dtype=torch.bfloat16)
    scale, shift = torch.randn(2, 1, 1, 1, 1, dim, device="cuda", dtype=torch.bfloat16) * 0.1
    monkeypatch.setattr(T, "SWIGLU_TILE_TOKENS", 32)
    monkeypatch.setattr(T, "_mlp_chunk", T.modulated_swiglu)
    monkeypatch.setattr(T, "_fused_residual_swiglu", None)
    with torch.inference_mode():
        expected = T.residual_modulating_mlp(x.clone(), mlp, norm, scale, shift)
        monkeypatch.setattr(T, "_fused_residual_swiglu", swiglu_triton.residual_swiglu_)
        fused = T.residual_modulating_mlp(x.clone(), mlp, norm, scale, shift)

    torch.testing.assert_close(fused, expected, atol=6e-2, rtol=2e-2)
