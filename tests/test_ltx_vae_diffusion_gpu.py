"""The decoder's fused diffusion-block kernels (``diffusion_triton`` and their ahead-of-time build) against
the torch code they replace, and their AMD-only routing. Needs a GPU and Triton; the kernel tests an AMD
GPU (their tiles are sized for RDNA)."""

import importlib.util

import pytest
import torch

from jasna.accelerator import is_amd_device
from jasna.models.ltx_vae import diffusion_hip
from jasna.models.ltx_vae import transformer as T

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None, reason="needs a GPU and Triton"
)


amd_only = pytest.mark.skipif(not torch.cuda.is_available() or not is_amd_device("cuda"), reason="needs an AMD GPU")


def _operands(rows: int):
    generator = torch.Generator(device="cuda").manual_seed(0)

    def randn(*shape, scale=1.0):
        return (torch.randn(shape, device="cuda", generator=generator) * scale).to(torch.bfloat16)

    dim, hidden = diffusion_hip.DIM, diffusion_hip.HIDDEN
    x = randn(rows, dim, scale=2.0)
    return x, randn(dim), randn(dim, scale=0.2), randn(dim, scale=0.2), randn(hidden, dim, scale=dim**-0.5), \
        randn(hidden, dim, scale=dim**-0.5), randn(dim, hidden, scale=hidden**-0.5)


def _reference(x, norm, scale, shift, w_gate, w_up, w_down):
    return x + T.modulated_swiglu(x, norm, scale, shift, torch.cat([w_gate, w_up]), w_down)


@amd_only
@pytest.mark.parametrize("rows", [1, 300, 16_384])
def test_fused_residual_swiglu_matches_the_torch_chunk(rows):
    from jasna.models.ltx_vae import diffusion_triton

    x, *weights = _operands(rows)
    expected = _reference(x, *weights)

    diffusion_triton.residual_swiglu_(x, *weights)

    torch.testing.assert_close(x, expected, atol=6e-2, rtol=2e-2)


@amd_only
def test_the_ahead_of_time_kernel_matches_the_jit_one():
    from jasna.models.ltx_vae import diffusion_triton

    x, *weights = _operands(5000)
    jit = x.clone()
    diffusion_triton.residual_swiglu_(jit, *weights)

    diffusion_hip.residual_swiglu_(x, *weights)

    torch.testing.assert_close(x, jit, atol=0, rtol=0)


def test_amd_routes_the_diffusion_blocks_through_the_fused_kernels_and_nvidia_keeps_torch(monkeypatch):
    from jasna.models.ltx_vae import diffusion_triton

    monkeypatch.setattr(T, "_fused_residual_swiglu", None)
    monkeypatch.setattr(T, "_fused_slab_qkv", None)
    T.enable_fused_kernels(torch.device("cuda"))

    amd = is_amd_device("cuda")
    assert T._fused_residual_swiglu is (diffusion_triton.residual_swiglu_ if amd else None)
    assert T._fused_slab_qkv is (diffusion_triton.slab_qkv if amd else None)


@amd_only
def test_the_fused_diffusion_mlp_covers_every_token_chunk(monkeypatch):
    from jasna.models.ltx_vae import diffusion_triton

    dim = diffusion_hip.DIM
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
        monkeypatch.setattr(T, "_fused_residual_swiglu", diffusion_triton.residual_swiglu_)
        fused = T.residual_modulating_mlp(x.clone(), mlp, norm, scale, shift)

    torch.testing.assert_close(fused, expected, atol=6e-2, rtol=2e-2)


def _slab(core_len: int = 6):
    torch.manual_seed(0)
    attn = T.NeighborhoodAttention3D(diffusion_hip.DIM, (3, 5, 11), head_dim=diffusion_hip.HEAD_DIM).cuda().bfloat16().eval()
    norm = torch.nn.RMSNorm(diffusion_hip.DIM).cuda().bfloat16()
    with torch.no_grad():
        for parameter in (attn.q_norm.weight, attn.k_norm.weight, norm.weight):
            parameter.uniform_(0.5, 1.5)
    x = torch.randn(1, 4, 6, 16, diffusion_hip.DIM, device="cuda", dtype=torch.bfloat16)
    scale, shift = torch.randn(2, 1, 1, 1, 1, diffusion_hip.DIM, device="cuda", dtype=torch.bfloat16) * 0.1
    left_halo = torch.randn(1, 4, 6, 5, diffusion_hip.DIM, device="cuda", dtype=torch.bfloat16)
    columns = torch.arange(3, 3 + core_len + 5, device="cuda").clamp(max=15)
    w_pos = torch.arange(core_len + 10, device="cuda", dtype=torch.float32) - 2
    return attn, norm, left_halo, x, columns, w_pos, scale, shift, core_len


@amd_only
@pytest.mark.parametrize("fused", ["triton", "aot"])
def test_fused_slab_qkv_matches_the_torch_slab(fused, monkeypatch):
    from jasna.models.ltx_vae import diffusion_triton

    monkeypatch.setattr(T, "_fused_na3d", None)
    slab = _slab()
    with torch.inference_mode():
        monkeypatch.setattr(T, "_fused_slab_qkv", None)
        expected = T._attn_on_w_slab(*slab)
        kernel = diffusion_triton.slab_qkv if fused == "triton" else diffusion_hip.slab_qkv
        monkeypatch.setattr(T, "_fused_slab_qkv", kernel)
        actual = T._attn_on_w_slab(*slab)

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=2e-2)


@amd_only
def test_the_fused_slab_qkv_compiles_into_the_slab_without_a_graph_break(monkeypatch):
    from jasna.models.ltx_vae import diffusion_triton

    monkeypatch.setattr(T, "_fused_slab_qkv", diffusion_triton.slab_qkv)
    torch._dynamo.reset()
    with torch.inference_mode():
        explanation = torch._dynamo.explain(T._attn_on_w_slab)(*_slab())

    assert explanation.graph_break_count == 0


def _deterministic_stage(dim: int):
    torch.manual_seed(0)
    attn = T.NeighborhoodAttention3D(dim, (3, 5, 5), head_dim=diffusion_hip.HEAD_DIM).cuda().bfloat16().eval()
    with torch.no_grad():
        for parameter in (attn.q_norm.weight, attn.k_norm.weight):
            parameter.uniform_(0.5, 1.5)
    return attn, torch.randn(1, 5, 6, 9, dim, device="cuda", dtype=torch.bfloat16)


@amd_only
@pytest.mark.parametrize("fused", ["triton", "aot"])
def test_the_512_wide_deterministic_stages_take_the_fused_qkv(fused, monkeypatch):
    from jasna.models.ltx_vae import diffusion_triton

    monkeypatch.setattr(T, "_fused_na3d", None)
    attn, x = _deterministic_stage(512)
    with torch.inference_mode():
        monkeypatch.setattr(T, "_fused_slab_qkv", None)
        expected = attn(x)
        monkeypatch.setattr(T, "_fused_slab_qkv", diffusion_triton.slab_qkv if fused == "triton" else diffusion_hip.slab_qkv)
        actual = attn(x)

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=2e-2)


def test_the_wider_deterministic_stages_keep_the_torch_qkv(monkeypatch):
    monkeypatch.setattr(T, "_fused_na3d", None)
    monkeypatch.setattr(T, "_fused_slab_qkv", lambda *_args: pytest.fail("no fused Q/K/V at width 1024"))
    attn, x = _deterministic_stage(1024)
    with torch.inference_mode():
        assert attn(x).shape == x.shape


@amd_only
@pytest.mark.parametrize("shape", [(1, 128, 121, 128, 128), (2, 70, 3, 5, 9)])
def test_the_ahead_of_time_channels_first_copy_matches_torch(shape):
    from jasna.models.ltx_vae import layout_hip

    x = torch.randn(shape, device="cuda", dtype=torch.float16).contiguous(memory_format=torch.channels_last_3d)

    out = layout_hip.channels_first_bf16(x)

    assert out.is_contiguous()
    assert torch.equal(out, x.to(dtype=torch.bfloat16, memory_format=torch.contiguous_format))
