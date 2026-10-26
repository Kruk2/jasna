"""The decoder's fused diffusion-block kernels (the ``diffusion_triton`` source, JIT here, and its ahead-of-time
build ``diffusion_hip``) against the torch code they replace, and their routing. Needs a GPU and Triton; the
MLP and ahead-of-time tests an AMD GPU (the MLP is fused on AMD only, the bundles are built for RDNA)."""

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


def _jit_slab_qkv(*args):
    from jasna.models.ltx_vae import diffusion_triton

    return diffusion_triton.slab_qkv(*args)


SLAB_QKV_KERNELS = [pytest.param(_jit_slab_qkv, id="jit"), pytest.param(diffusion_hip.slab_qkv, id="aot", marks=amd_only)]


def _operands(rows: int):
    generator = torch.Generator(device="cuda").manual_seed(0)

    def randn(*shape, scale=1.0):
        return (torch.randn(shape, device="cuda", generator=generator) * scale).to(torch.bfloat16)

    dim, hidden = T.DIFFUSION_DIM, T.DIFFUSION_HIDDEN
    x = randn(rows, dim, scale=2.0)
    return x, randn(dim), randn(dim, scale=0.2), randn(dim, scale=0.2), randn(hidden, dim, scale=dim**-0.5), \
        randn(hidden, dim, scale=dim**-0.5), randn(dim, hidden, scale=hidden**-0.5)


def _jit_residual_swiglu_(x, norm_weight, scale, shift, w_gate, w_up, w_down):
    """The ``diffusion_triton`` source compiled by the Triton JIT, launched as ``diffusion_hip`` launches its build."""
    import triton

    from jasna.models.ltx_vae import diffusion_triton

    rows, dim = x.shape
    diffusion_triton._residual_swiglu_kernel[(triton.cdiv(rows, diffusion_hip.SWIGLU_BLOCK_ROWS),)](
        x, norm_weight, scale, shift, w_gate, w_up, w_down, rows,
        DIM=dim, HIDDEN=w_gate.shape[0], BLOCK_ROWS=diffusion_hip.SWIGLU_BLOCK_ROWS,
        BLOCK_HIDDEN=diffusion_hip.SWIGLU_BLOCK_HIDDEN, num_warps=diffusion_hip.SWIGLU_WARPS, num_stages=1,
    )


def _reference(x, norm, scale, shift, w_gate, w_up, w_down):
    return x + T.modulated_swiglu(x, norm, scale, shift, torch.cat([w_gate, w_up]), w_down)


@amd_only
@pytest.mark.parametrize("rows", [1, 300, 16_384])
def test_fused_residual_swiglu_matches_the_torch_chunk(rows):
    x, *weights = _operands(rows)
    expected = _reference(x, *weights)

    diffusion_hip.residual_swiglu_(x, *weights)

    torch.testing.assert_close(x, expected, atol=6e-2, rtol=2e-2)


@amd_only
def test_the_ahead_of_time_kernel_matches_its_source():
    x, *weights = _operands(5000)
    jit = x.clone()
    _jit_residual_swiglu_(jit, *weights)

    diffusion_hip.residual_swiglu_(x, *weights)

    torch.testing.assert_close(x, jit, atol=0, rtol=0)


def test_both_vendors_fuse_the_slab_qkv_and_only_amd_fuses_the_mlp():
    from jasna.models.ltx_vae import diffusion_triton

    kernels = T.decoder_kernels(torch.device("cuda"))

    amd = is_amd_device("cuda")
    assert kernels.residual_swiglu is (diffusion_hip.residual_swiglu_ if amd else None)
    assert kernels.slab_qkv is (diffusion_hip.slab_qkv if amd else diffusion_triton.slab_qkv)


@amd_only
def test_the_fused_diffusion_mlp_covers_every_token_chunk(monkeypatch):
    dim = T.DIFFUSION_DIM
    torch.manual_seed(0)
    mlp = T.SwiGLU(dim, T._swiglu_hidden_dim(dim)).cuda().bfloat16()
    norm = torch.nn.RMSNorm(dim).cuda().bfloat16()
    x = torch.randn(1, 3, 5, 7, dim, device="cuda", dtype=torch.bfloat16)
    scale, shift = torch.randn(2, 1, 1, 1, 1, dim, device="cuda", dtype=torch.bfloat16) * 0.1
    monkeypatch.setattr(T, "SWIGLU_TILE_TOKENS", 32)
    monkeypatch.setattr(T, "_mlp_chunk", T.modulated_swiglu)
    with torch.inference_mode():
        expected = T.residual_modulating_mlp(x.clone(), mlp, norm, scale, shift)
        mlp.kernels = T.DecoderKernels(residual_swiglu=diffusion_hip.residual_swiglu_)
        fused = T.residual_modulating_mlp(x.clone(), mlp, norm, scale, shift)

    torch.testing.assert_close(fused, expected, atol=6e-2, rtol=2e-2)


def _slab(core_len: int = 6):
    torch.manual_seed(0)
    attn = T.NeighborhoodAttention3D(T.DIFFUSION_DIM, (3, 5, 11), head_dim=T.DIFFUSION_HEAD_DIM).cuda().bfloat16().eval()
    norm = torch.nn.RMSNorm(T.DIFFUSION_DIM).cuda().bfloat16()
    with torch.no_grad():
        for parameter in (attn.q_norm.weight, attn.k_norm.weight, norm.weight):
            parameter.uniform_(0.5, 1.5)
    x = torch.randn(1, 4, 6, 16, T.DIFFUSION_DIM, device="cuda", dtype=torch.bfloat16)
    scale, shift = torch.randn(2, 1, 1, 1, 1, T.DIFFUSION_DIM, device="cuda", dtype=torch.bfloat16) * 0.1
    left_halo = torch.randn(1, 4, 6, 5, T.DIFFUSION_DIM, device="cuda", dtype=torch.bfloat16)
    columns = torch.arange(3, 3 + core_len + 5, device="cuda").clamp(max=15)
    w_pos = torch.arange(core_len + 10, device="cuda", dtype=torch.float32) - 2
    return attn, norm, left_halo, x, columns, w_pos, scale, shift, core_len


@pytest.mark.parametrize("kernel", SLAB_QKV_KERNELS)
def test_fused_slab_qkv_matches_the_torch_slab(kernel):
    slab = _slab()
    with torch.inference_mode():
        expected = T._attn_on_w_slab(*slab)
        slab[0].kernels = T.DecoderKernels(slab_qkv=kernel)
        actual = T._attn_on_w_slab(*slab)

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=2e-2)


@pytest.mark.parametrize("kernel", SLAB_QKV_KERNELS)
def test_the_fused_slab_qkv_compiles_into_the_slab_without_a_graph_break(kernel):
    slab = _slab()
    slab[0].kernels = (T.decoder_kernels(torch.device("cuda")) if kernel is _jit_slab_qkv
                       else T.DecoderKernels(slab_qkv=kernel))
    torch._dynamo.reset()
    with torch.inference_mode():
        explanation = torch._dynamo.explain(T._attn_on_w_slab)(*slab)

    assert explanation.graph_break_count == 0


def _deterministic_stage(dim: int):
    torch.manual_seed(0)
    attn = T.NeighborhoodAttention3D(dim, (3, 5, 5), head_dim=T.DIFFUSION_HEAD_DIM).cuda().bfloat16().eval()
    with torch.no_grad():
        for parameter in (attn.q_norm.weight, attn.k_norm.weight):
            parameter.uniform_(0.5, 1.5)
    return attn, torch.randn(1, 5, 6, 9, dim, device="cuda", dtype=torch.bfloat16)


@pytest.mark.parametrize("kernel", SLAB_QKV_KERNELS)
def test_the_512_wide_deterministic_stages_take_the_fused_qkv(kernel):
    attn, x = _deterministic_stage(512)
    with torch.inference_mode():
        expected = attn(x)
        attn.kernels = T.DecoderKernels(slab_qkv=kernel)
        actual = attn(x)

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=2e-2)


def test_the_wider_deterministic_stages_keep_the_torch_qkv():
    attn, x = _deterministic_stage(1024)
    attn.kernels = T.DecoderKernels(slab_qkv=lambda *_args: pytest.fail("no fused Q/K/V at width 1024"))
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
