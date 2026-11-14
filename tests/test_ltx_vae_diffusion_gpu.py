"""The decoder's fused diffusion-block kernels (the ``diffusion_triton`` source, JIT here, and its ahead-of-time
build ``diffusion_hip``) against the torch code they replace, and their routing. Needs a GPU and Triton; the
ahead-of-time tests an AMD GPU (the bundles are built for RDNA)."""

import copy
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
nvidia_only = pytest.mark.skipif(not torch.cuda.is_available() or is_amd_device("cuda"), reason="needs an NVIDIA GPU")


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


def _reference(x, norm, scale, shift, w_gate, w_up, w_down):
    return x + T.modulated_swiglu(x, norm, scale, shift, torch.cat([w_gate, w_up]), w_down)


def _fused_residual_swiglu_():
    return T.decoder_kernels(torch.device("cuda")).residual_swiglu


@pytest.mark.parametrize("rows", [1, 300, 16_384])
def test_fused_residual_swiglu_matches_the_torch_chunk(rows):
    x, *weights = _operands(rows)
    expected = _reference(x, *weights)

    _fused_residual_swiglu_()(x, *weights)

    torch.testing.assert_close(x, expected, atol=6e-2, rtol=2e-2)


def test_both_vendors_fuse_the_slab_qkv_mlp_attention_projection_and_context_inject():
    from jasna.models.ltx_vae import diffusion_triton

    kernels = T.decoder_kernels(torch.device("cuda"))

    amd = is_amd_device("cuda")
    assert kernels.residual_swiglu is (diffusion_hip.residual_swiglu_ if amd else diffusion_triton.residual_swiglu_)
    assert kernels.proj_residual is (diffusion_hip.proj_residual_ if amd else diffusion_triton.proj_residual_)
    assert kernels.context_add is (diffusion_hip.context_add_ if amd else diffusion_triton.context_add_)
    assert kernels.slab_qkv is (diffusion_hip.slab_qkv if amd else diffusion_triton.slab_qkv)


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
        mlp.kernels = T.DecoderKernels(residual_swiglu=_fused_residual_swiglu_())
        fused = T.residual_modulating_mlp(x.clone(), mlp, norm, scale, shift)

    torch.testing.assert_close(fused, expected, atol=6e-2, rtol=2e-2)


def _slab(core_len: int = 6):
    torch.manual_seed(0)
    attn = T.NeighborhoodAttention3D(T.DIFFUSION_DIM, (3, 5, 5), head_dim=T.DIFFUSION_HEAD_DIM).cuda().bfloat16().eval()
    norm = torch.nn.RMSNorm(T.DIFFUSION_DIM).cuda().bfloat16()
    with torch.no_grad():
        for parameter in (attn.q_norm.weight, attn.k_norm.weight, norm.weight):
            parameter.uniform_(0.5, 1.5)
    x = torch.randn(1, 4, 6, 16, T.DIFFUSION_DIM, device="cuda", dtype=torch.bfloat16)
    scale, shift = torch.randn(2, 1, 1, 1, 1, T.DIFFUSION_DIM, device="cuda", dtype=torch.bfloat16) * 0.1
    left_halo = torch.randn(1, 4, 6, 2, T.DIFFUSION_DIM, device="cuda", dtype=torch.bfloat16)
    columns = torch.arange(3, 3 + core_len + 2, device="cuda").clamp(max=15)
    w_pos = torch.arange(core_len + 4, device="cuda", dtype=torch.float32) - 2
    return attn, norm, left_halo, x, columns, w_pos, scale, shift, core_len, True


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
    slab[0].kernels = T.decoder_kernels(torch.device("cuda"))._replace(slab_qkv=kernel)
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


def test_fused_projection_adds_into_its_slab_columns_only_as_the_torch_projection_does():
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(1, 3, 5, 40, 256, device="cuda", generator=generator).bfloat16()
    out = torch.randn(1, 3, 5, 9, 256, device="cuda", generator=generator).bfloat16()
    weight = (torch.randn(256, 256, device="cuda", generator=generator) / 16).bfloat16()
    bias = torch.randn(256, device="cuda", generator=generator).bfloat16()
    expected = x.clone()
    expected[:, :, :, 20:29] += torch.nn.functional.linear(out, weight, bias)

    T.decoder_kernels(torch.device("cuda")).proj_residual(x, out, weight, bias, 20)

    torch.testing.assert_close(x, expected, atol=0.04, rtol=0)


def test_the_chunked_attention_residual_takes_the_fused_projection():
    torch.manual_seed(0)
    attn = T.NeighborhoodAttention3D(T.DIFFUSION_DIM, (3, 5, 5), head_dim=T.DIFFUSION_HEAD_DIM).cuda().bfloat16()
    norm = torch.nn.RMSNorm(T.DIFFUSION_DIM).cuda().bfloat16()
    x = torch.randn(1, 4, 6, 40, T.DIFFUSION_DIM, device="cuda", dtype=torch.bfloat16)
    scale, shift = torch.randn(2, 1, 1, 1, 1, T.DIFFUSION_DIM, device="cuda", dtype=torch.bfloat16) * 0.1
    with torch.inference_mode():
        expected = T.w_chunked_attention_residual(x.clone(), attn, norm, scale, shift)
        attn.kernels = T.DecoderKernels(proj_residual=T.decoder_kernels(torch.device("cuda")).proj_residual)
        actual = T.w_chunked_attention_residual(x.clone(), attn, norm, scale, shift)

    torch.testing.assert_close(actual, expected, atol=0.04, rtol=0)


@pytest.mark.parametrize("dim", [512, 1024])
def test_the_deterministic_blocks_take_the_fused_mlp(dim):
    torch.manual_seed(0)
    block = T.NABlock(dim, (3, 5, 5), head_dim=T.DIFFUSION_HEAD_DIM).cuda().bfloat16().eval()
    with torch.no_grad():
        block.norm2.weight.uniform_(0.5, 1.5)
    x = torch.randn(1, 3, 6, 7, dim, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = block(x)
        block.mlp.kernels = T.DecoderKernels(residual_swiglu=_fused_residual_swiglu_())
        actual = block(x)

    torch.testing.assert_close(actual, expected, atol=6e-2, rtol=2e-2)


def _nvidia_launches(fp16_accumulate: bool):
    """Every NVIDIA launch entry's configs with fp32 or fp16 accumulation: each config must work both ways."""
    from jasna.models.ltx_vae import diffusion_triton

    entries = [*diffusion_triton.NVIDIA_LAUNCH.values(), diffusion_triton.NVIDIA_DEFAULT_LAUNCH]
    return [entry._replace(fp16_accumulate=fp16_accumulate) for entry in entries]


def _mlp_operands(dim: int):
    generator = torch.Generator(device="cuda").manual_seed(0)
    hidden = T._swiglu_hidden_dim(dim)

    def randn(*shape, scale=1.0):
        return (torch.randn(shape, device="cuda", generator=generator) * scale).to(torch.bfloat16)

    return randn(300, dim, scale=2.0), (randn(dim), randn(dim, scale=0.2), randn(dim, scale=0.2),
                                        randn(hidden, dim, scale=dim**-0.5), randn(hidden, dim, scale=dim**-0.5),
                                        randn(dim, hidden, scale=hidden**-0.5))


def _mean_distance(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return (actual.float() - reference.float()).abs().mean().item()


@nvidia_only
@pytest.mark.parametrize("launch", _nvidia_launches(False) + _nvidia_launches(True))
@pytest.mark.parametrize("dim", T.FUSED_MLP_DIMS)
def test_every_nvidia_launch_config_computes_the_torch_mlp(monkeypatch, launch, dim):
    from jasna.models.ltx_vae import diffusion_triton

    x, weights = _mlp_operands(dim)
    expected = _reference(x, *weights)
    monkeypatch.setattr(diffusion_triton, "nvidia_launch", lambda device: launch)

    diffusion_triton.residual_swiglu_(x, *weights)

    torch.testing.assert_close(x, expected, atol=6e-2, rtol=2e-2)


@nvidia_only
@pytest.mark.parametrize("launch", _nvidia_launches(True))
@pytest.mark.parametrize("dim", T.FUSED_MLP_DIMS)
def test_an_fp16_accumulating_mlp_is_as_close_to_an_fp32_mlp_as_the_bf16_torch_mlp(monkeypatch, launch, dim):
    from jasna.models.ltx_vae import diffusion_triton

    x, weights = _mlp_operands(dim)
    fp32 = _reference(x.float(), *(weight.float() for weight in weights))
    bf16 = _reference(x, *weights)
    monkeypatch.setattr(diffusion_triton, "nvidia_launch", lambda device: launch)

    diffusion_triton.residual_swiglu_(x, *weights)

    assert _mean_distance(x, fp32) <= 1.02 * _mean_distance(bf16, fp32)


def _projection_operands():
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(1, 3, 5, 40, 256, device="cuda", generator=generator).bfloat16()
    out = torch.randn(1, 3, 5, 9, 256, device="cuda", generator=generator).bfloat16()
    weight = (torch.randn(256, 256, device="cuda", generator=generator) / 16).bfloat16()
    bias = torch.randn(256, device="cuda", generator=generator).bfloat16()
    return x, out, weight, bias


def _projected(x, out, weight, bias):
    expected = x.clone()
    expected[:, :, :, 20:29] += torch.nn.functional.linear(out, weight, bias)
    return expected


@nvidia_only
@pytest.mark.parametrize("launch", _nvidia_launches(False) + _nvidia_launches(True))
@pytest.mark.parametrize("out_dtype", (torch.bfloat16, torch.float16))
def test_every_nvidia_launch_config_computes_the_torch_projection_of_the_rounded_attention_output(
    monkeypatch, launch, out_dtype
):
    from jasna.models.ltx_vae import diffusion_triton

    x, out, weight, bias = _projection_operands()
    expected = _projected(x, out, weight, bias)
    monkeypatch.setattr(diffusion_triton, "nvidia_launch", lambda device: launch)

    diffusion_triton.proj_residual_(x, out.to(out_dtype), weight, bias, 20)

    torch.testing.assert_close(x, expected, atol=0, rtol=0)


def _context(drop_leading_frame: bool):
    torch.manual_seed(0)
    upsample = T.LinearPixelShuffleUpsample(T.DIFFUSION_CONTEXT_DIM, T.DIFFUSION_CONTEXT_STRIDE, 2).cuda().bfloat16()
    context_proj = torch.nn.Linear(T.DIFFUSION_DIM, T.DIFFUSION_DIM).cuda().bfloat16()
    stage4 = torch.randn(1, 3, 4, 9, T.DIFFUSION_CONTEXT_DIM, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(1, 5 if drop_leading_frame else 6, 8, 17, T.DIFFUSION_DIM, device="cuda", dtype=torch.bfloat16)
    return x, stage4, upsample, context_proj


@pytest.mark.parametrize("drop_leading_frame", (True, False))
def test_the_fused_context_inject_adds_what_the_torch_inject_adds(drop_leading_frame):
    x, stage4, upsample, context_proj = _context(drop_leading_frame)
    with torch.inference_mode():
        expected = T.inject_deferred_context(x.clone(), stage4, upsample, context_proj, None,
                                             drop_leading_frame=drop_leading_frame)
        actual = T.inject_deferred_context(x.clone(), stage4, upsample, context_proj,
                                           T.decoder_kernels(torch.device("cuda")).context_add,
                                           drop_leading_frame=drop_leading_frame)

    torch.testing.assert_close(actual, expected, atol=0.04, rtol=0)


@nvidia_only
@pytest.mark.parametrize("launch", _nvidia_launches(False))
def test_every_nvidia_launch_config_computes_the_torch_context_inject(monkeypatch, launch):
    from jasna.models.ltx_vae import diffusion_triton

    x, stage4, upsample, context_proj = _context(True)
    with torch.inference_mode():
        expected = T.inject_deferred_context(x.clone(), stage4, upsample, context_proj, None, drop_leading_frame=True)
        monkeypatch.setattr(diffusion_triton, "nvidia_launch", lambda device: launch)
        actual = T.inject_deferred_context(x.clone(), stage4, upsample, context_proj, diffusion_triton.context_add_,
                                           drop_leading_frame=True)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@nvidia_only
@pytest.mark.parametrize("launch", _nvidia_launches(True))
def test_an_fp16_accumulating_context_inject_is_as_close_to_an_fp32_one_as_the_bf16_torch_inject(
    monkeypatch, launch
):
    from jasna.models.ltx_vae import diffusion_triton

    x, stage4, upsample, context_proj = _context(True)
    with torch.inference_mode():
        fp32 = T.inject_deferred_context(x.float(), stage4.float(), copy.deepcopy(upsample).float(),
                                         copy.deepcopy(context_proj).float(), None, drop_leading_frame=True)
        bf16 = T.inject_deferred_context(x.clone(), stage4, upsample, context_proj, None, drop_leading_frame=True)
        monkeypatch.setattr(diffusion_triton, "nvidia_launch", lambda device: launch)
        actual = T.inject_deferred_context(x.clone(), stage4, upsample, context_proj, diffusion_triton.context_add_,
                                           drop_leading_frame=True)

    assert _mean_distance(actual, fp32) <= 1.02 * _mean_distance(bf16, fp32)


def _in_fp32(part):
    return copy.deepcopy(part).float() if isinstance(part, torch.nn.Module) or (
        isinstance(part, torch.Tensor) and part.is_floating_point()) else part


@nvidia_only
@pytest.mark.parametrize("launch", _nvidia_launches(True))
def test_an_fp16_accumulating_slab_qkv_is_as_close_to_an_fp32_slab_as_the_bf16_torch_slab(monkeypatch, launch):
    from jasna.models.ltx_vae import diffusion_triton

    slab = _slab()
    with torch.inference_mode():
        fp32 = T._attn_on_w_slab(*map(_in_fp32, slab))
        bf16 = T._attn_on_w_slab(*slab)
        monkeypatch.setattr(diffusion_triton, "nvidia_launch", lambda device: launch)
        slab[0].kernels = T.DecoderKernels(slab_qkv=diffusion_triton.slab_qkv)
        actual = T._attn_on_w_slab(*slab)

    assert _mean_distance(actual, fp32) <= 1.02 * _mean_distance(bf16, fp32)


def test_the_2048_wide_stage_keeps_the_torch_mlp():
    block = T.NABlock(2048, (3, 5, 5), head_dim=T.DIFFUSION_HEAD_DIM).cuda().bfloat16().eval()
    block.mlp.kernels = T.DecoderKernels(residual_swiglu=lambda *_args: pytest.fail("no fused MLP at width 2048"))
    with torch.inference_mode():
        out = block(torch.randn(1, 3, 6, 7, 2048, device="cuda", dtype=torch.bfloat16))

    assert out.shape == (1, 3, 6, 7, 2048)
