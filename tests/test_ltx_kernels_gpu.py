"""The Triton kernels for the LTX quantized linears against the torch path they replace.
Needs a consumer Blackwell GPU (SM 12.x) and Triton."""

from __future__ import annotations

import importlib.util

import pytest
import torch
import torch.nn.functional as F

from jasna.ltx import transformer as T

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or torch.cuda.get_device_capability()[0] != 12
    or importlib.util.find_spec("triton") is None,
    reason="needs a consumer Blackwell GPU and Triton",
)


@pytest.fixture(scope="module")
def kernels():
    assert T.enable_quant_kernels(torch.device("cuda"))
    from jasna.ltx import kernels

    return kernels


def _activations(rows: int, cols: int, generator: torch.Generator) -> torch.Tensor:
    x = torch.randn(rows, cols, generator=generator)
    x[:, torch.randperm(cols, generator=generator)[:4]] *= 40
    return x.to("cuda", torch.bfloat16)


def test_rotate_quantize_matches_the_torch_reference(kernels):
    generator = torch.Generator().manual_seed(0)
    for rows, cols in ((300, 4096), (64, 16384)):
        x = _activations(rows, cols, generator)
        codes, scales = kernels.rotate_quantize(x)
        expected_codes, expected_scales = T.quantize_rows(T.rotate(x))
        differ = codes != expected_codes[:rows]
        assert differ.float().mean() < 1e-4
        assert (codes.int() - expected_codes[:rows].int()).abs().max() <= 1
        assert torch.allclose(scales, expected_scales[:rows], rtol=1e-6, atol=0)


def test_int8_scaled_mm_is_bit_exact_with_the_reference_rescale(kernels, monkeypatch):
    monkeypatch.setattr(T, "_quant_kernels", None)
    generator = torch.Generator().manual_seed(0)
    # 8192 x 4096 leaves a mostly idle last wave, so it takes the split-K tail path
    for rows, out_features in ((300, 1000), (300, 32), (8192, 4096)):
        x = _activations(rows, 4096, generator)
        codes, act_scales = T.quantize_rows(T.rotate(x))
        codes, act_scales = codes[:rows].contiguous(), act_scales[:rows].contiguous()
        qdata = torch.randint(-127, 128, (out_features, 4096), generator=generator, dtype=torch.int8).cuda()
        scales = torch.rand(out_features, generator=generator).cuda() * 4e-4
        bias = torch.randn(out_features, generator=generator).to("cuda", torch.bfloat16)
        for b in (bias, None):
            expected = T.int8_linear(x, qdata, scales, b)
            assert torch.equal(kernels.int8_scaled_mm(codes, act_scales, qdata, scales, b, False), expected)
        gelu = kernels.int8_scaled_mm(codes, act_scales, qdata, scales, bias, True)
        expected = F.gelu(T.int8_linear(x, qdata, scales, bias), approximate="tanh")
        assert torch.allclose(gelu, expected, rtol=2**-7, atol=2**-7)


def test_fused_norm_modulation_and_feed_forward(kernels):
    generator = torch.Generator().manual_seed(0)
    x = _activations(256, 1024, generator)
    shift, scale, gate = (torch.randn(3, 2, 1024, generator=generator) * 0.3).to("cuda", torch.bfloat16)
    halves = F.rms_norm(x.float(), (1024,), eps=T.EPS).view(2, 128, 1024)
    modulated = (halves * (1 + scale.float()[:, None]) + shift.float()[:, None]).view(256, 1024).to(torch.bfloat16)
    codes, scales = kernels.modulate_rotate_quantize(x, shift, scale)
    expected_codes, expected_scales = kernels.rotate_quantize(modulated)
    assert (codes != expected_codes).float().mean() < 1e-4
    assert torch.allclose(scales, expected_scales, rtol=1e-2, atol=0)

    def weights(out_features, in_features):
        qdata = torch.randint(-127, 128, (out_features, in_features), generator=generator, dtype=torch.int8)
        return qdata.cuda(), torch.full((out_features,), 4e-4, device="cuda"), None

    up, down = weights(4096, 1024), weights(1024, 4096)
    hidden = kernels.int8_scaled_mm(codes, scales, *up, True)
    down_out = kernels.int8_scaled_mm(*kernels.rotate_quantize(hidden), *down, False).float().view(2, 128, 1024)
    expected = (x.float() + (down_out * gate.float()[:, None]).view(256, 1024)).to(torch.bfloat16)
    assert torch.equal(kernels.int8_feed_forward(codes, scales, *up, *down, x, gate, 96), expected)


def test_quantize_fp4_is_bit_exact(kernels):
    generator = torch.Generator().manual_seed(0)
    for rows, cols in ((100, 512), (256, 16384)):
        x = _activations(rows, cols, generator)
        tensor_scale = T.fp4_tensor_scale(x)
        packed, block_scales = kernels.quantize_fp4(x, tensor_scale, False)
        expected_packed, expected_scales = T.quantize_fp4(x, tensor_scale)
        assert torch.equal(packed, expected_packed)
        assert torch.equal(block_scales.view(torch.uint8), expected_scales.view(torch.uint8))
        packed, block_scales = kernels.quantize_fp4(x, tensor_scale, True)
        expected_packed, expected_scales = T.quantize_fp4(F.gelu(x, approximate="tanh"), tensor_scale)
        assert torch.equal(packed, expected_packed)
        assert torch.equal(block_scales.view(torch.uint8), expected_scales.view(torch.uint8))


def _quantized_block(dim: int, heads: int, fp4: bool, generator: torch.Generator) -> dict[str, torch.Tensor]:
    def rand(*shape: int) -> torch.Tensor:
        return (torch.randn(*shape, generator=generator) * 0.05).to(torch.bfloat16)

    def linear_weights(name: str, out_features: int, in_features: int) -> dict[str, torch.Tensor]:
        if fp4:
            block_scales = torch.full((out_features, in_features // 16), 1.0).to(torch.float8_e4m3fn)
            return {
                f"{name}.weight": torch.randint(0, 256, (out_features, in_features // 2), generator=generator, dtype=torch.uint8),
                f"{name}.weight_scale": T.swizzle_block_scales(block_scales),
                f"{name}.weight_scale_2": torch.tensor(0.02),
            }
        return {
            f"{name}.qdata": torch.randint(-127, 128, (out_features, in_features), generator=generator, dtype=torch.int8),
            f"{name}.scales": torch.full((out_features,), 4e-4),
        }

    weights = {}
    for attn in ("attn1", "attn2"):
        for proj in ("to_q", "to_k", "to_v", "to_out.0"):
            weights |= linear_weights(f"{attn}.{proj}", dim, dim)
            weights[f"{attn}.{proj}.bias"] = rand(dim)
        weights |= linear_weights(f"{attn}.to_gate_logits", heads, dim)
        weights[f"{attn}.to_gate_logits.bias"] = rand(heads)
        weights[f"{attn}.q_norm.weight"] = rand(dim) + 1
        weights[f"{attn}.k_norm.weight"] = rand(dim) + 1
    weights |= linear_weights("ff.net.0.proj", 4 * dim, dim)
    weights |= linear_weights("ff.net.2", dim, 4 * dim)
    weights["scale_shift_table"] = rand(9, dim)
    weights["prompt_scale_shift_table"] = rand(2, dim)
    return {name: tensor.cuda() for name, tensor in weights.items()}


@pytest.mark.parametrize("fp4", [False, True], ids=["int8", "nvfp4"])
def test_block_forward_with_kernels_matches_the_torch_path(kernels, monkeypatch, fp4):
    generator = torch.Generator().manual_seed(0)
    dim, heads = 512, 16
    weights = _quantized_block(dim, heads, fp4, generator)
    rope = T.build_rope(2, 4, 4, dim=dim, heads=heads, device=torch.device("cuda"))
    cond = T.StepConditioning(
        block_rows=(torch.randn(1, 2, 9, dim, generator=generator) * 0.1).to("cuda", torch.bfloat16),
        output_rows=torch.zeros(1, 2, dim, dtype=torch.bfloat16, device="cuda"),
        prompt_rows=(torch.randn(1, 1, 2, dim, generator=generator) * 0.1).to("cuda", torch.bfloat16),
    )
    x = torch.randn(1, 64, dim, generator=generator).to("cuda", torch.bfloat16)
    context = torch.randn(1, 8, dim, generator=generator).to("cuda", torch.bfloat16)
    with_kernels = T.block_forward(weights, x, cond, context, rope, heads=heads)
    monkeypatch.setattr(T, "_quant_kernels", None)
    expected = T.block_forward(weights, x, cond, context, rope, heads=heads)
    assert (with_kernels.float() - expected.float()).norm() / expected.float().norm() < 1e-2


def test_kernels_reject_widths_they_would_compute_wrong(kernels):
    x = torch.zeros(8, 384, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError):
        kernels.rotate_quantize(x)
    with pytest.raises(ValueError):
        kernels.quantize_fp4(x, torch.ones((), device="cuda"), False)
    codes = torch.zeros(8, 320, dtype=torch.int8, device="cuda")
    weights = torch.zeros(256, 320, dtype=torch.int8, device="cuda")
    with pytest.raises(ValueError):
        kernels.int8_scaled_mm(codes, torch.ones(8, device="cuda"), weights, torch.ones(256, device="cuda"), None, False)
