"""The native LTX block against ``transformer.block_forward`` on the same trial-shaped weights
and inputs. Needs a GPU ``kernels.supports``, Triton, comfy-kitchen and the toolchain that
builds the native block; NVFP4 needs SM 12.x."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")
pytest.importorskip("comfy_kitchen")

from jasna.ltx import native
from jasna.ltx import transformer as T
from jasna.ltx.kernels import supports
from jasna.ltx.model_files import PlaceholderHandle, PlaceholderTensors

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None or not supports(torch.device("cuda")),
    reason="needs a GPU the LTX kernels support",
)
needs_nvfp4 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in native.FP4_ARCHS,
    reason="NVFP4 blocks need consumer Blackwell",
)
QUANTIZATIONS = ["int8", pytest.param("nvfp4", marks=needs_nvfp4)]
DEVICE = torch.device("cuda")
# canvas 512 and 768: 16 latent frames of 16x16 / 24x24 tokens, target and reference halves
CANVAS_SIDES = {8192: 16, 18432: 24}
# Relative error bounds against the eager block_forward on (block output, block update), set a
# little above how far the compiled block_forward the sampler runs is from it on an RTX 5090
# (INT8 3.8e-3 / 3.0e-2; NVFP4 1.8e-2 / 0.12: one rounding moves many FP4 codes).
TOLERANCE = {"int8": (5e-3, 3.5e-2), "nvfp4": (2.5e-2, 0.15)}


def _block(quantization: str) -> tuple[dict[str, torch.Tensor], dict]:
    handle = PlaceholderHandle(PlaceholderTensors(quantization, "distilled"))
    specs = {name.split(".", 2)[2]: spec for name, spec in handle.tensor_specs().items() if name.startswith("blocks.0.")}
    weights = {name: handle.get_tensor(f"blocks.0.{name}").to(DEVICE) for name in specs}
    return weights, specs


@pytest.fixture(scope="module", params=QUANTIZATIONS)
def block(request):
    T.require_kernels(DEVICE)
    assert T.enable_quant_kernels(DEVICE)
    T.enable_int8_attention()
    weights, specs = _block(request.param)
    forward = native.load(DEVICE, specs)
    assert forward is not None, "the native block did not build; see the log"
    return weights, forward, TOLERANCE[request.param]


def _inputs(weights: dict[str, torch.Tensor], rows: int, seed: int):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    dim = weights["scale_shift_table"].shape[1]
    side = CANVAS_SIDES[rows]
    rope = T.build_rope(16, side, side, dim=dim, heads=T.HEADS, device=DEVICE)
    cond = T.StepConditioning(
        block_rows=(torch.randn(1, 2, 9, dim, generator=generator, device=DEVICE) * 0.1).to(torch.bfloat16),
        output_rows=torch.zeros(1, 2, dim, dtype=torch.bfloat16, device=DEVICE),
        prompt_rows=(torch.randn(1, 1, 2, dim, generator=generator, device=DEVICE) * 0.1).to(torch.bfloat16),
    )
    x = torch.randn(1, rows, dim, generator=generator, device=DEVICE).to(torch.bfloat16)
    context = torch.randn(1, 1024, dim, generator=generator, device=DEVICE).to(torch.bfloat16)
    return x, cond, T.prompt_key_value(weights, cond, context), rope


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


def test_native_prompt_key_value_matches_the_python_heads_and_dtype(block):
    weights, forward, (tolerance, _) = block
    _x, cond, _prompt, _rope = _inputs(weights, min(CANVAS_SIDES), seed=3)
    context = torch.randn(1, 1024, weights["scale_shift_table"].shape[1], device=DEVICE).to(torch.bfloat16)
    expected = T.prompt_key_value(weights, cond, context)
    linears = [tensor for name in T.LINEAR_NAMES for tensor in native._linear_tensors(weights, name)]
    actual = forward.block.prompt_key_value(
        linears, weights["attn2.k_norm.weight"], weights["prompt_scale_shift_table"], cond.prompt_rows, context
    )
    for got, want in zip(actual, expected):
        assert got.dtype == want.dtype == T.prompt_attention_dtype(DEVICE)
        assert _relative_error(got, want) < tolerance


@pytest.mark.parametrize("rows", list(CANVAS_SIDES))
@pytest.mark.parametrize("skip", [False, True], ids=["full", "stg"])
def test_native_block_matches_block_forward(block, rows, skip):
    weights, forward, (output_tolerance, update_tolerance) = block
    x, cond, prompt, rope = _inputs(weights, rows, seed=rows)
    expected = T.block_forward(weights, x, cond, prompt, rope, skip_self_attention=skip)
    actual = forward(weights, x, cond, prompt, rope, skip_self_attention=skip)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.isfinite(actual.float()).all()
    assert _relative_error(actual, expected) < output_tolerance
    assert _relative_error(actual - x, expected - x) < update_tolerance


def test_native_block_honors_feed_forward_biases(block):
    weights, forward, (_, update_tolerance) = block
    generator = torch.Generator(device=DEVICE).manual_seed(1)
    biased = dict(weights)
    for name in ("ff.net.0.proj", "ff.net.2"):
        out_features = (weights.get(f"{name}.qdata") if f"{name}.qdata" in weights else weights[f"{name}.weight"]).shape[0]
        biased[f"{name}.bias"] = (torch.randn(out_features, generator=generator, device=DEVICE) * 2).to(torch.bfloat16)
    x, cond, prompt, rope = _inputs(weights, 8192, seed=2)
    expected = T.block_forward(biased, x, cond, prompt, rope)
    with_bias = _relative_error(forward(biased, x, cond, prompt, rope) - x, expected - x)
    without_bias = _relative_error(forward(weights, x, cond, prompt, rope) - x, expected - x)
    assert with_bias < update_tolerance and with_bias < without_bias / 3


@pytest.fixture(scope="module")
def int8_block():
    _, specs = _block("int8")
    forward = native.load(DEVICE, specs)
    assert forward is not None, "the native block did not build; see the log"
    return forward


# rows cover both key tiles, lengths off the tile, kitchen's short-K paths (<= 256 / 512)
# and its Blackwell 64-row query tile (4096-4608 rows at head dim 128)
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("rows", [200, 500, 1000, 1100, 4200, 8192, 18432])
def test_native_int8_attention_is_comfy_kitchens_bit_for_bit(int8_block, head_dim, rows):
    import comfy_kitchen

    heads = 4096 // head_dim
    generator = torch.Generator(device=DEVICE).manual_seed(rows * head_dim)
    q, k, v = (torch.randn(rows, heads * head_dim, generator=generator, device=DEVICE).to(torch.bfloat16) for _ in range(3))
    attention = int8_block.block.self_attention(q, k, v, rows, heads, head_dim, native.attention_cta_k(head_dim, rows))

    def heads_first(t: torch.Tensor) -> torch.Tensor:
        return t.view(1, rows, heads, head_dim).transpose(1, 2)

    expected = comfy_kitchen.int8_attention(heads_first(q), heads_first(k), heads_first(v))
    assert torch.equal(attention, expected[0].transpose(0, 1))
    exact = F.scaled_dot_product_attention(heads_first(q).float(), heads_first(k).float(), heads_first(v).float())
    assert _relative_error(attention, exact[0].transpose(0, 1)) < 3e-2


def test_native_block_is_deterministic(block):
    weights, forward, _ = block
    x, cond, prompt, rope = _inputs(weights, 8192, seed=4)
    assert torch.equal(forward(weights, x, cond, prompt, rope), forward(weights, x, cond, prompt, rope))


def test_native_block_rejects_an_odd_row_count(block):
    weights, forward, _ = block
    x, cond, prompt, rope = _inputs(weights, 8192, seed=5)
    with pytest.raises(RuntimeError):
        forward(weights, x[:, :-1], cond, prompt, rope)

