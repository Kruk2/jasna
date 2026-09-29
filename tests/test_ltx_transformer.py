import pytest
import torch

from jasna.ltx import transformer as T
from jasna.ltx.sampler import fuse_overlaps, patchify, unpatchify


def _block_weights(dim: int, heads: int, generator: torch.Generator) -> dict[str, torch.Tensor]:
    def rand(*shape: int) -> torch.Tensor:
        return (torch.randn(*shape, generator=generator) * 0.05).to(torch.bfloat16)

    weights = {}
    for attn in ("attn1", "attn2"):
        for proj in ("to_q", "to_k", "to_v", "to_out.0"):
            weights[f"{attn}.{proj}.weight"] = rand(dim, dim)
            weights[f"{attn}.{proj}.bias"] = rand(dim)
        weights[f"{attn}.to_gate_logits.weight"] = rand(heads, dim)
        weights[f"{attn}.to_gate_logits.bias"] = rand(heads)
        weights[f"{attn}.q_norm.weight"] = rand(dim) + 1
        weights[f"{attn}.k_norm.weight"] = rand(dim) + 1
    weights["ff.net.0.proj.weight"] = rand(4 * dim, dim)
    weights["ff.net.2.weight"] = rand(dim, 4 * dim)
    weights["scale_shift_table"] = rand(9, dim)
    weights["prompt_scale_shift_table"] = rand(2, dim)
    return weights


def test_rotation_is_its_own_inverse():
    x = torch.randn(4, 512, dtype=torch.float32)
    assert torch.allclose(T.rotate(T.rotate(x)), x, atol=1e-5)



def test_compiled_rotation_emits_no_cache_warning():
    import warnings

    torch._dynamo.reset()
    x = torch.randn(4, 512, dtype=torch.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rotated = torch.compile(T.rotate, backend="eager")(x)
    assert torch.equal(rotated, T.rotate(x))
    assert not [w for w in caught if "lru_cache" in str(w.message)]

def test_quantize_rows_pads_and_bounds():
    x = torch.randn(40, 64) * 3
    codes, scales = T.quantize_rows(x)
    assert codes.shape == (64, 64) and scales.shape == (64,)
    assert codes.abs().max() == T.ACT_QMAX
    assert torch.allclose(codes[:40].float() * scales[:40, None], x, atol=float(scales.max()))


def test_block_matches_across_token_chunking(monkeypatch):
    generator = torch.Generator().manual_seed(0)
    dim, heads = 64, 4
    weights = _block_weights(dim, heads, generator)
    rope = T.build_rope(3, 2, 2, dim=dim, heads=heads, device=torch.device("cpu"))
    cond = T.StepConditioning(
        block_rows=(torch.randn(1, 2, 9, dim, generator=generator) * 0.1).to(torch.bfloat16),
        output_rows=torch.zeros(1, 2, dim, dtype=torch.bfloat16),
        prompt_rows=(torch.randn(1, 1, 2, dim, generator=generator) * 0.1).to(torch.bfloat16),
    )
    x = torch.randn(1, 24, dim, generator=generator).to(torch.bfloat16)
    context = torch.randn(1, 5, dim, generator=generator).to(torch.bfloat16)
    whole = T.block_forward(weights, x, cond, context, rope, heads=heads)
    monkeypatch.setattr(T, "CHUNK_ROWS", 5)
    chunked = T.block_forward(weights, x, cond, context, rope, heads=heads)
    assert torch.equal(whole, chunked)
    skipped = T.block_forward(weights, x, cond, context, rope, heads=heads, skip_self_attention=True)
    assert not torch.equal(whole, skipped)


def test_latent_positions_are_causal_seconds():
    positions = T.latent_positions(3, 1, 1)
    assert positions[0].tolist() == pytest.approx([0.5 / 24, 5 / 24, 13 / 24])
    assert positions[1].tolist() == [16.0, 16.0, 16.0]


def test_patchify_round_trip():
    latent = torch.randn(1, 128, 16, 2, 3)
    assert torch.equal(unpatchify(patchify(latent), 16, 2, 3), latent)


def test_fuse_overlaps_writes_the_same_mean_to_both_windows():
    tokens_per_frame = 2
    left = torch.randn(1, 16 * tokens_per_frame, 4)
    right = torch.randn(1, 16 * tokens_per_frame, 4)
    right_before = right.clone()
    fuse_overlaps([left, right], tokens_per_frame, [0.0, 0.0, 0.0, 0.0])
    assert torch.equal(right[:, 2:10], left[:, 24:32])
    assert torch.equal(right[:, 10:], right_before[:, 10:])
    fuse_overlaps([left, right], tokens_per_frame, [0.2, 0.4, 0.6, 0.8])
    assert torch.equal(right[:, 2:10], left[:, 24:32])


def test_fp4_codes_round_half_to_even_and_saturate():
    values = torch.tensor([0.0, 0.25, 0.3, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.5, 9.0, -0.5, -6.0])
    expected = [0, 0, 1, 2, 2, 4, 4, 6, 6, 7, 7, 8 | 1, 8 | 7]
    assert T.fp4_codes(values).tolist() == expected


def test_block_scale_swizzle_matches_cublas_tile_addresses():
    rows, cols = 200, 6
    scales = torch.arange(rows * cols, dtype=torch.float32).remainder(200).view(rows, cols).to(torch.float8_e4m3fn)
    tiled = T.swizzle_block_scales(scales).view(torch.uint8).flatten()
    padded_cols = 8
    for row in (0, 31, 32, 127, 128, 199):
        for col in range(cols):
            tile = (row // 128) * (padded_cols // 4) + col // 4
            offset = tile * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + col % 4
            assert tiled[offset] == scales[row, col].view(torch.uint8)


def test_quantize_fp4_reconstructs_within_fp4_error():
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(130, 64, generator=generator)
    tensor_scale = T.fp4_tensor_scale(x)
    packed, tiled = T.quantize_fp4(x, tensor_scale)
    assert packed.shape == (130, 32) and packed.dtype == torch.uint8
    assert tiled.shape == (256, 4) and tiled.dtype == torch.float8_e4m3fn
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    codes = torch.stack([packed >> 4, packed & 15], dim=-1).flatten(1).long()
    values = table[codes & 7] * (1 - 2 * (codes >> 3)).float()
    blocks = x.view(130, 4, 16)
    scales = (blocks.abs().amax(-1) / T.FP4_MAX / tensor_scale).clamp(max=T.FP8_E4M3_MAX)
    scales = scales.to(torch.float8_e4m3fn).float()
    rebuilt = values.view(130, 4, 16) * scales.unsqueeze(-1) * tensor_scale
    assert (rebuilt - blocks).norm() / blocks.norm() < 0.15


def test_self_attention_falls_back_to_sdpa_off_gpu():
    generator = torch.Generator().manual_seed(0)
    q, k, v = torch.randn(3, 1, 2, 8, 16, generator=generator).unbind(0)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(T.self_attention(q, k, v), expected)


def test_block_forward_runs_eager_without_triton(monkeypatch):
    import importlib.util

    from jasna.ltx import sampler

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "triton" else real_find_spec(name))
    assert sampler.compiled_block_forward() is T.block_forward


def test_sampler_settings_come_from_the_model_file():
    from jasna.ltx.sampler import sampler_settings

    sigmas, stg = sampler_settings({"sigmas": "[1.0, 0.4, 0.0]", "stg_scale": "0.0"})
    assert sigmas.tolist() == pytest.approx([1.0, 0.4, 0.0]) and stg == 0.0
    with pytest.raises(KeyError):
        sampler_settings({"format": "jasna-ltx-restore"})


def test_window_state_halves_without_stg():
    from jasna.ltx import sampler

    assert sampler._window_state_bytes(100, stg=True) == 2 * sampler._window_state_bytes(100, stg=False)


def test_block_forward_compiles_without_timing_kernels(monkeypatch):
    import importlib.util

    from jasna.ltx import sampler

    if importlib.util.find_spec("triton") is None:
        pytest.skip("needs Triton")

    calls = []
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: calls.append(kwargs) or fn)

    assert sampler.compiled_block_forward() is T.block_forward
    assert calls == [{"dynamic": False, "options": {"deterministic": True}}]
