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
