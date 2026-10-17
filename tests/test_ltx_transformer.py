import os
import sys
from types import SimpleNamespace

import pytest
import torch

from jasna.ltx import transformer as T
from jasna.ltx.plan import LESS_OVERLAP_WINDOW_HOP, WINDOW_HOP, shared_latents
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
    prompt = T.prompt_key_value(weights, cond, context, heads=heads)
    whole = T.block_forward(weights, x, cond, prompt, rope, heads=heads)
    monkeypatch.setattr(T, "CHUNK_ROWS", 5)
    chunked = T.block_forward(weights, x, cond, prompt, rope, heads=heads)
    assert torch.equal(whole, chunked)
    skipped = T.block_forward(weights, x, cond, prompt, rope, heads=heads, skip_self_attention=True)
    assert not torch.equal(whole, skipped)


def test_fp16_prompt_attention_keeps_the_block_in_bf16():
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
    prompt = T.prompt_key_value(weights, cond, torch.randn(1, 5, dim, generator=generator).to(torch.bfloat16), heads=heads)
    bf16 = T.block_forward(weights, x, cond, prompt, rope, heads=heads)
    fp16 = T.block_forward(weights, x, cond, tuple(t.half() for t in prompt), rope, heads=heads)
    assert fp16.dtype == torch.bfloat16
    torch.testing.assert_close(fp16, bf16, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("major, dtype", [(7, torch.float16), (8, torch.bfloat16)])
def test_prompt_attention_runs_in_fp16_before_sm80(major, dtype, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(major=major))
    assert T.prompt_attention_dtype(torch.device("cuda")) == dtype
    assert T.prompt_attention_dtype(torch.device("cpu")) == torch.bfloat16


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
    shared = shared_latents(WINDOW_HOP)
    fuse_overlaps([left, right], tokens_per_frame, shared, [0.0, 0.0, 0.0, 0.0])
    assert torch.equal(right[:, 2:10], left[:, 24:32])
    assert torch.equal(right[:, 10:], right_before[:, 10:])
    fuse_overlaps([left, right], tokens_per_frame, shared, shared.ramp)
    assert torch.equal(right[:, 2:10], left[:, 24:32])


def test_less_overlap_fuses_two_latent_frames():
    tokens_per_frame = 2
    left = torch.randn(1, 16 * tokens_per_frame, 4)
    right = torch.randn(1, 16 * tokens_per_frame, 4)
    left_before, right_before = left.clone(), right.clone()
    shared = shared_latents(LESS_OVERLAP_WINDOW_HOP)
    fuse_overlaps([left, right], tokens_per_frame, shared, shared.ramp)
    assert torch.equal(right[:, 2:6], left[:, 28:32])
    assert torch.equal(right[:, 6:], right_before[:, 6:])
    assert torch.equal(left[:, :28], left_before[:, :28])


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


def test_public_block_forward_runs_on_plain_tensors_without_the_protection_module(monkeypatch):
    generator = torch.Generator().manual_seed(0)
    dim, heads = 256, 2
    weights = _block_weights(dim, heads, generator)
    for name in T.LINEAR_NAMES:
        weight = weights.pop(f"{name}.weight").float()
        codes, scales = T.quantize_rows(T.rotate(weight))
        weights[f"{name}.qdata"] = codes[: weight.shape[0]]
        weights[f"{name}.scales"] = scales[: weight.shape[0]]
    rope = T.build_rope(3, 2, 2, dim=dim, heads=heads, device=torch.device("cpu"))
    cond = T.StepConditioning(
        block_rows=(torch.randn(1, 2, 9, dim, generator=generator) * 0.1).to(torch.bfloat16),
        output_rows=torch.zeros(1, 2, dim, dtype=torch.bfloat16),
        prompt_rows=(torch.randn(1, 1, 2, dim, generator=generator) * 0.1).to(torch.bfloat16),
    )
    x = torch.randn(1, 24, dim, generator=generator).to(torch.bfloat16)
    context = torch.randn(1, 5, dim, generator=generator).to(torch.bfloat16)
    monkeypatch.setitem(sys.modules, "jasna.protection", None)
    out = T.block_forward(weights, x, cond, T.prompt_key_value(weights, cond, context, heads=heads), rope, heads=heads)
    assert out.shape == x.shape and torch.isfinite(out.float()).all()
    assert not torch.equal(out, x)


def test_self_attention_falls_back_to_sdpa_off_gpu():
    generator = torch.Generator().manual_seed(0)
    q, k, v = torch.randn(3, 1, 2, 8, 16, generator=generator).unbind(0)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(T.self_attention(q, k, v), expected)


@pytest.mark.parametrize("package", ["triton", "comfy_kitchen"])
def test_ltx_refuses_to_run_without_its_kernels(package, monkeypatch):
    import importlib.util

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == package else real_find_spec(name))
    with pytest.raises(RuntimeError, match=package):
        T.require_kernels(torch.device("cuda"))


def test_ltx_refuses_a_device_without_int8_attention():
    pytest.importorskip("comfy_kitchen")
    with pytest.raises(RuntimeError, match="INT8 attention"):
        T.require_kernels(torch.device("cpu"))


def test_ltx_names_why_the_int8_attention_extension_did_not_load(monkeypatch):
    """#510: a Linux container refused LTX without saying why comfy-kitchen's _C failed to load."""
    comfy_kitchen = pytest.importorskip("comfy_kitchen")
    from comfy_kitchen.backends import cuda as kitchen_cuda

    monkeypatch.setattr(comfy_kitchen, "int8_attention_is_available", lambda device: False)
    monkeypatch.setattr(kitchen_cuda, "_EXT_ERROR", "unverified runtime extension")
    monkeypatch.setattr(torch.version, "hip", None)
    with pytest.raises(RuntimeError, match=r"did not load \(unverified runtime extension\)\.$"):
        T.require_kernels(torch.device("cuda"))


def test_compiled_functions_fail_instead_of_running_eager_past_the_recompile_limit(monkeypatch):
    from jasna.tensor_utils import compile_static

    monkeypatch.setattr(torch._dynamo.config, "fail_on_recompile_limit_hit", False)
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    compile_static(T.block_forward, mode="default", device=torch.device("cpu"))
    assert torch._dynamo.config.fail_on_recompile_limit_hit is True


def test_sampler_settings_come_from_the_model_file():
    from jasna.ltx.sampler import sampler_settings

    sigmas, stg = sampler_settings({"sigmas": "[1.0, 0.4, 0.0]", "stg_scale": "0.0"})
    assert sigmas.tolist() == pytest.approx([1.0, 0.4, 0.0]) and stg == 0.0
    with pytest.raises(KeyError):
        sampler_settings({"format": "jasna-ltx-restore"})


def test_window_state_halves_without_stg():
    from jasna.ltx import sampler

    assert sampler._window_state_bytes(100, stg=True) == 2 * sampler._window_state_bytes(100, stg=False)


def test_block_forward_compiles_with_autotuning_and_no_cuda_graphs(monkeypatch):
    import importlib.util

    from jasna.ltx import sampler

    if importlib.util.find_spec("triton") is None:
        pytest.skip("needs Triton")

    calls = []
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: calls.append(kwargs) or fn)

    assert sampler.compiled_block_forward(torch.device("cpu")) is T.block_forward
    assert calls == [{"dynamic": False, "mode": "max-autotune-no-cudagraphs"}]


def test_an_unwritable_compile_cache_is_a_warning(tmp_path, monkeypatch, caplog):
    import os

    from jasna.engine_paths import configure_torch_compile_cache

    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(os, "access", lambda path, mode: path != tmp_path)
    with caplog.at_level("WARNING", logger="jasna.engine_paths"):
        configure_torch_compile_cache()
    assert "is not writable" in caplog.text


def test_compile_cache_lives_in_model_weights_unless_set(monkeypatch):
    from jasna.engine_paths import configure_torch_compile_cache, model_weights_dir

    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    configure_torch_compile_cache()
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str((model_weights_dir() / "torch-compile-cache").resolve())

    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/elsewhere")
    configure_torch_compile_cache()
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == "/elsewhere"
