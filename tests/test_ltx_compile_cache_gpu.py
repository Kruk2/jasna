"""A second process reuses the first one's autotuned LTX block kernels, so a seed gives
the same pixels on every run. Needs an NVIDIA GPU and Triton; compiles with autotuning
(~1-2 min)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="needs an NVIDIA GPU and Triton",
)

_RUN_BLOCK = """
import sys
import torch
from torch._dynamo.utils import counters
from jasna.ltx import transformer as T
from jasna.ltx.sampler import compiled_block_forward

dim, heads, device = 1024, 8, torch.device("cuda")
generator = torch.Generator().manual_seed(0)

def rand(*shape, scale=0.05):
    return (torch.randn(*shape, generator=generator) * scale).to(device, torch.bfloat16)

weights = {}
for name in T.LINEAR_NAMES:
    out_features = heads if name.endswith("to_gate_logits") else 4 * dim if name == "ff.net.0.proj" else dim
    in_features = 4 * dim if name == "ff.net.2" else dim
    weights[f"{name}.qdata"] = torch.randint(-127, 128, (out_features, in_features), generator=generator, dtype=torch.int8).to(device)
    weights[f"{name}.scales"] = torch.full((out_features,), 2e-4, device=device)
    weights[f"{name}.bias"] = rand(out_features)
for attn in ("attn1", "attn2"):
    weights[f"{attn}.q_norm.weight"] = rand(dim) + 1
    weights[f"{attn}.k_norm.weight"] = rand(dim) + 1
weights["scale_shift_table"] = rand(9, dim)
weights["prompt_scale_shift_table"] = rand(2, dim)
cond = T.StepConditioning(block_rows=rand(1, 2, 9, dim, scale=0.1), output_rows=rand(1, 2, dim), prompt_rows=rand(1, 1, 2, dim, scale=0.1))
rope = T.build_rope(4, 8, 8, dim=dim, heads=heads, device=device)
x = rand(1, 2 * 4 * 8 * 8, dim, scale=1.0)
context = rand(1, 64, dim, scale=1.0)

prompt = T.prompt_key_value(weights, cond, context, heads=heads)
out = compiled_block_forward(device)(weights, x, cond, prompt, rope, heads=heads)
torch.save(out.cpu(), sys.argv[1])
print(counters["inductor"]["fxgraph_cache_hit"])
"""


def _run(tmp_path, name: str) -> tuple[torch.Tensor, int]:
    env = {**os.environ, "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "cache")}
    result = subprocess.run(
        [sys.executable, "-c", _RUN_BLOCK, str(tmp_path / name)],
        env=env, capture_output=True, text=True, check=True, timeout=900,
    )
    return torch.load(tmp_path / name), int(result.stdout.strip().splitlines()[-1])


def test_second_process_reuses_the_autotuned_kernels(tmp_path):
    first, first_hits = _run(tmp_path, "first.pt")
    second, second_hits = _run(tmp_path, "second.pt")

    assert first_hits == 0 and second_hits > 0
    assert torch.equal(first, second)
