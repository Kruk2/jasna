"""LTX self-attention through comfy-kitchen's INT8 attention against exact SDPA.
Needs an NVIDIA GPU and comfy-kitchen."""

import importlib.util

import pytest
import torch
import torch.nn.functional as F

from jasna.ltx import transformer as T

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None or importlib.util.find_spec("comfy_kitchen") is None,
    reason="needs an NVIDIA GPU and comfy-kitchen",
)


def test_self_attention_runs_int8_attention_close_to_exact():
    T.require_kernels(torch.device("cuda"))
    T.enable_int8_attention()
    generator = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (
        torch.randn(1, T.HEADS, 1024, 128, device="cuda", generator=generator, dtype=torch.bfloat16) for _ in range(3)
    )
    out = T.self_attention(q, k, v)
    expected = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    assert out.shape == q.shape and out.is_contiguous()
    assert (out.float() - expected).norm() / expected.norm() < 3e-2
