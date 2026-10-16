"""Parity of the Triton ``na3d`` against the pure-torch ``eager_na.na3d``.

Runs on CUDA, or on CPU through Triton's interpreter:
    CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 python -m pytest tests/test_na_triton.py
"""

import os

import pytest
import torch

triton = pytest.importorskip("triton")

from jasna.models.ltx_vae.eager_na import na3d as eager_na3d  # noqa: E402
from jasna.models.ltx_vae.na_triton import (  # noqa: E402
    LARGE_WINDOW_CONFIG,
    SMALL_WINDOW_CONFIG,
    config_for,
    na3d_with_config,
    tile_config,
)

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cpu" if INTERPRET else "cuda"
pytestmark = pytest.mark.skipif(
    not INTERPRET and not torch.cuda.is_available(), reason="needs CUDA or TRITON_INTERPRET=1"
)
BF16_ATOL = 2e-2


@pytest.fixture(autouse=True)
def _interpreter_dot_in_fp32(monkeypatch):
    """Triton 3.7's interpreter multiplies bf16 dot operands as their raw uint16 bits."""
    if not INTERPRET:
        return
    from triton.runtime.interpreter import InterpreterBuilder

    original = InterpreterBuilder.create_dot

    def create_dot(self, a, b, d, input_precision, max_num_imprecise_acc):
        def fp32(x):
            return self.cast_impl(x, triton.language.float32) if x.dtype.scalar == triton.language.bfloat16 else x

        return original(self, fp32(a), fp32(b), d, input_precision, max_num_imprecise_acc)

    monkeypatch.setattr(InterpreterBuilder, "create_dot", create_dot)


@pytest.mark.parametrize(
    ("shape", "kernel", "query_w", "config"),
    [
        ((1, 12, 13, 14, 2, 16), (11, 11, 11), (0, 14), LARGE_WINDOW_CONFIG),
        ((1, 13, 13, 20, 2, 16), (11, 11, 11), (5, 10), LARGE_WINDOW_CONFIG),
        ((1, 12, 13, 14, 2, 16), (3, 5, 5), (0, 14), LARGE_WINDOW_CONFIG),
        ((2, 5, 7, 9, 3, 32), (3, 5, 7), (2, 5), SMALL_WINDOW_CONFIG),
        ((1, 3, 6, 17, 1, 64), (5, 7, 3), (0, 17), SMALL_WINDOW_CONFIG),
        ((1, 6, 9, 11, 2, 16), (5, 5, 5), (1, 9), tile_config((4, 4, 8), 64, 4, 2)),
    ],
)
def test_matches_eager_na3d(shape, kernel, query_w, config):
    generator = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(shape, generator=generator).to(DEVICE, torch.bfloat16) for _ in range(3))
    q = (q * shape[-1] ** -0.5).to(torch.bfloat16)
    w_start, w_len = query_w

    expected = eager_na3d(q.float(), k.float(), v.float(), kernel, 0)[:, :, :, w_start : w_start + w_len]
    out = na3d_with_config(q[:, :, :, w_start : w_start + w_len], k, v, kernel, w_start, config)

    assert out.dtype == torch.bfloat16
    assert (out.float() - expected).abs().max().item() < BF16_ATOL


def test_large_windows_take_the_large_tile_config():
    assert config_for((11, 11, 11)) is LARGE_WINDOW_CONFIG
    assert config_for((3, 5, 5)) is SMALL_WINDOW_CONFIG
    assert config_for((3, 7, 7)) is SMALL_WINDOW_CONFIG


def test_rejects_query_columns_outside_the_grid():
    q = torch.zeros(1, 4, 4, 3, 1, 16, dtype=torch.bfloat16)
    k = torch.zeros(1, 4, 4, 6, 1, 16, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="exceed W"):
        na3d_with_config(q, k, k, (3, 3, 3), 4, SMALL_WINDOW_CONFIG)
