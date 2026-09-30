from __future__ import annotations

import importlib.util
from collections.abc import Callable

import torch


def pad_batch_with_last(x: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    n = int(x.shape[0])
    bs = int(batch_size)
    if n <= 0:
        raise ValueError("cannot pad an empty batch")
    if bs <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if n > bs:
        raise ValueError(f"input batch {n} exceeds target batch_size {bs}")
    if n == bs:
        return x
    pad = x[-1:].expand(bs - n, *x.shape[1:])
    return torch.cat([x, pad], dim=0)


def compile_static(fn: Callable, mode: str) -> Callable:
    """``fn`` compiled with static shapes when Triton is importable, else ``fn`` itself.
    Raises dynamo's recompile limits so every shape a run meets gets its own graph."""
    if importlib.util.find_spec("triton") is None:
        return fn
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    torch._dynamo.config.accumulated_cache_size_limit = max(torch._dynamo.config.accumulated_cache_size_limit, 1024)
    return torch.compile(fn, dynamic=False, mode=mode)
