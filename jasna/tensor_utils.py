from __future__ import annotations

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


def compiles_bf16(device: torch.device) -> bool:
    """Inductor refuses bf16 graphs on CUDA before SM 8.0."""
    return device.type != "cuda" or torch.cuda.get_device_capability(device) >= (8, 0)


def compile_static(fn: Callable, mode: str, device: torch.device) -> Callable:
    """``fn`` compiled with static shapes. Raises dynamo's recompile limits so every shape a run
    meets gets its own graph, and makes hitting them an error instead of a silent eager run.
    Where inductor has no bf16 (:func:`compiles_bf16`) ``fn`` stays eager."""
    if not compiles_bf16(device):
        return fn
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    torch._dynamo.config.accumulated_cache_size_limit = max(torch._dynamo.config.accumulated_cache_size_limit, 1024)
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    return torch.compile(fn, dynamic=False, mode=mode)
