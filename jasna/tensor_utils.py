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


def half_dtype(device: torch.device) -> torch.dtype:
    """The 16-bit float for tensor-core math on ``device``: bf16, or fp16 on CUDA GPUs before
    SM 8.0, which have no bf16 tensor cores, no fused bf16 attention and no Inductor bf16."""
    if device.type == "cuda" and torch.cuda.get_device_properties(device).major < 8:
        return torch.float16
    return torch.bfloat16


def compile_static(fn: Callable, mode: str, device: torch.device, dtype: torch.dtype) -> Callable:
    """``fn`` (computing in ``dtype``) compiled with static shapes. Raises dynamo's recompile
    limits so every shape a run meets gets its own graph, and makes hitting them an error
    instead of a silent eager run. ``fn`` stays eager without Triton (Windows ROCm ships none, so
    inductor has no GPU backend there) and, in bf16, where inductor has no bf16 (:func:`half_dtype`
    is fp16)."""
    from torch.utils._triton import has_triton_package

    if not has_triton_package() or (dtype == torch.bfloat16 and half_dtype(device) != torch.bfloat16):
        return fn
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    torch._dynamo.config.accumulated_cache_size_limit = max(torch._dynamo.config.accumulated_cache_size_limit, 1024)
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    return torch.compile(fn, dynamic=False, mode=mode)
