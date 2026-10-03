"""Narrow MMEngine import compatibility for the Windows ROCm torch wheel.

The Windows ROCm build intentionally omits distributed c10d, while MMEngine
0.10.x imports a small distributed-only surface even for single-process
inference. Keep this shim at the BasicVSR++ dependency boundary so Linux ROCm,
CUDA, and distributed-capable torch installations are untouched.
"""

from __future__ import annotations

import sys
import types
from typing import Any


_FSDP_MODULE = "mmengine.model.wrappers.fully_sharded_distributed"


def _is_windows_rocm_without_distributed(
    torch_module: Any,
    platform: str,
) -> bool:
    if platform != "win32":
        return False
    torch_version = getattr(torch_module, "version", None)
    if not getattr(torch_version, "hip", None):
        return False
    torch_dist = getattr(torch_module, "distributed", None)
    is_available = getattr(torch_dist, "is_available", None)
    if not callable(is_available):
        return False
    try:
        return not bool(is_available())
    except Exception:
        return False


def prepare_mmengine_for_windows_rocm(
    torch_module: Any | None = None,
    *,
    platform: str | None = None,
) -> tuple[str, ...]:
    """Make MMEngine importable for single-process Windows ROCm inference.

    Returns the compatibility surfaces installed by this call. The optional
    inputs keep the platform gate directly unit-testable without replacing the
    active torch installation.
    """

    if torch_module is None:
        import torch as torch_module

    active_platform = sys.platform if platform is None else platform
    if not _is_windows_rocm_without_distributed(torch_module, active_platform):
        return ()

    installed: list[str] = []
    torch_dist = torch_module.distributed
    if not hasattr(torch_dist, "ReduceOp"):

        class _ReduceOp:
            SUM = "sum"
            PRODUCT = "product"
            MIN = "min"
            MAX = "max"
            BAND = "band"
            BOR = "bor"
            BXOR = "bxor"

        torch_dist.ReduceOp = _ReduceOp
        installed.append("torch.distributed.ReduceOp")

    if _FSDP_MODULE not in sys.modules:
        module = types.ModuleType(_FSDP_MODULE)

        class MMFullyShardedDataParallel:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                raise RuntimeError(
                    "MMEngine FSDP is unavailable in this Windows ROCm torch build"
                )

        module.MMFullyShardedDataParallel = MMFullyShardedDataParallel
        sys.modules[_FSDP_MODULE] = module
        installed.append("mmengine FSDP stub")

    return tuple(installed)
