from __future__ import annotations

from collections.abc import MutableMapping
from contextlib import nullcontext
from enum import StrEnum
import os
from typing import Any

import torch

# NORMAL benchmarks every unseen convolution problem. BasicVSR++ has fixed
# spatial dimensions but a variable temporal clip length (and therefore variable
# effective convolution batches), so FAST avoids repeated runtime profiling while
# still using MIOpen's system/user performance databases. Users can override this.
#
# Expandable segments release memory by unmapping virtual address ranges rather
# than by a device-synchronizing free, so a VramOffloader empty_cache() could
# pull pages out from under kernels another thread still had in flight — issue
# #252 caught a restorer fp16 GEMM faulting with "Page not present". The env
# names cover the versions in the field; whichever one the build reads wins.
def apply_rocm_env_defaults(environ: MutableMapping[str, str]) -> None:
    environ.setdefault("MIOPEN_FIND_MODE", "FAST")
    environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:False")
    environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:False")
    environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")


def configure_rocm_process_env() -> None:
    """Apply the ROCm defaults to this process; call at entry points before GPU work."""
    if torch.version.hip:
        apply_rocm_env_defaults(os.environ)


class AcceleratorVendor(StrEnum):
    NVIDIA = "nvidia"
    AMD = "amd"
    CPU = "cpu"


def vendor_for_device(device: torch.device | str | None = None) -> AcceleratorVendor:
    resolved = torch.device(device) if device is not None else None
    if resolved is not None and resolved.type == "cpu":
        return AcceleratorVendor.CPU
    if torch.version.hip:
        return AcceleratorVendor.AMD
    if torch.version.cuda:
        return AcceleratorVendor.NVIDIA
    return AcceleratorVendor.CPU


def is_nvidia_device(device: torch.device | str | None = None) -> bool:
    return vendor_for_device(device) is AcceleratorVendor.NVIDIA


def is_amd_device(device: torch.device | str | None = None) -> bool:
    return vendor_for_device(device) is AcceleratorVendor.AMD


def device_module(device: torch.device | str):
    return torch.get_device_module(torch.device(device))


def device_context(device: torch.device | str):
    resolved = torch.device(device)
    if resolved.type == "cpu":
        return nullcontext()
    return device_module(resolved).device(resolved)


def stream_context(stream: Any):
    if stream is None:
        return nullcontext()
    try:
        return device_module(stream.device).stream(stream)
    except (TypeError, ValueError):
        # Also supports lightweight stream doubles used by callers/tests.
        return torch.cuda.stream(stream)


def new_stream(device: torch.device | str):
    resolved = torch.device(device)
    return device_module(resolved).Stream(resolved)


def current_stream(device: torch.device | str):
    resolved = torch.device(device)
    return device_module(resolved).current_stream(resolved)


def new_event(device: torch.device | str):
    return device_module(torch.device(device)).Event()


def set_device(device: torch.device | str) -> None:
    resolved = torch.device(device)
    if resolved.type != "cpu":
        device_module(resolved).set_device(resolved)


def synchronize(device: torch.device | str | None = None) -> None:
    if device is None:
        torch.accelerator.synchronize()
        return
    resolved = torch.device(device)
    if resolved.type != "cpu":
        device_module(resolved).synchronize(resolved)


def empty_cache(device: torch.device | str | None = None) -> None:
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        torch.accelerator.empty_cache()
        return
    if device is not None:
        module = device_module(torch.device(device))
        if hasattr(module, "empty_cache"):
            module.empty_cache()


def ipc_collect(device: torch.device | str) -> None:
    module = device_module(torch.device(device))
    if hasattr(module, "ipc_collect"):
        module.ipc_collect()


def reset_peak_memory_stats(device: torch.device | str) -> None:
    module = device_module(torch.device(device))
    if hasattr(module, "reset_peak_memory_stats"):
        module.reset_peak_memory_stats(torch.device(device))


def mem_get_info(device: torch.device | str) -> tuple[int, int]:
    module = device_module(torch.device(device))
    return module.mem_get_info(torch.device(device))


def device_name(device: torch.device | str) -> str:
    resolved = torch.device(device)
    if resolved.type == "cpu":
        return "CPU"
    return str(device_module(resolved).get_device_name(resolved))
