from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from enum import StrEnum
import os
from typing import Any

import torch

# NORMAL benchmarks every unseen convolution problem. BasicVSR++ has fixed
# spatial dimensions but a variable temporal clip length (and therefore variable
# effective convolution batches), so FAST avoids repeated runtime profiling while
# still using MIOpen's system/user performance databases. Users can override this.
if getattr(torch.version, "hip", None):
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")


class AcceleratorVendor(StrEnum):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    CPU = "cpu"


@dataclass(frozen=True)
class AcceleratorCapabilities:
    vendor: AcceleratorVendor
    pytorch_device_type: str
    tensorrt: bool
    migraphx: bool
    nvcodec: bool
    amf: bool
    xpu: bool


def vendor_for_device(device: torch.device | str | None = None) -> AcceleratorVendor:
    resolved = torch.device(device) if device is not None else None
    if resolved is not None and resolved.type == "cpu":
        return AcceleratorVendor.CPU
    if resolved is not None and resolved.type == "xpu":
        return AcceleratorVendor.INTEL
    if getattr(torch.version, "hip", None):
        return AcceleratorVendor.AMD
    if resolved is None:
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            return AcceleratorVendor.INTEL
    if getattr(torch.version, "cuda", None):
        return AcceleratorVendor.NVIDIA
    return AcceleratorVendor.CPU


def capabilities_for_device(
    device: torch.device | str | None = None,
) -> AcceleratorCapabilities:
    vendor = vendor_for_device(device)
    device_type = (
        torch.device(device).type
        if device is not None
        else "xpu" if vendor is AcceleratorVendor.INTEL else "cuda" if vendor in {
            AcceleratorVendor.NVIDIA,
            AcceleratorVendor.AMD,
        } else "cpu"
    )
    return AcceleratorCapabilities(
        vendor=vendor,
        pytorch_device_type=device_type,
        tensorrt=vendor is AcceleratorVendor.NVIDIA,
        migraphx=vendor is AcceleratorVendor.AMD,
        nvcodec=vendor is AcceleratorVendor.NVIDIA,
        amf=vendor is AcceleratorVendor.AMD,
        xpu=vendor is AcceleratorVendor.INTEL,
    )


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
