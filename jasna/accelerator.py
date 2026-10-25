from __future__ import annotations

from collections.abc import MutableMapping
from contextlib import nullcontext
import ctypes
from enum import StrEnum
import functools
import logging
import os
from pathlib import Path
import sys
from typing import Any

import torch

log = logging.getLogger(__name__)

# BasicVSR++ meets a new convolution batch for every clip length. FAST never
# writes the user FindDb, and the Windows ROCm wheels ship no system FindDb
# (ROCm/TheRock#7958), so every miss falls back to slow heuristic kernels forever.
# HYBRID runs a real find once per new problem and keeps the result across runs
# (issue #481: 4x faster primary restoration once the DB is seeded).
#
# That first find also benchmarks the im2col/GEMM conv solvers, whose workspace
# is about 27x the conv input: 17 GiB for the LTX VAE encoder at 768, so a
# 16 GB card runs out of memory, the find never finishes and every run fails.
# The CK solvers win those benchmarks anyway; without GEMM the peak is 4.3 GiB.
#
# Expandable segments release memory by unmapping virtual address ranges rather
# than by a device-synchronizing free, so a VramOffloader empty_cache() could
# pull pages out from under kernels another thread still had in flight — issue
# #252 caught a restorer fp16 GEMM faulting with "Page not present". The env
# names cover the versions in the field; whichever one the build reads wins.
def apply_rocm_env_defaults(environ: MutableMapping[str, str]) -> None:
    environ.setdefault("MIOPEN_FIND_MODE", "HYBRID")
    environ.setdefault("MIOPEN_DEBUG_CONV_GEMM", "0")
    environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:False")
    environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:False")
    if sys.platform != "win32":
        environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")


def has_aotriton_flash_images(torch_lib: Path) -> bool:
    return any((torch_lib / "aotriton.images").glob("*/flash/attn_fwd.zip"))


def configure_rocm_process_env() -> None:
    """Apply the ROCm defaults to this process; call at entry points before GPU work."""
    if not torch.version.hip:
        return
    apply_rocm_env_defaults(os.environ)
    # MIOpen then runs channels-last 3D convs in NDHWC instead of transposing every input, weight
    # and output: the compiled LTX VAE encoder 5 % faster and 0.5 GiB lighter on an R9700. Eager
    # encoders (no Triton: Windows ROCm, AMD releases) run 1.6x slower with it, and BasicVSR++ and
    # RF-DETR pass no channels-last conv inputs.
    from torch.utils._triton import has_triton_package

    if has_triton_package():
        os.environ.setdefault("PYTORCH_MIOPEN_SUGGEST_NHWC", "1")
    # Windows ROCm ships the AOTriton kernel images in a separate device wheel;
    # without them the first flash/mem-efficient SDPA launch dies with a HIP error.
    if not has_aotriton_flash_images(Path(torch.__file__).parent / "lib"):
        log.warning("PyTorch has no AOTriton attention kernels installed; attention runs slower without them")
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)


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


def supports_licensed_ltx_on_amd() -> bool:
    """Whether this OS has the protected AMD worker; GPU kernels are checked separately."""
    return sys.platform in {"linux", "win32"}


def supports_nvfp4(device: torch.device | str) -> bool:
    return is_nvidia_device(device) and torch.cuda.get_device_capability(device)[0] >= 10


def amd_gpu_architecture(device: torch.device | str) -> str:
    """The AMD GPU's architecture without target features, e.g. ``gfx1201``."""
    return torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]


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


class _NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


@functools.cache
def _nvml_device(uuid: str):
    nvml = ctypes.WinDLL("nvml.dll")
    _nvml_check(nvml, nvml.nvmlInit_v2())
    handle = ctypes.c_void_p()
    _nvml_check(nvml, nvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}".encode(), ctypes.byref(handle)))
    return nvml, handle


def _nvml_check(nvml, status: int) -> None:
    if status:
        nvml.nvmlErrorString.restype = ctypes.c_char_p
        raise RuntimeError(f"NVML: {nvml.nvmlErrorString(status).decode()}")


def _nvml_free_bytes(device: torch.device) -> int:
    nvml, handle = _nvml_device(str(torch.cuda.get_device_properties(device).uuid))
    memory = _NvmlMemory()
    _nvml_check(nvml, nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)))
    return memory.free


def device_free_bytes(device: torch.device) -> int:
    """VRAM free on the whole GPU. On Windows, CUDA counts only this process's memory (WDDM
    hides every other process's), so NVML's device-wide figure caps it on NVIDIA there; AMD
    reads HIP's figure alone."""
    free = torch.cuda.mem_get_info(device)[0]
    if sys.platform == "win32" and is_nvidia_device(device):
        return min(free, _nvml_free_bytes(device))
    return free


def device_name(device: torch.device | str) -> str:
    resolved = torch.device(device)
    if resolved.type == "cpu":
        return "CPU"
    return str(device_module(resolved).get_device_name(resolved))
