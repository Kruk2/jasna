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

import psutil
import torch
import torch.nn.functional as F

from jasna._frozen import is_frozen
from jasna.rocm_bootstrap import TRITON_HIP_ARCHITECTURES

log = logging.getLogger(__name__)

HIP_KERNEL_ARCHITECTURES = frozenset({"gfx1030", *TRITON_HIP_ARCHITECTURES, "gfx908", "gfx90a", "gfx942", "gfx950"})
"""The targets of the ``jasna/media/*.cu`` bundles: the Triton set plus RDNA2 and Instinct."""

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
    if kernel_backend(torch.device("cuda")) is KernelBackend.AMD_TRITON:
        os.environ.setdefault("PYTORCH_MIOPEN_SUGGEST_NHWC", "1")
    # Windows ROCm ships the AOTriton kernel images in a separate device wheel;
    # without them the first flash/mem-efficient SDPA launch dies with a HIP error.
    if not has_aotriton_flash_images(Path(torch.__file__).parent / "lib"):
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


@functools.cache
def has_triton() -> bool:
    """Whether Triton is installed; shared by source runs and both vendors' releases."""
    from torch.utils._triton import has_triton_package

    return has_triton_package()


class KernelBackend(StrEnum):
    """Which GPU kernels a device runs."""

    NVIDIA = "nvidia"
    NVIDIA_PRE_SM80 = "nvidia-pre-sm80"
    """No bf16 and no Triton tensor-core dot: fp16 torch kernels where Triton would run."""
    AMD_TRITON = "amd-triton"
    AMD_PREBUILT = "amd-prebuilt"
    """AMD kernels built ahead of time; packaged builds compile UNet 4x separately."""
    CPU = "cpu"

    @property
    def is_amd(self) -> bool:
        return self in (KernelBackend.AMD_TRITON, KernelBackend.AMD_PREBUILT)


def kernel_backend(device: torch.device | str) -> KernelBackend:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return KernelBackend.CPU
    if is_amd_device(resolved):
        # A release ships Triton only for UNet 4x; LTX runs on prebuilt kernels without importing it.
        return KernelBackend.AMD_PREBUILT if is_frozen() or not has_triton() else KernelBackend.AMD_TRITON
    if torch.cuda.get_device_capability(resolved) < (8, 0):
        return KernelBackend.NVIDIA_PRE_SM80
    return KernelBackend.NVIDIA


def load_gpu_libraries_ahead(device: torch.device, dtype: torch.dtype) -> None:
    """Pay what a process's first conv, first ``x @ wᵀ`` in ``dtype``, first fp32 ``a @ b`` and first
    ``torch.compile`` (hashing the Triton install) cost once, off the critical path: on ROCm each loads
    a library (MIOpen, a GEMM library per dtype and layout) or hashes ~1 GB, 0.4-1.2 s apiece."""
    if device.type != "cuda":
        return
    square = torch.zeros(8, 8, device=device, dtype=dtype)
    F.conv2d(square[None, None], square[None, None, :3, :3])
    F.linear(square, square)
    square.float() @ square.float()
    if has_triton():
        from triton.runtime.cache import triton_key

        triton_key()


_AMD_TUNING_ENV = ("MIOPEN_FIND_MODE", "MIOPEN_DEBUG_CONV_GEMM", "PYTORCH_MIOPEN_SUGGEST_NHWC")


@functools.cache
def log_amd_backend(device: torch.device) -> None:
    """Log once per process what decides AMD speed; a missing attention or video runtime warns."""
    from jasna.os_utils import check_amd_amf_runtime

    aotriton = has_aotriton_flash_images(Path(torch.__file__).parent / "lib")
    amf, _ = check_amd_amf_runtime()
    log.log(
        logging.INFO if aotriton and amf else logging.WARNING,
        "AMD backend: %s (%s), torch %s, HIP %s, %s kernels, AOTriton attention %s, AMF video %s, %s",
        torch.cuda.get_device_name(device),
        amd_gpu_architecture(device),
        torch.__version__,
        torch.version.hip,
        "Triton" if has_triton() else "prebuilt",
        "yes" if aotriton else "missing (attention runs slower)",
        "yes" if amf else "missing",
        " ".join(f"{name}={os.environ.get(name, '-')}" for name in _AMD_TUNING_ENV),
    )


def supports_nvfp4(device: torch.device | str) -> bool:
    return is_nvidia_device(device) and torch.cuda.get_device_capability(device)[0] >= 10


def supports_frame_generation(device: torch.device | str) -> bool:
    """NVIDIA's Video Frame Generation runs on RTX 40-series (Ada) and newer only."""
    return is_nvidia_device(device) and torch.cuda.get_device_capability(device) >= (8, 9)


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


def _nvml_total_disagrees_with_cuda(nvml_total: int, cuda_total: int) -> bool:
    # A real dGPU's NVML total stays within the WDDM reservation, well under 5% of
    # CUDA's; SoC GPUs (NVIDIA GB10 and friends) answer the v1 query with a bogus
    # small pool, so their totals disagree outright.
    return nvml_total * 20 < cuda_total * 19


def _nvml_memory(device: torch.device):
    """NVML's device-wide memory info for ``device``, with the CUDA properties a caller needs
    to compare NVML's total against CUDA's (``_nvml_total_disagrees_with_cuda``), so the query
    is not repeated at each call site."""
    props = torch.cuda.get_device_properties(device)
    nvml, handle = _nvml_device(str(props.uuid))
    memory = _NvmlMemory()
    _nvml_check(nvml, nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)))
    return props, memory


def _nvml_free_bytes(device: torch.device) -> int:
    props, memory = _nvml_memory(device)
    if _nvml_total_disagrees_with_cuda(memory.total, props.total_memory):
        # CUDA's own free figure is pinned at ~99.5% of its total on these boxes, so
        # it cannot cap the budget either; the host's available RAM is the honest
        # bound on a unified-memory box, and the caller's min() caps the process
        # view at it.
        return psutil.virtual_memory().available
    return memory.free


def device_free_bytes(device: torch.device) -> int:
    """VRAM free on the whole GPU. On Windows, CUDA counts only this process's memory (WDDM
    hides every other process's), so NVML's device-wide figure caps it on NVIDIA there; AMD
    reads HIP's figure alone."""
    free = torch.cuda.mem_get_info(device)[0]
    if sys.platform == "win32" and is_nvidia_device(device):
        return min(free, _nvml_free_bytes(device))
    return free


_ALLOCATION_PROBE_CHUNK_BYTES = 64 << 20


def allocatable_bytes(device: torch.device, limit: int) -> int:
    """How much of ``limit`` this process can really allocate on ``device`` now, found by
    allocating it in chunks and freeing them again. Windows can allow less than the free figure
    says, and without sysmem fallback the extra fails instead of spilling to RAM (#678)."""
    torch.cuda.empty_cache()
    chunks = []
    allocated = 0
    while allocated < limit:
        size = min(_ALLOCATION_PROBE_CHUNK_BYTES, limit - allocated)
        try:
            chunks.append(torch.empty(size, dtype=torch.uint8, device=device))
        except torch.OutOfMemoryError:
            log.info("VRAM: %d of %d MiB free can be allocated", allocated >> 20, limit >> 20)
            break
        allocated += size
    chunks.clear()
    torch.cuda.empty_cache()
    return allocated


def device_name(device: torch.device | str) -> str:
    resolved = torch.device(device)
    if resolved.type == "cpu":
        return "CPU"
    return str(device_module(resolved).get_device_name(resolved))
