from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from contextlib import nullcontext
import ctypes
from enum import StrEnum
import functools
import hashlib
import importlib.util
import itertools
import logging
import ntpath
import os
from pathlib import Path
import sys
from typing import Any
import uuid

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


def rocm_libraries_dir() -> Path | None:
    """The ROCm wheels' ``_rocm_sdk_libraries`` folder: beside a frozen app's executable, else the installed package."""
    if is_frozen():
        from jasna.os_utils import get_application_executable

        return Path(get_application_executable()).resolve().parent / "_rocm_sdk_libraries"
    spec = importlib.util.find_spec("_rocm_sdk_libraries")
    return Path(next(iter(spec.submodule_search_locations))) if spec and spec.submodule_search_locations else None


def windows_miopen_system_db(platform: str, libraries: Path | None) -> Path | None:
    """The folder MIOpen should read its system databases (tuned solvers) from on Windows. The Windows ROCm
    wheels ship none, so Jasna's release build puts its own tuning in ``share/miopen/db``
    (``jasna.miopen_tuning``); MIOpen there looks beside MIOpen.dll in ``bin`` instead."""
    if platform != "win32" or libraries is None:
        return None
    database = libraries / "share" / "miopen" / "db"
    return database if database.is_dir() else None


def windows_ascii_miopen_user_folders(platform: str, environ: Mapping[str, str]) -> dict[str, str]:
    """MIOpen keeps its user database and kernel cache in ``%USERPROFILE%\\.miopen`` and fails every conv
    with miopenStatusInternalError when that path has non-ASCII letters, e.g. a Chinese user name
    (issue #808). Then both move to a per-user folder under ProgramData, or Public; values the user set win."""
    names = [name for name in ("MIOPEN_USER_DB_PATH", "MIOPEN_CUSTOM_CACHE_DIR") if name not in environ]
    profile = environ.get("USERPROFILE", "")
    if platform != "win32" or profile.isascii() or not names:
        return {}
    root = next((environ[name] for name in ("PROGRAMDATA", "PUBLIC") if environ.get(name) and environ[name].isascii()), None)
    if root is None:
        log.warning("MIOpen may fail: the user folder %s has non-ASCII letters and no ASCII folder was found for its cache", profile)
        return {}
    user = hashlib.sha256(profile.encode()).hexdigest()[:12]
    folders = {"MIOPEN_USER_DB_PATH": "db", "MIOPEN_CUSTOM_CACHE_DIR": "cache"}
    return {name: ntpath.join(root, "Jasna", "miopen", user, folders[name]) for name in names}


def configure_rocm_process_env() -> None:
    """Apply the ROCm defaults to this process; call at entry points before GPU work."""
    if not torch.version.hip:
        return
    apply_rocm_env_defaults(os.environ)
    system_db = windows_miopen_system_db(sys.platform, rocm_libraries_dir())
    if system_db is not None:
        os.environ.setdefault("MIOPEN_SYSTEM_DB_PATH", str(system_db))
    user_folders = windows_ascii_miopen_user_folders(sys.platform, os.environ)
    if user_folders:
        log.info("The user folder name has non-ASCII letters; MIOpen keeps its cache in %s", user_folders)
        os.environ.update(user_folders)
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


def release_blas_workspaces() -> None:
    """Free the workspace cuBLAS/hipBLAS keeps for every thread handle and stream it ran on
    (32 MiB each on an RTX 50). They stay allocated until the process ends otherwise, so passes
    on fresh threads and streams take more VRAM after every job of a session. Only while no
    other thread runs GPU work."""
    if torch.cuda.is_initialized():
        torch._C._cuda_clearCublasWorkspaces()


def ipc_collect(device: torch.device | str) -> None:
    module = device_module(torch.device(device))
    if hasattr(module, "ipc_collect"):
        module.ipc_collect()


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


class _Guid(ctypes.Structure):
    _fields_ = [("data1", ctypes.c_uint32), ("data2", ctypes.c_uint16), ("data3", ctypes.c_uint16),
                ("data4", ctypes.c_ubyte * 8)]


def _guid(text: str) -> _Guid:
    return _Guid.from_buffer_copy(uuid.UUID(text).bytes_le)


class _DxgiAdapterDesc1(ctypes.Structure):
    _fields_ = [("description", ctypes.c_wchar * 128), ("vendor_id", ctypes.c_uint32), ("device_id", ctypes.c_uint32),
                ("sub_sys_id", ctypes.c_uint32), ("revision", ctypes.c_uint32),
                ("dedicated_video_memory", ctypes.c_size_t), ("dedicated_system_memory", ctypes.c_size_t),
                ("shared_system_memory", ctypes.c_size_t), ("luid", ctypes.c_ubyte * 8), ("flags", ctypes.c_uint32)]


class _DxgiVideoMemoryInfo(ctypes.Structure):
    _fields_ = [("budget", ctypes.c_uint64), ("current_usage", ctypes.c_uint64),
                ("available_for_reservation", ctypes.c_uint64), ("current_reservation", ctypes.c_uint64)]


_IID_IDXGI_FACTORY1 = "770aae78-f26f-4dba-a829-253c83d1b387"
_IID_IDXGI_ADAPTER3 = "645967a4-1392-4310-a798-8053ce3e93fd"
_DXGI_ERROR_NOT_FOUND = 0x887A0002 - (1 << 32)
_DXGI_MEMORY_SEGMENT_GROUP_LOCAL = 0
# vtable slots: IUnknown 0-2, IDXGIObject 3-6, IDXGIFactory 7-11, IDXGIFactory1 12-13;
# IDXGIAdapter 7-9, IDXGIAdapter1 10, IDXGIAdapter2 11, IDXGIAdapter3 12-17
_QUERY_INTERFACE, _RELEASE = 0, 2
_FACTORY_ENUM_ADAPTERS1 = 12
_ADAPTER_GET_DESC1 = 10
_ADAPTER_QUERY_VIDEO_MEMORY_INFO = 14


def _com_method(interface: ctypes.c_void_p, slot: int, *argtypes):
    function = ctypes.cast(interface, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0][slot]
    method = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(function)
    return lambda *args: method(interface, *args)


def _check_hresult(result: int, call: str) -> None:
    if result < 0:
        raise OSError(f"DXGI {call} failed (HRESULT 0x{result & 0xFFFFFFFF:08X})")


def _adapter_video_memory(adapter: ctypes.c_void_p) -> tuple[bytes, str, int, int]:
    desc = _DxgiAdapterDesc1()
    _check_hresult(_com_method(adapter, _ADAPTER_GET_DESC1, ctypes.POINTER(_DxgiAdapterDesc1))(ctypes.byref(desc)),
                   "GetDesc1")
    adapter3 = ctypes.c_void_p()
    _check_hresult(_com_method(adapter, _QUERY_INTERFACE, ctypes.POINTER(_Guid), ctypes.POINTER(ctypes.c_void_p))(
        ctypes.byref(_guid(_IID_IDXGI_ADAPTER3)), ctypes.byref(adapter3)), "QueryInterface(IDXGIAdapter3)")
    info = _DxgiVideoMemoryInfo()
    try:
        _check_hresult(_com_method(adapter3, _ADAPTER_QUERY_VIDEO_MEMORY_INFO, ctypes.c_uint, ctypes.c_int,
                                   ctypes.POINTER(_DxgiVideoMemoryInfo))(
            0, _DXGI_MEMORY_SEGMENT_GROUP_LOCAL, ctypes.byref(info)), "QueryVideoMemoryInfo")
    finally:
        _com_method(adapter3, _RELEASE)()
    return bytes(desc.luid), desc.description, info.budget, info.current_usage


def windows_video_memory() -> dict[bytes, tuple[str, int, int]]:
    """Every display adapter's LUID with its name and this process's budget and use of its dedicated VRAM, as
    Windows (DXGI) reports them. The budget is what Windows lets this process keep in VRAM; past it Windows
    moves the process's allocations to system RAM instead of failing them."""
    factory = ctypes.c_void_p()
    _check_hresult(ctypes.WinDLL("dxgi").CreateDXGIFactory1(ctypes.byref(_guid(_IID_IDXGI_FACTORY1)),
                                                            ctypes.byref(factory)), "CreateDXGIFactory1")
    adapters = {}
    try:
        enum_adapters = _com_method(factory, _FACTORY_ENUM_ADAPTERS1, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))
        for index in itertools.count():
            adapter = ctypes.c_void_p()
            result = enum_adapters(index, ctypes.byref(adapter))
            if result == _DXGI_ERROR_NOT_FOUND:
                return adapters
            _check_hresult(result, "EnumAdapters1")
            try:
                luid, name, budget, usage = _adapter_video_memory(adapter)
            finally:
                _com_method(adapter, _RELEASE)()
            adapters[luid] = (name, budget, usage)
    finally:
        _com_method(factory, _RELEASE)()


# hipDeviceProp_t (R0600) starts with char name[256] and a 16-byte hipUUID, then char luid[8] (filled on Windows).
_HIP_PROPERTIES_LUID_OFFSET = 272
_HIP_PROPERTIES_BYTES = 4096


@functools.cache
def _hip_device_luid(index: int) -> bytes:
    from jasna.media.cuda_kernel import check_hip, hip_runtime

    hip = hip_runtime()
    hip.hipGetDevicePropertiesR0600.argtypes = [ctypes.c_void_p, ctypes.c_int]
    hip.hipGetDevicePropertiesR0600.restype = ctypes.c_int
    properties = ctypes.create_string_buffer(_HIP_PROPERTIES_BYTES)
    check_hip(hip.hipGetDevicePropertiesR0600(properties, index), "hipGetDevicePropertiesR0600")
    return properties.raw[_HIP_PROPERTIES_LUID_OFFSET:_HIP_PROPERTIES_LUID_OFFSET + 8]


@functools.cache
def _warn_no_dxgi_adapter(luid: bytes) -> None:
    log.warning("VRAM: Windows lists no display adapter for this GPU (LUID %s); sizing from the GPU's own free "
                "figure, which misses other programs' VRAM", luid.hex())


def _amd_windows_video_memory(device: torch.device) -> tuple[int, int] | None:
    """This process's Windows budget and use of ``device``'s VRAM, on Windows AMD only: HIP's free figure
    there counts only this process's VRAM and there is no NVML. ``None`` elsewhere or when DXGI lists no
    adapter with the device's LUID."""
    if sys.platform != "win32" or not is_amd_device(device):
        return None
    index = torch.device(device).index
    luid = _hip_device_luid(index if index is not None else torch.cuda.current_device())
    found = windows_video_memory().get(luid)
    if found is None:
        _warn_no_dxgi_adapter(luid)
        return None
    return found[1], found[2]


def budget_capped_free_bytes(free: int, budget: int, usage: int) -> int:
    """``free`` VRAM capped at what this process's Windows budget has left."""
    return max(0, min(free, budget - usage))


def amd_windows_budget_cap(device: torch.device, free: int) -> int:
    """``free`` capped by this process's Windows budget on Windows AMD: past the budget Windows quietly moves
    VRAM to system RAM and kernels wait on PCIe (#823). Elsewhere ``free`` unchanged."""
    memory = _amd_windows_video_memory(device)
    return free if memory is None else budget_capped_free_bytes(free, *memory)


def warn_over_windows_budget(device: torch.device) -> bool:
    """Warns, and returns True, when this process holds more of ``device``'s VRAM than its Windows budget
    (Windows AMD only)."""
    memory = _amd_windows_video_memory(device)
    if memory is None or memory[1] <= memory[0]:
        return False
    log.warning("VRAM: this process uses %d MiB, more than the %d MiB Windows lets it keep on the GPU; Windows "
                "moved the rest to system RAM, so the GPU runs slower. Closing other programs that use the GPU helps",
                memory[1] >> 20, memory[0] >> 20)
    return True


def device_free_bytes(device: torch.device) -> int:
    """VRAM free on the whole GPU. On Windows, CUDA and HIP count only this process's memory (WDDM
    hides every other process's), so NVML's device-wide figure caps it on NVIDIA there and the
    process's Windows budget on AMD."""
    free = torch.cuda.mem_get_info(device)[0]
    if sys.platform == "win32" and is_nvidia_device(device):
        return min(free, _nvml_free_bytes(device))
    return amd_windows_budget_cap(device, free)


def device_used_bytes(device: torch.device) -> int:
    """VRAM in use on the whole GPU, by every program on it: CUDA/HIP contexts, allocator
    caches and native allocations included."""
    return torch.cuda.mem_get_info(device)[1] - device_free_bytes(device)


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
