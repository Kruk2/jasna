"""Loading and launching ahead-of-time compiled GPU kernels.

Each ``.cu`` source next to this module is built twice (see
``docs/en/development.md``): into a ``.fatbin`` loaded through the CUDA driver
API on NVIDIA, and into a ``.hipfb`` offload bundle loaded through the HIP
module API on AMD. Neither needs a compiler at run time.
"""
import ctypes
import os
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import torch

from jasna._frozen import is_frozen
from jasna.accelerator import HIP_KERNEL_ARCHITECTURES, amd_gpu_architecture, is_amd_device, is_nvidia_device

_driver: ctypes.CDLL | None = None
_hip: ctypes.CDLL | None = None
_driver_lock = threading.Lock()
_modules: dict[tuple[int, str], ctypes.c_void_p] = {}

_LAUNCH_ARGTYPES = [
    ctypes.c_void_p,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_void_p),
]


def has_precompiled_kernels(device: torch.device | str) -> bool:
    """Whether ``device`` runs the precompiled kernels instead of their Torch equivalents."""
    if is_nvidia_device(device):
        return True
    if not is_amd_device(device):
        return False
    return amd_gpu_architecture(device) in HIP_KERNEL_ARCHITECTURES


def kernel_image_path(name: str) -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent / name
    return Path(__file__).resolve().with_name(name)


def cuda_driver() -> ctypes.CDLL:
    global _driver
    if _driver is not None:
        return _driver

    loader = ctypes.WinDLL if os.name == "nt" else ctypes.CDLL
    lib = loader("nvcuda.dll" if os.name == "nt" else "libcuda.so.1")
    lib.cuInit.argtypes = [ctypes.c_uint]
    lib.cuInit.restype = ctypes.c_int
    lib.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.cuCtxGetCurrent.restype = ctypes.c_int
    lib.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    lib.cuModuleLoadData.restype = ctypes.c_int
    lib.cuModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib.cuModuleGetFunction.restype = ctypes.c_int
    lib.cuLaunchKernel.argtypes = _LAUNCH_ARGTYPES
    lib.cuLaunchKernel.restype = ctypes.c_int
    lib.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    lib.cuGetErrorName.restype = ctypes.c_int
    lib.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    lib.cuGetErrorString.restype = ctypes.c_int
    lib.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    lib.cuStreamCreate.restype = ctypes.c_int
    lib.cuStreamDestroy.argtypes = [ctypes.c_void_p]
    lib.cuStreamDestroy.restype = ctypes.c_int
    lib.cuMemHostRegister_v2.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    lib.cuMemHostRegister_v2.restype = ctypes.c_int
    lib.cuMemHostUnregister.argtypes = [ctypes.c_void_p]
    lib.cuMemHostUnregister.restype = ctypes.c_int
    _driver = lib
    return lib


def _windows_torch_hip_runtime() -> ctypes.CDLL:
    """The amdhip64 copy torch_hip.dll imports. The frozen app maps more than one copy, and kernels launch only
    on the copy whose contexts torch made (another copy fails with hipErrorContextIsDestroyed)."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    kernel32.GetModuleHandleW.restype = ctypes.c_void_p
    kernel32.GetModuleHandleExW.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    kernel32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    base = kernel32.GetModuleHandleW("torch_hip.dll")
    if not base:
        raise RuntimeError("torch_hip.dll is not loaded")

    def u32(offset: int) -> int:
        return ctypes.c_uint32.from_address(base + offset).value

    # PE32+: the optional header follows the 24-byte file header; data directory 1 is the import table.
    descriptor = base + u32(u32(0x3C) + 24 + 112 + 8)
    while name := ctypes.c_uint32.from_address(descriptor + 12).value:
        if ctypes.string_at(base + name).lower().startswith(b"amdhip64"):
            function = ctypes.c_uint64.from_address(base + ctypes.c_uint32.from_address(descriptor + 16).value).value
            module = ctypes.c_void_p()
            from_address_unchanged_refcount = 0x4 | 0x2
            if not kernel32.GetModuleHandleExW(from_address_unchanged_refcount, function, ctypes.byref(module)):
                raise ctypes.WinError(ctypes.get_last_error())
            path = ctypes.create_unicode_buffer(32768)
            kernel32.GetModuleFileNameW(module, path, len(path))
            return ctypes.CDLL(path.value)
        descriptor += 20
    raise RuntimeError("torch_hip.dll does not import the HIP runtime")


def hip_runtime() -> ctypes.CDLL:
    """The HIP runtime Torch itself loaded (ROCm SDK wheel on Linux and Windows)."""
    global _hip
    if _hip is not None:
        return _hip

    if os.name == "nt":
        lib = _windows_torch_hip_runtime()
    else:
        from torch.cuda._utils import _get_hip_runtime_library

        lib = _get_hip_runtime_library()
    lib.hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.hipGetDevice.restype = ctypes.c_int
    lib.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    lib.hipModuleLoadData.restype = ctypes.c_int
    lib.hipModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib.hipModuleGetFunction.restype = ctypes.c_int
    lib.hipModuleLaunchKernel.argtypes = _LAUNCH_ARGTYPES
    lib.hipModuleLaunchKernel.restype = ctypes.c_int
    lib.hipGetErrorName.argtypes = [ctypes.c_int]
    lib.hipGetErrorName.restype = ctypes.c_char_p
    lib.hipGetErrorString.argtypes = [ctypes.c_int]
    lib.hipGetErrorString.restype = ctypes.c_char_p
    lib.hipHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    lib.hipHostRegister.restype = ctypes.c_int
    lib.hipHostUnregister.argtypes = [ctypes.c_void_p]
    lib.hipHostUnregister.restype = ctypes.c_int
    _hip = lib
    return lib


def check_hip(result: int, operation: str) -> None:
    if result == 0:
        return
    lib = hip_runtime()
    name = lib.hipGetErrorName(result)
    message = lib.hipGetErrorString(result)
    name_text = name.decode(errors="replace") if name else f"HIP error {result}"
    message_text = message.decode(errors="replace") if message else "unknown error"
    raise RuntimeError(f"{operation} failed: {name_text}: {message_text}")


def check_cuda(result: int, operation: str) -> None:
    if result == 0:
        return
    lib = cuda_driver()
    name = ctypes.c_char_p()
    message = ctypes.c_char_p()
    lib.cuGetErrorName(result, ctypes.byref(name))
    lib.cuGetErrorString(result, ctypes.byref(message))
    name_text = name.value.decode(errors="replace") if name.value else f"CUDA error {result}"
    message_text = message.value.decode(errors="replace") if message.value else "unknown error"
    raise RuntimeError(f"{operation} failed: {name_text}: {message_text}")


_HOST_REGISTER_PORTABLE = 1


def host_register(pointer: int, nbytes: int) -> int:
    """Page-locks ``nbytes`` of host memory at ``pointer`` for every GPU context; returns the
    driver status (0 on success) so the caller can stop at the system's page-lock cap."""
    if is_amd_device():
        return hip_runtime().hipHostRegister(pointer, nbytes, _HOST_REGISTER_PORTABLE)
    return cuda_driver().cuMemHostRegister_v2(pointer, nbytes, _HOST_REGISTER_PORTABLE)


def host_unregister(pointer: int) -> None:
    if is_amd_device():
        check_hip(hip_runtime().hipHostUnregister(pointer), "hipHostUnregister")
    else:
        check_cuda(cuda_driver().cuMemHostUnregister(pointer), "cuMemHostUnregister")


def current_context() -> ctypes.c_void_p:
    lib = cuda_driver()
    context = ctypes.c_void_p()
    check_cuda(lib.cuCtxGetCurrent(ctypes.byref(context)), "cuCtxGetCurrent")
    return context


def create_stream() -> int:
    """Create a blocking CUDA stream in the current context and return its handle."""
    handle = ctypes.c_void_p()
    check_cuda(cuda_driver().cuStreamCreate(ctypes.byref(handle), 0), "cuStreamCreate")
    return handle.value


def destroy_stream(stream: int) -> None:
    check_cuda(cuda_driver().cuStreamDestroy(ctypes.c_void_p(stream)), "cuStreamDestroy")


class ModuleApi(NamedTuple):
    """The vendor's module-loading entry points; HIP mirrors the CUDA driver signatures."""

    image_suffix: str
    owner: Callable[[], int]
    load_data: Callable
    get_function: Callable
    launch_kernel: Callable
    check: Callable[[int, str], None]


def _current_cuda_context() -> int:
    check_cuda(cuda_driver().cuInit(0), "cuInit")
    context = current_context()
    if not context.value:
        raise RuntimeError("No current CUDA context while loading a precompiled kernel")
    return context.value


def _current_hip_device() -> int:
    device = ctypes.c_int()
    check_hip(hip_runtime().hipGetDevice(ctypes.byref(device)), "hipGetDevice")
    return device.value


def module_api() -> ModuleApi:
    if is_amd_device():
        lib = hip_runtime()
        return ModuleApi(
            ".hipfb", _current_hip_device,
            lib.hipModuleLoadData, lib.hipModuleGetFunction, lib.hipModuleLaunchKernel, check_hip,
        )
    lib = cuda_driver()
    return ModuleApi(
        ".fatbin", _current_cuda_context,
        lib.cuModuleLoadData, lib.cuModuleGetFunction, lib.cuLaunchKernel, check_cuda,
    )


def load_module(api: ModuleApi, module_name: str) -> ctypes.c_void_p:
    image_name = module_name + api.image_suffix
    owner = api.owner()

    with _driver_lock:
        key = (owner, image_name)
        cached = _modules.get(key)
        if cached is not None:
            return cached

        path = kernel_image_path(image_name)
        try:
            image = ctypes.create_string_buffer(path.read_bytes())
        except OSError as exc:
            raise RuntimeError(f"Missing precompiled GPU kernel: {path}") from exc
        module = ctypes.c_void_p()
        api.check(api.load_data(ctypes.byref(module), image), api.load_data.__name__)
        _modules[key] = module
        return module


def resolve_function(api: ModuleApi, module_name: str, function_name: str) -> ctypes.c_void_p:
    module = load_module(api, module_name)
    function = ctypes.c_void_p()
    api.check(
        api.get_function(ctypes.byref(function), module, function_name.encode("ascii")),
        f"{api.get_function.__name__}({function_name})",
    )
    return function


def grid_size(extent: int, block: int) -> int:
    return (extent + block - 1) // block


class Kernel:
    """One precompiled function launched through a reused argument buffer.

    ``module_name`` is the ``.cu`` stem; the vendor picks ``.fatbin`` or
    ``.hipfb``. ``arg_types`` are the ctypes of the kernel parameters in order;
    ``launch`` fills them from plain Python ints/floats.
    ``shared_memory_bytes`` is the dynamic shared memory each block gets
    (Triton-built kernels declare theirs that way). The lock keeps one
    instance safe to share between threads.
    """

    def __init__(
        self,
        module_name: str,
        function_name: str,
        arg_types: Sequence[type],
        shared_memory_bytes: int = 0,
    ):
        self.module_name = module_name
        self.function_name = function_name
        self.shared_memory_bytes = shared_memory_bytes
        self._api: ModuleApi | None = None
        self._function: ctypes.c_void_p | None = None
        self._values = [arg_type() for arg_type in arg_types]
        self._params = (ctypes.c_void_p * len(self._values))(
            *(ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in self._values)
        )
        self._lock = threading.Lock()

    def launch(
        self,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: Sequence[int | float],
        stream: int,
    ) -> None:
        with self._lock:
            if self._function is None:
                self._api = module_api()
                self._function = resolve_function(self._api, self.module_name, self.function_name)
            for value, arg in zip(self._values, args, strict=True):
                value.value = arg
            api = self._api
            api.check(
                api.launch_kernel(
                    self._function, *grid, *block, self.shared_memory_bytes, ctypes.c_void_p(stream),
                    self._params, None,
                ),
                f"{api.launch_kernel.__name__}({self.function_name})",
            )


_TRITON_SCRATCH_ARGS = 2
_TRITON_SCRATCH = (0,) * _TRITON_SCRATCH_ARGS
TritonLaunch = tuple[str, tuple[int, int, int], int, int]
"""A launch-table entry: (function, grid, threads per block, dynamic shared memory bytes)."""


def _triton_ctype(kind: str) -> type:
    if kind.startswith("*"):
        return ctypes.c_void_p
    return {"i32": ctypes.c_int32, "fp32": ctypes.c_float}[kind.split(":")[0]]


class TritonKernel:
    """A Triton kernel built ahead of time by ``scripts/triton_hipfb.py``, with its launch-table entry.

    ``arguments`` maps each runtime argument, in order, to the Triton type it was built for (the
    build's own signature). ``launch`` takes tensors for the pointers and ints/floats for the rest,
    then passes Triton's scratch pointers, which the build guarantees are unused.
    """

    def __init__(self, module_name: str, launch: TritonLaunch, arguments: Mapping[str, str]):
        function, self.grid, threads, shared = launch
        self.module_name = module_name
        self.block = (threads, 1, 1)
        arg_types = (*map(_triton_ctype, arguments.values()), *(ctypes.c_void_p,) * _TRITON_SCRATCH_ARGS)
        self._kernel = Kernel(module_name, function, arg_types, shared)

    def launch(self, grid: tuple[int, int, int], stream: int, *args: torch.Tensor | int | float) -> None:
        values = (arg.data_ptr() if isinstance(arg, torch.Tensor) else arg for arg in args)
        self._kernel.launch(grid, self.block, (*values, *_TRITON_SCRATCH), stream)


_aot_kernels: dict[tuple[str, str], TritonKernel] = {}


def has_aot_kernel(launch_table: Mapping[str, Mapping[str, TritonLaunch]], module_name: str, device: torch.device) -> bool:
    """Whether ``device`` is an AMD GPU that ``launch_table`` (a generated ``KERNEL_LAUNCH``) has ``module_name`` for."""
    return is_amd_device(device) and module_name in launch_table.get(amd_gpu_architecture(device), {})


def aot_kernel(
    launch_table: Mapping[str, Mapping[str, TritonLaunch]],
    module_name: str,
    device: torch.device,
    arguments: Mapping[str, str],
) -> TritonKernel:
    """The ``module_name`` Triton kernel built ahead of time for ``device``'s AMD architecture, one per
    process and architecture."""
    architecture = amd_gpu_architecture(device)
    key = (module_name, architecture)
    kernel = _aot_kernels.get(key)
    if kernel is None:
        launch = launch_table.get(architecture, {}).get(module_name)
        if launch is None:
            raise RuntimeError(f"no ahead-of-time kernel {module_name} on {architecture}")
        kernel = _aot_kernels.setdefault(key, TritonKernel(module_name, launch, arguments))
    return kernel
