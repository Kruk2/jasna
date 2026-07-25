"""Loading and launching ahead-of-time compiled CUDA kernels.

Kernels ship as ``.fatbin`` files built from the ``.cu`` sources next to this
module (see ``docs/en/development.md``) and are loaded through the CUDA driver
API, so no toolkit is needed at run time.
"""
import ctypes
import os
import sys
import threading
from pathlib import Path

from jasna._frozen import is_frozen

_driver: ctypes.CDLL | None = None
_driver_lock = threading.Lock()
_modules: dict[tuple[int, str], ctypes.c_void_p] = {}


def fatbin_path(name: str) -> Path:
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
    lib.cuLaunchKernel.argtypes = [
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
    lib.cuLaunchKernel.restype = ctypes.c_int
    lib.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    lib.cuGetErrorName.restype = ctypes.c_int
    lib.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    lib.cuGetErrorString.restype = ctypes.c_int
    _driver = lib
    return lib


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


def current_context() -> ctypes.c_void_p:
    lib = cuda_driver()
    context = ctypes.c_void_p()
    check_cuda(lib.cuCtxGetCurrent(ctypes.byref(context)), "cuCtxGetCurrent")
    return context


def load_module(fatbin_name: str) -> ctypes.c_void_p:
    lib = cuda_driver()
    check_cuda(lib.cuInit(0), "cuInit")

    context = current_context()
    if not context.value:
        raise RuntimeError(f"No current CUDA context while loading {fatbin_name}")

    with _driver_lock:
        key = (context.value, fatbin_name)
        cached = _modules.get(key)
        if cached is not None:
            return cached

        path = fatbin_path(fatbin_name)
        try:
            image = ctypes.create_string_buffer(path.read_bytes())
        except OSError as exc:
            raise RuntimeError(f"Missing precompiled CUDA kernel: {path}") from exc
        module = ctypes.c_void_p()
        check_cuda(lib.cuModuleLoadData(ctypes.byref(module), image), "cuModuleLoadData")
        _modules[key] = module
        return module


def resolve_function(fatbin_name: str, function_name: str) -> ctypes.c_void_p:
    context = current_context()
    if not context.value:
        raise RuntimeError(f"No current CUDA context while resolving {function_name}")
    module = load_module(fatbin_name)
    function = ctypes.c_void_p()
    check_cuda(
        cuda_driver().cuModuleGetFunction(
            ctypes.byref(function), module, function_name.encode("ascii")
        ),
        f"cuModuleGetFunction({function_name})",
    )
    return function
