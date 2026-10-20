"""Loading and launching ahead-of-time compiled CUDA kernels.

Kernels ship as ``.fatbin`` files built from the ``.cu`` sources next to this
module (see ``docs/en/development.md``) and are loaded through the CUDA driver
API, so no toolkit is needed at run time.
"""
import ctypes
import hashlib
import logging
import os
import sys
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path

import torch

from jasna._frozen import is_frozen
from jasna.accelerator import is_nvidia_device
from jasna.os_utils import env_flag

log = logging.getLogger(__name__)

_HIP_KERNELS_ENV = "JASNA_HIP_KERNELS"

_driver: ctypes.CDLL | None = None
_hiprtc: ctypes.CDLL | None = None
_driver_lock = threading.Lock()
_modules: dict[tuple[int, str], ctypes.c_void_p] = {}

# Latched the first time a HIP kernel cannot be built or loaded on this machine.
# Every later request then takes the eager torch path instead of raising again:
# a card whose toolchain cannot build the kernels must degrade, not abort.
_hip_kernels_disabled: str | None = None

_HIPRTC_PREAMBLE = """
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef short int16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
"""


def _is_hip_runtime() -> bool:
    return getattr(torch.version, "hip", None) is not None


def disable_hip_kernels(reason: str) -> None:
    """Latch the fused HIP kernels off for the rest of the process.

    Called when a kernel cannot be compiled or loaded (a non-gfx1100 card whose
    hiprtc build fails, a missing driver/rtc DLL, ...).  After this every
    ``is_gpu_kernel_device`` returns False on HIP, so the callers keep using the
    eager torch implementation instead of raising again.
    """
    global _hip_kernels_disabled
    if _hip_kernels_disabled is None:
        _hip_kernels_disabled = reason
        log.warning("HIP kernels unavailable, using the torch fallback: %s", reason)


def handle_kernel_failure(operation: str, exc: BaseException) -> bool:
    """Latch HIP kernels off after a failure.

    Returns True when the caller should fall back to the eager implementation,
    False on a non-HIP runtime where the exception is a genuine error and must
    propagate (NVIDIA keeps its existing behaviour).
    """
    if not _is_hip_runtime():
        return False
    disable_hip_kernels(f"{operation}: {exc}")
    return True


def is_gpu_kernel_device(device: torch.device | str) -> bool:
    dev = torch.device(device)
    if dev.type != "cuda":
        return False
    if is_nvidia_device(dev):
        return True
    if (
        _is_hip_runtime()
        and _hip_kernels_disabled is None
        and env_flag(_HIP_KERNELS_ENV, default=True)
    ):
        return True
    return False


def fatbin_path(name: str) -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent / name
    return Path(__file__).resolve().with_name(name)


def _rocm_bin_dir() -> Path | None:
    torch_dir = Path(torch.__file__).resolve().parent
    candidate = torch_dir.parent / "_rocm_sdk_core" / "bin"
    if candidate.is_dir():
        return candidate
    return None


def _load_hip_driver() -> ctypes.CDLL:
    loader = ctypes.WinDLL if os.name == "nt" else ctypes.CDLL
    rocm_bin = _rocm_bin_dir()
    lib = None
    if os.name == "nt":
        names = ["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"]
        if rocm_bin is not None:
            for found in sorted(rocm_bin.glob("amdhip64*.dll"), reverse=True):
                names.insert(0, str(found))
        for name in names:
            try:
                lib = loader(name)
                break
            except OSError:
                continue
        if lib is None:
            raise RuntimeError("Could not load amdhip64 DLL for ROCm kernel execution")
    else:
        lib = loader("libamdhip64.so")

    lib.cuInit = lib.hipInit
    lib.cuInit.argtypes = [ctypes.c_uint]
    lib.cuInit.restype = ctypes.c_int

    lib.cuCtxGetCurrent = lib.hipCtxGetCurrent
    lib.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.cuCtxGetCurrent.restype = ctypes.c_int

    lib.cuModuleLoadData = lib.hipModuleLoadData
    lib.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    lib.cuModuleLoadData.restype = ctypes.c_int

    lib.cuModuleGetFunction = lib.hipModuleGetFunction
    lib.cuModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib.cuModuleGetFunction.restype = ctypes.c_int

    lib.cuLaunchKernel = lib.hipModuleLaunchKernel
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

    lib.cuGetErrorName = lib.hipGetErrorName
    lib.cuGetErrorName.argtypes = [ctypes.c_int]
    lib.cuGetErrorName.restype = ctypes.c_char_p

    lib.cuGetErrorString = lib.hipGetErrorString
    lib.cuGetErrorString.argtypes = [ctypes.c_int]
    lib.cuGetErrorString.restype = ctypes.c_char_p

    lib.cuStreamCreate = lib.hipStreamCreateWithFlags
    lib.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    lib.cuStreamCreate.restype = ctypes.c_int

    lib.cuStreamDestroy = lib.hipStreamDestroy
    lib.cuStreamDestroy.argtypes = [ctypes.c_void_p]
    lib.cuStreamDestroy.restype = ctypes.c_int
    return lib


def _load_hiprtc() -> ctypes.CDLL:
    global _hiprtc
    if _hiprtc is not None:
        return _hiprtc
    loader = ctypes.WinDLL if os.name == "nt" else ctypes.CDLL
    rocm_bin = _rocm_bin_dir()
    lib = None
    if os.name == "nt":
        names: list[str] = ["hiprtc0716.dll", "hiprtc.dll"]
        if rocm_bin is not None:
            for found in sorted(rocm_bin.glob("hiprtc*.dll"), reverse=True):
                if "builtins" not in found.name.lower():
                    names.insert(0, str(found))
        for name in names:
            try:
                lib = loader(name)
                break
            except OSError:
                continue
        if lib is None:
            raise RuntimeError("Could not load hiprtc DLL for ROCm kernel compilation")
    else:
        lib = loader("libhiprtc.so")

    lib.hiprtcCreateProgram.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.hiprtcCreateProgram.restype = ctypes.c_int
    lib.hiprtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.hiprtcDestroyProgram.restype = ctypes.c_int
    lib.hiprtcCompileProgram.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.hiprtcCompileProgram.restype = ctypes.c_int
    lib.hiprtcGetCodeSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    lib.hiprtcGetCodeSize.restype = ctypes.c_int
    lib.hiprtcGetCode.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.hiprtcGetCode.restype = ctypes.c_int
    lib.hiprtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    lib.hiprtcGetProgramLogSize.restype = ctypes.c_int
    lib.hiprtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.hiprtcGetProgramLog.restype = ctypes.c_int
    _hiprtc = lib
    return lib


def cuda_driver() -> ctypes.CDLL:
    global _driver
    if _driver is not None:
        return _driver

    if _is_hip_runtime():
        _driver = _load_hip_driver()
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


def check_cuda(result: int, operation: str) -> None:
    if result == 0:
        return
    lib = cuda_driver()
    if _is_hip_runtime():
        raw_name = lib.cuGetErrorName(result)
        raw_msg = lib.cuGetErrorString(result)
        name_text = raw_name.decode(errors="replace") if raw_name else f"HIP error {result}"
        message_text = raw_msg.decode(errors="replace") if raw_msg else "unknown error"
        raise RuntimeError(f"{operation} failed: {name_text}: {message_text}")
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


def create_stream() -> int:
    """Create a blocking CUDA/HIP stream in the current context and return its handle."""
    handle = ctypes.c_void_p()
    check_cuda(cuda_driver().cuStreamCreate(ctypes.byref(handle), 0), "cuStreamCreate")
    return handle.value


def destroy_stream(stream: int) -> None:
    check_cuda(cuda_driver().cuStreamDestroy(ctypes.c_void_p(stream)), "cuStreamDestroy")


def _runtime_tag() -> str:
    """ROCm build the cached binary was compiled against.

    A .hsaco only loads into the runtime that produced it, and the cache name carries
    the *source* digest, which does not change when ROCm does. Without this tag a
    binary left behind by another ROCm build is reused, hipModuleLoadData answers
    hipErrorNoBinaryForGpu, and that error latches onto the context: the torch fallback
    the caller logs right after it fails too. A stale artifact in the tree is enough to
    trigger it, so the name now says which runtime it belongs to.
    """
    hip = getattr(torch.version, "hip", None) or "unknown"
    return "".join(ch for ch in hip if ch.isalnum())


def _compile_or_load_hsaco(fatbin_name: str, dev_idx: int, *, refresh: bool = False) -> bytes:
    cu_name = Path(fatbin_name).with_suffix(".cu").name
    cu_path = fatbin_path(cu_name)
    src_bytes = cu_path.read_bytes()
    props = torch.cuda.get_device_properties(dev_idx)
    arch = getattr(props, "gcnArchName", "gfx1100").split(":")[0]
    digest = hashlib.sha256(src_bytes).hexdigest()[:8]
    hsaco_name = f"{cu_path.stem}.{arch}.{_runtime_tag()}.{digest}.hsaco"

    local_cache = cu_path.with_name(hsaco_name)
    temp_cache = Path(tempfile.gettempdir()) / "jasna-hiprtc-cache" / hsaco_name
    if refresh:
        for stale in (local_cache, temp_cache):
            try:
                stale.unlink()
            except OSError:
                pass
    else:
        if local_cache.is_file():
            return local_cache.read_bytes()
        if temp_cache.is_file():
            return temp_cache.read_bytes()

    hiprtc = _load_hiprtc()
    src = src_bytes.decode("utf-8")
    src_patched = src.replace("#include <stdint.h>", _HIPRTC_PREAMBLE).replace(
        "#include <cuda_fp16.h>", ""
    )
    prog = ctypes.c_void_p()
    res = hiprtc.hiprtcCreateProgram(
        ctypes.byref(prog),
        src_patched.encode("utf-8"),
        cu_name.encode("utf-8"),
        0,
        None,
        None,
    )
    if res != 0:
        raise RuntimeError(f"hiprtcCreateProgram({cu_name}) failed with code {res}")
    try:
        opts = (ctypes.c_char_p * 3)(
            f"--offload-arch={arch}".encode("utf-8"),
            b"-O3",
            b"-std=c++17",
        )
        res = hiprtc.hiprtcCompileProgram(prog, 3, opts)
        if res != 0:
            log_size = ctypes.c_size_t()
            hiprtc.hiprtcGetProgramLogSize(prog, ctypes.byref(log_size))
            log_buf = ctypes.create_string_buffer(log_size.value)
            hiprtc.hiprtcGetProgramLog(prog, log_buf)
            raise RuntimeError(
                f"hiprtcCompileProgram({cu_name}, {arch}) failed ({res}): "
                f"{log_buf.value.decode('utf-8', errors='replace')}"
            )
        code_size = ctypes.c_size_t()
        hiprtc.hiprtcGetCodeSize(prog, ctypes.byref(code_size))
        code_buf = ctypes.create_string_buffer(code_size.value)
        hiprtc.hiprtcGetCode(prog, code_buf)
        hsaco = code_buf.raw
    finally:
        hiprtc.hiprtcDestroyProgram(ctypes.byref(prog))

    for target in (local_cache, temp_cache):
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = target.with_suffix(f".tmp.{os.getpid()}")
            tmp_file.write_bytes(hsaco)
            tmp_file.replace(target)
            break
        except OSError:
            continue
    return hsaco


def load_module(fatbin_name: str, dev_idx: int | None = None) -> ctypes.c_void_p:
    lib = cuda_driver()
    check_cuda(lib.cuInit(0), "cuInit")

    if _is_hip_runtime():
        if dev_idx is None:
            dev_idx = torch.cuda.current_device()
        elif torch.cuda.current_device() != dev_idx:
            torch.cuda.set_device(dev_idx)
        with _driver_lock:
            key = (dev_idx, fatbin_name)
            cached = _modules.get(key)
            if cached is not None:
                return cached
            try:
                image = ctypes.create_string_buffer(_compile_or_load_hsaco(fatbin_name, dev_idx))
                module = ctypes.c_void_p()
                check_cuda(lib.cuModuleLoadData(ctypes.byref(module), image), "hipModuleLoadData")
            except RuntimeError:
                # A binary this runtime cannot load latches hipErrorNoBinaryForGpu onto the
                # context, so every later kernel - the caller's torch fallback included -
                # fails. Rebuild from source once and retry: a successful load clears it.
                image = ctypes.create_string_buffer(
                    _compile_or_load_hsaco(fatbin_name, dev_idx, refresh=True)
                )
                module = ctypes.c_void_p()
                check_cuda(lib.cuModuleLoadData(ctypes.byref(module), image), "hipModuleLoadData")
            _modules[key] = module
            return module

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


def resolve_function(
    fatbin_name: str, function_name: str, dev_idx: int | None = None
) -> ctypes.c_void_p:
    module = load_module(fatbin_name, dev_idx=dev_idx)
    function = ctypes.c_void_p()
    check_cuda(
        cuda_driver().cuModuleGetFunction(
            ctypes.byref(function), module, function_name.encode("ascii")
        ),
        f"cuModuleGetFunction({function_name})",
    )
    return function



def grid_size(extent: int, block: int) -> int:
    return (extent + block - 1) // block


class Kernel:
    """One fatbin function launched through a reused argument buffer.

    ``arg_types`` are the ctypes of the kernel parameters in order; ``launch``
    fills them from plain Python ints/floats. The lock keeps one instance safe
    to share between threads.
    """

    def __init__(self, fatbin_name: str, function_name: str, arg_types: Sequence[type]):
        self.fatbin_name = fatbin_name
        self.function_name = function_name
        self._function: ctypes.c_void_p | None = None
        self._functions_by_dev: dict[int, ctypes.c_void_p] = {}
        self._values = [arg_type() for arg_type in arg_types]
        self._params = (ctypes.c_void_p * len(self._values))(
            *(ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in self._values)
        )
        self._lock = threading.Lock()

    def _resolve(self, device: torch.device | None) -> ctypes.c_void_p:
        """The driver function handle, compiled/loaded on first use."""
        if _is_hip_runtime():
            dev_idx = (
                device.index
                if device is not None and device.index is not None
                else torch.cuda.current_device()
            )
            if torch.cuda.current_device() != dev_idx:
                torch.cuda.set_device(dev_idx)
            func = self._functions_by_dev.get(dev_idx)
            if func is None:
                func = resolve_function(self.fatbin_name, self.function_name, dev_idx=dev_idx)
                self._functions_by_dev[dev_idx] = func
            return func
        if self._function is None:
            self._function = resolve_function(self.fatbin_name, self.function_name)
        return self._function

    def ensure(self, device: torch.device | None = None) -> bool:
        """Build and load the kernel now, so a failure surfaces at construction.

        Returns True when the kernel is ready, False when the HIP path cannot
        build it (latched off for the rest of the process) so the caller keeps
        its eager implementation.  On a non-HIP runtime the exception propagates.
        """
        if _hip_kernels_disabled is not None:
            return False
        try:
            with self._lock:
                self._resolve(device)
            return True
        except Exception as exc:  # pragma: no cover - depends on the GPU/driver
            if not handle_kernel_failure(self.function_name, exc):
                raise
            return False

    def launch(
        self,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: Sequence[int | float],
        stream: int,
        device: torch.device | None = None,
    ) -> None:
        with self._lock:
            func = self._resolve(device)
            for value, arg in zip(self._values, args, strict=True):
                value.value = arg
            check_cuda(
                cuda_driver().cuLaunchKernel(
                    func, *grid, *block, 0, ctypes.c_void_p(stream), self._params, None
                ),
                f"cuLaunchKernel({self.function_name})",
            )


def build_kernel(
    fatbin_name: str,
    function_name: str,
    arg_types: Sequence[type],
    device: torch.device | str,
) -> Kernel | None:
    """A ``Kernel`` proven to work on ``device``, or None to stay on torch.

    ``Kernel`` construction alone does not compile anything: the HIP path
    compiles on first use.  We force that here so a card whose toolchain cannot
    build the kernel for its architecture (or whose driver/rtc DLLs are missing)
    degrades to the eager torch implementation instead of raising deep inside
    the pipeline.  Every caller already handles ``None`` as "no kernel".
    """
    if not is_gpu_kernel_device(device):
        return None
    kernel = Kernel(fatbin_name, function_name, arg_types)
    if not kernel.ensure(device):
        return None
    return kernel
