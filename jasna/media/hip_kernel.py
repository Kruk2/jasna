"""Loading and launching ahead-of-time compiled AMD HIP kernels.

The platform-specific code objects are built from the same ``.cu`` sources as
the NVIDIA fatbins. They are loaded through the HIP module API, so a ROCm
compiler is needed only when rebuilding the code objects, not at run time.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import sys
import threading
from collections.abc import Mapping
from pathlib import Path

import torch

from jasna._frozen import is_frozen
from jasna.accelerator import is_amd_device

AMD_HIP_COLOR_KERNELS_ENV = "JASNA_AMD_HIP_COLOR_KERNELS"
_SUPPORTED_ARCHITECTURES = frozenset({"gfx1100"})
_LINUX_COLOR_CODE_OBJECTS = (
    "yuv_to_rgb.gfx1100.hsaco",
    "rgb_to_yuv.gfx1100.hsaco",
)
_WINDOWS_COLOR_CODE_OBJECTS = (
    "yuv_to_rgb.gfx1100.windows.co",
    "rgb_to_yuv.gfx1100.windows.co",
)
# Kept as the Linux tuple for compatibility with the existing Linux build and
# tests. New code should use ``required_color_code_objects``.
_REQUIRED_COLOR_CODE_OBJECTS = _LINUX_COLOR_CODE_OBJECTS
_WINDOWS_MANIFEST = "hip_color_kernels.gfx1100.windows.json"
_WINDOWS_MANIFEST_SCHEMA = "jasna.hip-color-kernels.windows.v1"
_WINDOWS_PARAMETER_ABI = "jasna.hip-module-kernel-params.v1"

_runtime: ctypes.CDLL | None = None
_runtime_lock = threading.Lock()
_modules: dict[tuple[int, str], ctypes.c_void_p] = {}
_functions: dict[tuple[int, str, str], ctypes.c_void_p] = {}


def _override_from_environment(environ: Mapping[str, str]) -> bool | None:
    raw = environ.get(AMD_HIP_COLOR_KERNELS_ENV, "auto").strip().casefold()
    if raw in {"", "auto"}:
        return None
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"{AMD_HIP_COLOR_KERNELS_ENV} must be auto, 0/1, false/true, no/yes, or off/on"
    )


def hip_color_kernels_enabled(
    device: torch.device | str,
    *,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether the validated gfx1100 product route applies.

    Auto-selection remains limited to the validated Linux AMD target. Windows
    stays opt-in because its accepted build is pinned to one exact HIP runtime
    and has only a modest end-to-end gain. An explicit request fails closed
    instead of silently falling back on an unsupported, incomplete, or
    ABI-mismatched installation.
    """

    override = _override_from_environment(os.environ if environ is None else environ)
    if override is False:
        return False
    resolved = torch.device(device)
    supported_platform = sys.platform in {"linux", "win32"}
    eligible = (
        supported_platform
        and resolved.type == "cuda"
        and is_amd_device(resolved)
        and getattr(torch.version, "hip", None) is not None
        and torch.cuda.is_available()
    )
    if not eligible:
        if override is True:
            raise RuntimeError(
                f"{AMD_HIP_COLOR_KERNELS_ENV}=1 is supported only on an "
                "available Linux or Windows AMD/ROCm device"
            )
        return False
    if sys.platform == "win32" and override is None:
        # Windows remains explicit-only under its exact runtime contract.
        return False
    properties = torch.cuda.get_device_properties(resolved)
    architecture = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    if architecture not in _SUPPORTED_ARCHITECTURES:
        if override is True:
            raise RuntimeError(
                f"{AMD_HIP_COLOR_KERNELS_ENV}=1 supports "
                f"{sorted(_SUPPORTED_ARCHITECTURES)}, got {architecture!r}"
            )
        return False
    required = required_color_code_objects()
    missing = [
        str(code_object_path(name))
        for name in required
        if not code_object_path(name).is_file()
    ]
    if missing:
        if override is True:
            raise RuntimeError(
                f"{AMD_HIP_COLOR_KERNELS_ENV}=1 is missing precompiled kernels: {missing}"
            )
        return False
    if sys.platform == "win32":
        _validate_windows_bundle(architecture)
    return True


def required_color_code_objects(platform: str | None = None) -> tuple[str, str]:
    selected = sys.platform if platform is None else platform
    if selected == "win32":
        return _WINDOWS_COLOR_CODE_OBJECTS
    return _LINUX_COLOR_CODE_OBJECTS


def color_code_object_name(stem: str, platform: str | None = None) -> str:
    selected = sys.platform if platform is None else platform
    suffix = ".gfx1100.windows.co" if selected == "win32" else ".gfx1100.hsaco"
    return f"{stem}{suffix}"


def code_object_path(name: str) -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent / name
    return Path(__file__).resolve().with_name(name)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hip_major(version: object) -> int:
    text = str(version or "").strip()
    head, separator, _tail = text.partition(".")
    if not separator or not head.isdigit():
        raise RuntimeError(f"cannot determine HIP major version from {text!r}")
    return int(head)


def _validate_windows_bundle(architecture: str) -> None:
    manifest_path = code_object_path(_WINDOWS_MANIFEST)
    if not manifest_path.is_file():
        raise RuntimeError(f"missing Windows HIP colour-kernel manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot read Windows HIP colour-kernel manifest {manifest_path}: {exc}"
        ) from exc

    expected = {
        "schema": _WINDOWS_MANIFEST_SCHEMA,
        "platform": "win32",
        "architecture": architecture,
        "hip_major": _hip_major(getattr(torch.version, "hip", None)),
        "torch_hip_runtime": str(getattr(torch.version, "hip", "")),
        "code_object_abi": 4,
        "parameter_abi": _WINDOWS_PARAMETER_ABI,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"Windows HIP colour-kernel manifest {key} mismatch: "
                f"expected {value!r}, observed {manifest.get(key)!r}"
            )

    expected_dll = f"amdhip64_{expected['hip_major']}.dll"
    if str(manifest.get("runtime_dll", "")).casefold() != expected_dll.casefold():
        raise RuntimeError(
            "Windows HIP colour-kernel runtime DLL mismatch: "
            f"expected {expected_dll!r}, observed {manifest.get('runtime_dll')!r}"
        )
    runtime_file = _windows_runtime_file_identity()
    recorded_runtime_hash = manifest.get("runtime_dll_sha256")
    if runtime_file["runtime_dll"].casefold() != expected_dll.casefold():
        raise RuntimeError(
            "loaded Windows HIP runtime DLL mismatch: "
            f"expected {expected_dll!r}, observed {runtime_file['runtime_dll']!r}"
        )
    if not isinstance(recorded_runtime_hash, str) or len(recorded_runtime_hash) != 64:
        raise RuntimeError("Windows HIP colour-kernel manifest has no runtime DLL SHA256")
    if runtime_file["runtime_dll_sha256"] != recorded_runtime_hash.casefold():
        raise RuntimeError(
            "Windows HIP runtime DLL SHA256 mismatch: "
            f"expected {recorded_runtime_hash}, observed {runtime_file['runtime_dll_sha256']}"
        )
    recorded_runtime_api = manifest.get("runtime_api_version")
    if not isinstance(recorded_runtime_api, int):
        raise RuntimeError(
            "Windows HIP colour-kernel manifest has no runtime API version"
        )
    observed_runtime_api = _windows_runtime_api_version()
    if observed_runtime_api != recorded_runtime_api:
        raise RuntimeError(
            "Windows HIP runtime API version mismatch: "
            f"expected {recorded_runtime_api}, observed {observed_runtime_api}"
        )

    for section, names in (
        ("sources", ("yuv_to_rgb.cu", "rgb_to_yuv.cu")),
        ("artifacts", required_color_code_objects("win32")),
    ):
        recorded = manifest.get(section)
        if not isinstance(recorded, dict):
            raise RuntimeError(
                f"Windows HIP colour-kernel manifest is missing {section!r} hashes"
            )
        for name in names:
            path = code_object_path(name)
            expected_hash = recorded.get(name)
            if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                raise RuntimeError(
                    f"Windows HIP colour-kernel manifest has no SHA256 for {name}"
                )
            if section == "sources" and is_frozen() and not path.is_file():
                # Frozen products record the exact source identity in the
                # manifest but do not ship rebuild-only .cu files.
                continue
            observed_hash = _sha256_file(path)
            if observed_hash != expected_hash.casefold():
                raise RuntimeError(
                    f"Windows HIP colour-kernel SHA256 mismatch for {name}: "
                    f"expected {expected_hash}, observed {observed_hash}"
                )


def _windows_loaded_hip_runtime() -> tuple[str, int, Path]:
    hip_major = _hip_major(getattr(torch.version, "hip", None))
    dll_name = f"amdhip64_{hip_major}.dll"
    # PyTorch's ROCm wheel initializes and maps its matching HIP runtime before
    # this loader is used. Reuse that exact module instead of letting Windows
    # search PATH/System32 for another DLL with the same basename.
    torch.cuda.init()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    kernel32.GetModuleHandleW.restype = ctypes.c_void_p
    handle = kernel32.GetModuleHandleW(dll_name)
    if not handle:
        raise RuntimeError(
            f"PyTorch did not load its matching Windows HIP runtime {dll_name}"
        )
    kernel32.GetModuleFileNameW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_uint,
    ]
    kernel32.GetModuleFileNameW.restype = ctypes.c_uint
    buffer = ctypes.create_unicode_buffer(32768)
    length = kernel32.GetModuleFileNameW(handle, buffer, len(buffer))
    if not length:
        raise RuntimeError(f"cannot resolve the loaded Windows HIP runtime {dll_name}")
    return dll_name, int(handle), Path(buffer.value).resolve()


def _windows_runtime_file_identity() -> dict[str, str]:
    dll_name, _handle, path = _windows_loaded_hip_runtime()
    return {
        "runtime_dll": dll_name,
        "runtime_path": str(path),
        "runtime_dll_sha256": _sha256_file(path),
    }


def _windows_runtime_api_version() -> int:
    lib = hip_runtime()
    version = ctypes.c_int()
    check_hip(lib.hipRuntimeGetVersion(ctypes.byref(version)), "hipRuntimeGetVersion")
    return version.value


def _windows_hip_runtime() -> ctypes.CDLL:
    dll_name, handle, _path = _windows_loaded_hip_runtime()
    return ctypes.WinDLL(dll_name, handle=handle, use_last_error=True)


def hip_runtime() -> ctypes.CDLL:
    global _runtime
    if _runtime is not None:
        return _runtime

    with _runtime_lock:
        if _runtime is not None:
            return _runtime
        lib = (
            _windows_hip_runtime()
            if sys.platform == "win32"
            else ctypes.CDLL("libamdhip64.so")
        )
        lib.hipInit.argtypes = [ctypes.c_uint]
        lib.hipInit.restype = ctypes.c_int
        lib.hipCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.hipCtxGetCurrent.restype = ctypes.c_int
        lib.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        lib.hipModuleLoad.restype = ctypes.c_int
        lib.hipModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        lib.hipModuleGetFunction.restype = ctypes.c_int
        lib.hipModuleUnload.argtypes = [ctypes.c_void_p]
        lib.hipModuleUnload.restype = ctypes.c_int
        lib.hipModuleLaunchKernel.argtypes = [
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
        lib.hipModuleLaunchKernel.restype = ctypes.c_int
        lib.hipGetErrorName.argtypes = [ctypes.c_int]
        lib.hipGetErrorName.restype = ctypes.c_char_p
        lib.hipGetErrorString.argtypes = [ctypes.c_int]
        lib.hipGetErrorString.restype = ctypes.c_char_p
        lib.hipRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.hipRuntimeGetVersion.restype = ctypes.c_int
        runtime_version = ctypes.c_int()
        result = lib.hipRuntimeGetVersion(ctypes.byref(runtime_version))
        if result != 0:
            raise RuntimeError(
                f"hipRuntimeGetVersion failed while loading the HIP module API: {result}"
            )
        runtime_major = runtime_version.value // 10_000_000
        torch_major = _hip_major(getattr(torch.version, "hip", None))
        if runtime_major != torch_major:
            raise RuntimeError(
                "loaded HIP runtime major does not match PyTorch: "
                f"runtime={runtime_major}, torch={torch_major}"
            )
        _runtime = lib
        return lib


def hip_runtime_identity() -> dict[str, object]:
    """Return the exact HIP module runtime selected for probe evidence."""

    lib = hip_runtime()
    version = ctypes.c_int()
    check_hip(lib.hipRuntimeGetVersion(ctypes.byref(version)), "hipRuntimeGetVersion")
    identity: dict[str, object] = {
        "runtime_version": version.value,
        "torch_hip": str(getattr(torch.version, "hip", "")),
    }
    if sys.platform == "win32":
        identity.update(_windows_runtime_file_identity())
    else:
        identity["runtime_dll"] = "libamdhip64.so"
    return identity


def check_hip(result: int, operation: str) -> None:
    if result == 0:
        return
    lib = hip_runtime()
    name = lib.hipGetErrorName(result)
    message = lib.hipGetErrorString(result)
    name_text = name.decode(errors="replace") if name else f"HIP error {result}"
    message_text = message.decode(errors="replace") if message else "unknown error"
    raise RuntimeError(f"{operation} failed: {name_text}: {message_text}")


def current_context() -> ctypes.c_void_p:
    lib = hip_runtime()
    check_hip(lib.hipInit(0), "hipInit")
    context = ctypes.c_void_p()
    check_hip(lib.hipCtxGetCurrent(ctypes.byref(context)), "hipCtxGetCurrent")
    return context


def load_module(code_object_name: str) -> tuple[int, ctypes.c_void_p]:
    context = current_context()
    if not context.value:
        raise RuntimeError(
            f"No current HIP context while loading {code_object_name}; "
            "initialize the PyTorch ROCm device first"
        )
    path = code_object_path(code_object_name)
    key = (int(context.value), str(path))
    with _runtime_lock:
        cached = _modules.get(key)
        if cached is not None:
            return int(context.value), cached
        if not path.is_file():
            raise RuntimeError(f"Missing precompiled HIP kernel: {path}")
        module = ctypes.c_void_p()
        check_hip(
            hip_runtime().hipModuleLoad(
                ctypes.byref(module), os.fsencode(path)
            ),
            f"hipModuleLoad({path.name})",
        )
        _modules[key] = module
        # Modules intentionally remain cached for the HIP context lifetime.
        # Product converters keep raw function handles, so unloading an
        # apparently idle module would invalidate live converters. Process
        # teardown releases the context and its modules together.
        return int(context.value), module


def resolve_function(code_object_name: str, function_name: str) -> ctypes.c_void_p:
    context_value, module = load_module(code_object_name)
    path = code_object_path(code_object_name)
    key = (context_value, str(path), function_name)
    with _runtime_lock:
        cached = _functions.get(key)
        if cached is not None:
            return cached
        function = ctypes.c_void_p()
        check_hip(
            hip_runtime().hipModuleGetFunction(
                ctypes.byref(function), module, function_name.encode("ascii")
            ),
            f"hipModuleGetFunction({function_name})",
        )
        _functions[key] = function
        return function


def launch_kernel(
    function: ctypes.c_void_p,
    *,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    stream: int,
    params: ctypes.Array,
    operation: str,
) -> None:
    check_hip(
        hip_runtime().hipModuleLaunchKernel(
            function,
            *grid,
            *block,
            0,
            ctypes.c_void_p(stream),
            params,
            None,
        ),
        operation,
    )
