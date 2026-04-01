import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

MIN_GPU_COMPUTE = (7, 5)
MIN_DRIVER_VERSION = 590


def check_nvidia_gpu() -> tuple[bool, str] | tuple[bool, tuple[str, int, int]]:
    """Return (True, gpu_name) or (False, "no_cuda") or (False, ("compute_too_low", major, minor))."""
    try:
        from jasna._suppress_noise import install as _install_noise_filters
        _install_noise_filters()
        import torch
    except ImportError:
        return False, "no_cuda"
    if not torch.cuda.is_available():
        return False, "no_cuda"
    capability = torch.cuda.get_device_capability(0)
    if capability < MIN_GPU_COMPUTE:
        return False, ("compute_too_low", capability[0], capability[1])
    return True, torch.cuda.get_device_name(0)


def _bundled_exe_filename(name: str) -> str:
    if os.name == "nt" and not name.lower().endswith(".exe"):
        return f"{name}.exe"
    return name


def _find_bundled_executable(name: str) -> Path | None:
    if not getattr(sys, "frozen", False):
        return None

    exe = _bundled_exe_filename(name)
    base = Path(sys.executable).parent
    internal = base / "_internal"

    if name in {"ffmpeg", "ffprobe"}:
        p = internal / "tools" / exe
        return p if p.is_file() else None

    if name == "mkvmerge":
        root = internal / "mkvtoolnix"
        direct = root / exe
        if direct.is_file():
            return direct
        if root.is_dir():
            for candidate in root.rglob(exe):
                if candidate.is_file():
                    return candidate
        return None

    return None


def find_executable(name: str) -> str | None:
    bundled = _find_bundled_executable(name)
    if bundled is not None:
        return str(bundled)
    return shutil.which(name)


def resolve_executable(name: str) -> str:
    found = find_executable(name)
    return found if found is not None else name



def get_subprocess_startup_info():
    if os.name != "nt":
        return None
    startup_info = subprocess.STARTUPINFO()
    startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return startup_info


def _parse_ffmpeg_major_version(version_output: str) -> int:
    def parse_major_from_libavutil(out: str) -> int:
        m = re.search(r"(?m)^\s*libavutil\s+(\d+)\.", out)
        if not m:
            raise ValueError("Could not parse libavutil version from ffmpeg output")
        libavutil_major = int(m.group(1))
        return libavutil_major - 52

    first_line = version_output.splitlines()[0] if version_output else ""
    m = re.match(r"^\s*(?:ffmpeg|ffprobe)\s+version\s+(\S+)", first_line)
    if not m:
        raise ValueError(f"Unexpected ffmpeg/ffprobe version output: {first_line!r}")

    ver = m.group(1)
    if ver.startswith(("N-", "git-", "GIT-")):
        return parse_major_from_libavutil(version_output)

    m = re.search(r"(\d+)", ver)
    if not m:
        raise ValueError(f"Could not parse ffmpeg major version from: {ver!r}")
    return int(m.group(1))


FFMPEG_DOWNLOAD_LINKS = (
    "Linux: https://www.ffmpeg.org/download.html  "
    "Windows: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full.7z"
)


def check_required_executables(disable_ffmpeg_check: bool = False) -> None:
    """Check that required external tools are available in PATH and callable."""
    missing: list[str] = []
    wrong_version: list[str] = []
    checks = {
        "ffprobe": ["-version"],
        "ffmpeg": ["-version"],
        "mkvmerge": ["--version"],
    }
    if disable_ffmpeg_check:
        checks = {k: v for k, v in checks.items() if k != "ffprobe" and k != "ffmpeg"}
    for exe, args in checks.items():
        exe_path = find_executable(exe)
        if exe_path is None:
            missing.append(exe)
            continue
        cmd = [exe_path] + list(args)
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            missing.append(exe)
            continue
        if completed.returncode != 0:
            logger.error(
                "%s failed (exit code %s). stdout:\n%s\nstderr:\n%s",
                exe,
                completed.returncode,
                completed.stdout or "",
                completed.stderr or "",
            )
            missing.append(exe)
            continue

        if exe in {"ffprobe", "ffmpeg"}:
            try:
                major = _parse_ffmpeg_major_version((completed.stdout or "") + (completed.stderr or ""))
            except ValueError:
                wrong_version.append(f"{exe} (could not detect major version)")
                continue
            if major != 8:
                wrong_version.append(f"{exe} (detected major={major})")

    if missing:
        msg = f"Error: Required executable(s) not found in PATH or not callable: {', '.join(missing)}"
        print(msg)
        logger.info("%s", msg)
        print("Please install them and ensure they are available in your system PATH and runnable.")
        logger.info("Please install them and ensure they are available in your system PATH and runnable.")
        if "ffmpeg" in missing or "ffprobe" in missing:
            print(FFMPEG_DOWNLOAD_LINKS)
            logger.info("%s", FFMPEG_DOWNLOAD_LINKS)
        sys.exit(1)
    if wrong_version:
        msg = (
            "Error: ffmpeg/ffprobe major version must be exactly 8 (or detectable as 8): "
            + ", ".join(wrong_version)
        )
        print(msg)
        logger.info("%s", msg)
        print(FFMPEG_DOWNLOAD_LINKS)
        logger.info("%s", FFMPEG_DOWNLOAD_LINKS)
        sys.exit(1)


def warn_if_windows_hardware_accelerated_gpu_scheduling_enabled() -> None:
    ok, info = check_windows_hardware_accelerated_gpu_scheduling()
    if not ok:
        if "Enabled" in info:
            msg = (
                "Warning: Windows 'Hardware-accelerated GPU scheduling' is enabled. "
                "This will make Jasna slower and might add artifacts to the output video."
            )
        else:
            msg = f"Warning: Could not determine Windows 'Hardware-accelerated GPU scheduling' status: {info}"
        print(msg)
        logger.info("%s", msg)


def _check_hags_d3dkmt() -> tuple[bool, str] | None:
    """Query the actual runtime HAGS state via D3DKMTQueryAdapterInfo.

    Returns (ok, info) or None if the API is unavailable.
    """
    import ctypes
    import ctypes.wintypes as wt

    try:
        gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
    except OSError:
        return None

    class _LUID(ctypes.Structure):
        _fields_ = [("LowPart", wt.DWORD), ("HighPart", wt.LONG)]

    # D3DKMT_HANDLE is UINT, not a pointer
    class _AI(ctypes.Structure):
        _fields_ = [
            ("hAdapter", wt.UINT),
            ("AdapterLuid", _LUID),
            ("NumOfSources", wt.ULONG),
            ("bPrecisePresentRegionsPreferred", wt.BOOL),
        ]

    class _EA2(ctypes.Structure):
        _fields_ = [("NumAdapters", wt.ULONG), ("pAdapters", ctypes.c_void_p)]

    class _QAI(ctypes.Structure):
        _fields_ = [
            ("hAdapter", wt.UINT),
            ("Type", wt.UINT),
            ("pPrivateDriverData", ctypes.c_void_p),
            ("PrivateDriverDataSize", wt.UINT),
        ]

    class _CLOSE(ctypes.Structure):
        _fields_ = [("hAdapter", wt.UINT)]

    _KMTQAITYPE_ADAPTERTYPE = 15
    _KMTQAITYPE_WDDM_2_7_CAPS = 70
    _ADAPTER_TYPE_RENDER = 0x1
    _ADAPTER_TYPE_SOFTWARE = 0x4

    try:
        enum_fn = gdi32.D3DKMTEnumAdapters2
        enum_fn.restype = ctypes.c_long
        enum_fn.argtypes = [ctypes.c_void_p]
        query_fn = gdi32.D3DKMTQueryAdapterInfo
        query_fn.restype = ctypes.c_long
        query_fn.argtypes = [ctypes.c_void_p]
        close_fn = gdi32.D3DKMTCloseAdapter
        close_fn.restype = ctypes.c_long
        close_fn.argtypes = [ctypes.c_void_p]
    except AttributeError:
        return None

    ea = _EA2(NumAdapters=0, pAdapters=None)
    status = enum_fn(ctypes.byref(ea))
    if status != 0 or ea.NumAdapters == 0:
        logger.debug("D3DKMTEnumAdapters2 count call: status=0x%08X, n=%d", status & 0xFFFFFFFF, ea.NumAdapters)
        return None
    adapters = (_AI * ea.NumAdapters)()
    ea.pAdapters = ctypes.cast(adapters, ctypes.c_void_p)
    status = enum_fn(ctypes.byref(ea))
    if status != 0 or ea.NumAdapters == 0:
        return None

    for i in range(ea.NumAdapters):
        handle = adapters[i].hAdapter
        atype = wt.UINT(0)
        qai = _QAI(
            hAdapter=handle,
            Type=_KMTQAITYPE_ADAPTERTYPE,
            pPrivateDriverData=ctypes.cast(ctypes.pointer(atype), ctypes.c_void_p),
            PrivateDriverDataSize=ctypes.sizeof(atype),
        )
        if query_fn(ctypes.byref(qai)) != 0:
            continue
        if not (atype.value & _ADAPTER_TYPE_RENDER) or (atype.value & _ADAPTER_TYPE_SOFTWARE):
            continue

        caps = wt.UINT(0)
        qai = _QAI(
            hAdapter=handle,
            Type=_KMTQAITYPE_WDDM_2_7_CAPS,
            pPrivateDriverData=ctypes.cast(ctypes.pointer(caps), ctypes.c_void_p),
            PrivateDriverDataSize=ctypes.sizeof(caps),
        )
        status = query_fn(ctypes.byref(qai))
        close_fn(ctypes.byref(_CLOSE(hAdapter=handle)))
        if status != 0:
            logger.debug("D3DKMTQueryAdapterInfo WDDM_2_7_CAPS: status=0x%08X", status & 0xFFFFFFFF)
            return None

        if bool(caps.value & 0x2):
            return False, "Enabled (recommended OFF: can slow Jasna and add artifacts)"
        return True, "Off"

    return None


def check_windows_hardware_accelerated_gpu_scheduling() -> tuple[bool, str]:
    if sys.platform != "win32":
        return True, "N/A"

    result = _check_hags_d3dkmt()
    if result is not None:
        return result
    return False, "Could not query runtime HAGS state (D3DKMT API unavailable or no hardware GPU found)"


_CUDA_SYSMEM_FALLBACK_POLICY_ID = 0x10ECECC9
_PREFER_NO_SYSMEM_FALLBACK = 1
_PREFER_SYSMEM_FALLBACK = 2


def _read_drs_setting(setting_id: int) -> int | None:
    import struct

    drs_path = os.path.join(
        os.environ.get("ProgramData", r"C:\ProgramData"),
        "NVIDIA Corporation",
        "Drs",
        "nvdrsdb0.bin",
    )
    with open(drs_path, "rb") as f:
        data = f.read()
    target = struct.pack("<I", setting_id)
    pos = data.find(target)
    if pos < 0:
        return None
    # DRS binary record: ID(4) + type_flags(4) + value(4) + ...
    value_offset = pos + 8
    if value_offset + 4 > len(data):
        return None
    return struct.unpack("<I", data[value_offset : value_offset + 4])[0]


def check_windows_nvidia_sysmem_fallback_policy() -> tuple[bool, str]:
    if sys.platform != "win32":
        return True, "N/A"

    try:
        value = _read_drs_setting(_CUDA_SYSMEM_FALLBACK_POLICY_ID)
    except OSError as e:
        return False, str(e)

    if value == _PREFER_NO_SYSMEM_FALLBACK:
        return True, "Prefer No Sysmem Fallback"
    if value == _PREFER_SYSMEM_FALLBACK:
        return False, "Prefer Sysmem Fallback (recommended: Prefer No Sysmem Fallback)"
    return False, "Driver Default (recommended: Prefer No Sysmem Fallback)"


def check_gpu_driver_version() -> tuple[bool, str]:
    nvidia_smi = find_executable("nvidia-smi")
    if not nvidia_smi:
        return False, "nvidia-smi not found"
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            startupinfo=get_subprocess_startup_info(),
        )
    except OSError as e:
        return False, str(e)
    if result.returncode != 0:
        return False, f"nvidia-smi exited with code {result.returncode}"
    version_str = result.stdout.strip().split("\n")[0].strip()
    m = re.match(r"(\d+)\.(\d+)", version_str)
    if not m:
        return False, f"Could not parse driver version: {version_str!r}"
    major = int(m.group(1))
    if major < MIN_DRIVER_VERSION:
        return False, f"{version_str} (requires {MIN_DRIVER_VERSION}+)"
    return True, version_str


def check_ascii_install_path() -> tuple[bool, str]:
    if getattr(sys, "frozen", False):
        install_path = Path(sys.executable).parent
    else:
        install_path = Path(__file__).resolve().parent
    path_str = str(install_path)
    try:
        path_str.encode("ascii")
    except UnicodeEncodeError:
        return False, path_str
    return True, path_str


def get_user_config_dir(app_name: str) -> Path:
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / app_name
        return Path.home() / app_name

    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / app_name
    return Path.home() / ".config" / app_name

