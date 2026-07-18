import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from jasna._frozen import is_frozen

logger = logging.getLogger(__name__)

MIN_GPU_COMPUTE = (7, 5)
MIN_DRIVER_VERSION = 580 if sys.platform == "linux" else 610

def check_supported_gpu(
    device: str = "cuda:0",
) -> tuple[bool, str] | tuple[bool, tuple[str, int, int]]:
    """Validate the GPU supported by the current vendor-specific build."""
    try:
        from jasna._suppress_noise import install as _install_noise_filters
        _install_noise_filters()
        import torch
        from jasna.accelerator import AcceleratorVendor, vendor_for_device
    except ImportError:
        return False, "no_cuda"
    if not torch.cuda.is_available():
        return False, "no_cuda"
    if vendor_for_device(device) is AcceleratorVendor.AMD:
        return True, torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    if capability < MIN_GPU_COMPUTE:
        return False, ("compute_too_low", capability[0], capability[1])
    return True, torch.cuda.get_device_name(device)


def check_nvidia_gpu(
    device: str = "cuda:0",
) -> tuple[bool, str] | tuple[bool, tuple[str, int, int]]:
    """Backward-compatible alias for callers/tests predating AMD builds."""
    return check_supported_gpu(device)


def _bundled_exe_filename(name: str) -> str:
    if os.name == "nt" and not name.lower().endswith(".exe"):
        return f"{name}.exe"
    return name


def _find_bundled_executable(name: str) -> Path | None:
    if not is_frozen():
        return None

    exe = _bundled_exe_filename(name)
    # Nuitka standalone has no _internal/ (a PyInstaller-ism); bundled tools sit at the
    # dist root next to the binary, alongside model_weights/ and assets/.
    base = Path(sys.executable).parent

    if name in {"ffmpeg", "ffprobe"}:
        p = base / "tools" / exe
        return p if p.is_file() else None

    return None


_COMMON_EXECUTABLE_LOCATIONS: dict[str, tuple[str, ...]] = {
    "nvidia-smi": (
        # Windows
        r"C:\Windows\System32\nvidia-smi.exe",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        # Linux / WSL
        "/usr/bin/nvidia-smi",
        "/usr/local/bin/nvidia-smi",
        "/usr/local/nvidia/bin/nvidia-smi",
        "/usr/lib/wsl/lib/nvidia-smi",
    ),
}


def _find_in_common_locations(name: str) -> str | None:
    for candidate in _COMMON_EXECUTABLE_LOCATIONS.get(name, ()):
        if os.path.isfile(candidate):
            return candidate
    return None


def find_executable(name: str) -> str | None:
    bundled = _find_bundled_executable(name)
    if bundled is not None:
        return str(bundled)
    found = shutil.which(name)
    if found is not None:
        return found
    return _find_in_common_locations(name)


def resolve_executable(name: str) -> str:
    found = find_executable(name)
    return found if found is not None else name



def get_subprocess_startup_info():
    if os.name != "nt":
        return None
    startup_info = subprocess.STARTUPINFO()
    startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return startup_info


STD_INPUT_HANDLE = -10
STD_OUTPUT_HANDLE = -11
STD_ERROR_HANDLE = -12


def _redirect_std_streams_to_null() -> None:
    """Point sys.stdout/stderr/stdin AND the process's OS-level standard handles at NUL.

    After FreeConsole the inherited console handles are dead, and two things break.
    First, a stray `print()` or a logging StreamHandler writing to the real stream raises
    `OSError: [WinError 6] The handle is invalid` — repointing sys.* at NUL discards those
    writes. Second — and this is what randomly fails the system check and later crashes
    ffprobe — `subprocess.Popen` calls `GetStdHandle()` for every stream the
    caller left as None (our tools run with stdout/stderr=PIPE but stdin unset), then
    `DuplicateHandle`s that dangling handle and dies with `[WinError 6]`/`[WinError 50]`.
    Repointing the OS std handles at NUL gives each child a valid, inheritable handle to
    duplicate; fixing sys.* alone does not."""
    null_r = open(os.devnull, "r")
    null_w = open(os.devnull, "w")
    sys.stdin = null_r
    sys.stdout = null_w
    sys.stderr = null_w
    if os.name != "nt":
        return
    import ctypes
    import msvcrt

    kernel32 = ctypes.windll.kernel32
    kernel32.SetStdHandle(STD_INPUT_HANDLE, ctypes.c_void_p(msvcrt.get_osfhandle(null_r.fileno())))
    kernel32.SetStdHandle(STD_OUTPUT_HANDLE, ctypes.c_void_p(msvcrt.get_osfhandle(null_w.fileno())))
    kernel32.SetStdHandle(STD_ERROR_HANDLE, ctypes.c_void_p(msvcrt.get_osfhandle(null_w.fileno())))


def drop_console_window() -> None:
    """Detach the console on a frozen GUI launch (Windows). The frozen binary is built
    console-subsystem (`--windows-console-mode=force`) so the CLI blocks cmd and stdout/stderr
    work; for a GUI/double-click launch we don't want that console window, so release it.
    No-op off Windows and in dev, where the console is the developer's own terminal."""
    if sys.platform != "win32" or not is_frozen():
        return
    import ctypes

    ctypes.windll.kernel32.FreeConsole()
    _redirect_std_streams_to_null()


def subprocess_no_window_kwargs() -> dict:
    """Popen/run kwargs that suppress a child's console window on Windows.

    The GUI drops its own console (FreeConsole), so a console-subsystem child like
    ffmpeg/ffprobe would otherwise pop its own cmd window. We capture their output
    via pipes, so they never need an inherited console. No-op (empty) off Windows."""
    if os.name != "nt":
        return {}
    return {
        "startupinfo": get_subprocess_startup_info(),
        "creationflags": subprocess.CREATE_NO_WINDOW,
    }


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


def check_required_executables() -> None:
    """Check that required external tools are available in PATH and callable."""
    missing: list[str] = []
    wrong_version: list[str] = []
    checks = {
        "ffprobe": ["-version"],
    }
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

        if exe == "ffprobe":
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
        if "ffprobe" in missing:
            print(FFMPEG_DOWNLOAD_LINKS)
            logger.info("%s", FFMPEG_DOWNLOAD_LINKS)
        sys.exit(1)
    if wrong_version:
        msg = (
            "Error: ffprobe major version must be exactly 8 (or detectable as 8): "
            + ", ".join(wrong_version)
        )
        print(msg)
        logger.info("%s", msg)
        print(FFMPEG_DOWNLOAD_LINKS)
        logger.info("%s", FFMPEG_DOWNLOAD_LINKS)
        sys.exit(1)


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
    try:
        import torch

        hip_version = getattr(torch.version, "hip", None)
        if hip_version:
            if not torch.cuda.is_available():
                return False, f"ROCm {hip_version} is installed but no AMD GPU is available"
            return True, f"ROCm {hip_version}"
    except ImportError:
        pass

    nvidia_smi = find_executable("nvidia-smi")
    if not nvidia_smi:
        return False, "nvidia-smi not found"
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            **subprocess_no_window_kwargs(),
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
    if is_frozen():
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
