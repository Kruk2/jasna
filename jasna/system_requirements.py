"""Startup checks of the PC Jasna runs on: GPU, drivers, ffprobe, install folder and disk. The CLI
prints their English messages; the GUI wizard (``gui.system_checks``) translates the same results."""

import logging
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from jasna._frozen import is_frozen
from jasna.os_utils import (
    DATACENTER_NVIDIA_COMPUTE,
    find_executable,
    get_application_executable,
    subprocess_no_window_kwargs,
    windows_powershell,
)

logger = logging.getLogger(__name__)

MIN_GPU_COMPUTE = (7, 5)
MIN_DRIVER_VERSION = 580 if sys.platform == "linux" else 610
MIN_AMD_WINDOWS_DRIVER = (32, 0, 21017, 2052)
# AMD's loaded module version ends in the driver release: 30100200 = 30.10.2.
MIN_AMD_LINUX_DRIVER_RELEASE = 30100200
FFPROBE_MAJOR_VERSION = 8


def check_supported_gpu(
    device: str = "cuda:0",
) -> tuple[bool, str] | tuple[bool, tuple[str, int, int]]:
    """Validate the GPU supported by the current vendor-specific build."""
    try:
        import torch
        from jasna.accelerator import AcceleratorVendor, amd_gpu_architecture, vendor_for_device
        from jasna.rocm_bootstrap import kernel_pack_architectures
    except ImportError:
        logger.warning("PyTorch failed to import", exc_info=True)
        return False, "no_cuda"
    if not torch.cuda.is_available():
        return False, "no_cuda"
    if vendor_for_device(device) is AcceleratorVendor.AMD:
        # Without kernels for the GPU's arch the first kernel launch fails, and on Windows
        # crashes the process (issues #408, #661).
        arch = amd_gpu_architecture(device)
        built_archs = (kernel_pack_architectures(Path(torch.__file__).parent)
                       or {a.split(":")[0] for a in torch.cuda.get_arch_list()})
        if built_archs and arch not in built_archs:
            return False, ("arch_unsupported", arch)
        return True, torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    if capability < MIN_GPU_COMPUTE:
        return False, ("compute_too_low", capability[0], capability[1])
    if is_frozen() and capability in DATACENTER_NVIDIA_COMPUTE and not _ships_tensorrt_builder(capability):
        return False, ("datacenter_unsupported", torch.cuda.get_device_name(device))
    return True, torch.cuda.get_device_name(device)


def _ships_tensorrt_builder(capability: tuple[int, int]) -> bool:
    root = Path(get_application_executable()).parent
    pattern = f"*builder_resource_sm{capability[0]}{capability[1]}[._]*"
    return any(any(directory.glob(pattern)) for directory in (root, root / "tensorrt_libs"))


def gpu_check_error(result: str | tuple) -> str:
    """English CLI message for a failed check_supported_gpu result."""
    if result == "no_cuda":
        return "No compatible GPU was found for this Jasna build."
    if result[0] == "arch_unsupported":
        return f"This AMD GPU ({result[1]}) is not supported by this Jasna build."
    if result[0] == "datacenter_unsupported":
        return f"Jasna doesn't support data-center GPUs like A100, H100 or B200 (GPU: {result[1]}). Use an RTX GPU."
    _, major, minor = result
    return f"Compute capability 7.5+ required (GPU: {major}.{minor})."
def parse_ffmpeg_major_version(version_output: str) -> int:
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


@dataclass(frozen=True)
class FfprobeStatus:
    """The ffprobe Jasna would run: ``path`` None when there is none, ``runs`` False when it fails
    to start or exits with an error, ``major`` None when its version can't be read."""

    path: str | None
    runs: bool
    major: int | None

    @property
    def ok(self) -> bool:
        return self.major == FFPROBE_MAJOR_VERSION


def ffprobe_status() -> FfprobeStatus:
    path = find_executable("ffprobe")
    if path is None:
        return FfprobeStatus(None, False, None)
    try:
        completed = subprocess.run(
            [path, "-version"], capture_output=True, text=True, check=False, **subprocess_no_window_kwargs()
        )
    except OSError:
        logger.warning("ffprobe could not start", exc_info=True)
        return FfprobeStatus(path, False, None)
    if completed.returncode != 0:
        logger.error(
            "ffprobe failed (exit code %s). stdout:\n%s\nstderr:\n%s",
            completed.returncode,
            completed.stdout or "",
            completed.stderr or "",
        )
        return FfprobeStatus(path, False, None)
    try:
        major = parse_ffmpeg_major_version((completed.stdout or "") + (completed.stderr or ""))
    except ValueError:
        major = None
    return FfprobeStatus(path, True, major)


def check_required_executables() -> None:
    """Exit with what to install when ffprobe is missing, fails, or is not version 8."""
    status = ffprobe_status()
    if not status.runs:
        print("Error: Required executable(s) not found in PATH or not callable: ffprobe")
        print("Please install them and ensure they are available in your system PATH and runnable.")
        print(FFMPEG_DOWNLOAD_LINKS)
        sys.exit(1)
    if not status.ok:
        detected = "could not detect major version" if status.major is None else f"detected major={status.major}"
        print(f"Error: ffprobe major version must be exactly 8 (or detectable as 8): ffprobe ({detected})")
        print(FFMPEG_DOWNLOAD_LINKS)
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
    from jasna.accelerator import is_amd_device

    if is_amd_device():
        return True, "N/A (AMD)"

    try:
        value = _read_drs_setting(_CUDA_SYSMEM_FALLBACK_POLICY_ID)
    except OSError as e:
        return False, str(e)

    if value == _PREFER_NO_SYSMEM_FALLBACK:
        return True, "Prefer No Sysmem Fallback"
    if value == _PREFER_SYSMEM_FALLBACK:
        return False, "Prefer Sysmem Fallback (recommended: Prefer No Sysmem Fallback)"
    return False, "Driver Default (recommended: Prefer No Sysmem Fallback)"


_AMD_DRIVER_UNCHECKED = (None, "AMD driver version could not be checked; install the ROCm 10.1 driver")


def _windows_amd_driver_status(returncode: int, stdout: str) -> tuple[bool | None, str]:
    """The status of the enabled AMD adapters' Driver Store versions, one per line of ``stdout``."""
    versions = stdout.split()
    if returncode or not versions or any(not re.fullmatch(r"\d+\.\d+\.\d+\.\d+", v) for v in versions):
        return _AMD_DRIVER_UNCHECKED
    supported = {tuple(map(int, version.split("."))) >= MIN_AMD_WINDOWS_DRIVER for version in versions}
    info = "AMD " + ", ".join(sorted(set(versions)))
    if len(supported) != 1:
        return None, f"{info}; mixed driver versions, compatibility could not be checked"
    if False in supported:
        minimum = ".".join(map(str, MIN_AMD_WINDOWS_DRIVER))
        return None, f"{info} (older than {minimum}; if processing fails, install AMD's ROCm 10.1 driver and reboot)"
    return True, info


def _linux_amd_driver_status(version_text: str | None, module_dir_exists: bool) -> tuple[bool | None, str]:
    """The status of the loaded amdgpu module from its ``version`` file (None when unreadable)."""
    if version_text is None:
        # Only AMD's dkms driver publishes a version; the in-kernel driver is checked by GPU availability.
        if module_dir_exists:
            return True, f"in-kernel amdgpu (Linux {platform.release()})"
        return None, "AMD driver version could not be checked (hidden sysfs)"
    # In-kernel drivers have no AMD release number; kernel versions are not comparable.
    match = re.fullmatch(r"\d+\.\d+\.\d+\.(\d{5}|\d{8})", version_text)
    if not match:
        return None, f"amdgpu {version_text}; compatibility could not be checked"
    if int(match[1]) < MIN_AMD_LINUX_DRIVER_RELEASE:
        return None, f"amdgpu {version_text} (older than AMD driver 30.10.2 for ROCm 10.1; if processing fails, update the host driver and reboot)"
    return True, f"amdgpu {version_text}"


def _check_amd_driver_version() -> tuple[bool | None, str]:
    """None is a warning: the driver is older than tested, or its version cannot be compared."""
    if sys.platform == "win32":
        command = (
            "$ErrorActionPreference = 'Stop'; "
            "Get-CimInstance Win32_VideoController | "
            "Where-Object { $_.PNPDeviceID -like 'PCI\\VEN_1002&*' -and $_.ConfigManagerErrorCode -eq 0 } | "
            "Select-Object -ExpandProperty DriverVersion"
        )
        try:
            result = subprocess.run(
                [windows_powershell(), "-NoProfile", "-NonInteractive", "-Command", command],
                capture_output=True, text=True, errors="replace", check=False, timeout=10,
                **subprocess_no_window_kwargs(),
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            logger.warning("Could not query the AMD display driver: %s", error)
            return _AMD_DRIVER_UNCHECKED
        status = _windows_amd_driver_status(result.returncode, result.stdout)
        if status == _AMD_DRIVER_UNCHECKED:
            logger.warning("AMD driver query failed: %s %s", result.stdout, result.stderr)
        return status

    amdgpu_module = Path("/sys/module/amdgpu")
    try:
        version = (amdgpu_module / "version").read_text().strip()
    except OSError as error:
        if not amdgpu_module.is_dir():
            logger.warning("Could not read the loaded AMD driver version: %s", error)
        return _linux_amd_driver_status(None, amdgpu_module.is_dir())
    return _linux_amd_driver_status(version, True)


def check_gpu_driver_version() -> tuple[bool | None, str]:
    """Return pass, failure or an unverified-version warning (None)."""
    import torch

    hip_version = torch.version.hip
    if hip_version:
        driver_ok, driver_info = _check_amd_driver_version()
        if not torch.cuda.is_available():
            return False, f"ROCm {hip_version} is installed but no AMD GPU is available"
        return driver_ok, f"ROCm {hip_version}; {driver_info}"

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


AMF_LINUX_INSTALL_URL = "https://repo.radeon.com/amf/"
AMD_DRIVER_URL = "https://www.amd.com/en/support/download/drivers.html"


def _install_dir() -> Path:
    return Path(sys.executable).parent if is_frozen() else Path(__file__).resolve().parent


def check_ascii_install_path() -> tuple[bool, str]:
    path_str = str(_install_dir())
    try:
        path_str.encode("ascii")
    except UnicodeEncodeError:
        return False, path_str
    return True, path_str


LTX_MIN_LINUX_KERNEL = (5, 8)


def kernel_runs_ltx(release: str) -> bool:
    """Whether the Linux kernel ``release`` (``uname -r``, such as ``5.10.55+``) can run the LTX worker."""
    major, minor = re.match(r"(\d+)\.(\d+)", release).groups()
    return (int(major), int(minor)) >= LTX_MIN_LINUX_KERNEL


def old_linux_kernel_for_ltx() -> str | None:
    """The running Linux kernel's release when it is too old for LTX, else None."""
    if sys.platform != "linux":
        return None
    release = platform.release()
    return None if kernel_runs_ltx(release) else release


def leftover_install_dll() -> str | None:
    """A DLL an older version left in this Windows release folder (a release extracted over an
    older one), which the loader may pick instead of this version's; None when there is none."""
    if sys.platform != "win32" or not is_frozen():
        return None
    from jasna.packaging.windows_dll_paths import ambiguous_dll_names

    ambiguous = ambiguous_dll_names(_install_dir())
    if not ambiguous:
        return None
    logger.warning("DLLs that more than one Jasna folder provides: %s", "; ".join(ambiguous))
    return Path(ambiguous[0].split(" = ")[0]).name


def install_on_hard_disk() -> bool:
    """True when Jasna's folder is on a spinning hard disk (HDD); False on SSDs and when the
    drive type cannot be told (network shares, RAID, virtual disks)."""
    if sys.platform == "win32":
        return _windows_drive_has_seek_penalty(_install_dir())
    device = _linux_block_device(_install_dir())
    return sysfs_block_is_rotational(Path(f"/sys/dev/block/{os.major(device)}:{os.minor(device)}"))


def _linux_block_device(path: Path) -> int:
    """The block device holding ``path``. Btrfs reports an anonymous device (major 0), so then
    the source device of the mount holding ``path`` is used, when it exists here (a container
    sees the host's mount source without its device node)."""
    device = os.stat(path).st_dev
    if os.major(device):
        return device
    source = mount_source(Path("/proc/self/mountinfo").read_text(), path.resolve())
    return os.stat(source).st_rdev if source.startswith("/dev/") and os.path.exists(source) else device


def mount_source(mountinfo: str, path: Path) -> str:
    """Source of the innermost mount holding ``path``, from ``/proc/self/mountinfo`` text."""
    best, source = "", ""
    for line in mountinfo.splitlines():
        fields, _, tail = line.partition(" - ")
        mount_point = fields.split()[4].replace("\\040", " ")
        if path.is_relative_to(mount_point) and len(mount_point) >= len(best):
            best, source = mount_point, tail.split()[1]
    return source


def sysfs_block_is_rotational(block: Path) -> bool:
    """``block`` is a ``/sys/dev/block/MAJOR:MINOR`` link; a partition reports its disk's queue.
    Virtio disks report rotational whatever the host storage is, so they count as unknown."""
    if not block.exists():
        return False
    block = block.resolve()
    if (block / "partition").exists():
        block = block.parent
    if _sysfs_device_is_on_virtio_bus(block):
        return False
    rotational = block / "queue" / "rotational"
    return rotational.exists() and rotational.read_text().strip() == "1"


def _sysfs_device_is_on_virtio_bus(device: Path) -> bool:
    return any(re.fullmatch(r"virtio\d+", part) for part in device.parts)


def _windows_drive_has_seek_penalty(path: Path) -> bool:
    import ctypes
    from ctypes import wintypes

    class StoragePropertyQuery(ctypes.Structure):
        _fields_ = [("PropertyId", ctypes.c_int), ("QueryType", ctypes.c_int),
                    ("AdditionalParameters", ctypes.c_ubyte * 1)]

    class DeviceSeekPenaltyDescriptor(ctypes.Structure):
        _fields_ = [("Version", wintypes.DWORD), ("Size", wintypes.DWORD), ("IncursSeekPenalty", wintypes.BOOLEAN)]

    drive = os.path.splitdrive(str(path))[0]
    if len(drive) != 2:
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                     wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.DeviceIoControl.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                         ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                                         ctypes.c_void_p]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    share_read_write, open_existing = 0x1 | 0x2, 3
    volume = kernel32.CreateFileW(f"\\\\.\\{drive}", 0, share_read_write, None, open_existing, 0, None)
    if volume in (None, wintypes.HANDLE(-1).value):
        return False
    storage_device_seek_penalty_property, property_standard_query = 7, 0
    ioctl_storage_query_property = 0x2D1400
    query = StoragePropertyQuery(storage_device_seek_penalty_property, property_standard_query)
    descriptor = DeviceSeekPenaltyDescriptor()
    returned = wintypes.DWORD()
    try:
        ok = kernel32.DeviceIoControl(volume, ioctl_storage_query_property, ctypes.byref(query),
                                      ctypes.sizeof(query), ctypes.byref(descriptor), ctypes.sizeof(descriptor),
                                      ctypes.byref(returned), None)
        return bool(ok) and bool(descriptor.IncursSeekPenalty)
    finally:
        kernel32.CloseHandle(volume)
